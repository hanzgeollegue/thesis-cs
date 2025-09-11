import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from contextlib import contextmanager

# Version compatibility check
try:
    import sklearn
    sklearn_version = sklearn.__version__
    logger = logging.getLogger(__name__)
    logger.info(f"scikit-learn version: {sklearn_version}")
    
    # Check for known compatibility issues
    if sklearn_version < "0.24.0":
        logger.warning(f"scikit-learn version {sklearn_version} may have compatibility issues. Recommended: >=0.24.0")
        
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"scikit-learn not available: {e}. TF-IDF scoring will use fallback methods.")

# Use conditional imports to work both in Django and standalone contexts
try:
    # Try relative imports first (Django context)
    from .enhanced_pdf_parser import PDFParser
    from .llm_ranker import LLMRanker
    from .text_processor import SemanticEmbedding
    from .config import get_openai_config, get_llm_config, validate_config
except ImportError:
    # Fallback to absolute imports (standalone context)
    try:
        from enhanced_pdf_parser import PDFParser
        from llm_ranker import LLMRanker
        from text_processor import SemanticEmbedding
        from config import get_openai_config, get_llm_config, validate_config
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise

logger = logging.getLogger(__name__)

# --- Timing configuration ---
TIMING_ENABLED = os.getenv("TIMING", "0") in {"1", "true", "True"}

@contextmanager
def time_phase(name: str, bucket: Optional[Dict[str, float]] = None):
    """Context manager for timing phases."""
    if not TIMING_ENABLED:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {name}: {elapsed:.3f}s")
        if bucket is not None:
            bucket[name] = elapsed

@dataclass
class ResumeScores:
    """Data class for resume scoring results."""
    tfidf_section_score: float
    tfidf_taxonomy_score: float
    semantic_score: float
    cross_encoder: float = 0.0
    final_pre_llm: float = 0.0
    final_pre_llm_display: float = 0.0

@dataclass
class ParsedResume:
    """Data class for parsed resume data."""
    id: str
    sections: Dict[str, str]
    meta: Dict[str, Any]
    scores: ResumeScores
    matched_skills: List[Dict[str, str]]
    parsed: Dict[str, Any]

class BatchProcessor:
    """Comprehensive batch processor for resume ranking pipeline."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, disable_ocr: bool = False):
        # Initialize PDF parser with custom output directory to avoid Django settings dependency
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, 'batch_processing_output')
        self.pdf_parser = PDFParser(output_dir=output_dir, disable_ocr=disable_ocr)
        
        # Get provider-agnostic LLM config and apply overrides
        llm_cfg = get_llm_config()
        self.api_key = api_key or llm_cfg['api_key']
        self.model = model or llm_cfg['model']
        
        # Validate configuration
        config_issues = validate_config()
        if config_issues:
            for issue in config_issues:
                logger.warning(issue)
        
        # Initialize LLM ranker
        if self.api_key:
            self.llm_ranker = LLMRanker(api_key=self.api_key, model=self.model)
            logger.info(f"LLM ranking enabled with model: {self.model}")
        else:
            self.llm_ranker = None
            logger.warning("No API key provided. LLM ranking will use fallback methods.")
        
        # Initialize SBERT for semantic embeddings
        try:
            self.semantic_embedding = SemanticEmbedding()
            if self.semantic_embedding.model_available:
                logger.info("SBERT semantic embedding enabled")
            else:
                logger.warning("SBERT not available. Semantic scoring will use fallback methods.")
        except Exception as e:
            logger.warning(f"Failed to initialize SBERT: {e}")
            self.semantic_embedding = None
        
        # Section weights as specified
        self.section_weights = {
            'experience': 0.45,
            'skills': 0.35,
            'education': 0.15,
            'misc': 0.05
        }
        
        # Canonical section mapping
        self.section_mapping = {
            'experience': ['experience', 'work experience', 'employment', 'career history', 'work history', 'professional experience'],
            'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise', 'proficiencies'],
            'education': ['education', 'academic background', 'qualifications', 'academic qualifications'],
            'misc': ['summary', 'profile', 'objective', 'projects', 'certifications', 'awards', 'languages', 'interests', 'volunteer', 'leadership']
        }
        
        # Skill taxonomy (simplified - you can expand this)
        self.skill_taxonomy = {
            'SKILL_001': ['react', 'reactjs', 'react.js'],
            'SKILL_002': ['python', 'python3'],
            'SKILL_003': ['javascript', 'js', 'ecmascript'],
            'SKILL_004': ['java'],
            'SKILL_005': ['sql', 'mysql', 'postgresql', 'database'],
            'SKILL_006': ['aws', 'amazon web services', 'cloud'],
            'SKILL_007': ['docker', 'containerization'],
            'SKILL_008': ['kubernetes', 'k8s'],
            'SKILL_009': ['git', 'version control'],
            'SKILL_010': ['machine learning', 'ml', 'ai', 'artificial intelligence']
        }
        
        # TF-IDF vectorizers for each section
        self.tfidf_vectorizers = {}
        self._initialize_tfidf_vectorizers()
    
    def _initialize_tfidf_vectorizers(self):
        """Initialize TF-IDF vectorizers for each section."""
        for section in self.section_weights.keys():
            try:
                # Try with basic parameters for maximum compatibility
                self.tfidf_vectorizers[section] = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
            except Exception as e:
                logger.warning(f"Error initializing TF-IDF vectorizer for {section}: {e}")
                # Fallback to minimal configuration
                try:
                    self.tfidf_vectorizers[section] = TfidfVectorizer()
                except Exception as e2:
                    logger.error(f"Failed to initialize TF-IDF vectorizer for {section}: {e2}")
                    # Create a dummy vectorizer that won't break the pipeline
                    self.tfidf_vectorizers[section] = None
    
    def process_batch(self, resumes: List[str], job_description: str) -> Dict[str, Any]:
        """Main batch processing pipeline."""
        timing_bucket = {}
        
        try:
            # Validate inputs
            if not job_description.strip():
                return {"error": "job_description_required"}
            
            if len(resumes) > 25:
                return {"error": f"Batch limit exceeded. Maximum 25 resumes allowed, got {len(resumes)}"}
            
            # Step 1: Parse PDFs to JSON
            logger.info(f"Starting batch processing of {len(resumes)} resumes")
            with time_phase("parse_all", timing_bucket):
                parsed_resumes = self._parse_pdfs_to_json(resumes)
            
            # Step 2: Extract job description sections
            jd_sections = self._extract_jd_sections(job_description)
            
            # Step 3: Section-Aware TF-IDF scoring
            logger.info("Computing Section-Aware TF-IDF scores")
            with time_phase("tfidf_fit", timing_bucket):
                self._compute_section_tfidf_scores(parsed_resumes, jd_sections)
            
            # Step 4: Skill-Cluster TF-IDF scoring
            logger.info("Computing Skill-Cluster TF-IDF scores")
            with time_phase("tfidf_score", timing_bucket):
                self._compute_taxonomy_tfidf_scores(parsed_resumes, jd_sections)
            
            # Step 5: Semantic embeddings scoring
            logger.info("Computing semantic similarity scores")
            with time_phase("sbert_encode", timing_bucket):
                self._compute_semantic_scores(parsed_resumes, job_description)
            
            # Step 6: Cross-Encoder reranking + fusion to select top-50
            logger.info("Applying Cross-Encoder reranker and fusion scoring")
            try:
                top50_idx, final_pre = self._apply_cross_encoder(parsed_resumes, job_description)
                # Store final_pre_llm on each resume
                for i, score in enumerate(final_pre):
                    parsed_resumes[i].scores.final_pre_llm = float(score)
                llm_subset = [parsed_resumes[i] for i in top50_idx]
            except Exception as e:
                logger.warning(f"Cross-Encoder reranker failed, falling back to all resumes: {e}")
                llm_subset = parsed_resumes
                for r in parsed_resumes:
                    if not hasattr(r.scores, 'final_pre_llm'):
                        r.scores.final_pre_llm = 0.0

            # Step 7: LLM ranking on selected subset
            logger.info("Generating LLM-based rankings (top-50 subset)")
            with time_phase("llm_rank", timing_bucket):
                final_ranking = self._generate_llm_rankings(llm_subset, job_description)
            
            # Step 8: Assemble final output
            with time_phase("save_results", timing_bucket):
                output = self._assemble_output(parsed_resumes, final_ranking, job_description)
            
            logger.info("Batch processing completed successfully")
            
            # Log batch summary
            if TIMING_ENABLED and timing_bucket:
                total_time = sum(timing_bucket.values())
                logger.info(f"[TIMING] SUMMARY batch({len(resumes)} resumes): {total_time:.3f}s total")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {"error": str(e)}
    
    def _parse_pdfs_to_json(self, resume_paths: List[str]) -> List[ParsedResume]:
        """Parse PDF resumes to structured JSON format."""
        parsed_resumes = []

        if not resume_paths:
            return parsed_resumes

        max_workers = max(1, min(4, len(resume_paths)))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(self._parse_single_resume, path): path for path in resume_paths}

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        parsed_resume = future.result()
                        if parsed_resume:
                            parsed_resumes.append(parsed_resume)
                    except Exception as e:
                        logger.error(f"Error parsing {path}: {str(e)}")
                        # Create a minimal resume entry for failed parses
                        parsed_resumes.append(self._create_fallback_resume(path, str(e)))
        except RuntimeError as e:
            # Common when Django's autoreloader is restarting the interpreter
            logger.warning(f"Thread pool unavailable ({e}); falling back to sequential parsing")
            for path in resume_paths:
                try:
                    parsed_resume = self._parse_single_resume(path)
                    if parsed_resume:
                        parsed_resumes.append(parsed_resume)
                except Exception as ex:
                    logger.error(f"Error parsing {path} sequentially: {str(ex)}")
                    parsed_resumes.append(self._create_fallback_resume(path, str(ex)))

        return parsed_resumes
    
    def _parse_single_resume(self, resume_path: str) -> Optional[ParsedResume]:
        """Parse a single resume PDF."""
        try:
            # Extract structured data using the correct method
            structured_data = self.pdf_parser._extract_structured_data(resume_path)
            
            if not structured_data.get('success', False):
                logger.warning(f"Failed to parse {resume_path}: {structured_data.get('error', 'Unknown error')}")
                return None
            
            # Normalize sections to canonical format
            normalized_sections = self._normalize_sections(structured_data['sections'])
            
            # Create parsed resume object
            resume_id = str(uuid.uuid4())
            filename = os.path.basename(resume_path)
            
            parsed_resume = ParsedResume(
                id=resume_id,
                sections=normalized_sections,
                meta={
                    "source_file": filename,
                    "pages": len(structured_data.get('layout_metadata', {}).get('text_elements', [])),
                    "processing_status": "success"
                },
                scores=ResumeScores(0.0, 0.0, 0.0),
                matched_skills=[],
                parsed=self._extract_parsed_data(structured_data)
            )
            
            return parsed_resume
            
        except Exception as e:
            logger.error(f"Error parsing {resume_path}: {str(e)}")
            return None
    
    def _normalize_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, str]:
        """Normalize section headers to canonical keys."""
        normalized = {
            'experience': '',
            'skills': '',
            'education': '',
            'misc': ''
        }
        
        for section in sections:
            header = section['header'].lower()
            content = ' '.join(section['content'])
            
            # Map to canonical section
            mapped_section = None
            for canonical, aliases in self.section_mapping.items():
                if any(alias in header for alias in aliases):
                    mapped_section = canonical
                    break
            
            if mapped_section:
                normalized[mapped_section] = content
            else:
                # Add to misc if no clear mapping
                if normalized['misc']:
                    normalized['misc'] += ' ' + content
                else:
                    normalized['misc'] = content
        
        return normalized
    
    def _extract_parsed_data(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured parsed data from resume."""
        parsed = {
            'experience': [],
            'skills': [],
            'education': [],
            'misc': ''
        }
        
        # Extract experience (simplified parsing)
        experience_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['experience', 'work', 'employment']):
                experience_section = section
                break
        
        if experience_section:
            # Simple experience parsing (you can enhance this)
            for content in experience_section['content']:
                if content.strip():
                    parsed['experience'].append({
                        'role': 'Software Engineer',  # Placeholder - enhance parsing
                        'company': 'Company Name',   # Placeholder - enhance parsing
                        'dates': '2020-2023',       # Placeholder - enhance parsing
                        'bullets': [content.strip()]
                    })
        
        # Extract skills
        skills_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['skills', 'competencies']):
                skills_section = section
                break
        
        if skills_section:
            for content in skills_section['content']:
                if content.strip():
                    parsed['skills'].append(content.strip())
        
        # Extract education
        education_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['education', 'academic']):
                education_section = section
                break
        
        if education_section:
            for content in education_section['content']:
                if content.strip():
                    parsed['education'].append({
                        'degree': 'Bachelor\'s',  # Placeholder - enhance parsing
                        'institution': 'University',  # Placeholder - enhance parsing
                        'year': '2020'  # Placeholder - enhance parsing
                    })
        
        return parsed
    
    def _create_fallback_resume(self, path: str, error: str) -> ParsedResume:
        """Create a fallback resume entry for failed parses."""
        return ParsedResume(
            id=str(uuid.uuid4()),
            sections={'experience': '', 'skills': '', 'education': '', 'misc': ''},
            meta={
                "source_file": os.path.basename(path),
                "pages": 0,
                "processing_status": "failed",
                "error": error
            },
            scores=ResumeScores(0.0, 0.0, 0.0),
            matched_skills=[],
            parsed={'experience': [], 'skills': [], 'education': [], 'misc': ''}
        )
    
    def _extract_jd_sections(self, job_description: str) -> Dict[str, str]:
        """Extract job description sections via improved header heuristics."""
        if not job_description or not job_description.strip():
            return {'experience': '', 'skills': '', 'education': '', 'misc': ''}

        text = job_description.replace('\r', '\n')
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

        # Map common JD headers to canonical sections
        header_map = {
            'experience': ['experience', 'responsibilities', 'what you will do', 'role', 'duties', 'job description'],
            'skills': ['requirements', 'qualifications', 'skills', 'nice to have', 'must have', 'preferred', 'technical skills', 'technologies'],
            'education': ['education', 'academic', 'degree', 'requirements', 'qualifications'],
        }

        def to_canonical(h: str) -> Optional[str]:
            h_low = h.lower().strip(':').strip('-').strip()
            for canon, keys in header_map.items():
                if any(k in h_low for k in keys):
                    return canon
            return None

        sections = {'experience': [], 'skills': [], 'education': [], 'misc': []}
        current = 'misc'
        
        for ln in lines:
            # Enhanced header detection: look for common patterns
            is_header = False
            
            # Pattern 1: Short lines with common header words
            if len(ln) <= 80 and re.match(r'^[A-Za-z][A-Za-z\s/&-]{2,}$', ln):
                canon = to_canonical(ln)
                if canon:
                    current = canon
                    is_header = True
            
            # Pattern 2: Lines ending with colon
            elif ln.endswith(':'):
                canon = to_canonical(ln[:-1])
                if canon:
                    current = canon
                    is_header = True
            
            # Pattern 3: Lines starting with bullet points and common words
            elif re.match(r'^[-•*]\s*', ln):
                canon = to_canonical(ln[2:].strip())
                if canon:
                    current = canon
                    is_header = True
            
            if not is_header:
                sections[current].append(ln)

        # If no specific sections found, try to distribute content intelligently
        if not any(sections[k] for k in ['experience', 'skills', 'education']):
            # Look for skill-related keywords in the text
            full_text = ' '.join(lines).lower()
            
            # Enhanced skill keywords
            skill_keywords = [
                'python', 'javascript', 'react', 'sql', 'aws', 'git', 'programming', 'development', 
                'database', 'cloud', 'docker', 'kubernetes', 'node.js', 'angular', 'vue', 'typescript',
                'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala',
                'html', 'css', 'bootstrap', 'jquery', 'mongodb', 'postgresql', 'mysql', 'redis',
                'machine learning', 'ai', 'artificial intelligence', 'data science', 'analytics',
                'devops', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'kubernetes', 'microservices'
            ]
            
            # Enhanced experience keywords
            exp_keywords = [
                'years', 'experience', 'develop', 'build', 'create', 'design', 'implement',
                'architect', 'lead', 'manage', 'mentor', 'collaborate', 'deliver', 'deploy',
                'maintain', 'optimize', 'debug', 'test', 'refactor', 'scale', 'performance'
            ]
            
            # Education keywords
            edu_keywords = [
                'degree', 'bachelor', 'master', 'phd', 'university', 'college', 'certification',
                'diploma', 'course', 'training', 'academic', 'education', 'qualification'
            ]
            
            # Distribute content based on keyword presence
            if any(kw in full_text for kw in skill_keywords):
                sections['skills'] = lines
            if any(kw in full_text for kw in exp_keywords):
                sections['experience'] = lines
            if any(kw in full_text for kw in edu_keywords):
                sections['education'] = lines

        return {k: ' '.join(v).strip() for k, v in sections.items()}
    
    def _compute_section_tfidf_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute Section-Aware TF-IDF scores."""
        # Build corpus for each section
        section_corpus = {section: [] for section in self.section_weights.keys()}
        
        # Add JD sections to corpus
        for section, content in jd_sections.items():
            if section in section_corpus:
                section_corpus[section].append(content)
        
        # Add resume sections to corpus
        for resume in resumes:
            for section, content in resume.sections.items():
                if section in section_corpus:
                    section_corpus[section].append(content)
        
        # Debug logging
        logger.info(f"JD sections: {jd_sections}")
        logger.info(f"Section corpus sizes: {[(k, len(v)) for k, v in section_corpus.items()]}")
        logger.info(f"Number of resumes: {len(resumes)}")
        
        # Reset scores before accumulation
        for resume in resumes:
            resume.scores.tfidf_section_score = 0.0

        # Fit TF-IDF models and compute scores (accumulate weighted per-section similarity)
        for section, corpus in section_corpus.items():
            if len(corpus) > 1:  # Need at least 2 documents for TF-IDF
                try:
                    # Check if vectorizer is available
                    if self.tfidf_vectorizers[section] is None:
                        logger.warning(f"TF-IDF vectorizer for {section} is not available, using fallback scoring")
                        # Use fallback scoring
                        for resume in resumes:
                            sim = self._compute_fallback_score(
                                jd_sections[section], resume.sections.get(section, '')
                            )
                            resume.scores.tfidf_section_score += self.section_weights.get(section, 0.0) * sim
                        continue
                    
                    # Fit the vectorizer
                    vectors = self.tfidf_vectorizers[section].fit_transform(corpus)
                    
                    # Compute similarity between JD and each resume
                    jd_vector = vectors[0:1]  # First document is JD
                    resume_vectors = vectors[1:]  # Rest are resumes
                    
                    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
                    
                    # Assign scores to resumes
                    for i, resume in enumerate(resumes):
                        if i < len(similarities):
                            score_contribution = self.section_weights.get(section, 0.0) * float(similarities[i])
                            resume.scores.tfidf_section_score += score_contribution
                        else:
                            resume.scores.tfidf_section_score += 0.0
                    
                    # Log section summary
                    if similarities.size > 0:
                        avg_sim = similarities.mean()
                        max_sim = similarities.max()
                        logger.info(f"Section '{section}' - Avg similarity: {avg_sim:.4f}, Max: {max_sim:.4f}")
                            
                except Exception as e:
                    logger.warning(f"Error computing TF-IDF for {section}: {str(e)}")
                    # Use fallback scoring
                    for resume in resumes:
                        sim = self._compute_fallback_score(
                            jd_sections[section], resume.sections.get(section, '')
                        )
                        resume.scores.tfidf_section_score += self.section_weights.get(section, 0.0) * sim
            else:
                logger.warning(f"Section '{section}' has only {len(corpus)} documents, skipping TF-IDF")
        
        # Log final TF-IDF scores summary
        logger.info("Final TF-IDF Section Scores:")
        for i, resume in enumerate(resumes):
            logger.info(f"  Resume {i}: {resume.scores.tfidf_section_score:.4f}")
    
    def _compute_taxonomy_tfidf_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute Skill-Cluster TF-IDF scores using canonical skill tokens."""
        # Canonicalize skills in resumes and JD
        canon_resumes = []
        for resume in resumes:
            canon_resume = self._canonicalize_skills(resume)
            canon_resumes.append(canon_resume)
        
        canon_jd = self._canonicalize_jd_skills(jd_sections)
        
        # Recompute TF-IDF with canonicalized content
        self._compute_section_tfidf_scores_canonical(canon_resumes, canon_jd)
        
        # Update original resumes with taxonomy scores
        for i, resume in enumerate(resumes):
            if i < len(canon_resumes):
                resume.scores.tfidf_taxonomy_score = canon_resumes[i].scores.tfidf_section_score
                resume.matched_skills = canon_resumes[i].matched_skills
        
        # Log final Taxonomy scores summary
        logger.info("Final TF-IDF Taxonomy Scores:")
        for i, resume in enumerate(resumes):
            logger.info(f"  Resume {i}: {resume.scores.tfidf_taxonomy_score:.4f}")
    
    def _canonicalize_skills(self, resume: ParsedResume) -> ParsedResume:
        """Replace skill phrases with canonical SKILL_ID tokens."""
        canon_resume = ParsedResume(
            id=resume.id,
            sections=resume.sections.copy(),
            meta=resume.meta.copy(),
            scores=resume.scores,
            matched_skills=[],
            parsed=resume.parsed.copy()
        )
        
        # Aggregate unique matched skills and their surface forms
        matched: dict = {}
        
        for section_name, content in resume.sections.items():
            canon_content = content
            for skill_id, surface_forms in self.skill_taxonomy.items():
                for surface_form in surface_forms:
                    if surface_form.lower() in content.lower():
                        # Replace with canonical token
                        canon_content = re.sub(
                            re.escape(surface_form), 
                            skill_id, 
                            canon_content, 
                            flags=re.IGNORECASE
                        )
                        
                        # Track matched skills
                        if skill_id not in matched:
                            matched[skill_id] = set()
                        matched[skill_id].add(surface_form)
            
            canon_resume.sections[section_name] = canon_content
        
        # Convert to list of dicts with deduplicated surface forms
        canon_resume.matched_skills = [
            {'skill_id': sid, 'surface_forms': sorted(list(forms))}
            for sid, forms in matched.items()
        ]
        return canon_resume
    
    def _canonicalize_jd_skills(self, jd_sections: Dict[str, str]) -> Dict[str, str]:
        """Canonicalize skills in job description sections."""
        canon_jd = {}
        for section, content in jd_sections.items():
            canon_content = content
            for skill_id, surface_forms in self.skill_taxonomy.items():
                for surface_form in surface_forms:
                    if surface_form.lower() in content.lower():
                        canon_content = re.sub(
                            re.escape(surface_form), 
                            skill_id, 
                            canon_content, 
                            flags=re.IGNORECASE
                        )
            canon_jd[section] = canon_content
        
        return canon_jd
    
    def _compute_section_tfidf_scores_canonical(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute TF-IDF scores for canonicalized content (true TF-IDF + cosine)."""
        section_corpus: Dict[str, List[str]] = {section: [] for section in self.section_weights.keys()}

        # JD first per section
        for section, content in jd_sections.items():
            if section in section_corpus:
                section_corpus[section].append(content or '')

        # Then resume texts
        for resume in resumes:
            for section, content in resume.sections.items():
                if section in section_corpus:
                    section_corpus[section].append(content or '')

        # Per section TF-IDF + cosine (JD vs resumes)
        per_resume_scores = [0.0] * len(resumes)
        for section, corpus in section_corpus.items():
            if len(corpus) <= 1:
                continue
            try:
                vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.95)
                X = vec.fit_transform(corpus)
                jd_vec = X[0:1]
                res_vecs = X[1:]
                sims = cosine_similarity(jd_vec, res_vecs).flatten()
                weight = self.section_weights.get(section, 0.0)
                for i, s in enumerate(sims):
                    per_resume_scores[i] += weight * float(s)
            except Exception as e:
                logger.warning(f"Canonical TF-IDF error in section '{section}': {e}")
                jd_text = corpus[0] if corpus else ''
                for i, resume in enumerate(resumes):
                    sim = self._compute_text_similarity(jd_text, resume.sections.get(section, ''))
                    per_resume_scores[i] += self.section_weights.get(section, 0.0) * sim

        # Store into the canonical resume objects’ tfidf_section_score (pipeline expects this)
        for i, resume in enumerate(resumes):
            resume.scores.tfidf_section_score = float(min(1.0, per_resume_scores[i]))
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute basic text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_semantic_scores(self, resumes: List[ParsedResume], job_description: str):
        """Compute semantic similarity scores using SBERT embeddings (all-MiniLM-L6-v2, cosine)."""
        from sentence_transformers import SentenceTransformer
        import numpy as np
        try:
            if not hasattr(self, "_sem_model"):
                self._sem_model = SentenceTransformer("all-MiniLM-L6-v2")
            jd_emb = self._sem_model.encode(job_description, normalize_embeddings=True)
            for resume in resumes:
                full_resume_text = ' '.join(resume.sections.values())
                res_emb = self._sem_model.encode(full_resume_text, normalize_embeddings=True)
                resume.scores.semantic_score = float(np.dot(jd_emb, res_emb))
            
            # Log semantic scores summary
            logger.info("Final Semantic Scores:")
            for i, resume in enumerate(resumes):
                logger.info(f"  Resume {i}: {resume.scores.semantic_score:.4f}")
                
        except Exception as e:
            logger.warning(f"Semantic embedding scoring failed, falling back: {e}")
            for resume in resumes:
                resume.scores.semantic_score = self._compute_text_similarity(
                    ' '.join(resume.sections.values()), job_description
                )

    def _z(self, arr):
        import numpy as np
        a = np.asarray(arr, dtype=float)
        return (a - a.mean()) / (a.std() + 1e-6)

    def _apply_cross_encoder(self, resumes: List[ParsedResume], job_description: str):
        from .cross_encoder_reranker import CrossEncoderReranker
        # Preselect top-200 by the 3-signal blend
        base_blend = []
        for r in resumes:
            base_blend.append(0.5*r.scores.semantic_score +
                              0.3*r.scores.tfidf_taxonomy_score +
                              0.2*r.scores.tfidf_section_score)
        # Pick top-200 indexes
        idxs = sorted(range(len(resumes)), key=lambda i: base_blend[i], reverse=True)[:200]
        subset = [resumes[i] for i in idxs]
        texts = [' '.join(r.sections.values()) for r in subset]
        ce = CrossEncoderReranker()
        ce_scores = ce.score_pairs(job_description, texts)
        for r, s in zip(subset, ce_scores):
            setattr(r.scores, 'cross_encoder', s)

        # Compute final_pre_llm across ALL resumes (CE missing -> use min CE)
        all_ce = [getattr(r.scores, 'cross_encoder', None) for r in resumes]
        min_ce = min([s for s in all_ce if s is not None], default=0.0)
        all_ce = [s if s is not None else min_ce for s in all_ce]

        z_ce = self._z(all_ce)
        z_sem = self._z([r.scores.semantic_score for r in resumes])
        z_tax = self._z([r.scores.tfidf_taxonomy_score for r in resumes])
        z_sec = self._z([r.scores.tfidf_section_score for r in resumes])

        final_pre = [0.6*z_ce[i] + 0.2*z_sem[i] + 0.15*z_tax[i] + 0.05*z_sec[i]
                     for i in range(len(resumes))]

        # Keep top-50 for LLM stage
        top50_idx = sorted(range(len(resumes)), key=lambda i: final_pre[i], reverse=True)[:50]
        # Store display-normalized 0-100 for UI
        import numpy as np
        arr = np.asarray(final_pre, dtype=float)
        min_v, max_v = float(arr.min()), float(arr.max())
        rng = max_v - min_v if max_v > min_v else 1.0
        display = [float(100 * (v - min_v) / rng) for v in final_pre]
        for i, r in enumerate(resumes):
            r.scores.final_pre_llm_display = display[i]
        return top50_idx, final_pre
    
    def _generate_llm_rankings(self, resumes: List[ParsedResume], job_description: str) -> List[Dict[str, Any]]:
        """Generate LLM-based rankings for all resumes."""
        try:
            if not self.llm_ranker or not getattr(self.llm_ranker, 'enabled', False):
                logger.warning("LLM ranker not initialized, using fallback ranking")
                return self._fallback_ranking(resumes, job_description)
            
            logger.info("Generating LLM-based rankings...")
            
            # Prepare candidate data for LLM ranking
            candidates_data = []
            for resume in resumes:
                candidate_data = {
                    'candidate_id': resume.id,
                    'resume_data': {
                        'sections': resume.sections,
                        'parsed': resume.parsed,
                        'meta': resume.meta
                    },
                    'computed_scores': {
                        'tfidf_section_score': resume.scores.tfidf_section_score,
                        'tfidf_taxonomy_score': resume.scores.tfidf_taxonomy_score,
                        'semantic_score': resume.scores.semantic_score
                    }
                }
                candidates_data.append(candidate_data)
            
            # Generate rankings using LLM
            ranking_results = self.llm_ranker.rank_candidates(job_description, candidates_data)
            
            # Convert to required format
            final_ranking = []
            for result in ranking_results:
                final_ranking.append({
                    'id': result.candidate_id,
                    'rank': result.rank,
                    'reasoning': result.reasoning,
                    'scores_snapshot': {
                        'tfidf_section': next((r.scores.tfidf_section_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'tfidf_taxonomy': next((r.scores.tfidf_taxonomy_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'semantic': next((r.scores.semantic_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'cross_encoder': next((getattr(r.scores, 'cross_encoder', 0.0) for r in resumes if r.id == result.candidate_id), 0.0),
                        'final_pre_llm': next((getattr(r.scores, 'final_pre_llm', 0.0) for r in resumes if r.id == result.candidate_id), 0.0),
                        'final_pre_llm_display': next((getattr(r.scores, 'final_pre_llm_display', 0.0) for r in resumes if r.id == result.candidate_id), 0.0)
                    }
                })
            
            logger.info(f"LLM ranking completed for {len(final_ranking)} candidates")
            return final_ranking
            
        except Exception as e:
            logger.error(f"Error in LLM ranking: {str(e)}")
            logger.info("Falling back to computed score ranking")
            return self._fallback_ranking(resumes, job_description)
    
    def _fallback_ranking(self, resumes: List[ParsedResume], job_description: str) -> List[Dict[str, Any]]:
        """Generate fallback rankings based on computed scores when LLM is not available."""
        logger.info("Using fallback ranking based on computed scores")
        
        # Calculate composite scores for each resume
        scored_resumes = []
        for resume in resumes:
            # Weighted combination of scores
            composite_score = (
                resume.scores.tfidf_section_score * 0.4 +
                resume.scores.tfidf_taxonomy_score * 0.4 +
                resume.scores.semantic_score * 0.2
            )
            scored_resumes.append((resume, composite_score))
        
        # Sort by composite score (highest first)
        scored_resumes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate ranking
        final_ranking = []
        for i, (resume, score) in enumerate(scored_resumes):
            final_ranking.append({
                'id': resume.id,
                'rank': i + 1,
                'reasoning': f"Ranked based on computed scores: TF-IDF Section ({resume.scores.tfidf_section_score:.2f}), TF-IDF Taxonomy ({resume.scores.tfidf_taxonomy_score:.2f}), Semantic ({resume.scores.semantic_score:.2f})",
                'scores_snapshot': {
                    'tfidf_section': resume.scores.tfidf_section_score,
                    'tfidf_taxonomy': resume.scores.tfidf_taxonomy_score,
                    'semantic': resume.scores.semantic_score
                }
            })
        
        return final_ranking
    
    def _generate_fallback_rankings(self, resumes: List[ParsedResume]) -> List[Dict[str, Any]]:
        """Generate fallback rankings based on computed scores."""
        # Sort by weighted average of scores
        scored_resumes = []
        for resume in resumes:
            weighted_score = (
                0.4 * resume.scores.tfidf_section_score +
                0.3 * resume.scores.tfidf_taxonomy_score +
                0.3 * resume.scores.semantic_score
            )
            scored_resumes.append((resume, weighted_score))
        
        # Sort by score (descending)
        scored_resumes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate ranking
        final_ranking = []
        for i, (resume, score) in enumerate(scored_resumes):
            final_ranking.append({
                'id': resume.id,
                'rank': i + 1,
                'reasoning': f"Ranked based on computed scores: TF-IDF Section ({resume.scores.tfidf_section_score:.2f}), TF-IDF Taxonomy ({resume.scores.tfidf_taxonomy_score:.2f}), Semantic ({resume.scores.semantic_score:.2f})",
                'scores_snapshot': {
                    'tfidf_section': resume.scores.tfidf_section_score,
                    'tfidf_taxonomy': resume.scores.tfidf_taxonomy_score,
                    'semantic': resume.scores.semantic_score
                }
            })
        
        return final_ranking
    
    def _assemble_output(self, resumes: List[ParsedResume], final_ranking: List[Dict[str, Any]], job_description: str) -> Dict[str, Any]:
        """Assemble the final output JSON."""
        # Convert resumes to required format
        resumes_output = []
        for resume in resumes:
            resume_output = {
                'id': resume.id,
                'scores': asdict(resume.scores),
                'matched_skills': resume.matched_skills,
                'parsed': resume.parsed,
                'meta': resume.meta
            }
            resumes_output.append(resume_output)
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(resumes, final_ranking)
        
        # Assemble final output
        output = {
            'job_description_digest': {
                'tokens_summary': f"Job description with {len(job_description.split())} words",
                'top_skills': self._extract_top_skills(job_description),
                'embedding_info': {'model': 'sentence-bert' if self.semantic_embedding and self.semantic_embedding.model_available else 'fallback', 'dim': 384 if self.semantic_embedding and self.semantic_embedding.model_available else 0}
            },
            'resumes': resumes_output,
            'final_ranking': final_ranking,
            'batch_summary': batch_summary
        }
        
        return output
    
    def _generate_batch_summary(self, resumes: List[ParsedResume], final_ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate batch summary information."""
        # Top candidates
        top_candidates = [r['id'] for r in final_ranking[:3]] if final_ranking else []
        
        # Common gaps (simplified analysis)
        common_gaps = []
        if resumes:
            # Check for common missing skills
            missing_skills = set()
            for resume in resumes:
                if not resume.sections['skills'].strip():
                    missing_skills.add('missing skills section')
                if not resume.sections['experience'].strip():
                    missing_skills.add('missing experience section')
            
            common_gaps = list(missing_skills)
        
        # Processing notes
        notes = []
        failed_count = sum(1 for r in resumes if r.meta.get('processing_status') == 'failed')
        if failed_count > 0:
            notes.append(f"{failed_count} resumes failed to process")
        
        if not notes:
            notes.append("All resumes processed successfully")
        
        return {
            'top_candidates': top_candidates,
            'common_gaps': common_gaps,
            'notes': '; '.join(notes)
        }
    
    def _extract_top_skills(self, job_description: str) -> List[Dict[str, str]]:
        """Extract top skills from job description."""
        top_skills = []
        for skill_id, surface_forms in self.skill_taxonomy.items():
            for surface_form in surface_forms:
                if surface_form.lower() in job_description.lower():
                    top_skills.append({
                        'skill_id': skill_id,
                        'label': surface_forms[0].title()
                    })
                    break  # Only add each skill once
        
        return top_skills[:10]  # Limit to top 10 

    def _compute_fallback_score(self, jd_content: str, resume_content: str) -> float:
        """Compute a fallback similarity score when TF-IDF is not available."""
        try:
            # Simple text similarity as fallback
            return self._compute_text_similarity(jd_content, resume_content)
        except Exception as e:
            logger.warning(f"Fallback scoring failed: {e}")
            return 0.0 