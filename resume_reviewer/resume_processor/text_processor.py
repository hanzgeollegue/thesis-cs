import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)

class SectionAwareTFIDF:
    """Section-aware TF-IDF implementation for resume processing."""
    
    def __init__(self):
        # Section weights for different resume sections
        self.section_weights = {
            'skills': 0.35,
            'experience': 0.45,
            'education': 0.15,
            'summary': 0.25,
            'projects': 0.30,
            'certifications': 0.20,
            'awards': 0.15,
            'languages': 0.10,
            'interests': 0.05,
            'volunteer': 0.15,
            'leadership': 0.20,
            'availability': 0.05,
            'references': 0.05,
            'contact': 0.00,  # No weight for contact info
            'default': 0.10   # Default weight for unknown sections
        }
        
        # TF-IDF vectorizers for different section types
        self.vectorizers = {}
        self.section_vectors = {}
        
    def build_section_vectors(self, resume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Build TF-IDF vectors for each section of a resume."""
        sections = resume_data.get('sections', [])
        section_vectors = {}
        
        for section in sections:
            header = section['header'].lower()
            content = section['content']
            
            # Determine section type
            section_type = self._classify_section(header)
            
            # Combine content into single text
            section_text = ' '.join(content)
            
            # Create or get vectorizer for this section type
            if section_type not in self.vectorizers:
                self.vectorizers[section_type] = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
            
            # Fit and transform the section text
            try:
                vector = self.vectorizers[section_type].fit_transform([section_text])
                section_vectors[section_type] = vector.toarray()[0]
            except Exception as e:
                logger.warning(f"Error vectorizing section {section_type}: {str(e)}")
                section_vectors[section_type] = np.zeros(1000)  # Default empty vector
        
        return section_vectors
    
    def _classify_section(self, header: str) -> str:
        """Classify a section header into a standard category."""
        header_lower = header.lower()
        
        # Map headers to standard categories
        if any(word in header_lower for word in ['skill', 'competency', 'expertise', 'proficiency']):
            return 'skills'
        elif any(word in header_lower for word in ['experience', 'employment', 'work', 'career']):
            return 'experience'
        elif any(word in header_lower for word in ['education', 'academic', 'qualification']):
            return 'education'
        elif any(word in header_lower for word in ['summary', 'profile', 'objective']):
            return 'summary'
        elif any(word in header_lower for word in ['project']):
            return 'projects'
        elif any(word in header_lower for word in ['certification', 'certificate', 'license']):
            return 'certifications'
        elif any(word in header_lower for word in ['award', 'honor', 'achievement']):
            return 'awards'
        elif any(word in header_lower for word in ['language']):
            return 'languages'
        elif any(word in header_lower for word in ['interest', 'hobby']):
            return 'interests'
        elif any(word in header_lower for word in ['volunteer']):
            return 'volunteer'
        elif any(word in header_lower for word in ['leadership']):
            return 'leadership'
        elif any(word in header_lower for word in ['availability']):
            return 'availability'
        elif any(word in header_lower for word in ['reference', 'referee']):
            return 'references'
        elif any(word in header_lower for word in ['contact', 'personal']):
            return 'contact'
        else:
            return 'default'
    
    def combine_weighted_vectors(self, section_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine section vectors with appropriate weights."""
        if not section_vectors:
            return np.zeros(1000)
        
        # Get the maximum vector length
        max_length = max(len(vector) for vector in section_vectors.values())
        
        # Initialize combined vector
        combined_vector = np.zeros(max_length)
        total_weight = 0
        
        for section_type, vector in section_vectors.items():
            weight = self.section_weights.get(section_type, self.section_weights['default'])
            
            # Pad or truncate vector to match max_length
            if len(vector) < max_length:
                padded_vector = np.pad(vector, (0, max_length - len(vector)), 'constant')
            else:
                padded_vector = vector[:max_length]
            
            combined_vector += weight * padded_vector
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_vector /= total_weight
        
        return combined_vector
    
    def calculate_similarity(self, resume1_vectors: Dict[str, np.ndarray], 
                           resume2_vectors: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between two resumes using weighted section vectors."""
        combined1 = self.combine_weighted_vectors(resume1_vectors)
        combined2 = self.combine_weighted_vectors(resume2_vectors)
        
        # Reshape for cosine similarity
        combined1 = combined1.reshape(1, -1)
        combined2 = combined2.reshape(1, -1)
        
        similarity = cosine_similarity(combined1, combined2)[0][0]
        return similarity


class SkillTaxonomy:
    """Skill taxonomy integration for normalizing skill variations."""
    
    def __init__(self):
        # Skill taxonomy mapping (simplified version - can be expanded)
        self.skill_mapping = {
            # Programming Languages
            'python': ['python', 'python3', 'python 3', 'py'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es2015'],
            'java': ['java', 'j2ee', 'j2se'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp'],
            'php': ['php', 'php7', 'php8'],
            'ruby': ['ruby', 'ruby on rails', 'rails'],
            'go': ['go', 'golang'],
            'rust': ['rust'],
            'swift': ['swift', 'swiftui'],
            'kotlin': ['kotlin'],
            'scala': ['scala'],
            
            # Frameworks and Libraries
            'react': ['react', 'reactjs', 'react.js', 'reactjs', 'react native'],
            'angular': ['angular', 'angularjs', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'node': ['node', 'nodejs', 'node.js', 'express'],
            'django': ['django', 'django framework'],
            'flask': ['flask'],
            'spring': ['spring', 'spring boot', 'spring framework'],
            'laravel': ['laravel'],
            'asp.net': ['asp.net', 'aspnet', 'asp .net'],
            'jquery': ['jquery', 'jquery.js'],
            'bootstrap': ['bootstrap', 'bootstrap css'],
            'tailwind': ['tailwind', 'tailwind css'],
            
            # Databases
            'mysql': ['mysql', 'mariadb'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'mongodb': ['mongodb', 'mongo'],
            'redis': ['redis'],
            'sqlite': ['sqlite', 'sqlite3'],
            'oracle': ['oracle', 'oracle db'],
            'sql server': ['sql server', 'mssql', 'microsoft sql server'],
            
            # Cloud and DevOps
            'aws': ['aws', 'amazon web services', 'amazon aws'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'docker': ['docker', 'docker container'],
            'kubernetes': ['kubernetes', 'k8s'],
            'jenkins': ['jenkins'],
            'git': ['git', 'github', 'gitlab', 'bitbucket'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment'],
            
            # Data Science and ML
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
            'pandas': ['pandas', 'pd'],
            'numpy': ['numpy', 'np'],
            'matplotlib': ['matplotlib', 'plt'],
            'seaborn': ['seaborn'],
            'jupyter': ['jupyter', 'jupyter notebook'],
            
            # Other Technologies
            'html': ['html', 'html5'],
            'css': ['css', 'css3'],
            'sass': ['sass', 'scss'],
            'less': ['less'],
            'webpack': ['webpack'],
            'babel': ['babel'],
            'typescript': ['typescript', 'ts'],
            'graphql': ['graphql'],
            'rest': ['rest', 'rest api', 'restful'],
            'soap': ['soap', 'soap api'],
            'microservices': ['microservices', 'micro service'],
            'api': ['api', 'apis'],
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for canonical, variations in self.skill_mapping.items():
            for variation in variations:
                self.reverse_mapping[variation.lower()] = canonical
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill to its canonical form."""
        skill_lower = skill.lower().strip()
        
        # Direct match
        if skill_lower in self.reverse_mapping:
            return self.reverse_mapping[skill_lower]
        
        # Partial match (for multi-word skills)
        for variation, canonical in self.reverse_mapping.items():
            if skill_lower in variation or variation in skill_lower:
                return canonical
        
        # Return original if no match found
        return skill
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract and normalize skills from text content."""
        # Split text into words and phrases
        words = re.findall(r'\b\w+(?:\s+\w+)*\b', text.lower())
        
        skills = []
        for word in words:
            normalized = self.normalize_skill(word)
            if normalized != word:  # Only add if normalization occurred
                skills.append(normalized)
        
        return list(set(skills))  # Remove duplicates
    
    def get_skill_taxonomy_score(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill taxonomy overlap score."""
        if not resume_skills or not job_skills:
            return 0.0
        
        # Normalize all skills
        normalized_resume = [self.normalize_skill(skill) for skill in resume_skills]
        normalized_job = [self.normalize_skill(skill) for skill in job_skills]
        
        # Calculate overlap
        overlap = set(normalized_resume) & set(normalized_job)
        total_required = len(set(normalized_job))
        
        if total_required == 0:
            return 0.0
        
        return len(overlap) / total_required


class SemanticEmbedding:
    """Semantic embedding generation for resume content."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_available = True
        except ImportError:
            logger.warning("SentenceTransformer not available. Install with: pip install sentence-transformers")
            self.model_available = False
            self.model = None
    
    def generate_section_embeddings(self, resume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate embeddings for each section of a resume."""
        if not self.model_available:
            logger.error("SentenceTransformer model not available")
            return {}
        
        sections = resume_data.get('sections', [])
        embeddings = {}
        
        for section in sections:
            header = section['header']
            content = section['content']
            
            # Combine content into single text
            section_text = ' '.join(content)
            
            if section_text.strip():
                try:
                    # Generate embedding
                    embedding = self.model.encode(section_text)
                    embeddings[header] = embedding
                except Exception as e:
                    logger.warning(f"Error generating embedding for section {header}: {str(e)}")
        
        return embeddings
    
    def generate_resume_embedding(self, resume_data: Dict[str, Any]) -> np.ndarray:
        """Generate a single embedding for the entire resume."""
        if not self.model_available:
            logger.error("SentenceTransformer model not available")
            return np.zeros(384)  # Default embedding size
        
        # Combine all content
        all_content = []
        for section in resume_data.get('sections', []):
            all_content.extend(section['content'])
        
        combined_text = ' '.join(all_content)
        
        if combined_text.strip():
            try:
                embedding = self.model.encode(combined_text)
                return embedding
            except Exception as e:
                logger.error(f"Error generating resume embedding: {str(e)}")
                return np.zeros(384)
        
        return np.zeros(384)
    
    def calculate_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate semantic similarity between two embeddings."""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Ensure same dimensions
        if embedding1.size != embedding2.size:
            min_size = min(embedding1.size, embedding2.size)
            embedding1 = embedding1[:min_size]
            embedding2 = embedding2[:min_size]
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)


class ResumeProcessor:
    """Main processor that combines all text processing components."""
    
    def __init__(self):
        self.tfidf_processor = SectionAwareTFIDF()
        self.skill_taxonomy = SkillTaxonomy()
        self.semantic_embedding = SemanticEmbedding()
    
    def process_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a resume through all text processing components."""
        processed_data = {
            'original_data': resume_data,
            'tfidf_vectors': {},
            'skill_analysis': {},
            'semantic_embeddings': {},
            'combined_scores': {}
        }
        
        try:
            # Generate TF-IDF vectors
            section_vectors = self.tfidf_processor.build_section_vectors(resume_data)
            combined_vector = self.tfidf_processor.combine_weighted_vectors(section_vectors)
            processed_data['tfidf_vectors'] = {
                'section_vectors': section_vectors,
                'combined_vector': combined_vector.tolist()
            }
            
            # Extract and normalize skills
            all_content = []
            for section in resume_data.get('sections', []):
                all_content.extend(section['content'])
            
            combined_text = ' '.join(all_content)
            extracted_skills = self.skill_taxonomy.extract_skills_from_text(combined_text)
            
            processed_data['skill_analysis'] = {
                'extracted_skills': extracted_skills,
                'skill_count': len(extracted_skills)
            }
            
            # Generate semantic embeddings
            section_embeddings = self.semantic_embedding.generate_section_embeddings(resume_data)
            resume_embedding = self.semantic_embedding.generate_resume_embedding(resume_data)
            
            processed_data['semantic_embeddings'] = {
                'section_embeddings': {k: v.tolist() for k, v in section_embeddings.items()},
                'resume_embedding': resume_embedding.tolist()
            }
            
            # Calculate combined scores
            processed_data['combined_scores'] = {
                'tfidf_score': float(np.linalg.norm(combined_vector)),
                'skill_diversity': len(extracted_skills),
                'semantic_richness': float(np.linalg.norm(resume_embedding))
            }
            
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            processed_data['error'] = str(e)
        
        return processed_data
    
    def compare_resumes(self, resume1_data: Dict[str, Any], resume2_data: Dict[str, Any]) -> Dict[str, float]:
        """Compare two resumes using all similarity metrics."""
        # Process both resumes
        processed1 = self.process_resume(resume1_data)
        processed2 = self.process_resume(resume2_data)
        
        comparison_scores = {}
        
        try:
            # TF-IDF similarity
            if 'tfidf_vectors' in processed1 and 'tfidf_vectors' in processed2:
                tfidf_sim = self.tfidf_processor.calculate_similarity(
                    processed1['tfidf_vectors']['section_vectors'],
                    processed2['tfidf_vectors']['section_vectors']
                )
                comparison_scores['tfidf_similarity'] = tfidf_sim
            
            # Skill taxonomy similarity
            skills1 = processed1['skill_analysis'].get('extracted_skills', [])
            skills2 = processed2['skill_analysis'].get('extracted_skills', [])
            skill_sim = self.skill_taxonomy.get_skill_taxonomy_score(skills1, skills2)
            comparison_scores['skill_similarity'] = skill_sim
            
            # Semantic similarity
            if 'semantic_embeddings' in processed1 and 'semantic_embeddings' in processed2:
                emb1 = np.array(processed1['semantic_embeddings']['resume_embedding'])
                emb2 = np.array(processed2['semantic_embeddings']['resume_embedding'])
                semantic_sim = self.semantic_embedding.calculate_semantic_similarity(emb1, emb2)
                comparison_scores['semantic_similarity'] = semantic_sim
            
            # Overall similarity (weighted average)
            similarities = [v for v in comparison_scores.values() if isinstance(v, (int, float))]
            if similarities:
                comparison_scores['overall_similarity'] = sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.error(f"Error comparing resumes: {str(e)}")
            comparison_scores['error'] = str(e)
        
        return comparison_scores 