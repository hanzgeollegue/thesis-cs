# Design Patterns In This Project

This document highlights where and how core software design patterns are implemented in the resume-ranking application, with concise code excerpts and file references.

## Facade — Batch Orchestration

The `BatchProcessor` acts as a Facade: one cohesive entry point that orchestrates parsing, scoring (TF‑IDF, SBERT, Cross‑Encoder), meta‑combining, and output assembly.

File: `resume_reviewer/resume_processor/batch_processor.py:168`
```python
class BatchProcessor:
    """Comprehensive batch processor for resume ranking pipeline."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, disable_ocr: bool = False):
        self.pdf_parser = PDFParser(output_dir=output_dir, disable_ocr=disable_ocr)
        llm_cfg = get_llm_config()
        self.api_key = api_key or llm_cfg['api_key']
        self.model = model or llm_cfg['model']
        self.llm_ranker = LLMRanker(api_key=self.api_key, model=self.model) if self.api_key else None
        self.semantic_embedding = SemanticEmbedding()  # SBERT (guarded)
        self.ce_reranker = CEReranker()                # Cross-Encoder (guarded)
```

File: `resume_reviewer/resume_processor/batch_processor.py:311`
```python
def process_batch(self, resumes: List[str], job_description: str, jd_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parsed_resumes = self._parse_pdfs_to_json(resumes)
    # TF‑IDF
    section_tfidf_scores, skill_tfidf_scores = self._compute_section_tfidf_scores(parsed_resumes, jd_sections)
    # SBERT
    sbert_scores = self._compute_sbert_scores(parsed_resumes, job_description)
    # Cross‑Encoder reranking
    ce_results = self._apply_cross_encoder_reranking(
        parsed_resumes, job_description, section_tfidf_scores, skill_tfidf_scores, sbert_scores, ...
    )
    # Assemble final output
    output = self._assemble_output(parsed_resumes, final_ranking, job_description, jd_criteria=jd_criteria)
```

Used consistently from CLI and web flows:
- CLI: `resume_reviewer/resume_processor/management/commands/process_batch.py:65`
- Web: `resume_reviewer/resume_processor/views.py:914`

## Strategy — Pluggable Scoring Components

The scoring layer is decomposed into strategies that share narrow interfaces, allowing combination and substitution without changing the Facade.

### Strategy 1: Section‑Aware TF‑IDF

Invoked via a clear interface from the Facade:

File: `resume_reviewer/resume_processor/batch_processor.py:2140`
```python
from .text_processor import SectionAwareTFIDF
tfidf_processor = SectionAwareTFIDF()
result = tfidf_processor.compute_section_tfidf_scores(resume_dicts, jd_sections)
```

Implemented with section classification and weighted combination:

File: `resume_reviewer/resume_processor/text_processor.py:642`
```python
class SectionAwareTFIDF:
    def __init__(self):
        self.section_weights = {
            'skills': 0.35, 'experience': 0.45, 'education': 0.15, 'summary': 0.25, 'projects': 0.30, ...
        }
        self.vectorizers = {}

    def build_section_vectors(self, resume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # select/create TfidfVectorizer per section_type and fit/transform

    def combine_weighted_vectors(self, section_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        # weighted sum using section_weights, normalized by total weight

    def compute_section_tfidf_scores(self, resumes: List[Dict[str, Any]], jd_sections: Dict[str, str]) -> List[float]:
        # preprocess, build vectors, cosine similarity vs. JD channels
```

### Strategy 2: Semantic Embedding (SBERT)

Encapsulates semantic vector generation and similarity:

File: `resume_reviewer/resume_processor/text_processor.py:1527`
```python
class SemanticEmbedding:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_available = True

    def generate_resume_embedding(self, resume_data: Dict[str, Any]) -> np.ndarray:
        # encode combined resume content

    def calculate_semantic_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        # cosine similarity
```

### Strategy 3: Cross‑Encoder Reranker

Provides a reranking strategy that combines CE score with TF‑IDF and SBERT features:

File: `resume_reviewer/resume_processor/llm_ranker.py:56`
```python
class CEReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.model_name)
        self.model_available = True

    def rerank_candidates(self, job_description, candidates, section_tfidf_scores, sbert_scores) -> List[CERerankerResult]:
        # preprocess JD/candidate text, predict ce_score, combine with other signals, sort
```

Facade invocation:

File: `resume_reviewer/resume_processor/batch_processor.py:2242`
```python
def _apply_cross_encoder_reranking(...):
    result = self.ce_reranker.rerank_candidates(jd_input, resume_dicts, section_tfidf_scores, sbert_scores)
```

## Adapter / Parser Facade — Unified PDF Parsing

`PDFParser` provides a unified interface over multiple extraction backends (PyMuPDF fast path; pdfplumber fallback; optional OCR), adapting different libraries to a single structured output contract.

File: `resume_reviewer/resume_processor/enhanced_pdf_parser.py:237`
```python
class PDFParser:
    def __init__(self, output_dir: str = None, disable_ocr: bool = False):
        # chooses paths; tries PyMuPDF fast path, falls back to pdfplumber
```

Fast path usage and guarded import:

File: `resume_reviewer/resume_processor/enhanced_pdf_parser.py:23,106,496,499`
```python
import fitz  # PyMuPDF  # optional; may be None
USE_PYMUPDF = True
def _extract_text_fast(...):
    if not (USE_PYMUPDF and fitz is not None):
        return None
    doc = fitz.open(pdf_file_path)
    # extract blocks → text elements with (page, x, y)
```

## Provider Strategy — LLM Configuration

The LLM provider is selected via configuration without changing callers (OpenAI vs. Google Gemini), enabling/disablement toggles, and model settings.

File: `resume_reviewer/resume_processor/config.py:1`
```python
def get_llm_config() -> dict:
    if LLM_PROVIDER == 'google':
        return {'provider': 'google', 'api_key': EFFECTIVE_GOOGLE_API_KEY, 'model': GEMINI_MODEL, ...}
    return {'provider': 'openai', 'api_key': OPENAI_API_KEY, 'model': OPENAI_MODEL, ...}
```


## References At A Glance

- Facade: `resume_reviewer/resume_processor/batch_processor.py:168,311`
- Strategy (TF‑IDF): `resume_reviewer/resume_processor/text_processor.py:642`
- Strategy (SBERT): `resume_reviewer/resume_processor/text_processor.py:1527`
- Strategy (Cross‑Encoder): `resume_reviewer/resume_processor/llm_ranker.py:56,88`
- Adapter/Facade (Parser): `resume_reviewer/resume_processor/enhanced_pdf_parser.py:237`
- Provider Strategy: `resume_reviewer/resume_processor/config.py`

