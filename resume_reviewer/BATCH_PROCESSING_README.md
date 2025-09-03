# Batch Resume Processing Pipeline

This document describes the comprehensive batch processing pipeline for processing up to 25 PDF resumes against a job description, computing multiple scoring metrics, and generating AI-powered rankings.

## Overview

The pipeline implements a multi-stage approach to resume ranking:

1. **PDF Parsing** → Structured JSON with normalized sections
2. **Section-Aware TF-IDF** → Lexical matching with weighted sections
3. **Skill-Cluster TF-IDF** → Taxonomy-based skill matching
4. **Semantic Embeddings** → Meaning-based similarity
5. **LLM Ranking** → AI-powered final ranking with reasoning

## Features

- **Batch Processing**: Handle up to 25 resumes simultaneously
- **Multi-Modal Scoring**: Combine lexical, semantic, and AI-based approaches
- **Section Normalization**: Automatically map various section headers to canonical keys
- **Skill Taxonomy**: ESCO/O*NET-style skill clustering and matching
- **Parallel Processing**: Efficient handling of multiple PDFs
- **Comprehensive Output**: Machine-readable JSON + human summary
- **Error Handling**: Graceful fallbacks and detailed error reporting

## Architecture

### Core Components

- **`BatchProcessor`**: Main orchestration class
- **`PDFParser`**: Enhanced PDF parsing with layout analysis
- **`LLMRanker`**: GPT-based ranking and reasoning
- **Section Normalizers**: Map various headers to canonical sections
- **TF-IDF Engines**: Section-specific vectorization and scoring

### Section Mapping

The pipeline automatically normalizes section headers:

| Canonical Section | Common Aliases |
|------------------|----------------|
| `experience` | Work Experience, Employment, Career History, Work History |
| `skills` | Technical Skills, Core Competencies, Expertise, Proficiencies |
| `education` | Education, Academic Background, Qualifications |
| `misc` | Summary, Profile, Projects, Certifications, Awards, etc. |

### Scoring Weights

- **Experience**: 45% (most important)
- **Skills**: 35% (technical requirements)
- **Education**: 15% (qualifications)
- **Misc**: 5% (additional context)

## Usage

### 1. Django Management Command

```bash
python manage.py process_batch \
    --resumes resume1.pdf resume2.pdf resume3.pdf \
    --job-description "Software Engineer position..." \
    --output results.json \
    --api-key your_openai_api_key
```

### 2. Python API

```python
from resume_processor.batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(api_key="your_api_key")

# Process batch
results = processor.process_batch(
    resume_paths=["resume1.pdf", "resume2.pdf"],
    job_description="Job description text..."
)

# Access results
for resume in results['resumes']:
    print(f"Resume {resume['id']}: {resume['scores']}")
```

### 3. Test Script

```bash
python test_batch_processor.py
```

## Output Format

### Complete JSON Schema

```json
{
  "job_description_digest": {
    "tokens_summary": "Job description with 150 words",
    "top_skills": [
      {"skill_id": "SKILL_001", "label": "React"}
    ],
    "embedding_info": {"model": "sentence-bert", "dim": 768}
  },
  "resumes": [
    {
      "id": "uuid-123",
      "scores": {
        "tfidf_section_score": 0.85,
        "tfidf_taxonomy_score": 0.78,
        "semantic_score": 0.92
      },
      "matched_skills": [
        {"skill_id": "SKILL_001", "surface_forms": ["React", "ReactJS"]}
      ],
      "parsed": {
        "experience": [
          {
            "role": "Software Engineer",
            "company": "Tech Corp",
            "dates": "2020-2023",
            "bullets": ["Developed React applications"]
          }
        ],
        "skills": ["React", "Python", "SQL"],
        "education": [
          {
            "degree": "Bachelor's",
            "institution": "University",
            "year": "2020"
          }
        ],
        "misc": "Additional information..."
      },
      "meta": {
        "source_file": "resume.pdf",
        "pages": 2,
        "processing_status": "success"
      }
    }
  ],
  "final_ranking": [
    {
      "id": "uuid-123",
      "rank": 1,
      "reasoning": "Strong technical skills match with excellent experience...",
      "scores_snapshot": {
        "tfidf_section": 0.85,
        "tfidf_taxonomy": 0.78,
        "semantic": 0.92
      }
    }
  ],
  "batch_summary": {
    "top_candidates": ["uuid-123", "uuid-456", "uuid-789"],
    "common_gaps": ["missing cloud experience", "no production React"],
    "notes": "All resumes processed successfully"
  }
}
```

## Implementation Details

### TF-IDF Scoring

1. **Section-Aware**: Separate TF-IDF models for each canonical section
2. **Vocabulary Consistency**: Fixed vocabulary across batch for fair comparison
3. **Cosine Similarity**: Normalized scores in [0,1] range
4. **Weighted Aggregation**: Apply section-specific weights

### Skill Taxonomy

- **Canonical IDs**: SKILL_001, SKILL_002, etc.
- **Surface Forms**: Multiple ways to express the same skill
- **Longest Match**: Prefer exact matches over fuzzy matching
- **Fallback**: Unmatched skills remain as raw tokens

### Semantic Scoring

- **Sentence-BERT**: Default embedding model
- **Fallback**: Text similarity when embeddings unavailable
- **Normalization**: Cosine similarity in [0,1] range
- **Performance**: Sliding window + mean-pooling for long documents

### LLM Ranking

- **GPT-4**: Primary ranking model
- **Rubric**: Semantic alignment → Skill coverage → Evidence density
- **Tie-breakers**: Experience years, seniority, recency
- **Fallback**: Score-based ranking when LLM unavailable

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_api_key_here
DJANGO_SETTINGS_MODULE=resume_reviewer.settings
```

### Customization

```python
# Custom section weights
processor.section_weights = {
    'experience': 0.50,
    'skills': 0.30,
    'education': 0.15,
    'misc': 0.05
}

# Custom skill taxonomy
processor.skill_taxonomy = {
    'SKILL_CUSTOM': ['custom skill', 'alternative name']
}
```

## Performance Considerations

- **Parallel Processing**: Uses ThreadPoolExecutor for PDF parsing
- **Memory Management**: Processes resumes in batches
- **Caching**: TF-IDF models are fitted once per batch
- **Error Handling**: Continues processing even if individual resumes fail

## Error Handling

### Common Issues

1. **PDF Parsing Failures**: OCR fallback, graceful degradation
2. **Missing Sections**: Treated as empty strings, no penalties
3. **API Failures**: Fallback to score-based ranking
4. **File Issues**: Detailed error reporting per resume

### Validation

- **Input Validation**: File existence, format checking
- **Batch Limits**: Maximum 25 resumes enforced
- **Required Fields**: Job description must be provided
- **Output Validation**: Ensures valid JSON schema

## Testing

### Unit Tests

```bash
python manage.py test resume_processor
```

### Integration Tests

```bash
python test_batch_processor.py
```

### Performance Tests

```bash
# Test with different batch sizes
python manage.py process_batch --resumes $(ls *.pdf | head -10) --job-description "test"
```

## Future Enhancements

1. **Advanced JD Parsing**: Extract structured requirements
2. **Enhanced Skill Taxonomy**: Integration with ESCO/O*NET APIs
3. **Multi-Language Support**: International resume processing
4. **Real-time Processing**: WebSocket-based progress updates
5. **Custom Scoring**: User-defined scoring algorithms

## Troubleshooting

### Common Problems

1. **Memory Issues**: Reduce batch size, increase system memory
2. **API Rate Limits**: Add delays between LLM calls
3. **PDF Quality**: Ensure readable text, not just images
4. **Section Detection**: Check section header normalization

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger('resume_processor')
logger.setLevel(logging.DEBUG)
```

## Support

For issues and questions:

1. Check the debug logs in `debug.log`
2. Review the test examples
3. Verify input file formats
4. Check API key configuration

## License

This pipeline is part of the resume reviewer thesis project. 