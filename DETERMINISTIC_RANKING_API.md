# Deterministic Ranking Engine API

## Overview

A strict, deterministic ranking engine for resumes that processes only numeric features and limited metadata. Returns standardized rankings according to a precise algorithm with deterministic tie-breaking.

## API Endpoint

```
POST /resume/api/deterministic-ranking/
Content-Type: application/json
```

## Algorithm

1. **Normalize** raw scores to [0,1] using batch min/max per metric
2. **Calculate consistency bonus** based on score variance
3. **Compute base composite** score using weighted sum
4. **Apply penalties** for missing skills and low evidence
5. **Apply hard rules** (optional disqualification)
6. **Sort deterministically** with tie-breaking rules
7. **Scale to 0-100** and round to 2 decimals

## Input Schema

```json
{
  "job_profile": {
    "title": "Senior Software Engineer",
    "must_have_skills": ["python", "react", "sql"],
    "nice_to_have_skills": ["aws", "docker", "kubernetes"]
  },
  "weights": {
    "section_tfidf": 0.4,
    "skill_tfidf": 0.3,
    "sbert": 0.2,
    "consistency_bonus": 0.1
  },
  "penalties": {
    "missing_must_have": 0.1,
    "low_evidence": 0.05
  },
  "hard_rules": {
    "disqualify_if_missing_must_have": false
  },
  "batch_stats": {
    "section_tfidf_min": 0.1,
    "section_tfidf_max": 0.9,
    "skill_tfidf_min": 0.0,
    "skill_tfidf_max": 1.0,
    "sbert_min": 0.2,
    "sbert_max": 0.8
  },
  "candidates": [
    {
      "candidate_id": "candidate_001",
      "raw_scores": {
        "section_tfidf_overall": 0.8,
        "section_tfidf_by_section": {
          "experience": 0.85,
          "skills": 0.75,
          "education": 0.65,
          "projects": 0.70
        },
        "skill_tfidf_overall": 0.9,
        "skill_tfidf_by_cluster": {
          "python": 0.95,
          "react": 0.85,
          "sql": 0.90,
          "aws": 0.60
        },
        "sbert_overall": 0.75
      },
      "meta": {
        "missing_must_have_count": 0,
        "low_evidence": false,
        "years_experience": 5,
        "seniority_hint": "senior"
      }
    }
  ]
}
```

## Output Schema

```json
{
  "job_title": "Senior Software Engineer",
  "ranked_candidates": [
    {
      "candidate_id": "candidate_001",
      "composite_score": 85.75,
      "normalized": {
        "section_tfidf": 87.50,
        "skill_tfidf": 90.00,
        "sbert": 91.67
      },
      "penalties_applied": {
        "missing_must_have_count": 0,
        "low_evidence": false,
        "penalty_points": 0.00
      },
      "disqualified": false,
      "rationale": "S=0.88,K=0.90,B=0.92,0 missing must-have. Composite: 85.75/100."
    }
  ],
  "batch_notes": {
    "normalization_used": {
      "section_tfidf_min": 0.1,
      "section_tfidf_max": 0.9,
      "skill_tfidf_min": 0.0,
      "skill_tfidf_max": 1.0,
      "sbert_min": 0.2,
      "sbert_max": 0.8
    },
    "weights_used": {
      "section_tfidf": 0.4,
      "skill_tfidf": 0.3,
      "sbert": 0.2,
      "consistency_bonus": 0.1
    },
    "hard_rules": {
      "disqualify_if_missing_must_have": false
    }
  }
}
```

## Tie-Breaking Rules (Applied in Order)

1. **Higher composite score** (primary sort)
2. **Higher SBERT normalized score**
3. **Higher experience section TF-IDF** (raw score)
4. **Higher skill TF-IDF overall** (raw score)
5. **Lower missing must-have count**
6. **Alphabetical by candidate_id**

## Usage Example

### Using curl

```bash
curl -X POST http://127.0.0.1:8000/resume/api/deterministic-ranking/ \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Using Python

```python
import requests
import json

payload = {
    "job_profile": {...},
    "weights": {...},
    # ... rest of payload
}

response = requests.post(
    'http://127.0.0.1:8000/resume/api/deterministic-ranking/',
    json=payload
)

results = response.json()
```

## Testing

Run the test script to verify functionality:

```bash
cd resume_reviewer
python test_deterministic_ranking.py
```

## Features

- **Strict deterministic sorting** - Same input always produces same output
- **Batch normalization** - Fair comparison within each batch
- **Configurable weights** - Adjust importance of different metrics
- **Penalty system** - Deduct points for missing requirements
- **Hard disqualification rules** - Optional strict filtering
- **Consistent API** - Always returns valid JSON matching schema
- **No bias** - Uses only provided numeric features

## Error Handling

- Invalid JSON returns schema-compliant error response
- Missing fields are treated as zero values
- Min/max equality results in 0.5 normalization
- All scores are clamped to valid ranges

## Performance

- Processes batches of any size efficiently
- Deterministic sorting algorithm
- Minimal memory footprint
- Fast numeric computations only
