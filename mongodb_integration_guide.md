# MongoDB Integration Guide for Resume Processing Pipeline

## Overview
This guide outlines how to integrate MongoDB with the complete resume processing pipeline, replacing the current SQLite database.

## MongoDB Schema Design

### 1. Resume Collection
```javascript
{
  "_id": ObjectId,
  "candidate_id": String,
  "filename": String,
  "original_file_path": String,
  "uploaded_at": Date,
  "processed_at": Date,
  "processing_status": String, // "pending", "processing", "completed", "failed"
  "error_message": String,
  
  // Parsed data
  "parsed_data": {
    "success": Boolean,
    "sections": [
      {
        "header": String,
        "content": [String]
      }
    ],
    "summary": {
      "total_sections": Number,
      "section_names": [String],
      "has_contact_info": Boolean,
      "has_education": Boolean,
      "has_experience": Boolean,
      "has_skills": Boolean
    },
    "layout_metadata": {
      "text_elements": [
        {
          "text": String,
          "page": Number,
          "x": Number,
          "y": Number,
          "font_size": Number,
          "font_name": String,
          "is_bold": Boolean,
          "is_italic": Boolean
        }
      ],
      "font_statistics": {
        "mean_font_size": Number,
        "median_font_size": Number,
        "max_font_size": Number,
        "min_font_size": Number,
        "bold_elements_count": Number,
        "italic_elements_count": Number
      },
      "layout_analysis": {
        "total_elements": Number,
        "estimated_columns": Number,
        "column_boundaries": [Number]
      }
    }
  },
  
  // Processed data
  "processed_data": {
    "tfidf_vectors": {
      "section_vectors": Object, // Map of section types to vectors
      "combined_vector": [Number]
    },
    "skill_analysis": {
      "extracted_skills": [String],
      "skill_count": Number
    },
    "semantic_embeddings": {
      "section_embeddings": Object, // Map of section headers to embeddings
      "resume_embedding": [Number]
    },
    "combined_scores": {
      "tfidf_score": Number,
      "skill_diversity": Number,
      "semantic_richness": Number
    }
  },
  
  // Ranking data
  "ranking_data": {
    "rankings": [
      {
        "job_id": String,
        "score": Number,
        "rank": Number,
        "reasoning": String,
        "strengths": [String],
        "gaps": [String],
        "confidence": Number,
        "ranked_at": Date
      }
    ]
  }
}
```

### 2. Job Postings Collection
```javascript
{
  "_id": ObjectId,
  "job_id": String,
  "title": String,
  "company": String,
  "description": String,
  "requirements": [String],
  "nice_to_have": [String],
  "created_at": Date,
  "status": String, // "active", "closed", "draft"
  
  // Job analysis
  "job_analysis": {
    "required_skills": [String],
    "experience_level": String,
    "education_requirements": [String],
    "semantic_embedding": [Number]
  }
}
```

### 3. Ranking Sessions Collection
```javascript
{
  "_id": ObjectId,
  "session_id": String,
  "job_id": String,
  "job_description": String,
  "candidates_count": Number,
  "created_at": Date,
  "status": String, // "processing", "completed", "failed"
  
  "ranking_results": [
    {
      "candidate_id": String,
      "rank": Number,
      "score": Number,
      "reasoning": String,
      "strengths": [String],
      "gaps": [String],
      "confidence": Number
    }
  ],
  
  "summary": {
    "average_score": Number,
    "score_distribution": {
      "excellent": Number,
      "good": Number,
      "average": Number,
      "below_average": Number,
      "poor": Number
    },
    "top_candidates": [String] // Array of candidate IDs
  }
}
```

## Implementation Steps

### 1. Install MongoDB Dependencies
```bash
pip install pymongo
pip install djongo  # For Django ORM with MongoDB
```

### 2. Update Django Settings
```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'resume_processor_db',
        'ENFORCE_SCHEMA': True,
        'CLIENT': {
            'host': 'localhost',
            'port': 27017,
        }
    }
}
```

### 3. Create MongoDB Models
```python
# models.py
from djongo import models
from django.contrib.postgres.fields import JSONField

class Resume(models.Model):
    candidate_id = models.CharField(max_length=255, unique=True)
    filename = models.CharField(max_length=255)
    original_file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    processing_status = models.CharField(max_length=20, default='pending')
    error_message = models.TextField(blank=True, null=True)
    
    # JSON fields for complex data
    parsed_data = models.JSONField(default=dict)
    processed_data = models.JSONField(default=dict)
    ranking_data = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'resumes'

class JobPosting(models.Model):
    job_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255)
    description = models.TextField()
    requirements = models.JSONField(default=list)
    nice_to_have = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='active')
    job_analysis = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'job_postings'

class RankingSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    job_id = models.CharField(max_length=255)
    job_description = models.TextField()
    candidates_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='processing')
    ranking_results = models.JSONField(default=list)
    summary = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'ranking_sessions'
```

### 4. Update Views for MongoDB
```python
# views.py
from .models import Resume, JobPosting, RankingSession
from .text_processor import ResumeProcessor
from .llm_ranker import RankingManager

def process_resume_mongodb(request):
    """Process resume and store in MongoDB."""
    if request.method == 'POST':
        resume_file = request.FILES['resume_file']
        
        # Create resume record
        resume = Resume(
            candidate_id=f"candidate_{int(time.time())}",
            filename=resume_file.name,
            original_file_path=f"resumes/{resume_file.name}"
        )
        resume.save()
        
        # Process with pipeline
        pdf_parser = PDFParser()
        text_processor = ResumeProcessor()
        
        # Parse PDF
        json_file_path = pdf_parser.extract_to_json(resume_file.path, resume_file.name)
        with open(json_file_path, 'r') as f:
            parsed_data = json.load(f)
        
        # Process text
        processed_data = text_processor.process_resume(parsed_data)
        
        # Update MongoDB record
        resume.parsed_data = parsed_data
        resume.processed_data = processed_data
        resume.processing_status = 'completed'
        resume.processed_at = timezone.now()
        resume.save()
        
        return JsonResponse({'success': True, 'resume_id': resume.candidate_id})
```

### 5. Create Ranking API
```python
# views.py
def rank_candidates_mongodb(request):
    """Rank candidates for a job posting."""
    if request.method == 'POST':
        job_id = request.POST.get('job_id')
        candidate_ids = request.POST.getlist('candidate_ids')
        
        # Get job posting
        job = JobPosting.objects.get(job_id=job_id)
        
        # Get candidate resumes
        candidates = Resume.objects.filter(candidate_id__in=candidate_ids)
        
        # Prepare data for ranking
        candidates_data = []
        for candidate in candidates:
            candidates_data.append({
                'candidate_id': candidate.candidate_id,
                'resume_data': candidate.parsed_data,
                'computed_scores': candidate.processed_data.get('combined_scores', {})
            })
        
        # Rank candidates
        ranking_manager = RankingManager()
        session_data = ranking_manager.create_ranking_session(
            job.description, candidates_data
        )
        
        # Store ranking session
        ranking_session = RankingSession(
            session_id=session_data['session_id'],
            job_id=job_id,
            job_description=job.description,
            candidates_count=len(candidates_data),
            ranking_results=session_data['ranking_results'],
            summary=session_data.get('summary', {}),
            status='completed'
        )
        ranking_session.save()
        
        # Update candidate ranking data
        for result in session_data['ranking_results']:
            candidate = Resume.objects.get(candidate_id=result['candidate_id'])
            if 'rankings' not in candidate.ranking_data:
                candidate.ranking_data['rankings'] = []
            
            candidate.ranking_data['rankings'].append({
                'job_id': job_id,
                'score': result['score'],
                'rank': result['rank'],
                'reasoning': result['reasoning'],
                'strengths': result['strengths'],
                'gaps': result['gaps'],
                'confidence': result['confidence'],
                'ranked_at': timezone.now().isoformat()
            })
            candidate.save()
        
        return JsonResponse({
            'success': True,
            'session_id': session_data['session_id'],
            'ranking_results': session_data['ranking_results']
        })
```

## MongoDB Indexes for Performance

```javascript
// Resume collection indexes
db.resumes.createIndex({ "candidate_id": 1 })
db.resumes.createIndex({ "processing_status": 1 })
db.resumes.createIndex({ "uploaded_at": -1 })
db.resumes.createIndex({ "parsed_data.summary.has_skills": 1 })
db.resumes.createIndex({ "processed_data.skill_analysis.extracted_skills": 1 })

// Job postings indexes
db.job_postings.createIndex({ "job_id": 1 })
db.job_postings.createIndex({ "status": 1 })
db.job_postings.createIndex({ "created_at": -1 })

// Ranking sessions indexes
db.ranking_sessions.createIndex({ "session_id": 1 })
db.ranking_sessions.createIndex({ "job_id": 1 })
db.ranking_sessions.createIndex({ "created_at": -1 })
```

## Migration Strategy

### 1. Data Migration Script
```python
# migrate_to_mongodb.py
import json
import os
from django.core.management.base import BaseCommand
from resume_processor.models import Resume

class Command(BaseCommand):
    help = 'Migrate existing SQLite data to MongoDB'
    
    def handle(self, *args, **options):
        # Read existing JSON files
        structured_data_dir = 'media/structured_data'
        
        for filename in os.listdir(structured_data_dir):
            if filename.endswith('_structured.json'):
                with open(os.path.join(structured_data_dir, filename), 'r') as f:
                    parsed_data = json.load(f)
                
                # Create MongoDB record
                resume = Resume(
                    candidate_id=filename.replace('_structured.json', ''),
                    filename=filename,
                    original_file_path=f"resumes/{filename.replace('_structured.json', '.pdf')}",
                    parsed_data=parsed_data,
                    processing_status='completed'
                )
                resume.save()
                
                self.stdout.write(f"Migrated {filename}")
```

### 2. Run Migration
```bash
python manage.py migrate_to_mongodb
```

## Benefits of MongoDB Integration

1. **Flexible Schema**: Easy to add new fields without migrations
2. **JSON Storage**: Native support for complex nested data
3. **Scalability**: Better performance for large datasets
4. **Query Flexibility**: Rich querying capabilities for complex data
5. **Aggregation**: Powerful aggregation pipeline for analytics

## Next Steps

1. Install MongoDB and required dependencies
2. Update Django settings for MongoDB
3. Create new models and run migrations
4. Update views to use MongoDB models
5. Test the complete pipeline with MongoDB
6. Implement data migration from SQLite
7. Add MongoDB indexes for performance optimization

This MongoDB integration will provide a robust foundation for your complete resume processing pipeline with excellent scalability and flexibility for future enhancements. 