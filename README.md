## Tiny Evaluation Harness

We provide a minimal script to compute nDCG@10 and Recall@50 for your ranked outputs.

Run:

```bash
python tools/eval_ndcg.py --labels_csv labels.csv --runs_jsonl run.jsonl
```

Input formats:
- labels.csv: `jd_id,resume_id,label` where label is 0–3 (relevance).
- run.jsonl: one JSON per line `{"jd_id": "<id>", "ranked_ids": ["res1", "res2", ...]}`.

The script prints aggregated nDCG@10 and Recall@50 across JDs with ground-truth labels.

# Resume Processing & Analysis System

A comprehensive, AI-powered resume evaluation and ranking system built with Django. Process multiple resumes simultaneously, extract structured data, and rank candidates using advanced algorithms.

## 🚀 **Quick Start**

### 1. **Installation**
```bash
# Clone and setup
cd resume_reviewer
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### 2. **OpenAI API Setup (Required for AI Ranking)**
```bash
# Run the interactive setup script
python setup_openai.py

# Or set environment variable manually
export OPENAI_API_KEY="your-api-key-here"  # Linux/macOS
set OPENAI_API_KEY=your-api-key-here       # Windows
```

**Get your API key from:** https://platform.openai.com/api-keys

### 3. **Start the Server**
```bash
python manage.py runserver
```

### 4. **Access the Website**
Open your browser and go to: **http://127.0.0.1:8000**

## 🌟 **Key Features**

### **AI-Powered Ranking (GPT-4)**
- **Intelligent candidate evaluation** using OpenAI's GPT models
- **Detailed reasoning** for each ranking decision
- **Strengths and gaps analysis** for each candidate
- **Confidence scoring** for ranking reliability
- **Fallback ranking** when API is unavailable

### **Landing Page (Main Entry Point)**
- **Prominent batch upload form** - Upload up to 25 PDF resumes at once
- **Pre-filled job description** - Sample job description ready to customize
- **Modern, responsive design** - Clean interface with gradient background
- **Feature highlights** - Overview of system capabilities

### **Batch Processing Pipeline**
- **Multi-modal scoring**: TF-IDF lexical + semantic similarity + AI ranking
- **Section-aware analysis**: Experience (45%), Skills (35%), Education (15%), Misc (5%)
- **Skill taxonomy matching**: Canonical skill identifiers with surface form recognition
- **LLM-powered ranking**: GPT-based candidate evaluation with detailed reasoning

### **Advanced PDF Processing**
- **Intelligent text extraction**: PyPDF2 + OCR fallback with pytesseract
- **Smart section detection**: Header recognition with negative heuristics
- **Recursive parsing**: Embedded header detection and section splitting
- **Structured output**: JSON format for ML model consumption

## 🔑 **OpenAI API Configuration**

### **Setup Methods**

1. **Interactive Setup (Recommended)**
   ```bash
   python setup_openai.py
   ```

2. **Environment Variable**
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY="sk-your-key-here"
   
   # Windows
   set OPENAI_API_KEY=sk-your-key-here
   ```

3. **Direct in config.py**
   ```python
   # In resume_processor/config.py
   OPENAI_API_KEY = 'sk-your-key-here'
   ```

### **API Key Requirements**
- **Format**: Must start with `sk-`
- **Source**: https://platform.openai.com/api-keys
- **Cost**: Pay-per-use (typically $0.01-0.10 per batch)
- **Rate Limits**: 3,000 requests per minute

### **Models Available**
- **GPT-4**: Best quality, higher cost
- **GPT-3.5-turbo**: Good quality, lower cost
- **Configurable**: Change in config.py or setup script

## 🎯 **Website Structure**

### **Main Pages**
- **`/`** - Landing page with batch upload (main entry point)
- **`/resume/upload/`** - Single resume upload
- **`/resume/batch-upload/`** - Batch processing interface
- **`/resume/list/`** - View all processed resumes
- **`/admin/`** - Django admin panel

### **User Experience Flow**
1. **Land on homepage** → See batch upload form prominently
2. **Enter job description** → Pre-filled with sample content
3. **Select PDF files** → Up to 25 resumes allowed
4. **Process batch** → AI-powered analysis and ranking
5. **View results** → Detailed scoring and candidate insights

## 🛠️ **Technical Architecture**

### **Core Components**
- **`BatchProcessor`**: Orchestrates the entire pipeline
- **`PDFParser`**: Enhanced PDF text extraction and section detection
- **`LLMRanker`**: GPT-based candidate evaluation
- **`text_processor`**: TF-IDF scoring and skill taxonomy

### **Data Flow**
```
PDF Uploads → Text Extraction → Section Detection → 
TF-IDF Scoring → Skill Canonicalization → Semantic Embeddings → 
LLM Ranking → Structured Output → Results Display
```

### **Scoring Algorithms**
1. **Section-Aware TF-IDF**: Weighted by section importance
2. **Skill-Cluster TF-IDF**: Taxonomy-normalized skill matching
3. **Semantic Similarity**: Sentence-BERT embeddings
4. **Composite LLM Ranking**: GPT-based qualitative evaluation

## 📊 **Output Format**

### **Batch Processing Results**
```json
{
  "job_description_digest": {
    "tokens_summary": "...",
    "top_skills": [{"skill_id": "SKILL_001", "label": "React"}],
    "embedding_info": {"model": "sentence-transformers", "dim": 768}
  },
  "resumes": [
    {
      "id": "resume_1.pdf",
      "scores": {
        "tfidf_section_score": 0.85,
        "tfidf_taxonomy_score": 0.92,
        "semantic_score": 0.78
      },
      "matched_skills": [{"skill_id": "SKILL_001", "surface_forms": ["React", "ReactJS"]}],
      "parsed": {...},
      "meta": {...}
    }
  ],
  "final_ranking": [
    {
      "id": "resume_1.pdf",
      "rank": 1,
      "reasoning": "Strong technical skills and relevant experience...",
      "scores_snapshot": {...}
    }
  ],
  "batch_summary": {...}
}
```

## 🧪 **Testing**

### **Compatibility Test**
```bash
python test_compatibility.py
```

### **Project Test**
```bash
python test_project.py
```

### **Website Test**
```bash
python test_website.py
```

### **OpenAI API Test**
```bash
python setup_openai.py  # Includes API testing
```

### **Django Tests**
```bash
python manage.py test resume_processor
```

## 🔧 **Configuration**

### **Environment Variables**
- `OPENAI_API_KEY`: For LLM ranking (required for AI features)
- `DJANGO_SECRET_KEY`: Django secret key
- `DEBUG`: Development mode

### **Settings**
- **Batch limit**: 25 resumes maximum
- **Section weights**: Configurable in `BatchProcessor`
- **Skill taxonomy**: Expandable in `skill_taxonomy` dictionary
- **TF-IDF parameters**: Adjustable in `_initialize_tfidf_vectorizers`
- **LLM parameters**: Configurable in `config.py`

## 📁 **Project Structure**

```
resume_reviewer/
├── resume_processor/          # Main Django app
│   ├── templates/            # HTML templates
│   ├── management/           # Django commands
│   ├── tests/               # Test files
│   ├── config.py            # Configuration and API keys
│   └── views.py             # Web views
├── media/                   # Uploaded files
├── static/                  # Static assets
├── requirements.txt         # Dependencies
├── setup_openai.py         # OpenAI API setup script
├── manage.py               # Django management
└── README.md               # This file
```

## 🚀 **Usage Examples**

### **Command Line Batch Processing**
```bash
python manage.py process_batch \
  --resumes resume1.pdf resume2.pdf \
  --job-description "Software Engineer position..." \
  --output results.json \
  --api-key YOUR_OPENAI_KEY
```

### **Web Interface**
1. Navigate to http://127.0.0.1:8000
2. Enter job description
3. Select PDF files (max 25)
4. Click "Process Resumes"
5. View AI-powered results and rankings

## 🔍 **Troubleshooting**

### **Common Issues**
- **OpenAI API errors**: Check API key and billing status
- **scikit-learn compatibility**: Use `test_compatibility.py` to diagnose
- **PDF parsing errors**: Check OCR dependencies (pytesseract)
- **LLM ranking failures**: Verify API key and internet connection
- **Database errors**: Run `python manage.py migrate`

### **API Key Issues**
- **Invalid format**: Must start with `sk-`
- **Rate limiting**: Wait and retry
- **Billing issues**: Check OpenAI account status
- **Network errors**: Check internet connection

### **Performance Tips**
- **Batch size**: Optimal at 15-20 resumes for best performance
- **File size**: Large PDFs may slow processing
- **OCR usage**: Enable only when necessary for scanned documents
- **API costs**: Monitor usage in OpenAI dashboard

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

## 🆘 **Support**

For issues and questions:
1. Check the troubleshooting section
2. Run compatibility tests
3. Verify OpenAI API configuration
4. Review error logs in `debug.log`
5. Check Django admin for processing status

---

**Built with ❤️ using Django, OpenAI GPT, scikit-learn, and modern web technologies** 