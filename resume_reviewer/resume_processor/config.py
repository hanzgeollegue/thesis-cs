"""
Configuration file for Resume Processor application.
Contains API keys, model settings, and other configuration options.
"""

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

# Alternative: You can set the API key directly here (not recommended for production)
# OPENAI_API_KEY = 'your-api-key-here'

# LLM Ranking Configuration
LLM_RANKING_ENABLED = bool(OPENAI_API_KEY)
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1000

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# Section Weights for Scoring
SECTION_WEIGHTS = {
    'experience': 0.45,
    'skills': 0.35,
    'education': 0.15,
    'misc': 0.05
}

# Batch Processing Limits
MAX_BATCH_SIZE = 25
MAX_WORKERS = 4

# File Processing
SUPPORTED_FORMATS = ['.pdf']
MAX_FILE_SIZE_MB = 50

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_openai_config() -> dict:
    """Get OpenAI configuration dictionary."""
    return {
        'api_key': OPENAI_API_KEY,
        'model': OPENAI_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'enabled': LLM_RANKING_ENABLED
    }

def validate_config() -> list:
    """Validate configuration and return list of issues."""
    issues = []
    
    if not OPENAI_API_KEY:
        issues.append("OpenAI API key not set. LLM ranking will use fallback methods.")
        issues.append("Set OPENAI_API_KEY environment variable or add to config.py")
    
    if not OPENAI_API_KEY.startswith('sk-'):
        issues.append("OpenAI API key format appears invalid (should start with 'sk-')")
    
    return issues 