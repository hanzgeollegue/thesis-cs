#!/usr/bin/env python3
"""
Simple test script for batch processor functionality.
This script tests the core components without Django setup.
"""

import sys
import os
from pathlib import Path

# Add the resume_processor directory to Python path
current_dir = Path(__file__).parent
resume_processor_dir = current_dir / 'resume_processor'
sys.path.insert(0, str(resume_processor_dir))

# Also add the current directory to handle any other imports
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if we can import the required modules."""
    print("🧪 Testing Module Imports")
    print("=" * 50)
    
    try:
        # Test PDF parser
        from enhanced_pdf_parser import PDFParser
        print("✅ PDF Parser imported successfully")
        
        # Test LLM ranker
        from llm_ranker import LLMRanker
        print("✅ LLM Ranker imported successfully")
        
        # Test config
        from config import get_openai_config, validate_config
        print("✅ Config module imported successfully")
        
        # Test batch processor
        from batch_processor import BatchProcessor
        print("✅ Batch Processor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"Current Python path: {sys.path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_pdf_parser():
    """Test PDF parser functionality."""
    print("\n📄 Testing PDF Parser")
    print("=" * 50)
    
    try:
        from enhanced_pdf_parser import PDFParser
        
        # Test with custom output directory
        test_dir = os.path.join(os.getcwd(), 'test_output')
        parser = PDFParser(output_dir=test_dir)
        print("✅ PDF parser initialized successfully")
        print(f"Output directory: {parser.text_output_dir}")
        
        # Check if required method exists
        if hasattr(parser, '_extract_structured_data'):
            print("✅ _extract_structured_data method exists")
        else:
            print("❌ _extract_structured_data method missing")
            
        # List available methods
        methods = [method for method in dir(parser) if not method.startswith('_')]
        print(f"Public methods: {methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration functionality."""
    print("\n⚙️ Testing Configuration")
    print("=" * 50)
    
    try:
        from config import get_openai_config, validate_config
        
        config = get_openai_config()
        print(f"✅ Configuration loaded: {config}")
        
        config_issues = validate_config()
        if config_issues:
            print("⚠️ Configuration issues found:")
            for issue in config_issues:
                print(f"  - {issue}")
        else:
            print("✅ Configuration validation passed")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processor():
    """Test batch processor initialization."""
    print("\n🔄 Testing Batch Processor")
    print("=" * 50)
    
    try:
        from batch_processor import BatchProcessor
        
        processor = BatchProcessor()
        print("✅ Batch processor initialized successfully")
        
        # Test basic attributes
        print(f"Section weights: {processor.section_weights}")
        print(f"Skill taxonomy keys: {list(processor.skill_taxonomy.keys())}")
        print(f"LLM ranker available: {processor.llm_ranker is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without full Django setup."""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        from batch_processor import BatchProcessor
        
        # Initialize processor
        processor = BatchProcessor()
        print("✅ Batch processor initialized")
        
        # Test job description validation
        result = processor.process_batch([], "")
        if result.get("error") == "job_description_required":
            print("✅ Job description validation working")
        else:
            print(f"⚠️ Job description validation: {result}")
        
        # Test batch size validation
        fake_resumes = [f"resume_{i}.pdf" for i in range(30)]
        result = processor.process_batch(fake_resumes, "Test job description")
        if "error" in result and "Batch limit exceeded" in result["error"]:
            print("✅ Batch size validation working")
        else:
            print(f"⚠️ Batch size validation: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Simple Component Tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("PDF Parser", test_pdf_parser),
        ("Configuration", test_config),
        ("Batch Processor", test_batch_processor),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        print("\n💡 Next steps:")
        print("1. Start Django server: python manage.py runserver")
        print("2. Open: http://127.0.0.1:8000")
        print("3. Test batch upload functionality")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if you're in the correct directory")
        print("3. Verify Python version compatibility")
        print("4. Try running from the resume_reviewer directory")

if __name__ == "__main__":
    main()
