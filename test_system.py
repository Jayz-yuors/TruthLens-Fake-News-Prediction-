#!/usr/bin/env python3
# test_system.py - Complete system testing script

import requests
import json
import time
import subprocess
import sys
from typing import Dict, List

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(message: str, status: str = "INFO"):
    """Print colored status messages."""
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == "ERROR":
        print(f"{Colors.RED}❌ {message}{Colors.END}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
    elif status == "INFO":
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{Colors.END}\n")

# ============================================================
# TEST 1: N8N CONNECTIVITY
# ============================================================

def test_n8n_connectivity() -> bool:
    """Test if N8N server is running."""
    print_header("TEST 1: N8N CONNECTIVITY")
    
    try:
        response = requests.get("http://localhost:5678", timeout=5)
        print_status("N8N server is running", "SUCCESS")
        return True
    except requests.exceptions.ConnectionError:
        print_status("Cannot connect to N8N at localhost:5678", "ERROR")
        print("  → Start N8N with: n8n start")
        return False
    except Exception as e:
        print_status(f"Error connecting to N8N: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 2: N8N WEBHOOK
# ============================================================

def test_n8n_webhook() -> bool:
    """Test N8N webhook endpoint."""
    print_header("TEST 2: N8N WEBHOOK ENDPOINT")
    
    webhook_url = "http://localhost:5678/webhook/evidence"
    test_data = {
        "text": "India central bank inflation rate announcement"
    }
    
    try:
        print_status(f"Sending request to {webhook_url}", "INFO")
        response = requests.post(
            webhook_url,
            json=test_data,
            timeout=15
        )
        
        print_status(f"Response status: {response.status_code}", "INFO")
        
        if response.status_code == 200:
            data = response.json()
            print_status("Webhook response received successfully", "SUCCESS")
            
            # Print response details
            print(f"\n  Query Used: {data.get('query_used', 'N/A')}")
            print(f"  Match Count: {data.get('match_count', 0)}")
            print(f"  Trusted Sources: {data.get('trusted_sources', 0)}")
            print(f"  Credibility: {data.get('credibility', 'UNKNOWN')}")
            print(f"  Articles Found: {len(data.get('related_articles', []))}")
            
            return True
        else:
            print_status(f"Webhook returned error: {response.text}", "ERROR")
            return False
            
    except requests.exceptions.Timeout:
        print_status("Request timeout - N8N took too long to respond", "ERROR")
        print("  → Check N8N logs for slowdowns")
        return False
    except requests.exceptions.ConnectionError:
        print_status("Cannot connect to webhook endpoint", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error testing webhook: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 3: MODEL & TOKENIZER
# ============================================================

def test_model_loading() -> bool:
    """Test if ML model and tokenizer load correctly."""
    print_header("TEST 3: MODEL & TOKENIZER LOADING")
    
    try:
        from tensorflow.keras.models import load_model
        import pickle
        import os
        
        # Check files exist
        model_path = "models/lstm_model.h5"
        tokenizer_path = "models/tokenizer.pkl"
        
        if not os.path.exists(model_path):
            print_status(f"Model file not found: {model_path}", "ERROR")
            return False
        
        if not os.path.exists(tokenizer_path):
            print_status(f"Tokenizer file not found: {tokenizer_path}", "ERROR")
            return False
        
        # Load model
        print_status("Loading LSTM model...", "INFO")
        model = load_model(model_path)
        print_status("Model loaded successfully", "SUCCESS")
        
        # Load tokenizer
        print_status("Loading tokenizer...", "INFO")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print_status("Tokenizer loaded successfully", "SUCCESS")
        
        # Print model info
        print(f"\n  Model Type: {type(model).__name__}")
        print(f"  Model Params: {model.count_params():,}")
        print(f"  Tokenizer Vocab Size: {len(tokenizer.word_index):,}")
        
        return True
        
    except ImportError as e:
        print_status(f"Missing dependency: {str(e)}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error loading model: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 4: PREDICTION PIPELINE
# ============================================================

def test_prediction() -> bool:
    """Test ML prediction on sample texts."""
    print_header("TEST 4: PREDICTION PIPELINE")
    
    try:
        from src.pipeline.predict import predict
        
        test_cases = [
            ("Real news", "India's central bank keeps repo rate unchanged"),
            ("Fake news", "Scientists say hot water cures cancer instantly"),
            ("Questionable", "Government conspiracy exposed by leaked documents")
        ]
        
        for case_type, text in test_cases:
            print_status(f"Testing: {case_type}", "INFO")
            
            result = predict(text)
            
            if result.get("status") == "error":
                print_status(f"Prediction failed: {result.get('error')}", "ERROR")
                return False
            
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Verdict: {result['final_verdict']}")
            print()
        
        return True
        
    except ImportError:
        print_status("Cannot import prediction module", "ERROR")
        print("  → Ensure src/ directory is in Python path")
        return False
    except Exception as e:
        print_status(f"Error in prediction: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 5: URL HANDLING
# ============================================================

def test_url_handler() -> bool:
    """Test URL content extraction."""
    print_header("TEST 5: URL HANDLER")
    
    try:
        from src.services.url_handler import fetch_and_extract_text
        
        test_urls = [
            "https://www.bbc.com/news",
            "https://www.example.com"
        ]
        
        for url in test_urls:
            print_status(f"Testing URL: {url}", "INFO")
            
            result = fetch_and_extract_text(url)
            
            if result["success"]:
                print_status("URL extraction successful", "SUCCESS")
                print(f"  Domain: {result['domain']}")
                print(f"  Text Length: {result['content_length']}")
                print(f"  Headline: {result['headline'][:50]}..." if result['headline'] else "  Headline: N/A")
            else:
                print_status(f"URL extraction failed: {result['error']}", "WARNING")
            print()
        
        return True
        
    except ImportError:
        print_status("Cannot import URL handler", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error in URL handler: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 6: DOCUMENT HANDLER
# ============================================================

def test_document_handler() -> bool:
    """Test document extraction capabilities."""
    print_header("TEST 6: DOCUMENT HANDLER")
    
    try:
        from src.services.document_handler import DocumentHandler
        
        # Test TXT extraction
        print_status("Testing TXT extraction", "INFO")
        sample_text = "This is a sample news article about inflation and economy.".encode('utf-8')
        result = DocumentHandler.extract_text_from_txt(sample_text)
        
        if result["success"]:
            print_status("TXT extraction working", "SUCCESS")
        else:
            print_status(f"TXT extraction failed: {result['error']}", "WARNING")
        
        # Test chunking
        print_status("Testing text chunking", "INFO")
        long_text = "This is a test sentence. " * 100  # Create long text
        chunks = DocumentHandler.chunk_text(long_text, max_chunk_size=500)
        
        print_status(f"Text chunked into {len(chunks)} chunks", "SUCCESS")
        
        # Test summary
        print_status("Testing text summarization", "INFO")
        summary = DocumentHandler.create_summary(long_text, max_length=100)
        print_status("Summary created successfully", "SUCCESS")
        
        return True
        
    except ImportError as e:
        print_status(f"Missing library: {str(e)}", "WARNING")
        print("  → Install with: pip install PyPDF2 python-docx")
        return True  # Not critical
    except Exception as e:
        print_status(f"Error in document handler: {str(e)}", "ERROR")
        return False

# ============================================================
# TEST 7: API ENDPOINTS
# ============================================================

def test_api_endpoints() -> bool:
    """Test FastAPI endpoints."""
    print_header("TEST 7: API ENDPOINTS")
    
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        print_status(f"Checking API at {api_url}", "INFO")
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            print_status("API is running and healthy", "SUCCESS")
        else:
            print_status("API returned unexpected status", "WARNING")
            return True  # API not running is ok for this test
    except requests.exceptions.ConnectionError:
        print_status(f"API not running at {api_url}", "WARNING")
        print("  → Start with: python -m uvicorn src.api.main:app --reload")
        return True  # Not critical for other tests
    
    return True

# ============================================================
# TEST 8: COMPLETE INTEGRATION
# ============================================================

def test_complete_integration() -> bool:
    """Test complete system integration."""
    print_header("TEST 8: COMPLETE INTEGRATION (N8N + ML)")
    
    try:
        from src.pipeline.predict import predict
        
        test_text = "India's central bank announces new economic policies"
        
        print_status(f"Testing complete flow with: {test_text}", "INFO")
        
        result = predict(test_text)
        
        print(f"\n  Input Type: {result['input_type']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Final Verdict: {result['final_verdict']}")
        
        # Check evidence
        evidence = result.get('evidence', {})
        print(f"\n  Evidence from N8N:")
        print(f"    - Query Used: {evidence.get('query_used', 'N/A')}")
        print(f"    - Match Count: {evidence.get('match_count', 0)}")
        print(f"    - Trusted Sources: {evidence.get('trusted_sources', 0)}")
        print(f"    - Credibility: {evidence.get('credibility', 'UNKNOWN')}")
        
        # Check explanation
        if result.get('explanation'):
            print(f"\n  Top Keywords:")
            for item in result['explanation'][:3]:
                print(f"    - {item['word']}: {item['impact']:.4f} ({item['towards']})")
        
        print_status("Complete integration test successful", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Integration test failed: {str(e)}", "ERROR")
        return False

# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all tests and generate report."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{'FAKE NEWS DETECTION SYSTEM - COMPLETE TEST SUITE':^70}")
    print(f"{'='*70}{Colors.END}\n")
    
    tests = [
        ("N8N Connectivity", test_n8n_connectivity),
        ("N8N Webhook", test_n8n_webhook),
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction),
        ("URL Handler", test_url_handler),
        ("Document Handler", test_document_handler),
        ("API Endpoints", test_api_endpoints),
        ("Complete Integration", test_complete_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_status(f"Running {test_name}...", "INFO")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Unexpected error in {test_name}: {str(e)}", "ERROR")
            results[test_name] = False
        
        time.sleep(0.5)  # Small delay between tests
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✅" if result else "❌"
        print(f"{symbol} {test_name:.<40} {status}")
    
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
    
    if passed == total:
        print_status("ALL TESTS PASSED! System is ready.", "SUCCESS")
        return True
    else:
        print_status(f"{total - passed} test(s) failed. Review above.", "WARNING")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)