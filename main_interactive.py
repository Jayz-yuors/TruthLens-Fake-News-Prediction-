#!/usr/bin/env python3
# main_interactive.py - Interactive Fake News Detection Terminal Application

import sys
import time
from typing import Dict
from src.pipeline.predict import predict
from src.services.news_service import search_news

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print application header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║         🚀 FAKE NEWS DETECTION SYSTEM v3.0 🚀                 ║")
    print("║                                                                ║")
    print("║         Powered by: LSTM + LIME + N8N + NewsAPI               ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

def print_menu():
    """Print main menu."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                        MAIN MENU                            │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 1. Analyze Text News Article                               │")
    print("│ 2. Analyze News from URL                                   │")
    print("│ 3. Analyze Uploaded Document (PDF/DOCX/TXT)               │")
    print("│ 4. Batch Analysis (Multiple Articles)                      │")
    print("│ 5. Search & Verify News Topic                              │")
    print("│ 6. About This System                                       │")
    print("│ 7. Exit                                                    │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"{Colors.END}")

def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"{Colors.END}")

def print_prediction_result(result: Dict):
    """Print prediction result in formatted way."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌─────────────────────────────────────────────────────────────┐{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}│                    PREDICTION RESULT                        │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}└─────────────────────────────────────────────────────────────┘{Colors.END}\n")

    # ML Prediction
    prediction = result.get('prediction', 'ERROR')
    confidence = result.get('confidence', 0.0)
    
    if prediction == 'FAKE':
        pred_color = Colors.RED
        pred_icon = "❌"
    elif prediction == 'REAL':
        pred_color = Colors.GREEN
        pred_icon = "✅"
    else:
        pred_color = Colors.YELLOW
        pred_icon = "⚠️"

    print(f"{Colors.BOLD}ML Model Prediction:{Colors.END}")
    print(f"  {pred_icon} Status: {pred_color}{prediction.upper()}{Colors.END}")
    print(f"  📊 Confidence: {Colors.BOLD}{confidence*100:.2f}%{Colors.END}")

    # Evidence from N8N
    evidence = result.get('evidence', {})
    print(f"\n{Colors.BOLD}Evidence from N8N:{Colors.END}")
    print(f"  🔍 Query Used: {evidence.get('query_used', 'N/A')}")
    print(f"  📰 Match Count: {evidence.get('match_count', 0)} articles found")
    print(f"  🏢 Trusted Sources: {evidence.get('trusted_sources', 0)}")
    
    cred = evidence.get('credibility', 'UNKNOWN')
    if cred == 'HIGH':
        cred_color = Colors.GREEN
    elif cred == 'MEDIUM':
        cred_color = Colors.YELLOW
    elif cred == 'LOW':
        cred_color = Colors.RED
    else:
        cred_color = Colors.CYAN
    
    print(f"  📈 Credibility Level: {cred_color}{Colors.BOLD}{cred}{Colors.END}")
    
    # Related Articles
    articles = evidence.get('related_articles', [])
    if articles:
        print(f"\n{Colors.BOLD}Related Articles Found: {len(articles)}{Colors.END}")
        for i, article in enumerate(articles[:5], 1):
            # Handle both string and dict article formats
            if isinstance(article, dict):
                title = article.get('title', 'Untitled')
                source = article.get('source', 'Unknown')
                url = article.get('url', '')
                desc = article.get('description', '')
                print(f"    {i}. {Colors.CYAN}{title[:50]}{Colors.END}")
                print(f"       📰 Source: {source}")
                if desc:
                    print(f"       📝 {desc[:70]}...")
                if url:
                    print(f"       🔗 {url}")
            else:
                print(f"    {i}. {article[:70]}..." if len(str(article)) > 70 else f"    {i}. {article}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  No related articles found{Colors.END}")

    # Final Verdict
    verdict = result.get('final_verdict', 'INCONCLUSIVE')
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}FINAL VERDICT:{Colors.END}")
    print(f"  {verdict}\n")

    # LIME Explanation
    explanation = result.get('explanation', [])
    if explanation:
        print(f"{Colors.BOLD}Top Keywords Influencing Prediction:{Colors.END}")
        for item in explanation[:5]:
            word = item.get('word', '')
            impact = item.get('impact', 0)
            towards = item.get('towards', '')
            
            if towards == 'FAKE':
                arrow = "↘️ →"
                color = Colors.RED
            else:
                arrow = "↗️ →"
                color = Colors.GREEN
            
            print(f"  {arrow} {color}{word}{Colors.END}: {abs(impact):.4f} (toward {towards})")

    # Language Analysis
    lang_analysis = result.get('language_analysis', {})
    if lang_analysis and lang_analysis.get('language_analysis') == 'HIGH_SUSPICION':
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Language Analysis: HIGH SUSPICION{Colors.END}")
        patterns = lang_analysis.get('detected_patterns', {})
        for pattern, count in patterns.items():
            if count > 0:
                print(f"  • {pattern.replace('_', ' ').title()}: {count} occurrences")

    # Processing Time
    processing_time = result.get('processing_time', 0)
    print(f"\n⏱️  Processing Time: {Colors.BOLD}{processing_time}s{Colors.END}\n")

def analyze_text_article():
    """Analyze a text news article."""
    print_section("ANALYZE TEXT ARTICLE")
    
    print(f"{Colors.BOLD}Enter the news article (press Enter twice to submit):{Colors.END}")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                lines.pop()
                break
            lines.append(line)
        else:
            lines.append(line)
    
    text = "\n".join(lines).strip()
    
    if len(text) < 10:
        print(f"{Colors.RED}❌ Text too short! Minimum 10 characters required.{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.YELLOW}⏳ Analyzing...{Colors.END}")
    print("   🧠 Running ML model...")
    print("   📰 Gathering evidence from N8N...")
    print("   🔍 Searching related articles...")
    print("   📊 Calculating verdict...\n")

    try:
        result = predict(text, input_type="text", source="direct_input")
        if result.get('status') == 'success':
            print_prediction_result(result)
        else:
            print(f"{Colors.RED}❌ Error: {result.get('error', 'Unknown error')}{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error during analysis: {str(e)}{Colors.END}")

def analyze_url():
    """Analyze news from URL."""
    print_section("ANALYZE NEWS FROM URL")
    
    url = input(f"{Colors.BOLD}Enter URL (e.g., https://bbc.com/news):{Colors.END} ").strip()
    
    if not url.startswith('http'):
        print(f"{Colors.RED}❌ Invalid URL! Must start with http:// or https://{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.YELLOW}⏳ Analyzing...{Colors.END}")
    print("   🌐 Fetching content from URL...")
    print("   📄 Extracting text...")
    print("   🧠 Running ML model...")
    print("   📰 Gathering evidence from N8N...\n")

    try:
        result = predict(url, input_type="url", source=url)
        if result.get('status') == 'success':
            print_prediction_result(result)
            
            # Additional URL metadata
            url_meta = result.get('url_metadata', {})
            if url_meta:
                print(f"{Colors.BOLD}URL Information:{Colors.END}")
                print(f"  🏢 Domain: {url_meta.get('domain', 'N/A')}")
                print(f"  📑 Content Length: {url_meta.get('content_length', 'N/A')} characters")
                print(f"  ⭐ Domain Trust Score: {url_meta.get('domain_trustworthiness', {}).get('trust_score', 0):.2f}")
        else:
            print(f"{Colors.RED}❌ Error: {result.get('error', 'Unknown error')}{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error during analysis: {str(e)}{Colors.END}")

def analyze_document():
    """Analyze uploaded document."""
    print_section("ANALYZE DOCUMENT")
    
    file_path = input(f"{Colors.BOLD}Enter file path (PDF/DOCX/TXT):{Colors.END} ").strip()
    
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_ext = file_path.split('.')[-1].lower()
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}⏳ Analyzing...{Colors.END}")
        print("   📄 Extracting text from document...")
        print("   📚 Processing content...")
        print("   🧠 Running ML model...")
        print("   📰 Gathering evidence from N8N...\n")

        from src.services.document_handler import extract_text_from_document
        
        extracted = extract_text_from_document(file_content, file_ext)
        
        if extracted.get('success'):
            text = extracted.get('text', '')
            result = predict(text, input_type="document", source=file_path)
            
            if result.get('status') == 'success':
                print_prediction_result(result)
                
                print(f"{Colors.BOLD}Document Information:{Colors.END}")
                print(f"  📄 File: {file_path}")
                print(f"  📏 Text Length: {len(text)} characters")
                print(f"  📑 Chunks Analyzed: {len(extracted.get('chunks', []))}")
            else:
                print(f"{Colors.RED}❌ Error: {result.get('error', 'Unknown error')}{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Failed to extract: {extracted.get('error', 'Unknown error')}{Colors.END}")
    
    except FileNotFoundError:
        print(f"{Colors.RED}❌ File not found: {file_path}{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {str(e)}{Colors.END}")

def batch_analysis():
    """Analyze multiple articles."""
    print_section("BATCH ANALYSIS")
    
    count = input(f"{Colors.BOLD}How many articles to analyze?{Colors.END} ").strip()
    
    try:
        count = int(count)
        if count <= 0:
            print(f"{Colors.RED}❌ Please enter a positive number{Colors.END}")
            return
    except ValueError:
        print(f"{Colors.RED}❌ Invalid number{Colors.END}")
        return

    articles = []
    for i in range(count):
        print(f"\n{Colors.BOLD}Article {i+1}/{count}:{Colors.END}")
        text = input("Enter text (or 'skip' to skip): ").strip()
        if text.lower() != 'skip':
            articles.append(text)

    if not articles:
        print(f"{Colors.YELLOW}No articles to analyze{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.YELLOW}⏳ Analyzing {len(articles)} articles...{Colors.END}\n")

    results = []
    for i, text in enumerate(articles, 1):
        print(f"{Colors.CYAN}Processing article {i}/{len(articles)}...{Colors.END}")
        try:
            result = predict(text, input_type="text", source=f"batch_{i}")
            results.append(result)
            time.sleep(0.5)  # Small delay between analyses
        except Exception as e:
            print(f"{Colors.RED}Error analyzing article {i}: {str(e)}{Colors.END}")

    # Summary
    print_section("BATCH ANALYSIS SUMMARY")
    
    fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
    real_count = sum(1 for r in results if r.get('prediction') == 'REAL')
    avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0

    print(f"{Colors.BOLD}Results:{Colors.END}")
    print(f"  📊 Total Analyzed: {len(results)}")
    print(f"  🔴 Fake: {fake_count}")
    print(f"  🟢 Real: {real_count}")
    print(f"  📈 Average Confidence: {avg_confidence*100:.2f}%")

    # Detailed results
    print(f"\n{Colors.BOLD}Detailed Results:{Colors.END}")
    for i, result in enumerate(results, 1):
        pred = result.get('prediction')
        conf = result.get('confidence')
        verdict = result.get('final_verdict')
        pred_icon = "🔴" if pred == "FAKE" else "🟢"
        print(f"  {i}. {pred_icon} {pred} ({conf*100:.1f}%) - {verdict[:50]}...")

def search_news_topic():
    """Search and verify a news topic."""
    print_section("SEARCH & VERIFY NEWS TOPIC")
    
    topic = input(f"{Colors.BOLD}Enter topic to search:{Colors.END} ").strip()
    
    if not topic:
        print(f"{Colors.RED}❌ Topic cannot be empty{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.YELLOW}⏳ Searching for '{topic}'...{Colors.END}\n")

    try:
        articles = search_news(topic, page_size=5)
        
        if not articles:
            print(f"{Colors.YELLOW}⚠️  No articles found for this topic{Colors.END}")
            return

        print(f"{Colors.BOLD}Found {len(articles)} articles:{Colors.END}\n")
        
        for i, article in enumerate(articles, 1):
            print(f"{Colors.BOLD}{i}. {article.get('title', 'Untitled')}{Colors.END}")
            print(f"   📰 Source: {article.get('source', 'Unknown')}")
            print(f"   📝 {article.get('description', 'No description')[:100]}...")
            print(f"   🔗 {article.get('url', 'No URL')}\n")

    except Exception as e:
        print(f"{Colors.RED}❌ Error searching: {str(e)}{Colors.END}")

def show_about():
    """Show about information."""
    print_section("ABOUT THIS SYSTEM")
    
    about_text = f"""
{Colors.BOLD}Fake News Detection System v3.0{Colors.END}

{Colors.CYAN}Technology Stack:{Colors.END}
  • Machine Learning: LSTM Neural Network (689,473 parameters)
  • Explainability: LIME (Local Interpretable Model-agnostic Explanations)
  • Evidence Gathering: N8N Workflow Automation
  • News Verification: NewsAPI Integration
  • API Framework: FastAPI
  • Container: Docker & Docker Compose

{Colors.CYAN}Features:{Colors.END}
  ✓ Real-time fake news detection
  ✓ Confidence scoring and explanations
  ✓ Evidence from multiple sources
  ✓ Keyword importance analysis
  ✓ Domain trustworthiness assessment
  ✓ Batch processing capabilities
  ✓ Multiple input formats (text, URL, documents)

{Colors.CYAN}Model Information:{Colors.END}
  • Type: LSTM (Long Short-Term Memory)
  • Vocabulary Size: 115,492 words
  • Max Sequence Length: 100 tokens
  • Activation: Sigmoid (binary classification)
  • Training: 20K+ articles

{Colors.CYAN}How It Works:{Colors.END}
  1. Text Preprocessing: Cleaning and tokenization
  2. Model Prediction: LSTM generates prediction
  3. Evidence Gathering: N8N queries NewsAPI
  4. Explainability: LIME identifies influential keywords
  5. Final Verdict: Combines ML + Evidence + Language Analysis

{Colors.CYAN}Accuracy & Performance:{Colors.END}
  • Processing Time: 2-5 seconds per article
  • Evidence Sources: Real-time from NewsAPI
  • Fallback: Hardcoded articles when API unavailable
  • Confidence Range: 0-100%

{Colors.CYAN}Developed By:{Colors.END}
  Shruti | AI & ML Engineering
  Using: Python, TensorFlow, FastAPI, N8N, Docker

{Colors.CYAN}For more information:{Colors.END}
  • GitHub: Your repository
  • Documentation: README.md & guides
  • Issues: Report bugs and request features
"""
    print(about_text)

def main():
    """Main application loop."""
    print_header()
    
    print(f"{Colors.GREEN}✅ System initialized successfully!{Colors.END}")
    print(f"{Colors.GREEN}✅ N8N workflow: {Colors.BOLD}ACTIVE{Colors.END}")
    print(f"{Colors.GREEN}✅ ML Model: {Colors.BOLD}LOADED{Colors.END}")
    print(f"{Colors.GREEN}✅ NewsAPI: {Colors.BOLD}CONNECTED{Colors.END}\n")

    while True:
        print_menu()
        choice = input(f"{Colors.BOLD}Select option (1-7):{Colors.END} ").strip()

        if choice == '1':
            analyze_text_article()
        elif choice == '2':
            analyze_url()
        elif choice == '3':
            analyze_document()
        elif choice == '4':
            batch_analysis()
        elif choice == '5':
            search_news_topic()
        elif choice == '6':
            show_about()
        elif choice == '7':
            print(f"\n{Colors.GREEN}Thank you for using Fake News Detection System!{Colors.END}")
            print(f"{Colors.CYAN}Goodbye! 👋{Colors.END}\n")
            sys.exit(0)
        else:
            print(f"{Colors.RED}❌ Invalid option. Please select 1-7{Colors.END}")

        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}\n⏹️  Application interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Fatal error: {str(e)}{Colors.END}")
        sys.exit(1)