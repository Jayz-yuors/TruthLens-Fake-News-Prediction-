# src/services/url_handler.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from typing import Dict, List


class URLHandler:
    """
    Handles URL validation, content extraction, and text processing.
    """

    # Common boilerplate elements to remove
    BOILERPLATE_TAGS = [
        'script', 'style', 'nav', 'footer', 'header',
        'aside', 'noscript', 'meta', 'link'
    ]

    # Patterns for cleaning text
    COMMON_PATTERNS = [
        (r'\n+', ' '),  # Multiple newlines
        (r'\s+', ' '),  # Multiple spaces
        (r'<[^>]+>', ''),  # HTML tags
    ]

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "unknown"

    @staticmethod
    def fetch_url_content(url: str, timeout: int = 10) -> Dict:
        """
        Fetch URL content with proper error handling.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

            return {
                "success": True,
                "status_code": response.status_code,
                "content": response.text,
                "content_length": len(response.text),
                "encoding": response.encoding
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout - server took too long to respond",
                "status_code": 0
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - unable to reach the URL",
                "status_code": 0
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP error: {e.response.status_code}",
                "status_code": e.response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error fetching URL: {str(e)}",
                "status_code": 0
            }

    @staticmethod
    def extract_text_from_html(html_content: str) -> Dict:
        """
        Extract meaningful text from HTML content.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove boilerplate elements
            for tag in URLHandler.BOILERPLATE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extract headline
            headline = ""
            h1 = soup.find('h1')
            if h1:
                headline = h1.get_text(strip=True)
            else:
                title = soup.find('title')
                if title:
                    headline = title.get_text(strip=True)

            # Extract main content
            # Try common content containers
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|body|text|article'))

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback: get all text
                text = soup.get_text(separator=' ', strip=True)

            # Clean text
            for pattern, replacement in URLHandler.COMMON_PATTERNS:
                text = re.sub(pattern, replacement, text)

            text = text.strip()

            return {
                "success": True,
                "headline": headline,
                "text": text,
                "text_length": len(text)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error extracting text: {str(e)}",
                "headline": "",
                "text": ""
            }

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split long text into chunks for analysis.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    @staticmethod
    def assess_url_credibility(url: str) -> Dict:
        """
        Quick assessment of URL credibility based on domain patterns.
        """
        domain = URLHandler.get_domain(url)
        domain_lower = domain.lower()

        # High-trust domains
        high_trust = ["bbc", "reuters", "ap", "associated", "guardian", "nytimes", "washingtonpost", "economist", "ft.com"]
        
        # Suspicious patterns
        suspicious = ["conspiracy", "hoax", "scandal", "leaked", "secret", "exclusive"]

        trust_score = 0.5  # Neutral baseline

        if any(t in domain_lower for t in high_trust):
            trust_score = 0.9
        elif any(s in domain_lower for s in suspicious):
            trust_score = 0.2

        return {
            "domain": domain,
            "trust_score": trust_score,
            "is_suspicious": trust_score < 0.4
        }


# ============== MAIN FUNCTION ============== #

def fetch_and_extract_text(url: str) -> Dict:
    """
    Main function to fetch and extract text from URL.

    Returns:
        Dict with keys:
            - success: bool
            - text: str (extracted text)
            - headline: str
            - domain: str
            - status_code: int
            - content_length: int
            - error: str (if failed)
    """

    # Validate URL
    if not URLHandler.validate_url(url):
        return {
            "success": False,
            "error": "Invalid URL format",
            "text": "",
            "headline": "",
            "domain": "invalid"
        }

    # Fetch content
    fetch_result = URLHandler.fetch_url_content(url)

    if not fetch_result["success"]:
        return {
            "success": False,
            "error": fetch_result["error"],
            "text": "",
            "headline": "",
            "domain": URLHandler.get_domain(url),
            "status_code": fetch_result.get("status_code", 0)
        }

    # Extract text
    extract_result = URLHandler.extract_text_from_html(fetch_result["content"])

    if not extract_result["success"]:
        return {
            "success": False,
            "error": extract_result["error"],
            "text": "",
            "headline": "",
            "domain": URLHandler.get_domain(url)
        }

    # Check if content is meaningful
    if len(extract_result["text"]) < 50:
        return {
            "success": False,
            "error": "Page contains insufficient text content (< 50 characters)",
            "text": "",
            "headline": extract_result["headline"],
            "domain": URLHandler.get_domain(url),
            "content_length": len(extract_result["text"])
        }

    return {
        "success": True,
        "text": extract_result["text"],
        "headline": extract_result["headline"],
        "domain": URLHandler.get_domain(url),
        "status_code": fetch_result.get("status_code", 200),
        "content_length": len(extract_result["text"]),
        "chunks": URLHandler.chunk_text(extract_result["text"]),
        "credibility_assessment": URLHandler.assess_url_credibility(url)
    }


# ============== TEST ============== #

if __name__ == "__main__":
    test_urls = [
        "https://www.bbc.com/news",
        "https://www.reuters.com",
        "https://example.com"
    ]

    for url in test_urls:
        result = fetch_and_extract_text(url)
        print(f"\n🔗 URL: {url}")
        print(f"✅ Success: {result['success']}")
        if result['success']:
            print(f"📝 Text Length: {result['content_length']}")
            print(f"🏢 Domain: {result['domain']}")