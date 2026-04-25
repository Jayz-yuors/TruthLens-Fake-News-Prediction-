# src/services/news_service.py

import requests
from typing import List, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_KEY = "e70799b1025842f39f3593cea30a36bb"
BASE_URL = "https://newsapi.org/v2"


class NewsAPIHandler:
    """Handle interactions with NewsAPI with proper error handling."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.timeout = 10

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make HTTP request to NewsAPI.

        Args:
            endpoint: API endpoint (e.g., 'everything', 'top-headlines')
            params: Query parameters

        Returns:
            Response JSON or error dict
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            params["apiKey"] = self.api_key

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 401:
                logger.error("❌ Invalid API key")
                return {"error": "Invalid API key", "status": 401}

            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded")
                return {"error": "Rate limit exceeded", "status": 429}

            elif response.status_code == 500:
                logger.error("❌ NewsAPI server error")
                return {"error": "NewsAPI server error", "status": 500}

            elif response.status_code != 200:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}", "status": response.status_code}

            data = response.json()

            if data.get("status") != "ok":
                logger.warning(f"API returned non-OK status: {data.get('message')}")
                return {"error": data.get("message", "Unknown error"), "status": 400}

            return data

        except requests.exceptions.Timeout:
            logger.error("Request timeout to NewsAPI")
            return {"error": "Request timeout", "status": 0}

        except requests.exceptions.ConnectionError:
            logger.error("Connection error to NewsAPI")
            return {"error": "Connection error", "status": 0}

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": str(e), "status": 0}

    def fetch_top_headlines(self, category: str = "general", page_size: int = 10) -> List[Dict]:
        """
        Fetch top headlines.

        Args:
            category: News category
            page_size: Number of articles to fetch

        Returns:
            List of articles
        """
        params = {
            "category": category,
            "language": "en",
            "pageSize": min(page_size, 100),
            "sortBy": "publishedAt"
        }

        data = self._make_request("top-headlines", params)

        if "error" in data:
            logger.warning(f"Error fetching top headlines: {data['error']}")
            return []

        return self._parse_articles(data)

    def fetch_everything(self, query: str, page_size: int = 10) -> List[Dict]:
        """
        Search for articles matching a query.

        Args:
            query: Search query
            page_size: Number of articles to fetch

        Returns:
            List of articles
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Empty search query")
            return []

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100)
        }

        data = self._make_request("everything", params)

        if "error" in data:
            logger.warning(f"Error searching: {data['error']}")
            return []

        return self._parse_articles(data)

    @staticmethod
    def _parse_articles(data: Dict) -> List[Dict]:
        """
        Parse and clean article data from API response.

        Args:
            data: API response data

        Returns:
            List of cleaned articles
        """
        articles = []

        for article in data.get("articles", []):
            # Skip articles with missing critical data
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()

            if not title and not description:
                continue

            # Skip placeholder content
            if "[Removed]" in title or "[Removed]" in description:
                continue

            cleaned_article = {
                "title": title,
                "description": description,
                "content": article.get("content", "").strip(),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "author": article.get("author", ""),
                "published_at": article.get("publishedAt", ""),
                "image_url": article.get("urlToImage", "")
            }

            articles.append(cleaned_article)

        return articles


# ============== SINGLETON INSTANCE ============== #

_handler = NewsAPIHandler(API_KEY)


# ============== PUBLIC FUNCTIONS ============== #

def fetch_top_news(category: str = "general", page_size: int = 5) -> List[Dict]:
    """
    Fetch top news headlines.

    Args:
        category: News category (general, business, technology, health, etc.)
        page_size: Number of articles (default 5, max 100)

    Returns:
        List of article dictionaries with fields:
            - title
            - description
            - url
            - source
            - published_at
            - image_url
    """
    logger.info(f"Fetching top {page_size} headlines from category: {category}")
    return _handler.fetch_top_headlines(category, page_size)


def search_news(query: str, page_size: int = 10) -> List[Dict]:
    """
    Search for news articles matching a query.

    Args:
        query: Search query string
        page_size: Number of articles (default 10, max 100)

    Returns:
        List of matching articles
    """
    logger.info(f"Searching for: {query}")
    return _handler.fetch_everything(query, page_size)


def verify_article_with_news(text: str) -> Dict:
    """
    Search NewsAPI for articles related to the given text.
    Used to verify claims against news sources.

    Args:
        text: Text to verify

    Returns:
        Dict with:
            - found_articles: List of matching articles
            - trusted_sources: Count of trusted source articles
            - overall_credibility: Assessment
    """
    # Extract key terms from text for searching
    words = text.split()[:10]  # First 10 words
    query = " ".join(w for w in words if len(w) > 3)

    if not query:
        return {
            "found_articles": [],
            "trusted_sources": 0,
            "overall_credibility": "UNKNOWN"
        }

    articles = search_news(query, page_size=5)

    if not articles:
        return {
            "found_articles": [],
            "trusted_sources": 0,
            "overall_credibility": "UNKNOWN"
        }

    # Identify trusted sources
    TRUSTED_SOURCES = [
        "BBC News", "Reuters", "Associated Press", "CNN",
        "The Guardian", "The New York Times", "The Washington Post",
        "Financial Times", "The Economist", "NPR"
    ]

    trusted_count = sum(
        1 for article in articles
        if any(source in article.get("source", "") for source in TRUSTED_SOURCES)
    )

    # Determine credibility
    if trusted_count >= 3:
        credibility = "HIGH"
    elif trusted_count >= 1:
        credibility = "MEDIUM"
    else:
        credibility = "LOW"

    return {
        "found_articles": articles,
        "trusted_sources": trusted_count,
        "total_articles": len(articles),
        "overall_credibility": credibility
    }


# ============== TEST ============== #

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING NEWS SERVICE")
    print("="*80)

    # Test 1: Fetch top headlines
    print("\n📰 Test 1: Fetching top headlines...")
    headlines = fetch_top_news(category="technology", page_size=3)

    if headlines:
        for article in headlines:
            print(f"  ✓ {article['title'][:60]}...")
            print(f"    Source: {article['source']}")
    else:
        print("  ❌ No headlines fetched")

    # Test 2: Search news
    print("\n🔍 Test 2: Searching for 'AI'...")
    results = search_news("artificial intelligence", page_size=3)

    if results:
        for article in results:
            print(f"  ✓ {article['title'][:60]}...")
    else:
        print("  ❌ No results found")

    # Test 3: Verify article
    print("\n✅ Test 3: Verifying article...")
    verification = verify_article_with_news("India central bank inflation rate")

    print(f"  Found articles: {verification['total_articles']}")
    print(f"  Trusted sources: {verification['trusted_sources']}")
    print(f"  Credibility: {verification['overall_credibility']}")