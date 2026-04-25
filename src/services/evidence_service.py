# src/services/evidence_service.py

from src.services.news_service import fetch_top_news


# 🔥 Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    "breaking", "shocking", "secret", "leaked",
    "claim", "rumor", "unverified", "exclusive"
]


def analyze_keywords(text: str):
    text = text.lower()
    found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text]
    return found


def fetch_related_articles(query: str):
    try:
        articles = fetch_top_news(page_size=5)

        related_links = []
        trusted_sources = 0

        for article in articles:
            title = article.get("title", "")
            source = article.get("source", "")

            if query.lower()[:30] in title.lower():
                related_links.append(title)

            if any(src in str(article).lower() for src in ["bbc", "reuters", "ap"]):
                trusted_sources += 1

        return related_links, trusted_sources

    except:
        return [], 0


def build_evidence(text: str):

    # Keyword analysis
    keywords = analyze_keywords(text)

    # Related articles
    related_articles, trusted_count = fetch_related_articles(text)

    # Source credibility
    if trusted_count >= 2:
        credibility = "HIGH"
    elif trusted_count == 1:
        credibility = "MEDIUM"
    else:
        credibility = "LOW"

    return {
        "keywords_flagged": keywords,
        "related_articles": related_articles,
        "trusted_sources_found": trusted_count,
        "source_credibility": credibility
    }