# src/api/main.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from src.pipeline.predict import predict
from src.services.news_service import fetch_top_news
from src.services.url_handler import fetch_and_extract_text
from src.services.document_handler import extract_text_from_document


# ============== APP INIT ============== #

app = FastAPI(
    title="Fake News Detection API",
    description="LSTM + LIME + N8N Evidence Engine — Multi-input Fake News Detector",
    version="3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== SCHEMAS ============== #

class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str


# ============== ROOT ============== #

@app.get("/")
def home():
    return {
        "message": "Fake News Detection API v3.1 🚀",
        "endpoints": {
            "POST /predict/text": "Analyze raw text",
            "POST /predict/url": "Analyze webpage content",
            "POST /predict/document": "Analyze PDF / DOCX / TXT upload",
            "GET  /news": "Analyze live top headlines",
            "GET  /health": "Health check"
        }
    }


# ============== HEALTH ============== #

@app.get("/health")
def health_check():
    from src.pipeline.predict import model, tokenizer
    return {
        "status": "healthy",
        "api_version": "3.1",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "n8n_webhook": "http://localhost:5678/webhook/evidence"
    }


# ============== PREDICT TEXT ============== #

@app.post("/predict/text")
def predict_text(req: TextRequest):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(400, "Text must be at least 10 characters")
    result = predict(req.text, input_type="text", source="direct_input")
    if result["status"] == "error":
        raise HTTPException(500, result.get("error", "Prediction failed"))
    return result


# ============== PREDICT URL ============== #

@app.post("/predict/url")
def predict_url(req: URLRequest):
    if not req.url.startswith("http"):
        raise HTTPException(400, "Invalid URL — must start with http/https")

    extracted = fetch_and_extract_text(req.url)
    if not extracted["success"]:
        raise HTTPException(400, f"Could not fetch URL: {extracted.get('error')}")

    text = extracted["text"]
    domain = extracted.get("domain", "unknown")
    headline = extracted.get("headline", "")

    result = predict(text, input_type="url", source=req.url)

    result["url_metadata"] = {
        "domain": domain,
        "headline": headline,
        "content_length": extracted.get("content_length", 0),
        "domain_trust": _assess_domain_trust(domain)
    }
    return result


# ============== PREDICT DOCUMENT ============== #

@app.post("/predict/document")
async def predict_document(file: UploadFile = File(...)):
    allowed = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]
    if file.content_type not in allowed:
        raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")

    ext = file.filename.split(".")[-1].lower()
    extracted = extract_text_from_document(content, ext)

    if not extracted["success"]:
        raise HTTPException(400, f"Text extraction failed: {extracted.get('error')}")

    text = extracted["text"]
    chunks = extracted.get("chunks", [text])
    summary = extracted.get("summary", text[:500])

    if len(chunks) > 1:
        chunk_results = [
            predict(chunk, input_type="document_chunk", source=f"{file.filename} – part {i+1}")
            for i, chunk in enumerate(chunks)
        ]
        avg_conf = sum(r["confidence"] for r in chunk_results) / len(chunk_results)
        fake_count = sum(1 for r in chunk_results if r["prediction"] == "FAKE")
        overall = predict(summary, input_type="document_summary", source=file.filename)
        overall["chunk_analysis"] = {
            "total_chunks": len(chunks),
            "fake_chunks": fake_count,
            "fake_percentage": round((fake_count / len(chunks)) * 100, 1),
            "avg_confidence": round(avg_conf, 4),
        }
        return {**overall, "document_type": "multi_chunk"}

    result = predict(text, input_type="document", source=file.filename)
    return {**result, "document_type": "single", "text_length": len(text)}


# ============== LIVE NEWS ============== #

@app.get("/news")
def analyze_live_news(limit: int = 10):
    articles = fetch_top_news(page_size=limit)
    results = []
    for article in articles:
        text = article.get("title") or article.get("description")
        if not text:
            continue
        pred = predict(text, input_type="news_headline", source=article.get("source", "unknown"))
        results.append({
            "title": text,
            "source": article.get("source"),
            "url": article.get("url"),
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "final_verdict": pred["final_verdict"]
        })
    return {"total_articles": len(results), "news_analysis": results, "status": "success"}


# ============== HELPER ============== #

def _assess_domain_trust(domain: str) -> dict:
    TRUSTED = {
        "bbc.com": 0.95, "reuters.com": 0.95, "apnews.com": 0.93,
        "theguardian.com": 0.92, "nytimes.com": 0.92,
        "washingtonpost.com": 0.91, "cnn.com": 0.88, "ft.com": 0.90
    }
    d = domain.lower()
    for td, score in TRUSTED.items():
        if td in d:
            return {"is_trusted": True, "trust_score": score, "category": "established_news_outlet"}
    if any(t in d for t in ["news", "times", "press", "journal", "herald"]):
        return {"is_trusted": False, "trust_score": 0.5, "category": "potential_news_site"}
    return {"is_trusted": False, "trust_score": 0.3, "category": "unknown_source"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)