# src/pipeline/predict.py

import pickle
import numpy as np
import requests
import re
import time
from typing import Dict, List

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============== CONFIG ============== #

MODEL_PATH = "models/lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LEN = 100
TEMPERATURE = 1.2

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/evidence"
N8N_TIMEOUT = 15


# ============== LOAD MODEL & TOKENIZER ============== #

print("🔹 Loading model and tokenizer...")

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None
    tokenizer = None


# ============== PREPROCESS ============== #

def preprocess_text(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')


# ============== TEMPERATURE SCALING ============== #

def apply_temperature_scaling(prob, temperature=TEMPERATURE):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    logit = np.log(prob / (1 - prob))
    scaled = logit / temperature
    return 1 / (1 + np.exp(-scaled))


# ============== INPUT NORMALIZATION ============== #

def normalize_input(text: str) -> Dict:
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty")
    cleaned = re.sub(r'[^\w\s.!?,-]', '', text).lower()
    return {"type": "text", "value": cleaned}


# ============== N8N EVIDENCE RETRIEVAL ============== #

def get_evidence_from_n8n(text: str, retry_count: int = 3) -> Dict:
    """
    POST text to N8N webhook and return structured evidence.
    N8N workflow: Text Cleaning → Keyword Extraction → NewsAPI → Article Analysis → Response
    """
    print(f"\n🔍 Querying N8N evidence engine (text: {len(text)} chars)...")

    for attempt in range(retry_count):
        try:
            print(f"  Attempt {attempt + 1}/{retry_count}...")
            response = requests.post(
                N8N_WEBHOOK_URL,
                json={"text": text},
                timeout=N8N_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            print(f"  ✓ HTTP {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                result = {
                    "query_used": data.get("query_used", ""),
                    "match_count": data.get("match_count", 0),
                    "trusted_sources": data.get("trusted_sources", 0),
                    "credibility": data.get("credibility", "UNKNOWN"),
                    "related_articles": data.get("related_articles", []),
                    "n8n_status": "success"
                }
                print(f"  ✓ Credibility={result['credibility']} | Matches={result['match_count']} | Trusted={result['trusted_sources']}")
                return result
            else:
                print(f"  ⚠️ HTTP {response.status_code}: {response.text[:100]}")

        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️ Connection refused — is N8N running? (n8n start)")
        except Exception as e:
            print(f"  ⚠️ Error: {str(e)}")

        if attempt < retry_count - 1:
            time.sleep(1.5)

    print("⚠️ N8N unavailable — using fallback (ML-only mode)")
    return _fallback_evidence()


def _fallback_evidence() -> Dict:
    return {
        "query_used": "",
        "match_count": 0,
        "trusted_sources": 0,
        "credibility": "UNKNOWN",
        "related_articles": [],
        "n8n_status": "failed"
    }


# ============== LIME EXPLANATION ============== #

def get_lime_explanation(text: str) -> List[Dict]:
    """Run LIME explainer — gracefully skip if unavailable."""
    try:
        import numpy as np
        from lime.lime_text import LimeTextExplainer

        class_names = ["REAL", "FAKE"]
        explainer = LimeTextExplainer(class_names=class_names)

        def predict_proba(texts):
            seqs = tokenizer.texts_to_sequences(texts)
            from tensorflow.keras.preprocessing.sequence import pad_sequences as ps
            padded = ps(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
            preds = model.predict(padded, verbose=0)
            return np.hstack((1 - preds, preds))

        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        return [
            {"word": word, "impact": float(score), "towards": "FAKE" if score > 0 else "REAL"}
            for word, score in exp.as_list()[:10]
        ]
    except Exception as e:
        print(f"⚠️ LIME skipped: {str(e)}")
        return []


# ============== LANGUAGE ANALYSIS ============== #

def analyze_language_patterns(text: str) -> Dict:
    patterns = {
        "sensationalism": ["breaking", "shocking", "unbelievable", "you won't believe", "exclusive", "leaked"],
        "propaganda": ["everyone knows", "obviously", "clearly", "fake news", "hoax", "conspiracy"],
        "vague_claims": ["sources say", "allegedly", "reportedly", "some say", "rumor"],
        "emotional_manipulation": ["urgent", "act now", "danger", "threat", "crisis", "must read"]
    }
    text_lower = text.lower()
    detected = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in patterns.items()}
    total = sum(detected.values())
    suspicion_score = min(total / 10, 1.0)
    return {
        "detected_patterns": detected,
        "total_pattern_count": total,
        "suspicion_score": suspicion_score,
        "language_analysis": "HIGH_SUSPICION" if suspicion_score > 0.5 else "NORMAL"
    }


# ============== FINAL VERDICT ============== #

def combine_results(ml_label, ml_confidence, evidence, language_analysis) -> str:
    credibility = evidence.get("credibility", "UNKNOWN")
    trusted = evidence.get("trusted_sources", 0)
    matches = evidence.get("match_count", 0)
    suspicion = language_analysis.get("suspicion_score", 0)
    n8n_ok = evidence.get("n8n_status") == "success"

    if not n8n_ok:
        if ml_label == "FAKE" and ml_confidence > 0.85:
            return "🔴 LIKELY FAKE (High ML Confidence — No Evidence Available)"
        elif ml_label == "REAL" and ml_confidence > 0.85:
            return "🟢 LIKELY REAL (High ML Confidence — No Evidence Available)"
        return "⚠️ INCONCLUSIVE (Low Confidence + No Evidence)"

    if ml_label == "FAKE" and ml_confidence > 0.85:
        if credibility == "LOW" and suspicion > 0.5:
            return "🔴 STRONGLY FAKE (ML + Suspicious Language + Low Credibility)"
        elif credibility == "LOW":
            return "🔴 LIKELY FAKE (High ML Confidence + Low Credibility)"
        elif credibility == "MEDIUM":
            return "⚠️ POTENTIALLY FAKE (Mixed Evidence)"
        elif credibility == "HIGH":
            return "⚠️ CONFLICT: Model says FAKE but Evidence is Credible"

    if ml_label == "REAL" and ml_confidence > 0.85:
        if credibility == "HIGH" and trusted >= 2:
            return "🟢 VERIFIED REAL (ML + High Credibility + Trusted Sources)"
        elif credibility == "HIGH":
            return "🟢 LIKELY REAL (ML + High Credibility)"
        elif credibility == "MEDIUM":
            return "⚠️ WEAK SUPPORT (Model: Real, Evidence: Medium)"
        elif credibility == "LOW":
            return "⚠️ CONFLICT: Model says REAL but Low Credibility"

    if ml_confidence > 0.6:
        return f"⚠️ QUESTIONABLE (Moderate {ml_label} Prediction)"

    return "⚠️ INCONCLUSIVE (Low ML Confidence)"


# ============== MAIN PREDICT ============== #

def predict(text: str, input_type: str = "text", source: str = "unknown") -> Dict:
    """
    Full prediction pipeline:
    1. Validate & clean input
    2. LSTM model prediction
    3. LIME explanation
    4. N8N evidence retrieval
    5. Language pattern analysis
    6. Combined final verdict
    """
    start = time.time()

    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Input text too short (minimum 10 characters)")

        normalized = normalize_input(text)
        clean = normalized["value"]

        if model is None or tokenizer is None:
            raise Exception("Model not loaded. Check models/ directory.")

        # ML Prediction
        print(f"\n🧠 Running LSTM on {len(clean)} chars...")
        processed = preprocess_text(clean)
        raw = model.predict(processed, verbose=0)[0][0]
        calibrated = apply_temperature_scaling(raw)
        label = "FAKE" if calibrated > 0.5 else "REAL"
        confidence = calibrated if calibrated > 0.5 else (1 - calibrated)
        print(f"   → {label} | confidence={confidence:.4f}")

        # LIME
        explanation = get_lime_explanation(clean)

        # N8N Evidence
        evidence = get_evidence_from_n8n(clean)

        # Language Analysis
        lang = analyze_language_patterns(clean)

        # Final Verdict
        verdict = combine_results(label, confidence, evidence, lang)

        return {
            "input_type": input_type,
            "source": source,
            "prediction": label,
            "confidence": float(confidence),
            "raw_probability": float(calibrated),
            "final_verdict": verdict,
            "evidence": evidence,
            "explanation": explanation,
            "language_analysis": lang,
            "processing_time": round(time.time() - start, 2),
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            "input_type": input_type,
            "source": source,
            "prediction": "ERROR",
            "confidence": 0.0,
            "final_verdict": f"❌ ERROR: {str(e)}",
            "evidence": {},
            "explanation": [],
            "language_analysis": {},
            "processing_time": round(time.time() - start, 2),
            "status": "error",
            "error": str(e)
        }


def predict_batch(texts: List[str]) -> List[Dict]:
    return [predict(t) for t in texts]


# ============== QUICK TEST ============== #

if __name__ == "__main__":
    samples = [
        "India's central bank RBI keeps repo rate unchanged amid inflation concerns",
        "Scientists confirm that drinking hot water cures cancer instantly",
        "Breaking: Secret government conspiracy leaked online"
    ]
    for t in samples:
        r = predict(t)
        print(f"\n📝 {t[:60]}...")
        print(f"   Prediction : {r['prediction']} ({r['confidence']:.4f})")
        print(f"   Verdict    : {r['final_verdict']}")
        print(f"   N8N Status : {r['evidence'].get('n8n_status')}")
        print(f"   Time       : {r['processing_time']}s")