import os
import json
import re
import uuid
import requests
import torch
import joblib
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- LOAD ENV ---------------- #

load_dotenv()

# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
CORS(app)

GITHUB_TOKEN = os.getenv("AZURE_KEY")
ENDPOINT = "https://models.github.ai/inference/chat/completions"
MODEL = os.getenv("GPT_MODEL") or "openai/gpt-4.1"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}

torch.set_num_threads(1)

# ---------------- AUDIT STORAGE ---------------- #

AUDIT_FILE = "audit_history.json"

def load_audits():
    if not os.path.exists(AUDIT_FILE):
        return []
    with open(AUDIT_FILE, "r") as f:
        return json.load(f)

def save_audit(audit):
    audits = load_audits()
    audits.insert(0, audit)
    with open(AUDIT_FILE, "w") as f:
        json.dump(audits, f, indent=2)

# ---------------- LOAD LOCAL MODELS ---------------- #

tfidf = joblib.load("tfidf_vectorizer.pkl")
logistic = joblib.load("logistic_model.pkl")

tokenizer = BertTokenizer.from_pretrained("bert")
bert_model = BertForSequenceClassification.from_pretrained("bert")
bert_model.eval()

# ---------------- AI HELPERS ---------------- #

def ai_analyze_review(review_text):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict fake review detection system.\n\n"
                    "A review is FAKE if it is generic, promotional, vague, or exaggerated.\n"
                    "A review is GENUINE if it includes real usage details and balanced opinions.\n\n"
                    "Return ONLY valid JSON:\n"
                    "{\n"
                    '  "label": "Fake" | "Genuine",\n'
                    '  "confidence": number (0-100),\n'
                    '  "sentiment": "Positive" | "Neutral" | "Negative",\n'
                    '  "reason": "short explanation"\n'
                    "}"
                )
            },
            {"role": "user", "content": review_text}
        ],
        "temperature": 0.3
    }

    response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("Invalid AI JSON")

    return json.loads(match.group())

# ---------------- TRUE RATING ---------------- #

def calculate_true_rating(reviews, results):
    weighted_sum = 0
    weight_total = 0

    for r in results:
        review = next((rv for rv in reviews if rv["id"] == r["reviewId"]), None)
        if not review:
            continue

        rating = review["rating"]
        confidence = float(r["confidenceScore"]) / 100

        if r["label"] == "Genuine":
            weight = confidence
        else:
            weight = max(0.1, 1 - confidence) * 0.3

        weighted_sum += rating * weight
        weight_total += weight

    if weight_total == 0:
        return 0

    return round(weighted_sum / weight_total, 2)

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return jsonify({"status": "Backend running"})

# ---------- LOCAL ML PIPELINE ---------- #

@app.route("/predict", methods=["POST"])
def predict_local():
    data = request.get_json()
    reviews = data.get("reviews", [])

    texts = [r["text"] for r in reviews]
    ids = [r["id"] for r in reviews]

    vectors = tfidf.transform(texts)
    lr_preds = logistic.predict(vectors)
    lr_probs = logistic.predict_proba(vectors)

    results = []

    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        bert_pred = torch.argmax(probs).item()
        bert_conf = torch.max(probs).item() * 100

        lr_pred = lr_preds[i]
        lr_conf = max(lr_probs[i]) * 100

        final_label = "Fake" if (bert_pred == 1 and lr_pred == 1) else "Genuine"
        final_conf = round((bert_conf + lr_conf) / 2, 2)

        results.append({
            "reviewId": ids[i],
            "label": final_label,
            "confidenceScore": final_conf,
            "sentiment": "Neutral",
            "reason": "Local ML decision"
        })

    true_rating = calculate_true_rating(reviews, results)

    audit = {
        "auditId": str(uuid.uuid4()),
        "mode": "local",
        "timestamp": datetime.now().isoformat(),
        "reviews": reviews,
        "results": results,
        "trueRating": true_rating
    }

    save_audit(audit)

    return jsonify({
        "results": results,
        "trueRating": true_rating
    })

# ---------- AI PIPELINE ---------- #

@app.route("/predict-ai", methods=["POST"])
def predict_ai():
    data = request.get_json()
    reviews = data.get("reviews", [])

    results = []

    for r in reviews:
        try:
            ai = ai_analyze_review(r["text"])
            results.append({
                "reviewId": r["id"],
                "label": ai["label"],
                "confidenceScore": ai["confidence"],
                "sentiment": ai["sentiment"],
                "reason": ai["reason"]
            })
        except Exception as e:
            results.append({
                "reviewId": r["id"],
                "label": "Unknown",
                "confidenceScore": 0,
                "sentiment": "Neutral",
                "reason": str(e)
            })

    true_rating = calculate_true_rating(reviews, results)

    audit = {
        "auditId": str(uuid.uuid4()),
        "mode": "ai",
        "timestamp": datetime.now().isoformat(),
        "reviews": reviews,
        "results": results,
        "trueRating": true_rating
    }

    save_audit(audit)

    return jsonify({
        "results": results,
        "trueRating": true_rating
    })

# ---------- AUDIT HISTORY (MATCHES REACT UI) ---------- #

@app.route("/audits", methods=["GET"])
def get_audits():
    audits = load_audits()
    formatted = []

    for a in audits:
        reviews = a["reviews"]
        results = a["results"]

        fake_count = sum(1 for r in results if r["label"] == "Fake")
        genuine_count = sum(1 for r in results if r["label"] == "Genuine")

        original_avg = (
            sum(r["rating"] for r in reviews) / len(reviews)
            if reviews else 0
        )

        trust_score = round(
            (genuine_count / len(reviews)) * 100, 1
        ) if reviews else 0

        formatted.append({
            "id": a["auditId"],
            "projectName": "Fake Review Detection",
            "engine": "REVIEW SHIELD" if a["mode"] == "ai" else "LOCAL",
            "date": a["timestamp"],
            "reviewCount": len(reviews),
            "fakeCount": fake_count,
            "trustScore": trust_score,
            "summary": {
                "genuineCount": genuine_count,
                "originalAvgRating": round(original_avg, 1),
                "trueAvgRating": a["trueRating"]
            }
        })

    return jsonify(formatted)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    if not os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "w") as f:
            json.dump([], f)

    app.run(debug=True)
