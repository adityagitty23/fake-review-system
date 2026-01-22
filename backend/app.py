import os
import json
import re
import requests
import torch
import joblib
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
                    "You are an impartial review auditor. "
                    "Classify the review conservatively.\n\n"
                    "Return ONLY valid JSON:\n"
                    "{\n"
                    '  "label": "Genuine" | "Fake",\n'
                    '  "confidence": number (0-100),\n'
                    '  "sentiment": "Positive" | "Neutral" | "Negative",\n'
                    '  "reason": "brief explanation"\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": review_text
            }
        ],
        "temperature": 0.1
    }

    response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError("AI did not return valid JSON")

    return json.loads(match.group())


def ai_explain_prediction(review_text, label, confidence):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain clearly why a review was classified as Fake or Genuine. "
                    "Keep it short and user-friendly."
                )
            },
            {
                "role": "user",
                "content": (
                    f'Review: "{review_text}"\n'
                    f'Prediction: {label}\n'
                    f'Confidence: {confidence:.1f}%\n\n'
                    "Explain the reasoning."
                )
            }
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "AI explanation unavailable."

# ---------------- TRUE RATING ---------------- #

def calculate_true_rating(reviews, results):
    weighted_sum = 0.0
    weight_total = 0.0

    for r in results:
        review = next((rv for rv in reviews if rv["id"] == r["reviewId"]), None)
        if not review:
            continue

        rating = review["rating"]
        confidence = max(0.0, min(float(r["confidenceScore"]) / 100, 1.0))

        if r["label"] == "Genuine":
            weight = confidence
        else:
            weight = max(0.1, 1 - confidence) * 0.3

        weighted_sum += rating * weight
        weight_total += weight

    if weight_total == 0:
        return 0.0, "Not enough trustworthy reviews to compute true rating."

    return round(weighted_sum / weight_total, 2), (
        "True rating is calculated using confidence-weighted reviews. "
        "Fake reviews contribute significantly less."
    )

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return jsonify({
        "status": "Backend running",
        "modes": ["local", "ai"],
        "features": ["Local ML", "AI Explanation", "True Rating"]
    })

# ---------- LOCAL PIPELINE ---------- #

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

        final_label = "Fake" if (lr_pred == 1 and bert_pred == 1) else "Genuine"
        final_conf = round((lr_conf + bert_conf) / 2, 2)

        explanation = ai_explain_prediction(text, final_label, final_conf)

        results.append({
            "reviewId": ids[i],
            "label": final_label,
            "confidenceScore": final_conf,
            "sentiment": "Neutral",
            "reason": explanation
        })

    true_rating, rating_explanation = calculate_true_rating(reviews, results)

    return jsonify({
        "results": results,
        "trueRating": true_rating,
        "ratingExplanation": rating_explanation
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
                "label": ai.get("label", "Genuine"),
                "confidenceScore": float(ai.get("confidence", 50)),
                "sentiment": ai.get("sentiment", "Neutral"),
                "reason": ai.get("reason", "")
            })

        except Exception:
            results.append({
                "reviewId": r["id"],
                "label": "Genuine",
                "confidenceScore": 50,
                "sentiment": "Neutral",
                "reason": "AI fallback decision"
            })

    true_rating, rating_explanation = calculate_true_rating(reviews, results)

    return jsonify({
        "results": results,
        "trueRating": true_rating,
        "ratingExplanation": rating_explanation
    })

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
