import os
from dotenv import load_dotenv
from openai import OpenAI

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- SETUP ---------------- #

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    base_url=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_KEY")
)

torch.set_num_threads(1)

# ---------------- LOAD MODELS ---------------- #

tfidf = joblib.load("tfidf_vectorizer.pkl")
logistic = joblib.load("logistic_model.pkl")

tokenizer = BertTokenizer.from_pretrained("bert")
bert_model = BertForSequenceClassification.from_pretrained("bert")
bert_model.eval()

# ---------------- GPT EXPLANATION ---------------- #

def gpt_explain(review_text, prediction):
    try:
        response = client.chat.completions.create(
            model=os.getenv("GPT_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fake review detection system."
                },
                {
                    "role": "user",
                    "content": f"""
Review: "{review_text}"
Prediction: {prediction}

Explain briefly why this review is classified as {prediction}.
"""
                }
            ],
            temperature=0.2,
            max_tokens=80
        )
        return response.choices[0].message.content
    except Exception:
        return "Explanation unavailable"

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return jsonify({
        "status": "Backend running",
        "models": ["TF-IDF + Logistic", "BERT", "GPT-4.1 (explanations)"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    reviews = data.get("reviews", [])

    texts = [r["text"] for r in reviews]
    ids = [r["id"] for r in reviews]

    vectors = tfidf.transform(texts)
    lr_preds = logistic.predict(vectors)
    lr_probs = logistic.predict_proba(vectors)

    results = []

    for i, text in enumerate(texts):
        # ---- BERT ----
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        bert_pred = torch.argmax(probs).item()
        bert_conf = torch.max(probs).item() * 100

        # ---- Logistic ----
        lr_pred = lr_preds[i]
        lr_conf = max(lr_probs[i]) * 100

        # ---- Ensemble ----
        final_label = "Fake" if (lr_pred == 1 or bert_pred == 1) else "Genuine"
        final_conf = round((lr_conf + bert_conf) / 2, 2)

        explanation = gpt_explain(text, final_label)

        results.append({
            "reviewId": ids[i],
            "label": final_label,
            "confidenceScore": final_conf,
            "reason": explanation,
            "sentiment": "Neutral"
        })

    return jsonify(results)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
