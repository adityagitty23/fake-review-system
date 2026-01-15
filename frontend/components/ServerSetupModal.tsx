import React from 'react';
import { X, Copy, Terminal, Check } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

const ServerSetupModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const [copied, setCopied] = React.useState(false);

  if (!isOpen) return null;

  const pythonCode = `# backend/server.py
# Requirements: pip install fastapi uvicorn scikit-learn transformers torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import random

# --- ML SIMULATION (BERT + TF-IDF + LR) ---
# In a real app, you would load your trained .pkl models here.
# Since we don't have your dataset, we will simulate the pipeline logic.

from transformers import pipeline

print("Loading BERT Sentiment Model...")
# Using a small, fast BERT model for sentiment
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewInput(BaseModel):
    id: str
    text: string
    rating: int

class RequestBody(BaseModel):
    reviews: List[ReviewInput]

@app.post("/analyze")
async def analyze_reviews(body: RequestBody):
    results = []
    
    print(f"Processing {len(body.reviews)} reviews...")
    
    # Batch sentiment analysis for efficiency
    texts = [r.text for r in body.reviews]
    sentiments = sentiment_pipeline(texts) # BERT Step

    for i, review in enumerate(body.reviews):
        text_lower = review.text.lower()
        bert_sentiment = sentiments[i]
        
        # --- LOGISTIC REGRESSION / TF-IDF LOGIC SIMULATION ---
        # Heuristics mimicking what a TF-IDF+LR model looks for:
        
        score_modifiers = 0
        reasons = []

        # 1. Extreme Lengths (TF-IDF sparse/dense checks)
        if len(text_lower.split()) < 3:
            score_modifiers -= 20
            reasons.append("Review too short")
        
        # 2. Promotional Language (High coefficients in LR)
        promo_words = ["guarantee", "buy now", "click here", "amazing product", "100%"]
        if any(w in text_lower for w in promo_words):
            score_modifiers -= 30
            reasons.append("Promotional language detected")

        # 3. Sentiment Mismatch (BERT Cross-check)
        # If rating is 1 but BERT says POSITIVE, or Rating 5 and NEGATIVE
        mapped_sentiment = "Positive" if bert_sentiment['label'] == 'POSITIVE' else "Negative"
        
        if review.rating == 5 and mapped_sentiment == "Negative":
            score_modifiers -= 40
            reasons.append("Rating (5) contradicts Text Sentiment (Negative)")
        elif review.rating == 1 and mapped_sentiment == "Positive":
            score_modifiers -= 40
            reasons.append("Rating (1) contradicts Text Sentiment (Positive)")

        # Final Classification
        base_confidence = 85 + score_modifiers
        is_fake = base_confidence < 50
        
        results.append({
            "reviewId": review.id,
            "label": "Fake" if is_fake else "Genuine",
            "confidenceScore": min(max(int(base_confidence if not is_fake else (100 - base_confidence)), 0), 100),
            "reason": ", ".join(reasons) if reasons else "Consistent language patterns",
            "sentiment": "Neutral" if review.rating == 3 else mapped_sentiment
        })

    return {"results": results}

if __name__ == "__main__":
    print("Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;

  const handleCopy = () => {
    navigator.clipboard.writeText(pythonCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-3xl rounded-xl shadow-2xl flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/10 rounded-lg">
                <Terminal className="w-6 h-6 text-green-500" />
            </div>
            <div>
                <h2 className="text-xl font-bold text-white">Local ML Server Setup</h2>
                <p className="text-sm text-slate-400">Run BERT + TF-IDF + LR locally</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          <div className="space-y-6">
            <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg">
                <h3 className="text-blue-400 font-semibold mb-1">How it works</h3>
                <p className="text-sm text-slate-300">
                    To use local ML models, you need to run a Python backend. This code spins up a FastAPI server that uses 
                    <strong> HuggingFace Transformers (BERT)</strong> for sentiment analysis and mimics a 
                    <strong> Logistic Regression</strong> decision boundary for detection.
                </p>
            </div>

            <div className="space-y-2">
                <p className="text-sm font-medium text-slate-300">1. Install Dependencies</p>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-sm text-green-400">
                    pip install fastapi uvicorn scikit-learn transformers torch
                </div>
            </div>

            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <p className="text-sm font-medium text-slate-300">2. server.py</p>
                    <button 
                        onClick={handleCopy}
                        className="flex items-center gap-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded transition-colors"
                    >
                        {copied ? <Check className="w-3 h-3 text-green-400"/> : <Copy className="w-3 h-3"/>}
                        {copied ? "Copied!" : "Copy Code"}
                    </button>
                </div>
                <div className="relative">
                    <pre className="bg-slate-950 p-4 rounded-lg border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto h-64 custom-scrollbar">
                        {pythonCode}
                    </pre>
                </div>
            </div>

             <div className="space-y-2">
                <p className="text-sm font-medium text-slate-300">3. Run Server</p>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-sm text-green-400">
                    python server.py
                </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-slate-800 bg-slate-900/50 rounded-b-xl flex justify-end">
            <button 
                onClick={onClose}
                className="bg-slate-800 hover:bg-slate-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
            >
                Done
            </button>
        </div>
      </div>
    </div>
  );
};

export default ServerSetupModal;
