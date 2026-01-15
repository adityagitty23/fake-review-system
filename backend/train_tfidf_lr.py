import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# load dataset
df = pd.read_csv("reviews.csv")

X = df["review_text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_vec = vectorizer.fit_transform(X)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# save models
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_model.pkl")

print("TF-IDF + Logistic Regression model saved")
