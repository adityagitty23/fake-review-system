# Fake Review Detection System

A local-first AI system for detecting fake reviews using:
- TF-IDF + Logistic Regression
- BERT (local)
- GPT-4.1 (GitHub-hosted) for explanations

## Features
- Fully local ML inference
- Optional GPT-based explanations
- React + Flask architecture
- Privacy-first design

## Project Structure
fake-review-system/
├── frontend/
├── backend/
│ ├── app.py
│ ├── requirements.txt


## How to Run Locally

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

cd frontend
npm install
npm run dev
