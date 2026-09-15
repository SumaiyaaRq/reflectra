# Reflectra — AI-Powered Emotional Journaling Platform 💭

Reflectra is an intelligent journaling platform that combines Natural Language Processing (NLP), emotion analysis, and interactive analytics to help users better understand their emotional patterns over time.

Built with Flask, Reflectra transforms personal journal entries into meaningful emotional insights using a multi-tier sentiment analysis pipeline, trend forecasting, and smart reflection tools.

---

## ✨ Features

### 🤖 AI Emotion Analysis
- Multi-layer NLP pipeline for emotion detection
- Logistic Regression–based emotion classification
- Rule-based fallback sentiment engine
- Optional HuggingFace transformer integration

### 📈 Emotional Analytics
- Mood trend visualization
- Emotional distribution tracking
- Personal mood forecasting
- Anomaly detection for sudden emotional drops

### 🔍 Smart Reflection Tools
- Theme & keyword extraction using RAKE
- Similar journal entry retrieval using TF-IDF cosine similarity
- Entry filtering, searching, and timeline views

### 📝 Modern Journaling Experience
- Rich journaling interface
- Mood pre-selection
- Writing prompts & templates
- Auto-save drafts
- Focus writing mode
- Word count & writing timer

### 👤 Personalized User Experience
- Secure authentication system
- User profiles & achievements
- Dashboard analytics
- Streak tracking
- Responsive modern UI

---

# 🧠 ML Pipeline

Reflectra uses a layered emotion-analysis architecture:

## Tier 1 — Transformer Model (Optional)
Uses the `cardiffnlp/twitter-roberta-base-emotion` transformer model from HuggingFace for advanced emotion classification.

## Tier 2 — Logistic Regression Classifier
A lightweight sklearn-based classifier trained on handcrafted NLP features including:
- emotion lexicon scores
- sentiment polarity
- punctuation intensity
- negation handling
- text statistics

## Tier 3 — Rule-Based Engine
Fallback keyword and polarity-based emotion detection system that works without heavy ML dependencies.

---

# 📊 Model Evaluation

The project includes a built-in evaluation pipeline with:
- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance metrics

Current evaluation results:
- **Accuracy:** 77.1%
- **Macro F1 Score:** 0.79

---

# 🛠️ Tech Stack

## Backend
- Python
- Flask
- SQLAlchemy
- SQLite

## Machine Learning / NLP
- scikit-learn
- TextBlob
- NLTK
- NumPy
- TF-IDF Similarity
- RAKE Keyword Extraction

## Frontend
- HTML5
- CSS3
- JavaScript
- Responsive UI Design

---

# 📸 Screenshots

_Add screenshots here after deployment._

Suggested screenshots:
- Landing page
- Dashboard
- Journal page
- Analytics page
- Mood forecasting section

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/reflectra.git
cd reflectra
```

---

## 2. Create virtual environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create environment variables

Create a `.env` file:

```env
SECRET_KEY=your_secret_key_here
```

---

## 5. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 📂 Project Structure

```text
reflectra/
│
├── app.py
├── requirements.txt
├── evaluate.py
├── sentiment_engine.py
├── analyze_sentiment.py
├── motivational_response.py
│
├── templates/
│   ├── landing.html
│   ├── dashboard.html
│   ├── analytics.html
│   ├── journal.html
│   ├── profile.html
│   └── ...
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── instance/
│   └── database files
│
└── README.md
```

---

# 🔐 Privacy & Security

Reflectra is designed with user privacy in mind:
- Password hashing with bcrypt
- Session-based authentication
- Secure environment variables
- Private journal storage
- HTTPS-ready deployment configuration

---

# 🌱 Future Improvements

- Transformer-based production inference
- PostgreSQL deployment
- Emotion-aware recommendations
- Voice journaling
- Mobile-first PWA support
- Embedding-based semantic search
- AI-generated reflection summaries

---




