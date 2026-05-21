import re
import random
import json
import os
import math
from collections import Counter
 
# ── Configuration ────────────────────────────────────────────────────────────
USE_TRANSFORMERS = os.environ.get("USE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}
USE_SKLEARN_CLASSIFIER = True     # Logistic-regression tier (always available)
 
# Attempt HuggingFace import
_hf_pipeline = None
if USE_TRANSFORMERS:
    try:
        from transformers import pipeline as hf_pipeline
        _hf_pipeline = hf_pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            return_all_scores=True,
        )
    except Exception as e:
        print(f"[ML] HuggingFace unavailable: {e}. Falling back to sklearn tier.")
 
# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)           # normalise repeated chars
    text = re.sub(r"https?://\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
# ── Feature extraction (Tier 2) ───────────────────────────────────────────────
# Rich emotion lexicon — each entry is (keyword, emotion, weight)
EMOTION_LEXICON = [
    # happy / joy
    ("happy", "happy", 2), ("joy", "happy", 2), ("joyful", "happy", 2),
    ("excited", "excited", 2), ("thrilled", "excited", 2), ("elated", "happy", 2),
    ("grateful", "happy", 1.5), ("thankful", "happy", 1.5), ("love", "happy", 1.5),
    ("wonderful", "happy", 1.5), ("amazing", "happy", 1.5), ("fantastic", "happy", 1.5),
    ("delighted", "happy", 2), ("cheerful", "happy", 1.5), ("blissful", "happy", 2),
    ("content", "peaceful", 1.5), ("satisfied", "peaceful", 1),
    ("proud", "happy", 1.5), ("accomplished", "happy", 1.5),
 
    # sad / grief
    ("sad", "sad", 2), ("unhappy", "sad", 1.5), ("cry", "sad", 1.5),
    ("crying", "sad", 2), ("depressed", "sad", 2), ("lonely", "sad", 2),
    ("grief", "sad", 2), ("heartbroken", "sad", 2), ("miserable", "sad", 2),
    ("hopeless", "sad", 2), ("despair", "sad", 2), ("devastated", "sad", 2),
    ("sorrowful", "sad", 2), ("gloomy", "sad", 1.5), ("melancholy", "sad", 2),
    ("worthless", "sad", 2), ("empty", "sad", 1.5), ("broken", "sad", 1.5),
 
    # angry
    ("angry", "angry", 2), ("mad", "angry", 1.5), ("furious", "angry", 2),
    ("annoyed", "angry", 1.5), ("irritated", "angry", 1.5), ("rage", "angry", 2),
    ("frustrated", "angry", 1.5), ("outraged", "angry", 2), ("bitter", "angry", 1.5),
    ("resentful", "angry", 1.5), ("hostile", "angry", 2), ("enraged", "angry", 2),
 
    # anxious / stressed
    ("anxious", "anxious", 2), ("anxiety", "anxious", 2), ("worried", "anxious", 1.5),
    ("nervous", "anxious", 1.5), ("stressed", "anxious", 2), ("overwhelmed", "anxious", 2),
    ("panic", "anxious", 2), ("fear", "anxious", 1.5), ("scared", "anxious", 2),
    ("dread", "anxious", 2), ("tense", "anxious", 1.5), ("uneasy", "anxious", 1.5),
    ("apprehensive", "anxious", 1.5),
 
    # peaceful / calm
    ("calm", "peaceful", 2), ("peaceful", "peaceful", 2), ("serene", "peaceful", 2),
    ("relaxed", "peaceful", 2), ("tranquil", "peaceful", 2), ("centered", "peaceful", 1.5),
    ("mindful", "peaceful", 1.5), ("balanced", "peaceful", 1.5),
 
    # excited
    ("energetic", "excited", 1.5), ("pumped", "excited", 1.5),
    ("enthusiastic", "excited", 1.5), ("motivated", "excited", 1.5),
    ("inspired", "excited", 1.5), ("passionate", "excited", 1.5),
]
 
EMOTION_LABELS = ["happy", "sad", "angry", "anxious", "peaceful", "excited", "neutral"]
 
 
def extract_features(text: str) -> dict:
    """Return a feature dict useful for the sklearn tier and for analytics."""
    cleaned = clean_text(text)
    words = cleaned.split()
    word_set = set(words)
    n = max(len(words), 1)
 
    # Emotion scores via weighted lexicon
    emotion_scores = Counter()
    for keyword, emotion, weight in EMOTION_LEXICON:
        if keyword in word_set:
            emotion_scores[emotion] += weight
 
    # TextBlob polarity (still useful as a feature)
    polarity = 0.0
    try:
        from textblob import TextBlob
        polarity = TextBlob(cleaned).sentiment.polarity
    except Exception:
        pass
 
    # Negation detection — flip scores if preceded by negation
    negations = {"not", "never", "no", "don't", "doesn't", "didn't", "can't",
                 "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "wasn't"}
    neg_window = 3
    for i, w in enumerate(words):
        if w in negations:
            for j in range(i + 1, min(i + neg_window + 1, len(words))):
                target = words[j]
                for keyword, emotion, weight in EMOTION_LEXICON:
                    if keyword == target:
                        emotion_scores[emotion] -= weight * 0.8
 
    # Punctuation intensity clues
    excl = text.count("!")
    quest = text.count("?")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
 
    # Avg word length (longer words → more thoughtful / reflective)
    avg_word_len = sum(len(w) for w in words) / n
 
    # Sentence count
    sentences = re.split(r"[.!?]+", text)
    n_sentences = max(len([s for s in sentences if s.strip()]), 1)
    avg_sentence_len = n / n_sentences
 
    return {
        "emotion_scores": dict(emotion_scores),
        "polarity": polarity,
        "exclamations": excl,
        "questions": quest,
        "caps_ratio": caps_ratio,
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "word_count": n,
        "n_sentences": n_sentences,
    }
 
 
def rule_based_emotion(features: dict) -> tuple[str, float]:
    """
    Tier-2 rule classifier. Returns (emotion, confidence 0-1).
    Used when sklearn model isn't fitted yet.
    """
    scores = features["emotion_scores"]
    polarity = features["polarity"]
 
    if scores:
        best_emotion = max(scores, key=scores.get)
        total = sum(abs(v) for v in scores.values())
        confidence = min(scores[best_emotion] / max(total, 1), 1.0)
        return best_emotion, confidence
 
    # Fall back to polarity
    if polarity > 0.5:
        return "happy", 0.5
    elif polarity > 0.1:
        return "peaceful", 0.4
    elif polarity < -0.5:
        return "sad", 0.5
    elif polarity < -0.1:
        return "anxious", 0.4
    return "neutral", 0.3
 
 
# ── Tier-2 Logistic Regression classifier ────────────────────────────────────
class SklearnEmotionClassifier:
    """
    Lightweight logistic-regression emotion classifier.
    Trained on a synthetic seed dataset; grows better with real user labels.
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "lr_model.json")
 
    def __init__(self):
        self.fitted = False
        self.classes_ = EMOTION_LABELS
        self._weights = None
        self._bias = None
        self._load()
 
    def _feature_vector(self, features: dict) -> list:
        scores = features["emotion_scores"]
        return [
            scores.get("happy", 0),
            scores.get("sad", 0),
            scores.get("angry", 0),
            scores.get("anxious", 0),
            scores.get("peaceful", 0),
            scores.get("excited", 0),
            features["polarity"],
            features["exclamations"],
            features["caps_ratio"],
            features["avg_word_len"],
            features["avg_sentence_len"],
            features["word_count"] / 100,
        ]
 
    def _softmax(self, z: list) -> list:
        m = max(z)
        exps = [math.exp(v - m) for v in z]
        s = sum(exps)
        return [e / s for e in exps]
 
    def predict_proba(self, features: dict) -> dict:
        if not self.fitted:
            return {}
        x = self._feature_vector(features)
        logits = []
        for w_row, b in zip(self._weights, self._bias):
            logits.append(sum(xi * wi for xi, wi in zip(x, w_row)) + b)
        probs = self._softmax(logits)
        return dict(zip(self.classes_, probs))
 
    def fit(self, X_features: list, y_labels: list):
        """Train with sklearn LogisticRegression then serialise weights."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
            import numpy as np
 
            X = np.array([self._feature_vector(f) for f in X_features])
            le = LabelEncoder()
            le.fit(EMOTION_LABELS)
            y = le.transform(y_labels)
 
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X, y)
 
            self._weights = clf.coef_.tolist()
            self._bias = clf.intercept_.tolist()
            self.classes_ = list(le.classes_)
            self.fitted = True
            self._save()
            return True
        except Exception as e:
            print(f"[ML] sklearn fit failed: {e}")
            return False
 
    def _save(self):
        data = {
            "weights": self._weights,
            "bias": self._bias,
            "classes": self.classes_,
        }
        with open(self.MODEL_PATH, "w") as f:
            json.dump(data, f)
 
    def _load(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH) as f:
                    data = json.load(f)
                self._weights = data["weights"]
                self._bias = data["bias"]
                self.classes_ = data["classes"]
                self.fitted = True
            except Exception:
                pass
 
    def bootstrap_train(self):
        """
        Train on a curated seed dataset so the model works out-of-the-box
        without any user data. Call once on first run.
        """
        seed_data = [
            # happy
            ("Today was absolutely wonderful, I felt so joyful and grateful", "happy"),
            ("I'm so happy and excited about this amazing opportunity", "happy"),
            ("Feeling grateful and content, life is good right now", "happy"),
            ("Had such a fun day with friends, laughing and celebrating", "happy"),
            ("Finally accomplished my goal, I'm so proud and delighted", "happy"),
            # sad
            ("I feel so sad and lonely today, nothing seems to matter", "sad"),
            ("Crying all day, feeling hopeless and completely lost", "sad"),
            ("The grief is overwhelming, I miss them so much", "sad"),
            ("Feeling depressed and worthless, can't find any motivation", "sad"),
            ("Everything feels empty and pointless right now", "sad"),
            # angry
            ("I'm so furious and outraged at what happened today", "angry"),
            ("This is completely unfair, I'm absolutely enraged", "angry"),
            ("Fed up and frustrated with being treated this way", "angry"),
            ("So angry and resentful, I can't let it go", "angry"),
            ("This situation makes me mad and bitter every time", "angry"),
            # anxious
            ("Feeling so anxious and overwhelmed about everything", "anxious"),
            ("The panic attacks are getting worse, I'm terrified", "anxious"),
            ("Stressed and nervous about the upcoming presentation", "anxious"),
            ("Constant worry and dread, I can't sleep at night", "anxious"),
            ("Feeling tense and uneasy, scared of what comes next", "anxious"),
            # peaceful
            ("Feeling so calm and centered after my meditation session", "peaceful"),
            ("A tranquil morning walk, feeling serene and balanced", "peaceful"),
            ("Relaxed and at peace with everything in my life", "peaceful"),
            ("Mindful and present, enjoying this quiet moment", "peaceful"),
            ("Feeling grounded and content with where I am", "peaceful"),
            # excited
            ("So pumped and energetic about my new project!", "excited"),
            ("Thrilled and enthusiastic, can't wait to get started", "excited"),
            ("Feeling motivated and inspired by new possibilities", "excited"),
            ("Super excited and passionate about this opportunity!", "excited"),
            ("Energised and enthusiastic, ready to take on the world", "excited"),
            # neutral
            ("Had a regular day, nothing particularly notable happened", "neutral"),
            ("Just going through the motions, feeling neither good nor bad", "neutral"),
            ("A quiet and ordinary day, thinking about nothing specific", "neutral"),
            ("Didn't feel much today, just existing and going through the day", "neutral"),
            ("Normal day at work, everything was as expected", "neutral"),
        ]
        X_features = [extract_features(text) for text, _ in seed_data]
        y_labels = [label for _, label in seed_data]
        self.fit(X_features, y_labels)
 
 
# Singleton
_classifier = SklearnEmotionClassifier()
if not _classifier.fitted:
    _classifier.bootstrap_train()
 
 
# ── Key theme extraction ───────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "it", "they", "his",
    "her", "its", "their", "which", "who", "whom", "when", "where", "how",
    "what", "not", "no", "so", "just", "very", "really", "quite", "about",
    "up", "out", "if", "from", "by", "more", "also", "than", "then", "like",
    "get", "got", "feel", "felt", "feeling", "day", "time", "today",
}
 
def extract_key_themes(text: str, n: int = 5) -> list[str]:
    """
    RAKE-inspired keyword extraction using word co-occurrence scoring.
    Far more meaningful than filtering for words > 5 chars.
    """
    cleaned = clean_text(text)
    # Split on stopwords to get candidate phrases
    stopword_pattern = r"\b(?:" + "|".join(re.escape(s) for s in STOPWORDS) + r")\b"
    candidates_raw = re.split(stopword_pattern, cleaned)
    candidates = [c.strip() for c in candidates_raw if c.strip()]
 
    # Score each candidate phrase
    word_freq = Counter()
    word_degree = Counter()
    for phrase in candidates:
        words = phrase.split()
        for w in words:
            word_freq[w] += 1
            word_degree[w] += len(words) - 1
 
    def score_word(w):
        freq = word_freq[w]
        if freq == 0:
            return 0
        return (word_degree[w] + freq) / freq
 
    phrase_scores = {}
    for phrase in candidates:
        words = phrase.split()
        if not words:
            continue
        # Filter short / numeric words
        words = [w for w in words if len(w) > 2 and not w.isnumeric()]
        if not words:
            continue
        score = sum(score_word(w) for w in words)
        phrase_scores[phrase] = score
 
    # Sort and deduplicate
    ranked = sorted(phrase_scores, key=phrase_scores.get, reverse=True)
    seen_words: set = set()
    themes = []
    for phrase in ranked:
        p_words = set(phrase.split())
        if not p_words & seen_words:  # no overlap with already chosen
            themes.append(phrase)
            seen_words |= p_words
        if len(themes) >= n:
            break
 
    return themes if themes else ["reflection", "thoughts", "feelings"]
 
 
# ── Sentence-BERT style similarity (cosine on TF-IDF) ─────────────────────────
def build_tfidf_vector(text: str, vocab: dict, idf: dict) -> dict:
    """Sparse TF-IDF vector."""
    words = clean_text(text).split()
    tf = Counter(words)
    n = max(len(words), 1)
    vec = {}
    for w, count in tf.items():
        if w in vocab and w in idf:
            vec[w] = (count / n) * idf[w]
    return vec
 
 
def cosine_similarity_sparse(v1: dict, v2: dict) -> float:
    dot = sum(v1.get(w, 0) * v2.get(w, 0) for w in v1)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)
 
 
def find_similar_entries(query_text: str, all_entries: list, top_k: int = 3) -> list:
    """
    Returns top_k most similar entries to query_text.
    Uses TF-IDF cosine similarity — conceptually identical to embedding search.
    
    To upgrade to Sentence-BERT:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embs = model.encode([e.text for e in all_entries])
        query_emb = model.encode([query_text])
        scores = util.cos_sim(query_emb, embs)[0]
    """
    if len(all_entries) < 2:
        return []
 
    # Build corpus vocabulary + IDF
    corpus = [clean_text(e.text) for e in all_entries]
    query_clean = clean_text(query_text)
 
    vocab: set = set()
    for doc in corpus:
        vocab.update(doc.split())
 
    N = len(corpus)
    idf = {}
    for w in vocab:
        df = sum(1 for doc in corpus if w in doc.split())
        idf[w] = math.log((N + 1) / (df + 1)) + 1
 
    query_vec = build_tfidf_vector(query_clean, vocab, idf)
 
    results = []
    for entry, doc in zip(all_entries, corpus):
        doc_vec = build_tfidf_vector(doc, vocab, idf)
        sim = cosine_similarity_sparse(query_vec, doc_vec)
        results.append((entry, sim))
 
    results.sort(key=lambda x: x[1], reverse=True)
    # Exclude exact match (similarity = 1.0) and return top_k
    return [(e, s) for e, s in results if s < 0.99][:top_k]
 
 
# ── Mood trend & forecasting ───────────────────────────────────────────────────
def compute_rolling_sentiment(entries: list, window: int = 7) -> list:
    """
    Returns list of {date, avg_polarity, emotion} for each entry,
    plus a rolling average column.
    """
    if not entries:
        return []
 
    points = []
    for e in sorted(entries, key=lambda x: x.date):
        points.append({
            "date": e.date.strftime("%Y-%m-%d"),
            "polarity": round(e.polarity or 0, 3),
            "emotion": e.emotion,
            "intensity": e.intensity or 0,
        })
 
    # Rolling mean
    for i, p in enumerate(points):
        window_vals = [points[j]["polarity"] for j in range(max(0, i - window + 1), i + 1)]
        p["rolling_avg"] = round(sum(window_vals) / len(window_vals), 3)
 
    return points
 
 
def forecast_next_mood(entries: list) -> dict:
    """
    Simple mood forecasting using weighted recency average.
    
    To upgrade: replace with LSTM trained on user's polarity time-series:
        model = Sequential([LSTM(32, input_shape=(7,1)), Dense(1)])
    """
    if len(entries) < 3:
        return {"forecast": "neutral", "confidence": 0.3, "trend": "stable"}
 
    recent = sorted(entries, key=lambda x: x.date, reverse=True)[:7]
    polarities = [e.polarity or 0 for e in recent]
 
    # Exponentially weighted average (recent = more weight)
    weights = [2 ** i for i in range(len(polarities))]
    weighted_avg = sum(p * w for p, w in zip(polarities, weights)) / sum(weights)
 
    # Trend: slope of last 3 points
    if len(polarities) >= 3:
        slope = (polarities[0] - polarities[2]) / 2
    else:
        slope = 0
 
    # Map to emotion
    if weighted_avg > 0.4:
        forecast = "happy"
    elif weighted_avg > 0.15:
        forecast = "peaceful"
    elif weighted_avg < -0.4:
        forecast = "sad"
    elif weighted_avg < -0.15:
        forecast = "anxious"
    else:
        forecast = "neutral"
 
    trend = "improving" if slope > 0.05 else "declining" if slope < -0.05 else "stable"
    confidence = min(0.4 + 0.1 * len(entries), 0.85)
 
    return {
        "forecast": forecast,
        "confidence": round(confidence, 2),
        "trend": trend,
        "weighted_polarity": round(weighted_avg, 3),
    }
 
 
def detect_anomaly(entries: list) -> dict | None:
    """Flag if recent mood drops significantly below user's baseline."""
    if len(entries) < 5:
        return None
    all_pol = [e.polarity or 0 for e in entries]
    baseline = sum(all_pol) / len(all_pol)
    std = math.sqrt(sum((p - baseline) ** 2 for p in all_pol) / len(all_pol))
    recent_3 = sum(all_pol[:3]) / 3
    if recent_3 < baseline - 1.5 * std and std > 0.05:
        return {
            "detected": True,
            "baseline": round(baseline, 3),
            "recent": round(recent_3, 3),
            "message": "Your mood has been notably lower than your usual baseline recently.",
        }
    return None
 
 
# ── Model evaluation ──────────────────────────────────────────────────────────
def evaluate_model(labelled_entries: list) -> dict:
    """
    labelled_entries: list of (text, true_emotion_label)
    Returns accuracy, per-class F1, and confusion matrix dict.
    Saves results to ml/eval_results.json.
    """
    if not labelled_entries:
        return {}
 
    y_true, y_pred = [], []
    for text, true_label in labelled_entries:
        result = analyze_sentiment(text)
        y_true.append(true_label)
        y_pred.append(result["emotion"])
 
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
 
    # Per-class precision / recall / F1
    labels = sorted(set(y_true) | set(y_pred))
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
 
    # Macro F1
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics) if metrics else 0
 
    result = {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": metrics,
        "n_samples": len(y_true),
    }
 
    eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)
 
    return result
 
 
# ── Main analysis function ─────────────────────────────────────────────────────
MOTIVATION_BANK = {
    "happy": [
        "Your joy is contagious — hold onto this light and let it guide your next steps. 🌟",
        "Happiness shared is happiness multiplied! Celebrate this moment. ✨",
        "What a beautiful mindset. Keep glowing and inspiring those around you. 💫",
    ],
    "sad": [
        "It's okay to feel this way. Storms don't last forever, and you're stronger than you know. 🌧️",
        "Your feelings are valid. Be gentle with yourself — brighter days are ahead. 💙",
        "Even in sadness, there's growth. You're doing better than you think. 🌱",
    ],
    "angry": [
        "Take a deep breath. Channel this energy into something positive — you've got this. 🌿",
        "Anger is natural. Acknowledge it, let it pass, and peace will follow. 🕊️",
        "Transform this fire into fuel for meaningful change. Your calm is your power. 🔥",
    ],
    "anxious": [
        "You're stronger than your fears. Take one breath at a time — you are safe. 💪",
        "Worry doesn't define you. Ground yourself in this moment and trust the process. 🌄",
        "Your courage in facing these feelings is admirable. You're braver than you realise. 🦋",
    ],
    "peaceful": [
        "This calm is a gift. Embrace this peace and let it recharge your spirit. 🌸",
        "Inner peace is your superpower. Carry this tranquility with you. 🌼",
        "You're exactly where you need to be. This stillness is beautiful. 🌙",
    ],
    "excited": [
        "Your enthusiasm is electric! Ride this wave and make something amazing happen. ⚡",
        "This energy is powerful — channel it wisely and watch great things unfold. 🎉",
        "Let your excitement propel you toward your dreams! 🚀",
    ],
    "neutral": [
        "Sometimes stillness is exactly what we need. Reflection brings clarity. 🪞",
        "Even calm days are part of your beautiful journey. Every moment matters. 🌻",
        "Balance is beautiful. Breathe and enjoy this moment of equilibrium. ⚖️",
    ],
}
 
SUGGESTIONS = {
    "happy": "Celebrate this moment! Share your joy or channel it into something creative.",
    "sad": "Try gentle activities: a warm drink, your favourite music, or a conversation with a friend.",
    "angry": "Practice box breathing (4-4-4-4), take a walk, or write out exactly what's bothering you.",
    "anxious": "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you hear, 3 you can touch.",
    "peaceful": "Keep doing what you're doing! Consider journaling this feeling or a short meditation.",
    "excited": "Channel this energy into a project or goal you care about — strike while the iron is hot!",
    "neutral": "Reflect on your priorities, try something new, or revisit a goal you've been putting off.",
}
 
A_B_COMPARISON = {}  # Stores last A/B comparison result for the UI
 
 
def analyze_sentiment(text: str, mode: str = "ml") -> dict:
    """
    Main entry point. mode: 'ml' (default), 'textblob', or 'compare'.
    Returns a rich dict consumed by the Flask routes.
    """
    cleaned = clean_text(text)
    features = extract_features(text)
    polarity = features["polarity"]
 
    # ── Tier 1: HuggingFace ────────────────────────────────────────────────
    if _hf_pipeline is not None and mode != "textblob":
        try:
            raw_hf_out = _hf_pipeline(text[:512])
            hf_out = raw_hf_out[0] if raw_hf_out and isinstance(raw_hf_out[0], list) else raw_hf_out
            if isinstance(hf_out, dict):
                hf_out = [hf_out]
            label_map = {
                "joy": "happy", "sadness": "sad", "anger": "angry",
                "fear": "anxious", "surprise": "excited", "disgust": "angry",
            }
            best = max(hf_out, key=lambda x: x["score"])
            emotion = label_map.get(best["label"].lower(), "neutral")
            confidence = round(best["score"], 3)
            tier = "huggingface"
            prob_map = {label_map.get(r["label"].lower(), r["label"]): round(r["score"], 3) for r in hf_out}
        except Exception as e:
            print(f"[ML] HF inference error: {e}")
            emotion, confidence, tier, prob_map = _sklearn_or_rule(features)
 
    # ── Tier 2/3: sklearn or rule-based ───────────────────────────────────
    else:
        emotion, confidence, tier, prob_map = _sklearn_or_rule(features)
 
    # TextBlob baseline for A/B mode
    if mode == "textblob" or mode == "compare":
        tb_emotion, tb_confidence = rule_based_emotion(features)
        if mode == "textblob":
            emotion, confidence, tier = tb_emotion, tb_confidence, "textblob"
 
    # Sentiment label
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
 
    # Intensity (0–10): weighted combination of lexicon scores, punctuation, caps, word count
    score_sum = sum(abs(v) for v in features["emotion_scores"].values())
    intensity = min(10, max(1, round(
        abs(polarity) * 4                   # polarity contribution (0 if TextBlob unavailable)
        + score_sum * 0.9                   # emotion lexicon score (primary driver)
        + features["exclamations"] * 0.5    # punctuation expressiveness
        + features["caps_ratio"] * 3        # capitalisation intensity
        + min(features["word_count"] / 80, 2)  # length bonus (longer = more expressive)
    )))
 
    # Themes
    themes = extract_key_themes(text)
 
    result = {
        "cleaned": cleaned,
        "polarity": round(polarity, 4),
        "sentiment": sentiment,
        "emotion": emotion,
        "confidence": confidence,
        "intensity": intensity,
        "key_themes": ", ".join(themes[:4]),
        "key_themes_list": themes[:4],
        "motivation": random.choice(MOTIVATION_BANK[emotion]),
        "suggestion": SUGGESTIONS[emotion],
        "tier": tier,
        "prob_map": prob_map,
    }
 
    # Store for A/B comparison
    if mode == "compare":
        result["ab_comparison"] = {
            "ml": {"emotion": emotion, "confidence": confidence, "tier": tier},
            "textblob": {"emotion": tb_emotion, "confidence": tb_confidence, "tier": "textblob"},
            "agree": emotion == tb_emotion,
        }
 
    return result
 
 
def _sklearn_or_rule(features):
    if _classifier.fitted:
        prob_map = _classifier.predict_proba(features)
        if prob_map:
            emotion = max(prob_map, key=prob_map.get)
            confidence = round(prob_map[emotion], 3)
            return emotion, confidence, "sklearn_lr", prob_map
    emotion, confidence = rule_based_emotion(features)
    return emotion, confidence, "rule_based", {}
