from dotenv import load_dotenv
import os
import json
import csv
import io
import secrets
from hmac import compare_digest
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, Response, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

load_dotenv()
app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("SECRET_KEY must be set in production.")
    secret_key = secrets.token_hex(32)
app.secret_key = secret_key

app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
 
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///journals.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
 
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


# CSRF protection for all browser-submitted state changes.
def get_csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": get_csrf_token}


@app.before_request
def csrf_protect():
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return

    expected = session.get("_csrf_token")
    submitted = request.form.get("_csrf_token") or request.headers.get("X-CSRFToken")
    if not expected or not submitted or not compare_digest(expected, submitted):
        abort(400, description="Invalid or missing CSRF token.")
 
# ── Import ML engine ──────────────────────────────────────────────────────────
from ml.sentiment_engine import (
    analyze_sentiment,
    find_similar_entries,
    compute_rolling_sentiment,
    forecast_next_mood,
    detect_anomaly,
    evaluate_model,
    extract_key_themes,
)
 
 
# ── Database Models ───────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    entries = db.relationship(
        "JournalEntry", backref="author", lazy=True, cascade="all, delete-orphan"
    )
 
 
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    text = db.Column(db.Text, nullable=False)
    cleaned = db.Column(db.Text)
    polarity = db.Column(db.Float, default=0.0)
    sentiment = db.Column(db.String(20), default="Neutral")
    emotion = db.Column(db.String(20), default="neutral")
    confidence = db.Column(db.Float, default=0.0)
    intensity = db.Column(db.Integer, default=5)
    key_themes = db.Column(db.String(400))
    motivation = db.Column(db.Text)
    suggestion = db.Column(db.Text)
    ml_tier = db.Column(db.String(30), default="rule_based")
    date = db.Column(db.DateTime, default=datetime.utcnow)
 
 
with app.app_context():
    db.create_all()
 
 
# ── Auth decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated
 
 
def get_current_user():
    if "user_id" in session:
        return User.query.get(session["user_id"])
    return None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_streak(entries):
    """Return current consecutive day streak."""
    if not entries:
        return 0
    dates = sorted({e.date.date() for e in entries}, reverse=True)
    streak = 1
    today = datetime.utcnow().date()
    if dates[0] < today - timedelta(days=1):
        return 0
    for i in range(1, len(dates)):
        if (dates[i - 1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    return streak
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.route("/")
def index():
    if "user_id" in session:
        user = get_current_user()
        latest = (
            JournalEntry.query
            .filter_by(user_id=user.id)
            .order_by(JournalEntry.date.desc())
            .first()
        )
        emotion = latest.emotion if latest else "neutral"
        quote = latest.motivation if latest else "Start your journaling journey today. Your thoughts matter. 🪶"
        return render_template("index.html", user=user, emotion=emotion,
                               quote=quote, latest_entry=latest)
    return render_template("landing.html")
 
 
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
 
        if not all([username, email, password]):
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
            return redirect(url_for("register"))
 
        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        db.session.add(User(username=username, email=email, password_hash=pw_hash))
        db.session.commit()
        flash("Account created! Please sign in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")
 
 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash(f"Welcome back, {user.username}! ✨", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")
 
 
@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out. See you soon! 👋", "info")
    return redirect(url_for("index"))
 
 
@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    entries = (
        JournalEntry.query
        .filter_by(user_id=user.id)
        .order_by(JournalEntry.date.desc())
        .all()
    )
    total_entries = len(entries)
    recent_entries = entries[:10]
 
    # BUG FIX: compute emotion_stats and recent_emotion properly
    emotion_stats = {}
    for e in entries:
        emotion_stats[e.emotion] = emotion_stats.get(e.emotion, 0) + 1
 
    recent_emotion = entries[0].emotion if entries else None
    streak = compute_streak(entries)
    progress_width = min(total_entries * 10, 100)
 
    # ML: mood forecast
    forecast = forecast_next_mood(entries) if total_entries >= 3 else None
    anomaly = detect_anomaly(entries) if total_entries >= 5 else None
 
    return render_template(
        "dashboard.html",
        user=user,
        entries=recent_entries,
        total_entries=total_entries,
        emotion_stats=emotion_stats,
        recent_emotion=recent_emotion,
        streak=streak,
        progress_width=progress_width,
        forecast=forecast,
        anomaly=anomaly,
    )
 
 
@app.route("/journal", methods=["GET", "POST"])
@login_required
def journal():
    if request.method == "POST":
        text = request.form.get("journal", "").strip()
        analysis_mode = request.form.get("analysis_mode", "ml")
 
        if not text:
            flash("Please write something before submitting.", "warning")
            return redirect(url_for("journal"))
 
        analysis = analyze_sentiment(text, mode=analysis_mode)
 
        entry = JournalEntry(
            user_id=session["user_id"],
            text=text,
            cleaned=analysis["cleaned"],
            polarity=analysis["polarity"],
            sentiment=analysis["sentiment"],
            emotion=analysis["emotion"],
            confidence=analysis["confidence"],
            intensity=analysis["intensity"],
            key_themes=analysis["key_themes"],
            motivation=analysis["motivation"],
            suggestion=analysis["suggestion"],
            ml_tier=analysis["tier"],
        )
        db.session.add(entry)
        db.session.commit()
 
        # Find similar past entries
        all_entries = (
            JournalEntry.query
            .filter_by(user_id=session["user_id"])
            .order_by(JournalEntry.date.desc())
            .all()
        )
        similar = find_similar_entries(text, all_entries[1:], top_k=3)
 
        flash("Journal saved and analysed! 💌", "success")
        session["last_analysis"] = {
            "emotion": analysis["emotion"],
            "confidence": analysis["confidence"],
            "intensity": analysis["intensity"],
            "motivation": analysis["motivation"],
            "suggestion": analysis["suggestion"],
            "key_themes": analysis["key_themes"],
            "key_themes_list": analysis["key_themes_list"],
            "tier": analysis["tier"],
            "prob_map": analysis.get("prob_map", {}),
            "ab_comparison": analysis.get("ab_comparison"),
            "similar_ids": [e.id for e, _ in similar],
        }
        return redirect(url_for("journal"))
 
    result = session.pop("last_analysis", None)
    similar_entries = []
    if result and result.get("similar_ids"):
        for sid in result["similar_ids"]:
            e = JournalEntry.query.get(sid)
            if e:
                similar_entries.append(e)
 
    return render_template("journal.html", result=result, similar_entries=similar_entries)
 
 
@app.route("/past_entries")
@login_required
def past_entries():
    # BUG FIX: was using current_user (Flask-Login) which was never imported
    user = get_current_user()
    entries = (
        JournalEntry.query
        .filter_by(user_id=user.id)
        .order_by(JournalEntry.date.desc())
        .all()
    )
    all_text = " ".join(e.text for e in entries if e.text)
    total_words = len(all_text.split())
    return render_template("past_entries.html", entries=entries, total_words=total_words)
 
 
@app.route("/delete/<int:entry_id>", methods=["POST"])
@login_required
def delete(entry_id):
    entry = JournalEntry.query.get_or_404(entry_id)
    if entry.user_id != session["user_id"]:
        flash("Unauthorised.", "danger")
        return redirect(url_for("past_entries"))
    db.session.delete(entry)
    db.session.commit()
    flash("Entry deleted. 🗑️", "info")
    return redirect(url_for("past_entries"))
 
 
@app.route("/analytics")
@login_required
def analytics():
    user = get_current_user()
    entries = (
        JournalEntry.query
        .filter_by(user_id=user.id)
        .order_by(JournalEntry.date.asc())
        .all()
    )
 
    emotion_data, sentiment_data, monthly_data = {}, {"Positive": 0, "Negative": 0, "Neutral": 0}, {}
    for e in entries:
        emotion_data[e.emotion] = emotion_data.get(e.emotion, 0) + 1
        sentiment_data[e.sentiment] = sentiment_data.get(e.sentiment, 0) + 1
        mk = e.date.strftime("%Y-%m")
        monthly_data[mk] = monthly_data.get(mk, 0) + 1
    total_words = sum(len((e.text or "").split()) for e in entries)
    avg_words = round(total_words / len(entries)) if entries else 0
 
    # ML analytics
    rolling = compute_rolling_sentiment(entries)
    forecast = forecast_next_mood(entries) if len(entries) >= 3 else None
    anomaly = detect_anomaly(entries) if len(entries) >= 5 else None
 
    # Model performance (load from file if available)
    eval_path = os.path.join(os.path.dirname(__file__), "ml", "eval_results.json")
    model_metrics = None
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            model_metrics = json.load(f)
 
    return render_template(
        "analytics.html",
        emotion_data=emotion_data,
        sentiment_data=sentiment_data,
        monthly_data=monthly_data,
        rolling_data=json.dumps(rolling),
        forecast=forecast,
        anomaly=anomaly,
        model_metrics=model_metrics,
        entries=entries,
        avg_words=avg_words,
    )
 
 
@app.route("/profile")
@login_required
def profile():
    user = get_current_user()
    return render_template("profile.html", user=user, now=datetime.utcnow())
 
 
@app.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    user = get_current_user()
    username = request.form.get("username", "").strip()
    if username and username != user.username:
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
        else:
            user.username = username
            session["username"] = username
            db.session.commit()
            flash("Profile updated! ✅", "success")
    return redirect(url_for("profile"))
 
 
@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    user = get_current_user()
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")
    confirm_pw = request.form.get("confirm_password", "")
 
    if not bcrypt.check_password_hash(user.password_hash, current_pw):
        flash("Current password is incorrect.", "danger")
        return redirect(url_for("profile"))
    if new_pw != confirm_pw:
        flash("New passwords do not match.", "danger")
        return redirect(url_for("profile"))
    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "danger")
        return redirect(url_for("profile"))
 
    user.password_hash = bcrypt.generate_password_hash(new_pw).decode("utf-8")
    db.session.commit()
    flash("Password changed! 🔒", "success")
    return redirect(url_for("profile"))
 
 
# ── API Routes (JSON) ─────────────────────────────────────────────────────────
 
@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    """Live analysis endpoint called via JS for A/B mode comparison."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "compare")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_sentiment(text, mode=mode)
    return jsonify(result)
 
 
@app.route("/api/similar", methods=["POST"])
@login_required
def api_similar():
    """Return similar entries for a given text snippet."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).all()
    similar = find_similar_entries(text, entries, top_k=3)
    return jsonify([
        {
            "id": e.id,
            "text_preview": e.text[:120],
            "emotion": e.emotion,
            "date": e.date.strftime("%b %d, %Y"),
            "similarity": round(s, 3),
        }
        for e, s in similar
    ])
 
 
@app.route("/api/export")
@login_required
def api_export():
    """Export all entries as CSV."""
    user = get_current_user()
    entries = JournalEntry.query.filter_by(user_id=user.id).order_by(JournalEntry.date.desc()).all()
 
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "emotion", "sentiment", "polarity", "intensity",
                     "confidence", "ml_tier", "key_themes", "text"])
    for e in entries:
        writer.writerow([
            e.date.strftime("%Y-%m-%d %H:%M"),
            e.emotion, e.sentiment,
            round(e.polarity or 0, 4),
            e.intensity, round(e.confidence or 0, 3),
            e.ml_tier, e.key_themes, e.text,
        ])
 
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=reflectra_journal.csv"},
    )
 
 
@app.route("/api/rolling_sentiment")
@login_required
def api_rolling_sentiment():
    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).order_by(JournalEntry.date.asc()).all()
    return jsonify(compute_rolling_sentiment(entries))
 
 
@app.route("/api/forecast")
@login_required
def api_forecast():
    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).all()
    return jsonify(forecast_next_mood(entries))
 
 
@app.route("/api/model_info")
@login_required
def api_model_info():
    """Return current ML tier info."""
    from ml.sentiment_engine import _classifier, _hf_pipeline, USE_TRANSFORMERS
    return jsonify({
        "tier": "huggingface" if _hf_pipeline else ("sklearn_lr" if _classifier.fitted else "rule_based"),
        "transformers_available": _hf_pipeline is not None,
        "sklearn_fitted": _classifier.fitted,
        "upgrade_command": "pip install transformers torch sentence-transformers keybert",
    })
 
 
if __name__ == "__main__":
    app.run(debug=False)
