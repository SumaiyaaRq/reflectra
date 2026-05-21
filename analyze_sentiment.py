import random
from textblob import TextBlob
import re

def clean_text(text):
    # Normalize repeated letters and remove punctuation
    text = re.sub(r'(.)\1{2,}', r'\1', text.lower())
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def analyze_sentiment(text):
    cleaned = clean_text(text)
    blob = TextBlob(cleaned)
    polarity = blob.sentiment.polarity

    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "grateful", "love"],
        "sad": ["sad", "unhappy", "cry", "lonely", "depressed"],
        "angry": ["angry", "mad", "furious", "annoyed"],
        "fear": ["scared", "afraid", "fear", "worried", "anxious"],
        "stressed": ["stress", "tired", "pressure", "overwhelmed"],
    }
     # Default emotion
    detected_emotion = "neutral"

    for emotion, keywords in emotion_keywords.items():
        if any(word in cleaned for word in keywords):
            detected_emotion = emotion
            break

    # If TextBlob says it's highly positive/negative, adjust emotion
    if detected_emotion == "neutral":
        if polarity > 0.3:
            detected_emotion = "happy"
        elif polarity < -0.3:
            detected_emotion = "sad"


    # if polarity > 0:
    #     sentiment = "Positive 😊"
    # elif polarity < 0:
    #     sentiment = "Negative 😔"
    # else:
    #     sentiment = "Neutral 😐"

    # Select motivation or story
    motivation_bank = {
        "happy": [
            "Keep shining — your joy inspires others 🌟",
            "Happiness shared is happiness multiplied!",
            "Stay grateful, stay glowing ✨"
        ],
        "sad": [
            "It's okay to feel sad — let it flow, then rise again 💪",
            "Remember, storms make trees take deeper roots 🌧️",
            "You are not alone — better days are on the way ❤️"
        ],
        "angry": [
            "Take a deep breath — calm brings clarity 🕊️",
            "Use your energy to create, not destroy 🔥",
            "Anger is natural; balance it with peace 🌿"
        ],
        "fear": [
            "Courage is not the absence of fear — it’s action despite it 🌄",
            "You are stronger than your fears 🦋",
            "Trust the process, you’re doing great 💫"
        ],
        "stressed": [
            "Pause. Breathe. You’ve survived every tough day so far 💙",
            "You deserve rest, not just resilience 🌸",
            "Take it one step at a time — progress, not perfection 🌱"
        ],
        "neutral": [
            "A calm mind is a powerful mind 🌼",
            "Every ordinary day is part of your beautiful journey 🌻",
            "Keep journaling — reflection brings clarity 🪞"
        ]
    }

    motivation = random.choice(motivation_bank[detected_emotion])

    # Return results
    return cleaned, polarity, detected_emotion.title(), motivation
