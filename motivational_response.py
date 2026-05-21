import random

responses = {
    "strong_positive": [
        "You sound truly joyful — hold onto this light and share it 🌞",
        "Such strong positivity radiates from you! Let it inspire your next steps 💪",
        "Your spirit feels alive today — what a beautiful thing to feel 💫"
    ],
    "mild_positive": [
        "You seem content — let this quiet happiness anchor your day 🌿",
        "Things are going well; cherish this calm, positive energy ✨",
        "Keep that gentle optimism alive — small joys make big differences 🌸"
    ],
    "neutral": [
        "You seem balanced — reflection can help uncover subtle emotions 🌙",
        "This feels like a calm pause; breathe and notice the present moment 🪞",
        "Even peaceful neutrality has beauty — enjoy the stillness 🌼"
    ],
    "mild_negative": [
        "Seems like a low day — take it slow and treat yourself gently 🌧️",
        "You might be feeling off; that’s okay, you’re allowed to rest 💙",
        "Write out what’s bothering you — it often helps release the weight 🕊️"
    ],
    "strong_negative": [
        "You sound deeply hurt or stressed — please take care of yourself 🌧️",
        "It’s okay to reach out or pause; you don’t have to carry it alone 💞",
        "Pain is temporary, even when it feels endless. Hold on — you matter 🌈"
    ]
}

def get_response(sentiment):
    if sentiment not in responses:
        sentiment = "neutral"
    return random.choice(responses[sentiment])
