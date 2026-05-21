import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.sentiment_engine import evaluate_model, _classifier

# ── Evaluation dataset ────────────────────────────────────────────────────────
# Evaluation dataset spanning 7 emotion classes
EVAL_DATA = [
    # happy (5)
    ("Today was absolutely wonderful, felt so joyful and grateful for everything", "happy"),
    ("I'm thrilled and excited about this amazing new opportunity at work", "happy"),
    ("Finally achieved my goal after months of effort, feeling so proud and delighted", "happy"),
    ("Spent the day laughing with friends, life feels beautiful right now", "happy"),
    ("Grateful for the small wins today, cheerful and content", "happy"),

    # sad (5)
    ("Crying all afternoon, feel so hopeless and completely lost in life", "sad"),
    ("The grief of losing them is overwhelming, I miss everything about them", "sad"),
    ("Feeling deeply depressed and worthless, nothing seems to matter anymore", "sad"),
    ("So lonely and heartbroken, can't find any reason to feel better", "sad"),
    ("Everything feels empty and hollow, just going through the motions", "sad"),

    # angry (5)
    ("Absolutely furious at what happened, this is completely unfair and outrageous", "angry"),
    ("Fed up and enraged, I can't keep letting people treat me this way", "angry"),
    ("So mad and resentful today, the frustration is building up inside", "angry"),
    ("Bitter and hostile after what they did, I'm seething right now", "angry"),
    ("This situation makes me so irritated and annoyed every single time", "angry"),

    # anxious (5)
    ("Feeling so anxious and overwhelmed, the panic is hard to control", "anxious"),
    ("Constant worry and dread about the upcoming exam, can't sleep at night", "anxious"),
    ("Nervous and stressed about the presentation tomorrow, my hands are shaking", "anxious"),
    ("Scared and apprehensive about what comes next, feeling tense all over", "anxious"),
    ("The anxiety is getting worse, feeling uneasy and afraid of everything", "anxious"),

    # peaceful (5)
    ("Such a calm and serene morning walk, feeling completely at peace", "peaceful"),
    ("Meditated for 30 minutes and feel grounded, centered and tranquil", "peaceful"),
    ("Relaxed and content reading by the window, a mindful and balanced day", "peaceful"),
    ("Feeling deeply peaceful and settled within myself after journaling", "peaceful"),
    ("Everything feels harmonious and still today, beautifully calm", "peaceful"),

    # excited (5)
    ("So pumped and energetic about the new project launching next week!", "excited"),
    ("Thrilled and enthusiastic, I can't wait to get started on this!", "excited"),
    ("Feeling motivated and inspired by all the new possibilities ahead of me", "excited"),
    ("Super excited and passionate about the changes happening in my life", "excited"),
    ("Energised and enthusiastic, absolutely buzzing with anticipation!", "excited"),

    # neutral (5)
    ("Had an ordinary day at work, nothing particularly notable happened today", "neutral"),
    ("Just going through the motions, feeling neither particularly good nor bad", "neutral"),
    ("A quiet and regular Tuesday, thinking about nothing specific right now", "neutral"),
    ("Completed my tasks for the day, everything was as expected", "neutral"),
    ("Average day, woke up, worked, came home, nothing eventful", "neutral"),
]


def print_report(results: dict):
    """Pretty-print evaluation results."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  REFLECTRA EMOTION MODEL — EVALUATION REPORT")
    print(sep)
    print(f"  Samples     : {results['n_samples']}")
    print(f"  Accuracy    : {results['accuracy']:.1%}")
    print(f"  Macro F1    : {results['macro_f1']:.1%}")
    print(f"{sep}")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*8}")
    for label, m in sorted(results["per_class"].items()):
        bar_len = int(m["f1"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label:<12} {m['precision']:>10.1%} {m['recall']:>8.1%} {m['f1']:>8.1%}  {bar}")
    print(sep)
    print("  Results saved to ml/eval_results.json")
    print("  Open Analytics page to view the results in the dashboard.\n")


if __name__ == "__main__":
    print("Running evaluation...")
    results = evaluate_model(EVAL_DATA)
    if results:
        print_report(results)
    else:
        print("Evaluation failed — check that the ML engine loaded correctly.")