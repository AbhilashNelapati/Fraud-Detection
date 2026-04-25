import numpy as np

def compute_risk_score(probability: float) -> int:
    """Convert fraud probability (0-1) to a risk score 0-100.
    Clamps the input to [0, 1] and scales linearly.
    """
    prob = np.clip(probability, 0.0, 1.0)
    return int(prob * 100)

def risk_level(score: int) -> str:
    """Map a numeric risk score to a categorical risk level.
    0‑30   → Low
    31‑70  → Medium
    71‑100 → High
    """
    if score <= 30:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"

def get_risk(probability: float) -> dict:
    """Convenient helper returning score and level together."""
    score = compute_risk_score(probability)
    return {"score": score, "level": risk_level(score)}
