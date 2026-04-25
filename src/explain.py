import pandas as pd

def get_explanation(transaction_data: dict, risk_score: int) -> dict:
    """
    Generates a simple rule-based explanation for the transaction.
    In a full production environment, SHAP values can be integrated here.
    """
    reasons = []
    insight = "Normal transaction pattern"
    recommendation = "Approve"

    amt = transaction_data.get("amt", 0)
    category = transaction_data.get("category", "")
    hour = transaction_data.get("hour", 12)

    # Basic rules for explanation
    if amt > 300:
        reasons.append("High transaction amount")
        insight = "High amount anomaly"
    
    if (hour >= 22 or hour <= 4):
        reasons.append("Unusual night time transaction")
    
    high_risk_categories = ["misc_net", "grocery_pos", "shopping_net"]
    if category in high_risk_categories:
        reasons.append(f"Risky merchant category ({category})")

    explanation = " + ".join(reasons) if reasons else "Transaction aligns with user history"

    if risk_score > 70:
        recommendation = "Block immediately and send alert"
    elif risk_score > 30:
        recommendation = "Require 2FA or verify"

    return {
        "insight": insight,
        "explanation": explanation,
        "recommendation": recommendation
    }
