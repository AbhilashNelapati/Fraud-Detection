"""
agents.py
---------
Lightweight multi-agent implementation using simple LLM prompts
to avoid the heavy overhead of CrewAI if not needed, while still
demonstrating the "Agentic" design pattern.
"""

class AnalystAgent:
    def __init__(self):
        self.role = "Data Analyst"
    
    def analyze(self, transaction: dict) -> str:
        amt = transaction.get("amt", 0)
        hour = transaction.get("hour", 12)
        if amt > 250 and (hour < 5 or hour > 22):
            return "Found high transaction amount during unusual hours."
        return "Transaction appears normal based on surface features."

class RiskAgent:
    def __init__(self):
        self.role = "Risk Assessor"
        
    def score(self, probability: float) -> int:
        # Scale probability (0-1) to risk score (0-100)
        return int(min(probability * 100, 100))

class ExplainAgent:
    def __init__(self):
        self.role = "Explainability Agent"
        
    def explain(self, risk_score: int, insight: str) -> str:
        if risk_score > 75:
            return f"High risk alert ({risk_score}/100) triggered by: {insight}."
        elif risk_score > 30:
            return f"Medium risk ({risk_score}/100) noted: {insight}."
        return "Low risk transaction."

def run_agentic_pipeline(transaction: dict, probability: float) -> dict:
    """Run the lightweight multi-agent pipeline."""
    analyst = AnalystAgent()
    risk_assessor = RiskAgent()
    explainer = ExplainAgent()
    
    insight = analyst.analyze(transaction)
    score = risk_assessor.score(probability)
    explanation = explainer.explain(score, insight)
    
    return {
        "insight": insight,
        "risk_score": score,
        "explanation": explanation
    }
