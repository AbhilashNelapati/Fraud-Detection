from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path so we can import src modules
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Dummy implementations for API if real models aren't ready/trained yet
# In production, you'd import from src.model, src.risk, src.explain
from src.risk import get_risk
from src.explain import get_explanation

app = FastAPI(title="Fraud Analytics & AI Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup UI templates
UI_DIR = ROOT_DIR / "ui"
UI_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(UI_DIR))

# Mount dashboard directory for static charts
DASHBOARD_DIR = ROOT_DIR / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)
app.mount("/dashboard_data", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard_data")

class TransactionRequest(BaseModel):
    amt: float
    category: str
    hour: int
    is_night: int = 0
    distance_km: float = 0.0

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    # We serve the dashboard directly from the dashboard directory
    dashboard_html = DASHBOARD_DIR / "index.html"
    if dashboard_html.exists():
        return HTMLResponse(content=dashboard_html.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard not generated yet. Run EDA first.</h1>", status_code=404)

@app.post("/api/predict")
async def predict_fraud(transaction: TransactionRequest):
    # Dummy ML logic (Replace with actual model prediction)
    data = transaction.model_dump()
    
    # Simple dummy probability calculation for demonstration
    base_prob = 0.05
    if data["amt"] > 200: base_prob += 0.4
    if data["is_night"] == 1: base_prob += 0.2
    if data["distance_km"] > 50: base_prob += 0.15
    
    prob = min(base_prob, 0.99)
    prediction = "Fraud" if prob > 0.5 else "Legitimate"
    
    risk_info = get_risk(prob)
    explanation = get_explanation(data, risk_info["score"])
    
    return {
        "prediction": prediction,
        "risk_score": risk_info["score"],
        "risk_level": risk_info["level"],
        "insight": explanation["insight"],
        "explanation": explanation["explanation"],
        "recommendation": explanation["recommendation"]
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
