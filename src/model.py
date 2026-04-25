"""
model.py
--------
Trains Logistic Regression, Random Forest, and XGBoost fraud classifiers.
Saves best model to /models/best_model.pkl with scaler and feature list.
"""

import pandas as pd
import numpy as np
import pickle, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed — skipping.")

ROOT       = Path(__file__).resolve().parents[1]
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PKL    = MODEL_DIR / "best_model.pkl"
SCALER_PKL   = MODEL_DIR / "scaler.pkl"
FEATURES_JSON= MODEL_DIR / "features.json"
METRICS_JSON = MODEL_DIR / "model_metrics.json"

TRAIN_CSV = Path(r"C:\Users\nelap\OneDrive\Desktop\Data Project\Datasets\fraudTrain.csv")

# ── Features used by the model ────────────────────────────────────────────
FEATURE_COLS = [
    "amt", "city_pop", "hour", "month", "weekday",
    "is_weekend", "is_night", "age", "distance_km",
    "high_risk_category", "lat", "long",
    "merch_lat", "merch_long",
]

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    from src.preprocessing import feature_engineer, encode_categoricals
    df = feature_engineer(df)
    df = encode_categoricals(df)
    return df


def get_features_target(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["is_fraud"]
    return X, y, available


def train(nrows: int = 300_000):
    print(f"📦 Loading {nrows:,} training rows …")
    raw = pd.read_csv(TRAIN_CSV, nrows=nrows, index_col=0, low_memory=False)
    df  = _prep(raw)

    X, y, feat_cols = get_features_target(df)

    # Undersample majority to balance (ratio 1:5  legit:fraud)
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].sample(n=min(len(fraud_idx) * 5, len(y[y==0])),
                                  random_state=42).index
    idx  = fraud_idx.append(legit_idx)
    X, y = X.loc[idx], y.loc[idx]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "RandomForest":       RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                      class_weight="balanced", random_state=42),
    }
    if HAS_XGB:
        scale_pos = int((y == 0).sum() / (y == 1).sum())
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )

    best_auc, best_name, best_model = 0, None, None
    all_metrics = {}

    for name, clf in models.items():
        print(f"\n🔧 Training {name} …")
        if name == "LogisticRegression":
            clf.fit(X_tr_s, y_tr)
            y_prob = clf.predict_proba(X_val_s)[:, 1]
            y_pred = clf.predict(X_val_s)
        else:
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = clf.predict(X_val)

        auc = roc_auc_score(y_val, y_prob)
        metrics = {
            "roc_auc":   round(auc, 4),
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall":    round(recall_score(y_val, y_pred), 4),
            "f1":        round(f1_score(y_val, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        }
        all_metrics[name] = metrics
        print(f"  ROC-AUC={auc:.4f}  F1={metrics['f1']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}")

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, clf

    # ── Save artefacts ─────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:  pickle.dump(best_model, f)
    with open(SCALER_PKL,"wb") as f:  pickle.dump(scaler, f)
    FEATURES_JSON.write_text(json.dumps({
        "features": feat_cols,
        "best_model": best_name,
    }))
    METRICS_JSON.write_text(json.dumps(all_metrics, indent=2))

    print(f"\n🏆 Best model: {best_name}  (ROC-AUC={best_auc:.4f})")
    print(f"   Saved → {MODEL_PKL}")
    return best_model, scaler, feat_cols, all_metrics


def load_model():
    """Load saved model, scaler, and feature list."""
    with open(MODEL_PKL,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PKL, "rb") as f: scaler = pickle.load(f)
    feat_info = json.loads(FEATURES_JSON.read_text())
    return model, scaler, feat_info["features"], feat_info["best_model"]


def predict_single(transaction: dict):
    """
    Predict fraud for a single transaction dict.
    Returns {"fraud_prob": float, "prediction": str}.
    """
    model, scaler, feat_cols, model_name = load_model()
    row = pd.DataFrame([transaction])

    # Minimal feature engineering on raw input
    for col in feat_cols:
        if col not in row.columns:
            row[col] = 0

    X = row[feat_cols].fillna(0)

    if model_name == "LogisticRegression":
        X_s = scaler.transform(X)
        prob = float(model.predict_proba(X_s)[0, 1])
    else:
        prob = float(model.predict_proba(X)[0, 1])

    return {
        "fraud_prob": round(prob, 4),
        "prediction": "Fraud" if prob >= 0.5 else "Legitimate",
    }


if __name__ == "__main__":
    train(nrows=300_000)
