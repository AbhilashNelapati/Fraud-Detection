"""
eda_analysis.py
---------------
Runs full Exploratory Data Analysis on the fraud dataset.
Generates publication-quality charts saved to /dashboard/eda_charts/.
Also exports a summary dict used by the API analytics endpoint.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
TRAIN_CSV = Path(r"C:\Users\nelap\OneDrive\Desktop\Data Project\Datasets\fraudTrain.csv")
CHART_DIR = ROOT / "dashboard" / "eda_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = ROOT / "dashboard" / "eda_summary.json"

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE  = {"fraud": "#ef4444", "legit": "#22d3ee"}
DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
TEXT_CLR = "#e2e8f0"

def _style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "axes.edgecolor":   "#334155",
        "axes.labelcolor":  TEXT_CLR,
        "xtick.color":      TEXT_CLR,
        "ytick.color":      TEXT_CLR,
        "text.color":       TEXT_CLR,
        "grid.color":       "#334155",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })

def _save(fig, name: str):
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅ Saved: {path.name}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════════
def run_eda(nrows: int = 200_000) -> dict:
    """
    Run full EDA. Returns a summary dict and saves charts.
    nrows: use None for full dataset (slow); 200k is a good balance.
    """
    _style()
    print(f"📊 Loading {nrows or 'ALL'} rows from fraudTrain.csv …")
    df = pd.read_csv(TRAIN_CSV, nrows=nrows, index_col=0, low_memory=False)

    # ── Datetime engineering ───────────────────────────────────────────────
    df["trans_datetime"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]    = df["trans_datetime"].dt.hour
    df["month"]   = df["trans_datetime"].dt.month
    df["weekday"] = df["trans_datetime"].dt.dayofweek
    df["is_night"]= ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    total        = len(df)
    fraud_count  = df["is_fraud"].sum()
    legit_count  = total - fraud_count
    fraud_pct    = round(fraud_count / total * 100, 2)

    charts = {}

    # ── 1. Fraud vs Non-Fraud pie ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        [legit_count, fraud_count],
        labels=["Legitimate", "Fraud"],
        autopct="%1.1f%%",
        colors=[PALETTE["legit"], PALETTE["fraud"]],
        startangle=140,
        wedgeprops=dict(edgecolor=DARK_BG, linewidth=2),
    )
    for at in autotexts: at.set_fontsize(14); at.set_color("white")
    ax.set_title("Fraud vs Legitimate Transactions", fontsize=15, pad=20)
    charts["fraud_ratio"] = _save(fig, "01_fraud_ratio")

    # ── 2. Transaction Amount Distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, color in zip(axes, ["Legitimate", "Fraud"], [PALETTE["legit"], PALETTE["fraud"]]):
        data = df[df["is_fraud"] == (1 if label == "Fraud" else 0)]["amt"]
        ax.hist(data.clip(upper=data.quantile(0.99)), bins=60, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(f"Amount Distribution — {label}", fontsize=13)
        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout(pad=2)
    charts["amount_dist"] = _save(fig, "02_amount_distribution")

    # ── 3. Hourly Fraud Heatmap ────────────────────────────────────────────
    hourly = df.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0)
    hourly["fraud_rate"] = (hourly[1] / (hourly[0] + hourly[1]) * 100).round(2)
    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(hourly.index, hourly["fraud_rate"],
                  color=[PALETTE["fraud"] if r > hourly["fraud_rate"].mean() else PALETTE["legit"]
                         for r in hourly["fraud_rate"]], alpha=0.88)
    ax.axhline(hourly["fraud_rate"].mean(), color="#facc15", linestyle="--", linewidth=1.5, label="Average")
    ax.set_title("Fraud Rate by Hour of Day", fontsize=14)
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xticks(range(24))
    ax.legend()
    charts["hourly_fraud"] = _save(fig, "03_hourly_fraud_rate")

    # ── 4. Category Risk Analysis ──────────────────────────────────────────
    cat_fraud = (
        df.groupby("category")["is_fraud"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud", "count": "total"})
    )
    cat_fraud["fraud_rate"] = (cat_fraud["fraud"] / cat_fraud["total"] * 100).round(2)
    cat_fraud = cat_fraud.sort_values("fraud_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["fraud"] if r > cat_fraud["fraud_rate"].median() else PALETTE["legit"]
              for r in cat_fraud["fraud_rate"]]
    ax.barh(cat_fraud.index, cat_fraud["fraud_rate"], color=colors, alpha=0.88)
    ax.set_title("Fraud Rate by Merchant Category", fontsize=14)
    ax.set_xlabel("Fraud Rate (%)")
    ax.axvline(cat_fraud["fraud_rate"].median(), color="#facc15", linestyle="--", linewidth=1.5, label="Median")
    ax.legend()
    charts["category_risk"] = _save(fig, "04_category_fraud_rate")

    # ── 5. Monthly Trend ───────────────────────────────────────────────────
    monthly = df.groupby("month").agg(
        total=("is_fraud","count"), fraud=("is_fraud","sum")
    )
    monthly["fraud_rate"] = monthly["fraud"] / monthly["total"] * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["fraud_rate"], marker="o",
            color=PALETTE["fraud"], linewidth=2.5, markersize=7)
    ax.fill_between(monthly.index, monthly["fraud_rate"], alpha=0.2, color=PALETTE["fraud"])
    ax.set_title("Monthly Fraud Rate Trend", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xticks(monthly.index)
    charts["monthly_trend"] = _save(fig, "05_monthly_trend")

    # ── 6. Gender Analysis ─────────────────────────────────────────────────
    gender_fraud = df.groupby("gender")["is_fraud"].agg(["sum","count"])
    gender_fraud["rate"] = gender_fraud["sum"] / gender_fraud["count"] * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Female (F)", "Male (M)"], gender_fraud["rate"],
           color=[PALETTE["fraud"], PALETTE["legit"]], alpha=0.85, width=0.4)
    ax.set_title("Fraud Rate by Gender", fontsize=14)
    ax.set_ylabel("Fraud Rate (%)")
    charts["gender_fraud"] = _save(fig, "06_gender_fraud")

    # ── 7. Top 10 Risky States ─────────────────────────────────────────────
    state_fraud = (
        df.groupby("state")["is_fraud"]
        .agg(["sum","count"])
        .assign(rate=lambda x: x["sum"]/x["count"]*100)
        .nlargest(10, "rate")
        .sort_values("rate")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(state_fraud.index, state_fraud["rate"],
            color=PALETTE["fraud"], alpha=0.85)
    ax.set_title("Top 10 Highest Fraud Rate States", fontsize=14)
    ax.set_xlabel("Fraud Rate (%)")
    charts["state_risk"] = _save(fig, "07_top_states_fraud")

    # ── 8. Age Distribution ────────────────────────────────────────────────
    df["dob_dt"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"]    = (df["trans_datetime"] - df["dob_dt"]).dt.days // 365

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, val, color in [("Legitimate", 0, PALETTE["legit"]), ("Fraud", 1, PALETTE["fraud"])]:
        subset = df[df["is_fraud"] == val]["age"].dropna()
        ax.hist(subset, bins=40, alpha=0.65, color=color, label=label, density=True)
    ax.set_title("Age Distribution: Fraud vs Legitimate", fontsize=14)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")
    ax.legend()
    charts["age_dist"] = _save(fig, "08_age_distribution")

    # ── 9. Correlation Heatmap ─────────────────────────────────────────────
    num_cols = ["amt", "city_pop", "hour", "month", "weekday", "is_night", "is_fraud"]
    corr = df[[c for c in num_cols if c in df.columns]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    charts["correlation"] = _save(fig, "09_correlation_heatmap")

    # ── 10. Amount Box Plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data_plot = [
        df[df["is_fraud"] == 0]["amt"].clip(upper=df["amt"].quantile(0.99)),
        df[df["is_fraud"] == 1]["amt"].clip(upper=df["amt"].quantile(0.99)),
    ]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    labels=["Legitimate", "Fraud"],
                    boxprops=dict(facecolor=CARD_BG),
                    medianprops=dict(color="#facc15", linewidth=2),
                    whiskerprops=dict(color=TEXT_CLR),
                    capprops=dict(color=TEXT_CLR),
                    flierprops=dict(marker=".", color="#94a3b8", markersize=3))
    bp["boxes"][0].set_facecolor(PALETTE["legit"] + "44")
    bp["boxes"][1].set_facecolor(PALETTE["fraud"] + "44")
    ax.set_title("Transaction Amount: Fraud vs Legitimate", fontsize=14)
    ax.set_ylabel("Amount ($)")
    charts["amount_box"] = _save(fig, "10_amount_boxplot")

    # ── Summary dict ───────────────────────────────────────────────────────
    top_risky_cats = cat_fraud.sort_values("fraud_rate", ascending=False).head(5).to_dict()["fraud_rate"]
    peak_hour = int(hourly["fraud_rate"].idxmax())

    summary = {
        "total_transactions":   int(total),
        "fraud_transactions":   int(fraud_count),
        "legit_transactions":   int(legit_count),
        "fraud_percentage":     fraud_pct,
        "avg_fraud_amount":     round(float(df[df["is_fraud"]==1]["amt"].mean()), 2),
        "avg_legit_amount":     round(float(df[df["is_fraud"]==0]["amt"].mean()), 2),
        "peak_fraud_hour":      peak_hour,
        "top_risky_categories": top_risky_cats,
        "top_risky_states":     state_fraud["rate"].to_dict(),
        "chart_paths":          charts,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n✅ EDA complete! Summary → {SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    summary = run_eda(nrows=200_000)
    print("\n📊 Key Insights:")
    print(f"  • Fraud rate:         {summary['fraud_percentage']}%")
    print(f"  • Avg fraud amount:   ${summary['avg_fraud_amount']}")
    print(f"  • Avg legit amount:   ${summary['avg_legit_amount']}")
    print(f"  • Peak fraud hour:    {summary['peak_fraud_hour']}:00")
    print(f"  • Top risky category: {list(summary['top_risky_categories'].keys())[0]}")
