# Fraud Analytics & AI Intelligence System (EDA + Dashboard + ML + Explainable AI)

This project is a comprehensive Data Analytics and AI Intelligence system focused on identifying, predicting, and explaining fraudulent transactions. 

It is designed to showcase strong Data Analyst skills combined with practical AI/ML application.

## 🎯 Goal
A balanced, realistic, and job-ready system that:
- Analyzes fraud patterns (Data Analyst core)
- Generates insights for a Dashboard
- Predicts fraud using ML models
- Generates risk scores
- Provides simple AI explanation (Explainability)

## 📁 Structure

- `data/` : Raw datasets (Not uploaded to GitHub due to size limits. [Download from Kaggle](https://www.kaggle.com/))
- `notebooks/` : EDA notebooks (Data Analysis focus)
- `dashboard/` : Dashboard insights and charts
- `models/` : Saved ML models
- `src/` : Source code for processing, EDA, ML, and Explainability
- `agents/` : Lightweight multi-agent pipeline
- `api/` : FastAPI application
- `ui/` : Frontend UI

## 🚀 How to Run

1. **Download the Dataset**:
   Download the fraud dataset from Kaggle and place `fraudTrain.csv` and `fraudTest.csv` into the `data/` folder.

2. **Perform EDA**:
   Run the `src/eda_analysis.py` or the `notebooks/eda.ipynb` to generate charts and insights.
   
2. **Train Models**:
   Run `src/model.py` to train and save the ML classifiers.
   
3. **Run the API & UI**:
   Start the FastAPI server:
   ```bash
   python api/main.py
   ```
   Open your browser to `http://127.0.0.1:8000` to interact with the system.

## 🏆 Resume Highlight
**Data Analyst Role**:
> Analyzed fraud transaction dataset to identify risk patterns and built interactive dashboards (Power BI) for business insights, supported by ML-based fraud prediction and explainable risk scoring.

**AI Role**:
> Developed fraud detection system using ML models, explainable AI (SHAP), and lightweight multi-agent architecture, deployed via FastAPI with interactive UI.
