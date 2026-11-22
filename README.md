# A.R.G.U.S. — AI-Powered Credit Card Fraud Detection

An intelligent fraud detection prototype built using Machine Learning and Python to identify suspicious credit-card transactions, assess risk levels and explain model predictions.

<p align="center">
  <a href="https://argus-fraud-detection.streamlit.app/">Live Demo</a> •
  <a href="https://github.com/Usebonded/ARGUS-Fraud-Detection">GitHub Repository</a>
</p>

## Overview

A.R.G.U.S. analyses transaction data using a trained Random Forest model to predict potential fraud. It provides an interactive Streamlit dashboard with transaction analysis, risk classification and SHAP-based explainability.

## Key Features

- Machine-learning-based fraud detection.
- LOW, MEDIUM and HIGH risk classification.
- Interactive transaction monitoring dashboard.
- CSV upload and batch analysis.
- SHAP-based prediction explanations.
- Model performance and feature information.
- Downloadable analysis results.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Visualisation | Plotly |
| Deployment | Streamlit Community Cloud |

## Model Performance

Recorded test evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 92.13% |
| Recall | 83.67% |
| F1-Score | 87.70% |
| PR-AUC | 84.08% |
| ROC-AUC | 96.20% |

**Selected Model:** Random Forest  
**Decision Threshold:** 0.4083

## Application Modules

1. **Overview:** Transaction statistics, fraud distribution and risk summaries.
2. **Fraud Detection:** CSV uploads, batch predictions and downloadable results.
3. **Model Information:** Model details, evaluation metrics and threshold configuration.
4. **SHAP Explainability:** Feature contributions for individual predictions.

## Installation

```bash
git clone https://github.com/Usebonded/ARGUS-Fraud-Detection.git
cd ARGUS-Fraud-Detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Information

- **Institution:** Symbiosis Skills and Professional University
- **Programme:** B.Tech – Artificial Intelligence & Machine Learning
- **Academic Year:** 2026–27
- **Project Guide:** Nikhil Patankar (CSIT)

## Future Scope

- Real-time fraud monitoring.
- Banking API integration.
- Model drift detection and retraining.
- Advanced security and investigation workflows.
- POS and IoT integration.

---

**A.R.G.U.S. — Intelligent Detection. Explainable Decisions.**

*Academic prototype developed for educational and research purposes. Not intended for production banking decisions.*
