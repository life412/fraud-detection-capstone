# Fraud Detection Capstone Project

![CI](https://github.com/HUAWEI/fraud-detection-capstone/actions/workflows/ci.yml/badge.svg)

## Business Overview
In the financial sector, fraud detection is a mission-critical task. False negatives (missed fraud) result in direct financial loss and reputational damage. False positives (flagging legitimate transactions) lead to poor customer experience and operational overhead.

This project implements a robust, production-grade fraud detection system designed to:
- **Maximize Detection**: Identify fraudulent patterns in e-commerce and credit card transactions.
- **Minimize False Alarms**: Maintain high precision effectively using advanced ensemble techniques.
- **Ensure Explainability**: Provide transparent reasoning for every flagged transaction using SHAP, meeting regulatory requirements for AI in finance.

## Key Features
- **Modular Engineering**: Clean, tested, and reusable Python code.
- **Advanced Modelling**: Ensemble learning (XGBoost/Random Forest) with SMOTE for class imbalance.
- **Explainable AI**: SHAP (SHapley Additive exPlanations) for global and local interpretability.
- **Robust Pipeline**: Automated preprocessing, training, and evaluation workflows.

## Success Metrics
- **AUC-PR (Area Under the Precision-Recall Curve)**: Primary metric due to high class imbalance.
- **F1-Score**: To balance Precision and Recall.
- **Recall**: To ensure we capture the majority of fraud cases.

## Project Structure
```
fraud-detection-capstone/
├── data/               # Data files (ignored by git for security)
│   └── raw/            # Place your data here
├── notebooks/          # Exploration and experiments
├── src/                # Source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── tests/              # Unit tests
├── outputs/            # Model artifacts and plots
└── requirements.txt
```

## Setup & Usage

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone <repository_url>
cd fraud-detection-capstone
pip install -r requirements.txt
```

### 2. Data Preparation
**CRITICAL**: This project does not contain raw data files to verify data privacy compliance.
Place your datasets in `data/raw/`:
- `creditcard.csv` (Credit Card Fraud Detection dataset)

### 3. Run Pipeline
To train the model and generate metrics:
```bash
python -m src.train --model_type xgboost --smote
```

Arguments:
- `--model_type`: Choose between `xgboost` (default), `random_forest`, `logistic`.
- `--smote`: Flag to enable SMOTE oversampling.

### 4. Run Tests
Ensure code integrity by running the test suite:
```bash
pytest tests/
```

## Explainability
The pipeline automatically generates SHAP summary plots in `outputs/shap_summary.png` to help analysts understand key drivers of fraud (e.g., transaction amount, V14, V12 features).

---
*Note: This project is designed as a template for production deployment in a secure environment.*
