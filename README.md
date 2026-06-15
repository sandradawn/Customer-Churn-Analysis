# 📊 Telco Customer Churn Advisor

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange" alt="ML Stack">
  <img src="https://img.shields.io/badge/Explainability-SHAP-9cf" alt="SHAP">
</div>

<br>

A professional, end-to-end Machine Learning web application designed to predict and analyze customer churn for telecommunication providers. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this application features an intuitive UI and powerful Explainable AI (XAI) capabilities using **SHAP** to help business stakeholders understand the exact drivers behind customer attrition.

---

## 🎯 Features

- **Real-Time Predictions**: Instantly calculate the probability of a customer cancelling their service based on their demographic, service usage, and financial data.
- **Explainable AI (XAI)**: View dynamic SHAP waterfall plots for every prediction to understand *why* the model made its decision.
- **Premium UI**: A responsive, beautifully designed Streamlit interface featuring glassmorphism elements, custom typography, and clear risk assessment cards.
- **Robust Modeling**: Utilizes an optimized pipeline (handling data scaling and one-hot encoding) paired with class-imbalanced learning strategies.

---

## 🏗️ Project Structure

```text
customer-churn-analysis/
├── app/
│   ├── app.py                  # Main Streamlit web application
│   ├── utils.py                # Helper functions (Model loading & SHAP plotting)
│   └── assets/
│       └── style.css           # Custom CSS for premium UI styling
├── data/
│   ├── telco_churn.csv         # Raw dataset (Kaggle Telco Customer Churn)
│   └── split_data.pkl          # Pickled train/test splits
├── models/
│   ├── preprocessor.pkl        # Data preprocessing pipeline
│   └── best_model.pkl          # Trained classification model pipeline
├── notebooks/                  # Jupyter notebooks for EDA and model training
├── .streamlit/
│   └── config.toml             # Streamlit global theme configuration
├── requirements.txt            # Minimal dependencies to run the app
├── requirements-dev.txt        # Development dependencies (Jupyter, etc.)
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/sandradawn/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the core packages required to run the web application:

```bash
pip install -r requirements.txt
```

*(Note: If you plan to run or modify the Jupyter notebooks in the `notebooks/` directory, install the development dependencies using `pip install -r requirements-dev.txt`)*

### 4. Run the Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app/app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🧠 Machine Learning Insights

The underlying model is trained on the Kaggle Telco Customer Churn dataset. Key global drivers of churn identified during the modeling phase include:
1. **Month-to-Month Contracts**: The strongest positive driver of churn. Customers without long-term commitments are highly volatile.
2. **Short Tenure**: Customers with low tenure (<10 months) show significantly elevated churn risk.
3. **Fiber Optic Service**: Surprisingly correlated with increased churn, often pointing to pricing sensitivity or competitor targeting.
4. **Lack of Support Services**: Customers lacking Tech Support or Online Security packages churn at higher rates.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📝 License

This project is licensed under the MIT License.
