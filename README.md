# ⚡ Telco Churn Intelligence Dashboard

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Plotly-5.22.0-3F4F75.svg" alt="Plotly">
  <img src="https://img.shields.io/badge/Explainability-SHAP-9cf.svg" alt="SHAP">
</div>

<br>

An end-to-end Machine Learning web application designed to predict and analyze customer churn for telecommunication providers. Features a bespoke, SaaS-level real-time dashboard powered by **Explainable AI (SHAP)** and **Plotly** interactive analytics, built on top of an optimized **Scikit-Learn** preprocessing and classification pipeline.

---

<img width="1918" height="906" alt="image" src="https://github.com/user-attachments/assets/bc193779-4eae-42a7-a7de-f53319b8a9bd" />
<img width="1915" height="912" alt="image" src="https://github.com/user-attachments/assets/be02e669-2cf8-460e-8e59-5596c4a3a6bd" />
<img width="1917" height="735" alt="image" src="https://github.com/user-attachments/assets/9dfd5e2a-6eac-4beb-9554-0da41c129b90" />
<img width="1915" height="917" alt="image" src="https://github.com/user-attachments/assets/1471ed00-0696-4d18-8df2-a77350b654cc" />


## 🌟 Key Features

*   **Real-Time Churn Analysis**: No manual submission buttons. The dashboard processes profile updates and recalculates predictions instantly as user controls are modified.
*   **Interactive Risk Assessment**: Displays a custom Plotly Gauge Speedometer that dynamically reflects stable, monitoring, or critical risk thresholds.
*   **Explainable AI (XAI)**: Generates live SHAP waterfall plots for individual customer records to break down exactly how each attribute influences the risk score.
*   **Key Churn Drivers**: Programmatically extracts and styles the top 3 attributes contributing to or reducing the customer's churn risk.
*   **Ultra-Premium SaaS UI**: Styled using custom CSS with a clean glassmorphism layout, fluid hover transitions, and a clean typography hierarchy.
*   **Hidden Branding**: Customized container settings completely hide the standard Streamlit header, footer, and deploy buttons to preserve a bespoke product aesthetic.

---

## 🏗️ Project Architecture

```text
customer-churn-analysis/
├── app/
│   ├── app.py                  # Streamlit entry point and UI layout
│   ├── utils.py                # Preprocessor/Model loading & SHAP plot generators
│   └── assets/
│       └── style.css           # SaaS-level styling rules & glassmorphism configurations
├── data/
│   ├── split_data.pkl          # Pickled training and validation datasets
│   └── telco_churn.csv         # Raw customer churn dataset
├── models/
│   ├── best_model.pkl          # Trained Logistic Regression pipeline
│   └── preprocessor.pkl        # Fitted scaling & one-hot encoding transformer
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis & visualizations
│   ├── 02_preprocessing.ipynb  # Missing value handling & feature engineering
│   ├── 03_model_training.ipynb # Hyperparameter tuning & model evaluation
│   └── 04_explainability.ipynb # Global and local SHAP explanation tests
├── requirements.txt            # App dependency list
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- Git

### 1. Clone & Navigate
```bash
git clone https://github.com/sandradawn/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

### 2. Configure Environment
Set up a clean virtual environment to avoid package conflicts:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app/app.py
```
Your default browser will automatically open to `http://localhost:8501`.

---

## 🔬 Machine Learning Pipeline

### Data Preprocessing
Categorical features are dynamically one-hot encoded while numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled. These steps are wrapped in a Scikit-Learn `ColumnTransformer` to prevent data leakage during training and inference.

### Model Details
The system utilizes a **Logistic Regression** classifier with balanced class weights (`class_weight='balanced'`) to mitigate the inherent class imbalance in customer retention datasets. It achieves optimized performance for both precision and recall, allowing business analysts to catch high-risk customers without excessive false alerts.

### Explainability (SHAP)
Instead of treating the ML pipeline as a black box, the dashboard integrates SHAP (SHapley Additive exPlanations) values to build trust:
- **Baseline (E[f(X)])**: Represents the average model output across the training population.
- **Attributions (Arrows)**: Positive values (red) show variables pushing the customer towards churning, while negative values (blue) show attributes encouraging retention.
- **Individual Output (f(X))**: Represents the customer's raw log-odds score, which is sigmoid-transformed into the final churn percentage.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have enhancements or optimizations to share.

## 📄 License
Licensed under the [MIT License](LICENSE).
