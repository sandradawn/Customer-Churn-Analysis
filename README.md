# Customer Churn Analysis & Prediction Project

This project delivers a complete, end-to-end Machine Learning solution to predict and analyze customer churn for a telecommunications provider. Using demographic, service usage, and financial data from the Kaggle Telco Customer Churn dataset, we construct a data preprocessing and modeling pipeline, evaluate multiple machine learning classifiers, apply SHAP (SHapley Additive Explanations) for explainability, and deploy a premium, interactive Streamlit web dashboard.

---

## 📌 Project Folder Structure
```
customer-churn-analysis/
├── data/
│   ├── telco_churn.csv         # Raw Telco Customer Churn dataset (7043 rows, 21 columns)
│   └── split_data.pkl          # Pickled training and testing splits (80/20 stratified split)
├── notebooks/
│   ├── 01_eda.ipynb            # Step 1: Exploratory Data Analysis & Visualizations
│   ├── 02_preprocessing.ipynb  # Step 2: Data Cleansing, Imputation & Feature Engineering
│   ├── 03_model_training.ipynb # Step 3: Model Training, Evaluation & Optimization
│   └── 04_explainability.ipynb # Step 4: SHAP Global and Local Explainability Analysis
├── models/
│   ├── preprocessor.pkl        # Fitted ColumnTransformer (StandardScaler & OneHotEncoder)
│   └── best_model.pkl          # Saved best performing model pipeline (Logistic Regression)
├── app/
│   ├── app.py                  # Streamlit web application entrypoint
│   └── utils.py                # Helper functions for app caching and SHAP plotting
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git exclusion rules (preserving data and best_model.pkl)
└── README.md                   # Project documentation (this file)
```

---

## 💻 Tech Stack
- **Programming Language**: Python 3.11
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Model Explainability**: SHAP (SHapley Additive Explanations)
- **Data Visualization**: Matplotlib, Seaborn
- **Serialization**: Joblib
- **Web App Framework**: Streamlit
- **Notebook Environment**: Jupyter Notebook

---

## 📊 Dataset Description
The project utilizes the **Kaggle Telco Customer Churn** dataset.
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 rows, 21 columns
- **Target Variable**: `Churn` (Yes / No) - indicating whether a customer cancelled their service within the last month. Churn rate is approximately **26.5%**, representing a moderate class imbalance.
- **Features**:
  - **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - **Account & Financials**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 📈 Model Performance Comparison
Three models were trained with imbalance-aware strategies: **Logistic Regression** (balanced weights), **Random Forest** (balanced weights), and **XGBoost** (using `scale_pos_weight`). Performance on the 20% stratified test set is detailed below:

| Model | Accuracy | ROC-AUC | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.738** | **0.842** | **0.614** | **0.504** | **0.783** |
| **Random Forest** | 0.762 | 0.823 | 0.589 | 0.544 | 0.642 |
| **XGBoost** | 0.757 | 0.819 | 0.603 | 0.533 | 0.695 |

### Key Modeling Takeaways:
- **Logistic Regression (with Balanced Class Weights)** was selected as the **best performing model** due to achieving the highest **ROC-AUC (0.842)** and **F1 Score (0.614)**.
- It exhibits a high **Recall (78.3%)**, meaning it successfully identifies ~78% of the actual churning customers, making it a highly effective tool for proactive customer retention strategies.

---

## 🔍 Key Insights from SHAP (Top Churn Drivers)
Using SHAP values on the test set, we uncovered the following global drivers of customer churn:
1. **Month-to-Month Contracts**: Having a Month-to-month contract is the single strongest positive driver of churn. Customers without long-term commitments are highly volatile.
2. **Short Tenure**: Customers with low tenure (<10 months) show significantly elevated churn risk. Retention risk decreases exponentially as tenure grows.
3. **Fiber Optic Service**: Subscribing to Fiber Optic internet is strongly correlated with increased churn, possibly pointing to pricing issues or competitors' targeted discounts.
4. **Lack of Tech Support**: Customers who do not subscribe to Tech Support or Online Security services churn at much higher rates. Support packages serve as key relationship anchors.
5. **High Monthly Charges**: High billing amounts push customers to cancel services when their expenses exceed their perceived value.

---

## 💡 Business Recommendations
Based on the machine learning findings, the following actions should be taken:
- **Contract Migration**: Offer financial incentives (e.g., a free month or monthly discounts) to migrate Month-to-Month customers onto 1-Year or 2-Year contracts.
- **Onboarding Retention Programs**: Focus customer success efforts on new sign-ups within their first 6 months. Implement targeted feedback calls at months 1, 3, and 6.
- **Feature Bundling**: Promote and pre-bundle Online Security and Tech Support services with internet plans. Since these services reduce churn, offering them as a discount package increases stickiness.
- **Fiber Optic Plan Review**: Conduct customer satisfaction surveys among Fiber Optic subscribers to identify billing disputes or quality issues and match competitor pricing.

---

## 🚀 How to Install and Run

### 1. Prerequisites
Ensure you have **Python 3.11** installed. Locate your terminal and navigate to the project directory:
```bash
cd "C:\Users\saira anna dawn\.gemini\antigravity\scratch\customer-churn-analysis"
```

### 2. Set Up Virtual Environment & Install Dependencies
Create and activate the virtual environment, then install requirements:
```bash
# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 3. Run Notebooks in Order
You can run the notebooks in your IDE or launch Jupyter Notebook:
```bash
jupyter notebook
```
Execute the notebooks in this order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_model_training.ipynb`
4. `notebooks/04_explainability.ipynb`

Alternatively, you can run all notebooks programmatically using our generator script:
```bash
python generate_notebooks.py
```

### 4. Run the Streamlit Application
Launch the Streamlit dashboard:
```bash
streamlit run app/app.py
```
Open the provided URL in your web browser (usually `http://localhost:8501`) to interactively assess customer churn risk and view real-time SHAP waterfall charts.
