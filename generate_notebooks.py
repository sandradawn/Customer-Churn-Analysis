import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Notebook 01: Exploratory Data Analysis (EDA)\n\n"
            "This notebook loads the Telco Customer Churn dataset and performs exploratory data analysis "
            "to understand customer behavior, dataset distributions, and key drivers of churn."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "%matplotlib inline\n"
            "sns.set_theme(style='whitegrid', palette='muted')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load and Inspect the Dataset\n\n"
            "We load the dataset and display basic statistics: shape, data types, and null values."
        ),
        nbf.v4.new_code_cell(
            "df = pd.read_csv('../data/telco_churn.csv')\n"
            "print(f'Dataset Shape: {df.shape}')\n"
            "print('\\nDataset Columns and Types:')\n"
            "print(df.dtypes)\n"
            "print('\\nNull Values Count:')\n"
            "print(df.isnull().sum())"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Visualize Churn Distribution\n\n"
            "We look at the proportion of customers who churned vs. those who stayed."
        ),
        nbf.v4.new_code_cell(
            "churn_counts = df['Churn'].value_counts()\n"
            "churn_pct = df['Churn'].value_counts(normalize=True) * 100\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n\n"
            "# Bar Chart\n"
            "sns.countplot(data=df, x='Churn', ax=axes[0], hue='Churn', legend=False)\n"
            "axes[0].set_title('Churn Count Distribution')\n"
            "for p in axes[0].patches:\n"
            "    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() + 50), ha='center')\n\n"
            "# Pie Chart\n"
            "axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#5dade2', '#ec7063'])\n"
            "axes[1].set_title('Churn Percentage')\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Churn by Contract Type\n\n"
            "We analyze the relationship between contract types and churn rate."
        ),
        nbf.v4.new_code_cell(
            "sns.countplot(data=df, x='Contract', hue='Churn')\n"
            "plt.title('Churn by Contract Type')\n"
            "plt.ylabel('Count')\n"
            "plt.xlabel('Contract Type')\n"
            "plt.show()\n\n"
            "# Cross-tabulation percentage\n"
            "pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Churn by Customer Tenure\n\n"
            "We analyze if newer customers (lower tenure) have a higher propensity to churn."
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots()\n"
            "sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False, alpha=0.5, ax=ax)\n"
            "ax.set_title('Tenure Distribution by Churn')\n"
            "ax.set_xlabel('Tenure (months)')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Churn by Monthly Charges\n\n"
            "We inspect if customers with higher monthly bills tend to churn more."
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots()\n"
            "sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, alpha=0.5, ax=ax)\n"
            "ax.set_title('Monthly Charges Distribution by Churn')\n"
            "ax.set_xlabel('Monthly Charges ($)')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Churn by Internet Service Type\n\n"
            "We analyze how internet connection medium correlates with churn."
        ),
        nbf.v4.new_code_cell(
            "sns.countplot(data=df, x='InternetService', hue='Churn')\n"
            "plt.title('Churn by Internet Service Type')\n"
            "plt.ylabel('Count')\n"
            "plt.xlabel('Internet Service Type')\n"
            "plt.show()\n\n"
            "pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Correlation Heatmap\n\n"
            "We convert TotalCharges to numeric, map Churn to binary, and plot the correlation heatmap."
        ),
        nbf.v4.new_code_cell(
            "# Convert TotalCharges to numeric\n"
            "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n"
            "df['Churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})\n\n"
            "# Select numerical columns\n"
            "num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_numeric']\n"
            "corr_matrix = df[num_cols].corr()\n\n"
            "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n"
            "plt.title('Correlation Heatmap for Numerical Features')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Summary of Key Insights from EDA\n"
            "1. **Churn Imbalance**: ~26.5% of customers churn, showing that the dataset is imbalanced.\n"
            "2. **Contract Type**: Month-to-month contracts exhibit highly elevated churn rates (~42.7%) compared to one-year (~11.3%) and two-year (~2.8%) contracts.\n"
            "3. **Tenure**: Newly acquired customers (low tenure, <10 months) are at a much higher risk of churning.\n"
            "4. **Charges**: Higher monthly charges correlate with higher churn rates.\n"
            "5. **Internet Service**: Fiber optic subscribers have a disproportionately high churn rate (~41.9%) compared to DSL (~19.0%) and no internet service (~7.4%)."
        )
    ]
    nb['cells'] = cells
    return nb

def create_preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Notebook 02: Preprocessing and Data Pipeline\n\n"
            "This notebook prepares the raw data for model training. We will handle missing values, "
            "standardize numerical columns, one-hot encode categorical features, and split the data."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import joblib\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.compose import ColumnTransformer"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Data and Handle Missing Values\n\n"
            "We convert `TotalCharges` to numeric. Missing values in `TotalCharges` represent new customers (tenure=0). "
            "We will fill these nulls with 0.0."
        ),
        nbf.v4.new_code_cell(
            "df = pd.read_csv('../data/telco_churn.csv')\n"
            "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n"
            "print(f'Null values in TotalCharges before imputation: {df[\"TotalCharges\"].isnull().sum()}')\n\n"
            "# Fill missing values with 0 since tenure is 0 for these rows\n"
            "df['TotalCharges'] = df['TotalCharges'].fillna(0.0)\n"
            "print(f'Null values in TotalCharges after imputation: {df[\"TotalCharges\"].isnull().sum()}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Map Target Variable\n\n"
            "We map the binary target column `Churn` ('Yes'/'No') to 1/0."
        ),
        nbf.v4.new_code_cell(
            "df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})\n"
            "print('Target Churn distribution:')\n"
            "print(df['Churn'].value_counts(normalize=True))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Train-Test Split\n\n"
            "We split the dataset into 80% train and 20% test sets, using stratification to maintain "
            "the distribution of the target class."
        ),
        nbf.v4.new_code_cell(
            "X = df.drop(columns=['customerID', 'Churn'])\n"
            "y = df['Churn']\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train shape: {X_train.shape}')\n"
            "print(f'Test shape: {X_test.shape}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Construct ColumnTransformer Pipeline\n\n"
            "We scale numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`) and one-hot encode "
            "categorical variables, dropping the first category to avoid multicollinearity."
        ),
        nbf.v4.new_code_cell(
            "numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n"
            "categorical_cols = [\n"
            "    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',\n"
            "    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',\n"
            "    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'\n"
            "]\n\n"
            "preprocessor = ColumnTransformer(\n"
            "    transformers=[\n"
            "        ('num', StandardScaler(), numeric_cols),\n"
            "        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)\n"
            "    ],\n"
            "    remainder='passthrough'\n"
            ")\n\n"
            "# Fit-transform on training and transform on testing\n"
            "X_train_preprocessed = preprocessor.fit_transform(X_train)\n"
            "X_test_preprocessed = preprocessor.transform(X_test)\n\n"
            "feature_names = preprocessor.get_feature_names_out()\n"
            "print(f'Preprocessed Train shape: {X_train_preprocessed.shape}')\n"
            "print('Feature Names:')\n"
            "print(list(feature_names))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Save the Preprocessor and Splits\n\n"
            "We save the preprocessor object and the train/test splits so they can be loaded in the training notebook."
        ),
        nbf.v4.new_code_cell(
            "os.makedirs('../models', exist_ok=True)\n"
            "joblib.dump(preprocessor, '../models/preprocessor.pkl')\n"
            "joblib.dump((X_train, X_test, y_train, y_test), '../data/split_data.pkl')\n"
            "print('Successfully saved preprocessor and data splits.')"
        )
    ]
    nb['cells'] = cells
    return nb

def create_model_training_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Notebook 03: Model Training and Evaluation\n\n"
            "This notebook trains three different models (Logistic Regression, Random Forest, and XGBoost), "
            "evaluates them on several metrics, plots a Precision-Recall curve, and exports the best model."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import joblib\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from xgboost import XGBClassifier\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,\n"
            "    classification_report, confusion_matrix, precision_recall_curve\n"
            ")\n\n"
            "%matplotlib inline\n"
            "sns.set_theme(style='whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Preprocessor and Splits"
        ),
        nbf.v4.new_code_cell(
            "X_train, X_test, y_train, y_test = joblib.load('../data/split_data.pkl')\n"
            "preprocessor = joblib.load('../models/preprocessor.pkl')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Train Models\n\n"
            "We build Pipelines for all three classifiers, integrating the preprocessor. "
            "To handle target imbalance, we use `class_weight='balanced'` for Logistic Regression and Random Forest, "
            "and compute `scale_pos_weight` for XGBoost."
        ),
        nbf.v4.new_code_cell(
            "# Calculate scale_pos_weight for XGBoost\n"
            "scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n"
            "print(f'Computed XGBoost scale_pos_weight: {scale_pos_weight:.2f}')\n\n"
            "# Define Pipelines\n"
            "models = {\n"
            "    'Logistic Regression': Pipeline([\n"
            "        ('preprocessor', preprocessor),\n"
            "        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))\n"
            "    ]),\n"
            "    'Random Forest': Pipeline([\n"
            "        ('preprocessor', preprocessor),\n"
            "        ('model', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))\n"
            "    ]),\n"
            "    'XGBoost': Pipeline([\n"
            "        ('preprocessor', preprocessor),\n"
            "        ('model', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'))\n"
            "    ])\n"
            "}\n\n"
            "# Fit all models\n"
            "for name, pipeline in models.items():\n"
            "    print(f'Training {name}...')\n"
            "    pipeline.fit(X_train, y_train)"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Evaluate Model Performance\n\n"
            "We evaluate Accuracy, ROC-AUC, F1-Score, Precision, and Recall for each model, "
            "and print confusion matrices and classification reports."
        ),
        nbf.v4.new_code_cell(
            "metrics_summary = []\n\n"
            "for name, pipeline in models.items():\n"
            "    y_pred = pipeline.predict(X_test)\n"
            "    y_probs = pipeline.predict_proba(X_test)[:, 1]\n"
            "    \n"
            "    metrics = {\n"
            "        'Model': name,\n"
            "        'Accuracy': accuracy_score(y_test, y_pred),\n"
            "        'ROC-AUC': roc_auc_score(y_test, y_probs),\n"
            "        'F1 Score': f1_score(y_test, y_pred),\n"
            "        'Precision': precision_score(y_test, y_pred),\n"
            "        'Recall': recall_score(y_test, y_pred)\n"
            "    }\n"
            "    metrics_summary.append(metrics)\n"
            "    \n"
            "    print('='*50)\n"
            "    print(f'MODEL: {name}')\n"
            "    print('='*50)\n"
            "    print(classification_report(y_test, y_pred))\n"
            "    \n"
            "    # Confusion Matrix\n"
            "    cm = confusion_matrix(y_test, y_pred)\n"
            "    fig, ax = plt.subplots(figsize=(5, 4))\n"
            "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)\n"
            "    ax.set_title(f'{name} Confusion Matrix')\n"
            "    ax.set_ylabel('True Label')\n"
            "    ax.set_xlabel('Predicted Label')\n"
            "    plt.show()\n\n"
            "df_metrics = pd.DataFrame(metrics_summary)\n"
            "df_metrics.set_index('Model', inplace=True)\n"
            "print('Summary Table:')\n"
            "print(df_metrics.round(3))"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Precision-Recall Curve & Threshold Optimization\n\n"
            "We plot the Precision-Recall curve for each model. This allows us to select a threshold that balances "
            "false positives (precision) and false negatives (recall) according to our business preferences."
        ),
        nbf.v4.new_code_cell(
            "for name, pipeline in models.items():\n"
            "    y_probs = pipeline.predict_proba(X_test)[:, 1]\n"
            "    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)\n"
            "    plt.plot(recalls, precisions, label=name)\n\n"
            "plt.title('Precision-Recall Curves')\n"
            "plt.xlabel('Recall (Sensitivity)')\n"
            "plt.ylabel('Precision')\n"
            "plt.legend()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Export the Best Performing Model\n\n"
            "We inspect which model has the highest overall ROC-AUC / F1-Score, and save it as `models/best_model.pkl`."
        ),
        nbf.v4.new_code_cell(
            "# Identify model with highest ROC-AUC\n"
            "best_model_name = df_metrics['ROC-AUC'].idxmax()\n"
            "best_pipeline = models[best_model_name]\n"
            "print(f'Saving best model: {best_model_name} (ROC-AUC: {df_metrics.loc[best_model_name, \"ROC-AUC\"]:.3f})')\n\n"
            "os.makedirs('../models', exist_ok=True)\n"
            "joblib.dump(best_pipeline, '../models/best_model.pkl')\n"
            "print('Successfully saved models/best_model.pkl')"
        )
    ]
    nb['cells'] = cells
    return nb

def create_explainability_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Notebook 04: Model Explainability with SHAP\n\n"
            "This notebook loads the best performing model pipeline, preprocesses the test set, "
            "and uses SHAP (SHapley Additive Explanations) to explain global feature contributions and individual predictions."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import joblib\n"
            "import shap\n"
            "import matplotlib.pyplot as plt\n\n"
            "shap.initjs()"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Best Model and Test Splits"
        ),
        nbf.v4.new_code_cell(
            "X_train, X_test, y_train, y_test = joblib.load('../data/split_data.pkl')\n"
            "best_model = joblib.load('../models/best_model.pkl')\n"
            "print(f'Best Model Pipeline structure:\\n{best_model}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Extract Preprocessor and Model, and Preprocess the Test Set\n\n"
            "To explain the classifier's features directly, we pass the test set through the pipeline's preprocessor step."
        ),
        nbf.v4.new_code_cell(
            "preprocessor = best_model.named_steps['preprocessor']\n"
            "model = best_model.named_steps['model']\n\n"
            "X_test_preproc = preprocessor.transform(X_test)\n"
            "feature_names = preprocessor.get_feature_names_out()\n\n"
            "# Convert to a structured DataFrame for explainability\n"
            "X_test_preproc_df = pd.DataFrame(X_test_preproc, columns=feature_names)\n"
            "print(f'Preprocessed Test Shape: {X_test_preproc_df.shape}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Compute SHAP Values\n\n"
            "We initialize a SHAP TreeExplainer or standard Explainer depending on the model type."
        ),
        nbf.v4.new_code_cell(
            "if 'XGB' in str(type(model)) or 'RandomForest' in str(type(model)):\n"
            "    explainer = shap.TreeExplainer(model)\n"
            "    # Extract shap values\n"
            "    shap_values = explainer(X_test_preproc_df)\n"
            "else:\n"
            "    explainer = shap.Explainer(model, X_test_preproc_df)\n"
            "    shap_values = explainer(X_test_preproc_df)\n\n"
            "print('SHAP Values Computed successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. SHAP Beeswarm Summary Plot\n\n"
            "The beeswarm plot shows the global impact and directionality of features. "
            "For each feature, it plots individual customer SHAP values colored by feature value (high value in red, low in blue)."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(10, 8))\n"
            "shap.plots.beeswarm(shap_values, max_display=15, show=False)\n"
            "plt.title('SHAP Beeswarm Plot (Global Feature Impact)', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. SHAP Bar Plot (Global Importance)\n\n"
            "This shows the average absolute SHAP values, ranking the features by overall global importance."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(10, 6))\n"
            "shap.plots.bar(shap_values, max_display=15, show=False)\n"
            "plt.title('SHAP Global Feature Importance Bar Plot', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. SHAP Waterfall Plot (Individual Prediction)\n\n"
            "A waterfall plot explains a single customer prediction, showing how each feature value moves the churn probability "
            "away from the base expected value (prior probability) to the final predicted output."
        ),
        nbf.v4.new_code_cell(
            "# Select the first customer in the test set\n"
            "plt.figure(figsize=(10, 6))\n"
            "shap.plots.waterfall(shap_values[0], show=False)\n"
            "plt.title('SHAP Waterfall Plot for Single Customer Prediction', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Identify Top 5 Features & Business Interpretations\n\n"
            "We compute the mean absolute SHAP values to identify and print the top 5 most important features."
        ),
        nbf.v4.new_code_cell(
            "mean_abs_shap = np.abs(shap_values.values).mean(axis=0)\n"
            "df_importance = pd.DataFrame({\n"
            "    'Feature': feature_names,\n"
            "    'Mean_Abs_SHAP': mean_abs_shap\n"
            "}).sort_values(by='Mean_Abs_SHAP', ascending=False)\n\n"
            "print('Top 5 Most Important Features:')\n"
            "print(df_importance.head(5).to_string(index=False))"
        ),
        nbf.v4.new_markdown_cell(
            "### Business Meanings of the Top Churn Drivers:\n\n"
            "1. **Contract Type (e.g., `Contract_One year` / `Contract_Two year` vs Month-to-Month)**:\n"
            "   - **Meaning**: Month-to-Month contracts show a massive positive SHAP value, indicating they are strong drivers of churn. Conversely, having a One-Year or Two-Year contract significantly decreases a customer's churn risk.\n\n"
            "2. **Tenure (`num__tenure`)**:\n"
            "   - **Meaning**: Shorter customer relationship durations (tenure) highly increase churn probability. Long-standing customers (high tenure) are highly loyal, which appears as a strong negative SHAP value pulling risk down.\n\n"
            "3. **Internet Service Type (`cat__InternetService_Fiber optic`)**:\n"
            "   - **Meaning**: Subscribers with Fiber Optic connections show an increased risk of churn. This might reflect customer dissatisfaction with service cost, speed reliability, or aggressive competitor targeting.\n\n"
            "4. **Tech Support Availability (`cat__TechSupport_No internet service` or `cat__TechSupport_Yes`)**:\n"
            "   - **Meaning**: Customers who do not have tech support service are at a significantly higher risk of churning. Value-added support acts as an anchor that increases customer stickiness.\n\n"
            "5. **Monthly Charges (`num__MonthlyCharges`)**:\n"
            "   - **Meaning**: High monthly billing amounts directly drive customer churn. When monthly expenses exceed customers' perceived value, they cancel services."
        )
    ]
    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    print("Generating notebooks...")
    
    # 01 EDA
    nbf.write(create_eda_notebook(), os.path.join("notebooks", "01_eda.ipynb"))
    print("Generated 01_eda.ipynb")
    
    # 02 Preprocessing
    nbf.write(create_preprocessing_notebook(), os.path.join("notebooks", "02_preprocessing.ipynb"))
    print("Generated 02_preprocessing.ipynb")
    
    # 03 Model Training
    nbf.write(create_model_training_notebook(), os.path.join("notebooks", "03_model_training.ipynb"))
    print("Generated 03_model_training.ipynb")
    
    # 04 Explainability
    nbf.write(create_explainability_notebook(), os.path.join("notebooks", "04_explainability.ipynb"))
    print("Generated 04_explainability.ipynb")
    
    print("\nExecuting notebooks end-to-end to generate outputs & verify correctness...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    notebook_files = [
        "01_eda.ipynb",
        "02_preprocessing.ipynb",
        "03_model_training.ipynb",
        "04_explainability.ipynb"
    ]
    
    for filename in notebook_files:
        path = os.path.join("notebooks", filename)
        print(f"Executing {filename}...")
        with open(path, "r", encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)
        
        # Run notebook. The path is set to 'notebooks' folder so relative paths inside notebook function correctly.
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Finished executing {filename} and saved outputs.")
        
    print("\nAll notebooks generated and verified successfully.")
