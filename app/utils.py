import os
import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    """Loads the pre-trained best performing model pipeline from disk."""
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run notebooks first.")
    return joblib.load(model_path)

@st.cache_resource
def get_shap_explainer(_best_model):
    """
    Creates and caches a SHAP explainer based on the best model type.
    Uses the preprocessed training set as background.
    """
    preprocessor = _best_model.named_steps['preprocessor']
    model = _best_model.named_steps['model']
    
    # Load training split to get background distribution
    split_path = os.path.join("data", "split_data.pkl")
    if not os.path.exists(split_path):
         raise FileNotFoundError(f"Split data not found at {split_path}. Please run notebooks first.")
    
    X_train, _, _, _ = joblib.load(split_path)
    X_train_preproc = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_preproc_df = pd.DataFrame(X_train_preproc, columns=feature_names)
    
    # Clean up feature names to make them human readable in SHAP plots
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "").replace("_", " ")
        for name in feature_names
    ]
    X_train_preproc_df.columns = clean_feature_names
    
    # Select appropriate explainer based on model class
    model_type = str(type(model))
    if "LogisticRegression" in model_type:
        explainer = shap.LinearExplainer(model, X_train_preproc_df)
    elif "XGB" in model_type or "RandomForest" in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_train_preproc_df)
        
    return explainer, clean_feature_names

def plot_shap_waterfall(_best_model, _explainer, clean_feature_names, input_df):
    """
    Generates a SHAP waterfall plot for a single row prediction.
    """
    preprocessor = _best_model.named_steps['preprocessor']
    
    # Transform input row
    input_preproc = preprocessor.transform(input_df)
    input_preproc_df = pd.DataFrame(input_preproc, columns=clean_feature_names)
    
    # Compute SHAP values
    shap_values = _explainer(input_preproc_df)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For newer versions of SHAP, shap_values[0] is an Explanation object
    # We display it using shap.plots.waterfall
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    
    # Clean up aesthetics
    plt.title("SHAP Feature Attribution (Why this prediction?)", fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    
    return fig
