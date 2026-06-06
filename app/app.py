import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_model, get_shap_explainer, plot_shap_waterfall

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3498db, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .card-low {
        background-color: #ebfaf2;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #2ecc71;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }
    
    .card-medium {
        background-color: #fef5e7;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #f39c12;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }
    
    .card-high {
        background-color: #fdedec;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #e74c3c;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }
    
    .risk-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .risk-prob {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.markdown('<div class="main-title">Telco Customer Churn Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict customer churn probability and inspect individual risk drivers using SHAP explainability.</div>', unsafe_allow_html=True)

# Load best model and SHAP explainer
try:
    best_model = load_model()
    explainer, clean_feature_names = get_shap_explainer(best_model)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or training data: {e}")
    st.info("Please make sure you run the notebooks (specifically Notebook 03 and 02) to generate split_data.pkl and best_model.pkl before launching the Streamlit app.")
    model_loaded = False

if model_loaded:
    # Sidebar: Input widgets
    st.sidebar.markdown("### 👤 Customer Profile Input")
    st.sidebar.markdown("Provide details about the customer below:")

    # Demographics
    st.sidebar.markdown("**Demographics**")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen_str = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_str == "Yes" else 0
    partner = st.sidebar.selectbox("Partner (Married/Cohabiting)", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

    # Services
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Services Subscribed**")
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    
    # Optional services depend on internet connection
    if internet_service != "No":
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    # Contract & Billing
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Contract & Billing**")
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Usage / Charges
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Usage & Charges**")
    tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
    
    # Estimate total charges based on tenure and monthly charges, but allow custom override
    suggested_total = float(tenure * monthly_charges)
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 
        min_value=0.0, 
        max_value=10000.0, 
        value=suggested_total
    )

    # Create Input DataFrame matching raw column names and types
    input_data = {
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    }
    input_df = pd.DataFrame(input_data)

    # Prediction Action
    st.markdown("### 📊 Churn Prediction & Risk Analysis")
    
    predict_clicked = st.button("Predict Churn Risk", type="primary", use_container_width=True)

    if predict_clicked or 'prediction_done' in st.session_state:
        st.session_state.prediction_done = True
        
        # Run prediction
        try:
            # Predict Churn Probability
            probs = best_model.predict_proba(input_df)[0]
            churn_prob = probs[1] * 100
            
            # Setup layout columns
            col1, col2 = st.columns([1, 2], gap="large")
            
            with col1:
                st.markdown("#### Risk Assessment")
                
                # Determine risk level and custom CSS card class
                if churn_prob < 40.0:
                    card_class = "card-low"
                    risk_level = "LOW RISK"
                    text_color = "#27ae60"
                    explanation = "This customer is stable. Continue standard engagement and loyalty programs."
                elif churn_prob <= 70.0:
                    card_class = "card-medium"
                    risk_level = "MODERATE RISK"
                    text_color = "#d35400"
                    explanation = "This customer shows early signs of churn risk. Consider offering proactive support or discounts."
                else:
                    card_class = "card-high"
                    risk_level = "HIGH RISK"
                    text_color = "#c0392b"
                    explanation = "This customer is highly likely to churn. Immediate retention action recommended (e.g. contract upgrades, specialized promotions)."
                
                # Render the risk card
                st.markdown(f"""
                    <div class="{card_class}">
                        <div class="risk-title" style="color: {text_color};">{risk_level}</div>
                        <div class="risk-prob" style="color: {text_color};">{churn_prob:.1f}%</div>
                        <div style="font-size: 0.95rem; color: #2c3e50; line-height: 1.4;">
                            This customer has a <b>{churn_prob:.1f}%</b> probability of churning in the next month.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Actionable recommendations
                st.info(f"**Recommendation:** {explanation}")
                
                # Display individual inputs for transparency
                with st.expander("🔎 View Submitted Profile details"):
                    st.json(input_data)
                    
            with col2:
                st.markdown("#### Explainable AI (SHAP Plot)")
                st.write("The waterfall plot below explains which features pushed the prediction up (red) or down (blue) relative to the average churn rate.")
                
                # Plot and display SHAP waterfall
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        fig = plot_shap_waterfall(best_model, explainer, clean_feature_names, input_df)
                        st.pyplot(fig)
                    except Exception as shap_err:
                        st.error(f"Failed to render SHAP plot: {shap_err}")
                        st.write("Ensure model inputs match the expected training data columns.")
                        
        except Exception as pred_err:
            st.error(f"An error occurred during prediction: {pred_err}")
            st.exception(pred_err)
else:
    st.warning("Application is currently offline. Please run the model training step first to generate models and datasets.")
