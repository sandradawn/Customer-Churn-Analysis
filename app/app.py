import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_model, get_shap_explainer, plot_shap_waterfall

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
css_path = os.path.join("app", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    st.info("Please make sure you run the notebooks to generate split_data.pkl and best_model.pkl before launching the Streamlit app.")
    model_loaded = False

if model_loaded:
    # Sidebar: Input widgets
    with st.sidebar:
        st.markdown("### 👤 Customer Profile Input")
        st.markdown("Provide details about the customer below:")
        
        with st.expander("Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior_citizen = 1 if senior_citizen_str == "Yes" else 0
            partner = st.selectbox("Partner (Married/Cohabiting)", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with st.expander("Services Subscribed", expanded=False):
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
            
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"

        with st.expander("Contract & Billing", expanded=False):
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

        with st.expander("Usage & Charges", expanded=True):
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.slider("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
            
            suggested_total = float(tenure * monthly_charges)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=suggested_total)

    # Create Input DataFrame
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
        
        try:
            probs = best_model.predict_proba(input_df)[0]
            churn_prob = probs[1] * 100
            
            col1, col2 = st.columns([1, 2], gap="large")
            
            with col1:
                st.markdown("#### Risk Assessment")
                
                if churn_prob < 40.0:
                    card_class = "card-low"
                    risk_level = "LOW RISK"
                    text_color = "#16a34a"
                    explanation = "This customer is stable. Continue standard engagement and loyalty programs."
                elif churn_prob <= 70.0:
                    card_class = "card-medium"
                    risk_level = "MODERATE RISK"
                    text_color = "#d97706"
                    explanation = "This customer shows early signs of churn risk. Consider offering proactive support or discounts."
                else:
                    card_class = "card-high"
                    risk_level = "HIGH RISK"
                    text_color = "#dc2626"
                    explanation = "This customer is highly likely to churn. Immediate retention action recommended (e.g. contract upgrades, specialized promotions)."
                
                st.markdown(f"""
                    <div class="risk-card {card_class}">
                        <div class="risk-title" style="color: {text_color};">{risk_level}</div>
                        <div class="risk-prob" style="color: {text_color};">{churn_prob:.1f}%</div>
                        <div class="risk-desc">
                            This customer has a <b>{churn_prob:.1f}%</b> probability of churning in the next month.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**Recommendation:** {explanation}")
                
                with st.expander("🔎 View Submitted Profile details"):
                    st.json(input_data)
                    
            with col2:
                st.markdown("#### Explainable AI (SHAP Plot)")
                st.write("The waterfall plot below explains which features pushed the prediction up (red) or down (blue) relative to the average churn rate.")
                
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
