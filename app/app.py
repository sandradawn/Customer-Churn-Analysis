import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from utils import load_model, get_shap_explainer, plot_shap_waterfall

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load external CSS
css_path = os.path.join("app", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predictive insights and real-time risk assessment driven by Explainable AI.</div>', unsafe_allow_html=True)

try:
    best_model = load_model()
    explainer, clean_feature_names = get_shap_explainer(best_model)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or training data: {e}")
    st.info("Please make sure you run the notebooks to generate split_data.pkl and best_model.pkl before launching the Streamlit app.")
    model_loaded = False

if model_loaded:
    
    st.markdown("### 1. Build Customer Profile")
    st.markdown("Configure the customer's demographics, services, and billing attributes below:")
    
    # Main Tabs for Input
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Demographics", "📦 Services", "📄 Contract & Billing", "💳 Usage & Charges"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior_citizen = 1 if senior_citizen_str == "Yes" else 0
        with col2:
            partner = st.selectbox("Partner (Married/Cohabiting)", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
        with col2:
            if internet_service != "No":
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            else:
                st.info("Additional services disabled since Internet Service is 'No'.")
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col2:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.slider("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
        with col2:
            suggested_total = float(tenure * monthly_charges)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=suggested_total)
            st.caption(f"Suggested Total based on tenure & monthly: ${suggested_total:.2f}")

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

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
    with col_btn_2:
        predict_clicked = st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True)

    if predict_clicked or 'prediction_done' in st.session_state:
        st.session_state.prediction_done = True
        
        st.markdown("---")
        st.markdown("### 2. Predictive Dashboard")
        
        try:
            probs = best_model.predict_proba(input_df)[0]
            churn_prob = probs[1] * 100
            
            # Extract top drivers using explainer directly
            preprocessor = best_model.named_steps['preprocessor']
            input_preproc = preprocessor.transform(input_df)
            input_preproc_df = pd.DataFrame(input_preproc, columns=clean_feature_names)
            shap_values = explainer(input_preproc_df)
            
            # Map SHAP values
            val_array = shap_values[0].values
            feature_impacts = [(name, val) for name, val in zip(clean_feature_names, val_array)]
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            top_drivers = feature_impacts[:3]

            col_res1, col_res2 = st.columns([1.2, 2], gap="large")
            
            with col_res1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("#### Real-time Risk Score", unsafe_allow_html=True)
                
                # Plotly Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = churn_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1e293b"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#10b981'},
                            {'range': [40, 70], 'color': '#f59e0b'},
                            {'range': [70, 100], 'color': '#ef4444'}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': churn_prob}
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1e293b", 'family': "Inter"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recommendation
                if churn_prob < 40.0:
                    text_color = "#10b981"
                    status = "STABLE"
                    rec = "Continue standard engagement."
                elif churn_prob <= 70.0:
                    text_color = "#d97706"
                    status = "MONITOR"
                    rec = "Offer proactive support or discounts."
                else:
                    text_color = "#dc2626"
                    status = "CRITICAL"
                    rec = "Immediate retention action required!"
                    
                st.markdown(f"<h3 style='text-align: center; color: {text_color}; margin-top: -20px;'>STATUS: {status}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-weight: 500;'>{rec}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("#### Key Drivers", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.9rem; color: #64748b;'>The top factors influencing this specific customer's score:</p>", unsafe_allow_html=True)
                for name, val in top_drivers:
                    direction = "Increased Risk" if val > 0 else "Decreased Risk"
                    badge_class = "driver-high" if val > 0 else "driver-low"
                    st.markdown(f"• <b>{name}</b> <span class='driver-badge {badge_class}'>{direction}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_res2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("#### Explainable AI: SHAP Waterfall", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.95rem; color: #475569;'>This visual breaks down how each feature pushed the risk score up from the baseline average.</p>", unsafe_allow_html=True)
                
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        fig_shap = plot_shap_waterfall(best_model, explainer, clean_feature_names, input_df)
                        st.pyplot(fig_shap)
                    except Exception as shap_err:
                        st.error(f"Failed to render SHAP plot: {shap_err}")
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as pred_err:
            st.error(f"An error occurred during prediction: {pred_err}")
            st.exception(pred_err)
else:
    st.warning("Application is currently offline. Please run the model training step first to generate models and datasets.")
