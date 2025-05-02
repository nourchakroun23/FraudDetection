import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image
from lime.lime_tabular import LimeTabularExplainer

# Configuration
st.set_page_config(
    layout="wide",
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️"
)

# CSS personnalisé
def load_css():
    with open("assets/css/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Titre avec logo
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/images/logo.png", width=80)
with col2:
    st.title("Advanced Fraud Detection Dashboard")

# Chargement des données
@st.cache_data
def load_data():
    return {
        "X_test": np.load("data/processed/X_test_fused.npy"),
        "y_test": np.load("data/processed/y_test.npy"),
        "feature_names": pickle.load(open("assets/feature_names.pkl", "rb")),
        "sample_data": pd.read_csv("data/raw/data_sample.csv")
    }

# Chargement des modèles
@st.cache_resource
def load_models():
    return {
        "xgb": joblib.load("data/models/xgboost_model.pkl"),
        "autoencoder": load_model(
            "data/models/autoencoder_classifier.h5",
            custom_objects={'mse': MeanSquaredError()}
        ),
        "fl_models": [joblib.load(f"data/federated_assets/local_model_client_{i}.pkl") for i in range(5)]
    }

# Chargement des assets FL
@st.cache_data
def load_fl_assets():
    return {
        "metrics": pickle.load(open("data/federated_assets/global_metrics.pkl", "rb")),
        "roc_curve": Image.open("data/federated_assets/roc_curve_fl.png"),
        "conf_matrix": Image.open("data/federated_assets/confusion_matrix_fl.png")
    }

# Initialisation
data = load_data()
models = load_models()
fl_assets = load_fl_assets()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("", ["🏠 Overview", "🔍 Model Analysis", "🌐 Federated Learning", "🤖 AI Explainability"])

# Align features function
def align_features(X_input, feature_names, expected_num_features=40):
    X_df = pd.DataFrame(X_input)

    # Truncate if more than expected
    if X_df.shape[1] > expected_num_features:
        X_df = X_df.iloc[:, :expected_num_features]

    # Add missing columns
    missing_count = expected_num_features - X_df.shape[1]
    if missing_count > 0:
        for i in range(missing_count):
            col_name = f"missing_feature_{i}"
            X_df[col_name] = 0

    # Re-assign column names
    all_names = feature_names[:min(len(feature_names), expected_num_features)]
    while len(all_names) < expected_num_features:
        all_names.append(f"missing_feature_{len(all_names)}")
    X_df.columns = all_names

    return X_df

# Pages
if page == "🏠 Overview":
    st.header("Project Overview")
    st.markdown("""
    This dashboard presents a **hybrid fraud detection system** combining:
    - Traditional ML models (XGBoost, Random Forest)
    - Deep Learning (Autoencoder classifier)
    - Federated Learning (5 clients simulation)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("assets/images/roc_curves.png", caption="Model Comparison (ROC Curves)")
    with col2:
        st.image("assets/images/confusion_matrix_xgboost.png", caption="Best Model (XGBoost)")

elif page == "🔍 Model Analysis":
    st.header("Model Performance Analysis")
    tab1, tab2 = st.tabs(["📈 Metrics", "📊 Data Explorer"])
    
    with tab1:
        model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest", "LightGBM"])
        st.image(f"assets/images/confusion_matrix_{model_choice.lower().replace(' ', '_')}.png")
        
    with tab2:
        st.dataframe(data["sample_data"].style.highlight_max(axis=0), height=600)

elif page == "🌐 Federated Learning":
    st.header("Federated Learning Simulation")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", f"{fl_assets['metrics']['roc_auc']:.2%}")
    col2.metric("Precision", f"{fl_assets['metrics']['precision']:.2%}")
    col3.metric("Recall", f"{fl_assets['metrics']['recall']:.2%}")
    
    st.subheader("Global Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.image(fl_assets["roc_curve"])
    with col2:
        st.image(fl_assets["conf_matrix"])
    
    st.subheader("Client-Specific Analysis")
    client = st.select_slider("Select Client", options=range(5))
    instance_idx = st.slider("Test Instance", 0, 100, 0)

    # No need for align_features here, skip this part
    X_df = data["X_test"]  # Directly use data without alignment for Federated Learning
    local_pred = models["fl_models"][client].predict_proba([X_df[instance_idx]])[0][1]
    global_pred = np.mean([m.predict_proba([X_df[instance_idx]])[0][1] for m in models["fl_models"]])

    st.write(f"**Client {client} Prediction** vs **Global Model**")
    st.progress(int(local_pred * 100))
    st.caption(f"Local probability: {local_pred:.2%}")
    st.progress(int(global_pred * 100))
    st.caption(f"Global aggregated probability: {global_pred:.2%}")

elif page == "🤖 AI Explainability":
    st.header("Model Explainability")
    tab1, tab2 = st.tabs(["SHAP Analysis", "LIME Explanations"])
    
    with tab1:
        st.subheader("Feature Importance (SHAP)")
        # Apply align_features for SHAP
        X_df = align_features(data["X_test"], data["feature_names"])
        
        explainer = shap.TreeExplainer(models["xgb"])
        shap_values = explainer.shap_values(X_df.iloc[:100])
        fig = plt.figure()
        shap.summary_plot(shap_values, X_df.iloc[:100], feature_names=X_df.columns)
        st.pyplot(fig)

    with tab2:
        st.subheader("Instance-Level Explanation (LIME)")
        # Apply align_features for LIME
        X_df = align_features(data["X_test"], data["feature_names"])
        instance_idx = st.number_input("Instance Index", 0, len(X_df) - 1, 0)

        explainer = LimeTabularExplainer(
            training_data=X_df.values,
            feature_names=X_df.columns.tolist(),
            class_names=["Legit", "Fraud"],
            mode='classification'
        )

        exp = explainer.explain_instance(
            X_df.iloc[instance_idx].values,
            models["xgb"].predict_proba,
            num_features=10
        )

        st.components.v1.html(exp.as_html(), height=800)

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
<p>Fraud Detection System - © 2025 | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
