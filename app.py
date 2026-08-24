import streamlit as st
import pandas as pd

from src.preprocessor import load_and_prepare_data, get_features_and_target
from src.model import train_model
from src.risk_engine import get_risk_message, calculate_feature_importance


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Alzheimer Risk Prediction",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------
# Load data and train model
# -----------------------------
@st.cache_resource
def build_model():
    df = load_and_prepare_data("data/alzheimer_data.csv")
    X, y = get_features_and_target(df)

    model, accuracy, y_test, predictions = train_model(X, y)

    return df, model, accuracy


df, model, accuracy = build_model()


# -----------------------------
# Header
# -----------------------------
st.title("🧠 Alzheimer Risk Prediction System")

st.markdown(
    """
    ### AI-powered cognitive health screening prototype

    This application uses machine learning to analyze selected
    demographic and cognitive-assessment features and estimate
    a risk category.

    **⚠️ Educational/research prototype — not a medical diagnosis.**
    """
)


# -----------------------------
# Model performance
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Records",
        len(df)
    )

with col3:
    st.metric(
        "ML Algorithm",
        "Random Forest"
    )


st.divider()


# -----------------------------
# Patient assessment
# -----------------------------
st.subheader("👤 Patient Assessment")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=40,
        max_value=95,
        value=65
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    education = st.slider(
        "Education Years",
        min_value=0,
        max_value=25,
        value=12
    )

    mmse = st.slider(
        "MMSE Score",
        min_value=0,
        max_value=30,
        value=24
    )

with col2:

    memory = st.selectbox(
        "Memory Complaints",
        ["No", "Yes"]
    )

    behavioral = st.selectbox(
        "Behavioral Changes",
        ["No", "Yes"]
    )

    activity = st.slider(
        "Daily Activity Score",
        min_value=0,
        max_value=10,
        value=7
    )

    family_history = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )


# -----------------------------
# Convert inputs
# -----------------------------
gender_value = 1 if gender == "Female" else 0
memory_value = 1 if memory == "Yes" else 0
behavioral_value = 1 if behavioral == "Yes" else 0
family_value = 1 if family_history == "Yes" else 0


patient_data = pd.DataFrame(
    [[
        age,
        gender_value,
        education,
        mmse,
        memory_value,
        behavioral_value,
        activity,
        family_value
    ]],
    columns=[
        "Age",
        "Gender",
        "EducationYears",
        "MMSE",
        "MemoryComplaints",
        "BehavioralChanges",
        "DailyActivityScore",
        "FamilyHistory"
    ]
)


# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button(
    "🔍 Analyze Patient Risk",
    type="primary",
    use_container_width=True
):

    prediction = model.predict(patient_data)[0]

    probabilities = model.predict_proba(patient_data)[0]

    confidence = max(probabilities) * 100

    risk = get_risk_message(prediction)

    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Risk Category",
            f"{risk['indicator']} {risk['level']}"
        )

    with result_col2:

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

    st.info(risk["message"])

    st.divider()

    # -----------------------------
    # Feature importance
    # -----------------------------
    st.subheader("🔎 Model Feature Importance")

    features = [
        "Age",
        "Gender",
        "EducationYears",
        "MMSE",
        "MemoryComplaints",
        "BehavioralChanges",
        "DailyActivityScore",
        "FamilyHistory"
    ]

    importance = calculate_feature_importance(
        model,
        features
    )

    importance_df = pd.DataFrame(
        importance,
        columns=["Feature", "Importance"]
    )

    importance_df["Importance"] = (
        importance_df["Importance"] * 100
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.caption(
        "Feature importance indicates how much each feature "
        "contributed to the Random Forest's decisions."
    )


# -----------------------------
# Dataset preview
# -----------------------------
with st.expander("📁 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# -----------------------------
# Disclaimer
# -----------------------------
st.divider()

st.caption(
    "This application is an educational machine-learning prototype "
    "using synthetic demonstration data. It must not be used for "
    "medical diagnosis or treatment decisions."
)