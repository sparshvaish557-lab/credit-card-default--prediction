import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Default Prediction", page_icon="💳", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("credit_default_model.joblib")

st.title("💳 Credit Card Default Prediction")
st.write("Enter customer information to estimate the risk of credit card payment default.")

try:
    bundle = load_bundle()
except Exception:
    st.error("The trained model file could not be loaded. Make sure credit_default_model.joblib is uploaded in the same GitHub repository as app.py.")
    st.stop()

model = bundle["model"]
model_name = bundle.get("model_name", "Machine learning model")
features = bundle["features"]
medians = bundle.get("medians", {})

st.caption(f"Model used: {model_name}")

with st.form("prediction_form"):
    values = {}
    for feature in features:
        default_value = medians.get(feature, 0)
        try:
            default_value = float(default_value)
        except (TypeError, ValueError):
            default_value = 0.0
        values[feature] = st.number_input(
            feature.replace("_", " "),
            value=default_value,
            step=1.0,
            format="%.2f",
        )
    submitted = st.form_submit_button("Predict default risk", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
        elif len(probabilities) > 1:
            probability = float(probabilities[-1])

    if prediction == 1:
        st.warning("Prediction: Customer may default on payment.")
    else:
        st.success("Prediction: Customer is not predicted to default on payment.")

    if probability is not None:
        st.metric("Estimated probability of default", f"{probability:.1%}")

st.divider()
st.caption("Educational project only. This prediction should not be used as the sole basis for real lending or financial decisions.")
