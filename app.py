import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(
    page_title="Credit Default Risk Analyzer",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {padding: 1.5rem; border-radius: 16px; background: linear-gradient(120deg, #102a43, #2563a6); color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin-bottom: .4rem;}
.hero p {color: #e8f1fb; margin-bottom: 0;}
.section {padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25); border-radius: 14px; margin-bottom: 1rem;}
.small-note {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load("credit_default_model.joblib")

def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

st.markdown("""
<div class="hero">
  <h1>💳 Credit Default Risk Analyzer</h1>
  <p>Estimate the likelihood of a credit card payment default using a trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

try:
    bundle = load_bundle()
    model = bundle["model"]
    model_name = bundle.get("model_name", "Trained machine-learning model")
    features = bundle["features"]
    medians = bundle.get("medians", {})
except Exception as exc:
    st.error("The model could not be loaded. Upload credit_default_model.joblib to the same GitHub repository as app.py.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

st.caption(f"Model: **{model_name}**  ·  Complete the fields below, then select **Predict default risk**.")
st.info("Use the customer's available information. This is an educational project, not a real lending decision system.")

# Categorical values are displayed as readable labels but sent to the model as dataset codes.
categorical_options = {
    "sex": {
        "Male": 1,
        "Female": 2,
    },
    "education": {
        "Unknown / not specified (0)": 0,
        "Graduate school (1)": 1,
        "University (2)": 2,
        "High school (3)": 3,
        "Others (4)": 4,
        "Unknown (5)": 5,
        "Unknown (6)": 6,
    },
    "marriage": {
        "Unknown / not specified (0)": 0,
        "Married (1)": 1,
        "Single (2)": 2,
        "Others (3)": 3,
    },
}
pay_status_options = {
    "No consumption (-2)": -2,
    "Paid in full (-1)": -1,
    "Use of revolving credit (0)": 0,
    "Payment delayed 1 month (1)": 1,
    "Payment delayed 2 months (2)": 2,
    "Payment delayed 3 months (3)": 3,
    "Payment delayed 4 months (4)": 4,
    "Payment delayed 5 months (5)": 5,
    "Payment delayed 6 months (6)": 6,
    "Payment delayed 7 months (7)": 7,
    "Payment delayed 8 months or more (8)": 8,
}

def nice_label(feature):
    aliases = {
        "limitbal": "Credit limit",
        "age": "Age (years)",
        "sex": "Gender",
        "education": "Education level",
        "marriage": "Marital status",
        "pay0": "Most recent repayment status",
        "pay2": "Repayment status (2 months ago)",
        "pay3": "Repayment status (3 months ago)",
        "pay4": "Repayment status (4 months ago)",
        "pay5": "Repayment status (5 months ago)",
        "pay6": "Repayment status (6 months ago)",
    }
    key = norm(feature)
    if key in aliases:
        return aliases[key]
    label = str(feature).replace("_", " ").strip()
    return label.replace("Amt", "Amount").replace("amt", "amount").title()

def default_number(feature):
    try:
        return float(medians.get(feature, 0))
    except (TypeError, ValueError):
        return 0.0

input_values = {}
with st.form("prediction_form"):
    # Basic details
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("👤 Customer Information")
    basic_cols = st.columns(3)
    for feature in features:
        key = norm(feature)
        if key in ("sex", "education", "marriage", "age", "limitbal"):
            col = basic_cols[0] if key in ("sex", "age") else basic_cols[1] if key in ("education", "marriage") else basic_cols[2]
            with col:
                if key in categorical_options:
                    options = categorical_options[key]
                    selected_label = st.selectbox(nice_label(feature), list(options.keys()), key=f"cat_{key}")
                    input_values[feature] = options[selected_label]
                else:
                    minimum = 18.0 if key == "age" else 0.0
                    maximum = 120.0 if key == "age" else 100000000.0
                    step = 1.0 if key == "age" else 1000.0
                    input_values[feature] = st.number_input(
                        nice_label(feature),
                        min_value=minimum,
                        max_value=maximum,
                        value=max(minimum, min(maximum, default_number(feature))),
                        step=step,
                        key=f"num_{key}",
                    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Payment history
    pay_features = [f for f in features if norm(f).startswith("pay") and norm(f)[3:].isdigit()]
    if pay_features:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("📅 Repayment History")
        st.caption("Choose the recorded repayment status for each period.")
        cols = st.columns(3)
        for i, feature in enumerate(pay_features):
            with cols[i % 3]:
                selected = st.selectbox(
                    nice_label(feature),
                    list(pay_status_options.keys()),
                    index=next((j for j, v in enumerate(pay_status_options.values()) if v == int(default_number(feature))), 2),
                    key=f"pay_{norm(feature)}",
                )
                input_values[feature] = pay_status_options[selected]
        st.markdown('</div>', unsafe_allow_html=True)

    # Remaining numeric financial features
    remaining = [f for f in features if f not in input_values]
    if remaining:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("💰 Billing and Payment Details")
        st.caption("Enter the amounts shown on the customer's statements and payment records.")
        cols = st.columns(3)
        for i, feature in enumerate(remaining):
            with cols[i % 3]:
                value = default_number(feature)
                input_values[feature] = st.number_input(
                    nice_label(feature),
                    value=value,
                    step=100.0,
                    format="%.2f",
                    key=f"amount_{norm(feature)}",
                )
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict default risk", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[input_values[f] for f in features]], columns=features)
    try:
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                probability = float(probs[classes.index(1)])
            elif len(probs) > 1:
                probability = float(probs[-1])

        st.divider()
        st.subheader("📊 Prediction Result")
        left, right = st.columns([1, 2])
        with left:
            if prediction == 1:
                st.error("Higher predicted default risk")
                st.markdown("The model predicts **possible payment default**.")
            else:
                st.success("Lower predicted default risk")
                st.markdown("The model predicts **no payment default**.")
        with right:
            if probability is not None:
                st.metric("Estimated probability of default", f"{probability:.1%}")
                st.progress(max(0.0, min(1.0, probability)))
                st.caption("This is a model estimate, not a guarantee of future behavior.")
            else:
                st.write("The model returned a class prediction but does not provide probability estimates.")
        with st.expander("Review the values sent to the model"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error("Prediction failed. Please check that the input features match the model used during training.")
        st.caption(f"Technical detail: {exc}")

st.divider()
st.caption("Educational use only. Do not use this application as the sole basis for real lending or financial decisions.")
