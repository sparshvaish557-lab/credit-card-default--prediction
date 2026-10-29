# Credit Card Default Prediction

A Streamlit web app that uses a trained machine-learning model to estimate credit card payment default risk.

## Files needed

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `credit_default_model.joblib` — exported model bundle from Google Colab

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload all three required files to a GitHub repository, then create an app on Streamlit Community Cloud using `app.py` as the main file.

The model bundle must contain `model`, `model_name`, `features`, and `medians` keys.

Educational use only; not for real lending decisions.
