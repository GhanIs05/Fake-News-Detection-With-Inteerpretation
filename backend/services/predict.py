import os
import joblib
import numpy as np

PROJECT_ROOT = os.getcwd()

# -------- Logistic --------
LOG_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "logistic_model.pkl"
)
LOG_VEC_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "tfidf_vectorizer.pkl"
)

log_model = joblib.load(LOG_MODEL_PATH)
log_vectorizer = joblib.load(LOG_VEC_PATH)

# -------- XGBoost --------
XGB_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "xgboost_model.pkl"
)
XGB_VEC_PATH = os.path.join(
    PROJECT_ROOT, "backend", "models", "tfidf_vectorizer_xgb.pkl"
)

xgb_model = joblib.load(XGB_MODEL_PATH)
xgb_vectorizer = joblib.load(XGB_VEC_PATH)


def predict_text(text: str, model_type: str = "logistic"):
    if model_type == "xgboost":
        vec = xgb_vectorizer.transform([text])
        probs = xgb_model.predict_proba(vec)[0]

        fake_prob = float(probs[0])
        real_prob = float(probs[1])

        label = "Real" if real_prob > fake_prob else "Fake"

    else:  # logistic
        vec = log_vectorizer.transform([text])
        probs = log_model.predict_proba(vec)[0]

        fake_prob = float(probs[0])
        real_prob = float(probs[1])


        label = "Real" if real_prob > fake_prob else "Fake"

    # ✅ HONEST CONFIDENCE (difference between classes)
    confidence = abs(real_prob - fake_prob)

    return {
        "model": model_type,
        "label": label,
        "confidence": round(confidence * 100, 2),  # percentage
        "scores": {
            "fake": round(fake_prob * 100, 2),
            "real": round(real_prob * 100, 2)
        }
    }

