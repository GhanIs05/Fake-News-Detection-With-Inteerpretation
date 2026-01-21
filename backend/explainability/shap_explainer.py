import os
import shap
import joblib

PROJECT_ROOT = os.getcwd()

# -------- Logistic --------
log_model = joblib.load(
    os.path.join(PROJECT_ROOT, "backend", "models", "logistic_model.pkl")
)
log_vectorizer = joblib.load(
    os.path.join(PROJECT_ROOT, "backend", "models", "tfidf_vectorizer.pkl")
)

log_explainer = shap.LinearExplainer(
    log_model,
    log_vectorizer.transform(["dummy"])
)

# -------- XGBoost --------
xgb_model = joblib.load(
    os.path.join(PROJECT_ROOT, "backend", "models", "xgboost_model.pkl")
)
xgb_vectorizer = joblib.load(
    os.path.join(PROJECT_ROOT, "backend", "models", "tfidf_vectorizer_xgb.pkl")
)

xgb_explainer = shap.TreeExplainer(xgb_model)


def explain_text(text: str, model_type: str = "logistic", top_k: int = 10):

    if model_type == "xgboost":
        vec = xgb_vectorizer.transform([text])
        shap_values = xgb_explainer.shap_values(vec)[0]
        feature_names = xgb_vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]

    else:  # logistic
        vec = log_vectorizer.transform([text])
        shap_values = log_explainer(vec).values[0]
        feature_names = log_vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]

    words = []
    for i in indices:
        words.append({
            "word": feature_names[i],
            "impact": round(float(shap_values[i]), 4)
        })

    words = sorted(words, key=lambda x: abs(x["impact"]), reverse=True)

    return words[:top_k]
