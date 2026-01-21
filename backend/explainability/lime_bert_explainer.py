import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

# -------------------------
# Load BERT
# -------------------------
PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "bert")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class_names = ["Fake", "Real"]

# -------------------------
# Prediction function for LIME
# -------------------------
# def bert_predict_proba(texts):
#     probs = []

#     for text in texts:
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=128
#         )

#         with torch.no_grad():
#             outputs = model(**inputs)
#             softmax = torch.softmax(outputs.logits, dim=1)[0]

#         probs.append(softmax.numpy())

#     return np.array(probs)

def bert_predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


# -------------------------
# LIME Explainer
# -------------------------
explainer = LimeTextExplainer(class_names=class_names)

def explain_bert(text, num_features=8):
    # explanation = explainer.explain_instance(
    #     text_instance=text,
    #     classifier_fn=bert_predict_proba,
    #     num_features=num_features
    # )
    explanation = explainer.explain_instance(
    text_instance=text,
    classifier_fn=bert_predict_proba,
    num_features=6,
    num_samples=500   # 🔥 KEY FIX
)


    words = []
    for word, weight in explanation.as_list():
        words.append({
            "word": word,
            "impact": round(float(weight), 4)
        })

    return words
