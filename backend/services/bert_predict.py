import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------
# Load model & tokenizer
# -------------------------
PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "bert")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_bert(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    label = "Real" if real_prob > fake_prob else "Fake"
    confidence = abs(real_prob - fake_prob)

    return {
        "model": "bert",
        "label": label,
        "confidence": round(confidence * 100, 2),
        "scores": {
            "fake": round(fake_prob * 100, 2),
            "real": round(real_prob * 100, 2)
        }
    }
