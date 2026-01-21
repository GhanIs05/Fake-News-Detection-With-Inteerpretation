# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models

# from PIL import Image
# from io import BytesIO
# import numpy as np
# import base64

# from backend.explainability.occlusion import OcclusionExplainer
# from backend.services.bert_predict import predict_bert
# from backend.explainability.lime_bert_explainer import explain_bert
# from backend.services.predict import predict_text
# from backend.explainability.shap_explainer import explain_text

# # -----------------------------
# # IMAGE MODEL SETUP
# # -----------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 2)

# model.load_state_dict(torch.load("backend/models/image_cnn.pth", map_location=DEVICE))
# model = model.to(DEVICE)
# model.eval()

# print("✅ Image CNN loaded")

# # -----------------------------
# # FASTAPI
# # -----------------------------
# app = FastAPI(title="Fake News Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class TextInput(BaseModel):
#     text: str


# # -----------------------------
# # IMAGE PREDICTION + EXPLAINABILITY
# # -----------------------------
# @app.post("/predict/image")
# async def predict_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     img = Image.open(BytesIO(contents)).convert("RGB")

#     img_tensor = transform(img).unsqueeze(0).to(DEVICE)

#     # Prediction
#     with torch.no_grad():
#         output = model(img_tensor)
#         probs = torch.softmax(output, dim=1)[0]
#         pred = torch.argmax(probs).item()

#     label = "fake" if pred == 1 else "real"
#     confidence = round(float(probs[pred]) * 100, 2)

#     # Occlusion explainability
#     explainer = OcclusionExplainer(model)
#     heatmap = explainer.generate(img_tensor, pred)

#     heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
#     heatmap_img = heatmap_img.resize(img.size)
#     heatmap_img = heatmap_img.convert("RGB")

#     blended = Image.blend(img, heatmap_img, alpha=0.5)

#     buf = BytesIO()
#     blended.save(buf, format="PNG")
#     heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

#     return {
#         "label": label,
#         "confidence": confidence,
#         "heatmap": heatmap_base64
#     }


# # -----------------------------
# # TEXT PREDICTION
# # -----------------------------
# @app.post("/predict/text")
# def predict(input: TextInput, model: str = "logistic"):
#     if model == "bert":
#         return predict_bert(input.text)
#     return predict_text(input.text, model)


# @app.post("/explain/text")
# def explain(input: TextInput, model: str = "logistic"):
#     if model == "bert":
#         return {
#             "model": "bert",
#             "important_words": explain_bert(input.text)
#         }

#     return {
#         "model": model,
#         "important_words": explain_text(input.text, model)
#     }
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image
from io import BytesIO
import numpy as np
import base64

from backend.explainability.occlusion import OcclusionExplainer
from backend.services.bert_predict import predict_bert
from backend.explainability.lime_bert_explainer import explain_bert
from backend.services.predict import predict_text
from backend.explainability.shap_explainer import explain_text

# -----------------------------
# IMAGE MODEL SETUP
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

# ✅ IMPORTANT: ENV-SAFE MODEL PATH
MODEL_PATH = os.getenv(
    "IMAGE_MODEL_PATH",
    "backend/models/image_cnn.pth"   # local fallback
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Image CNN loaded from:", MODEL_PATH)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Fake News Detection API")

# ✅ PRODUCTION-SAFE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # frontend URL can change after deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# -----------------------------
# IMAGE PREDICTION + EXPLAINABILITY
# -----------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # ---- Prediction (fast & safe) ----
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "fake" if pred == 1 else "real"
    confidence = round(float(probs[pred]) * 100, 2)

    # ---- Occlusion Explainability (stable) ----
    explainer = OcclusionExplainer(model)
    heatmap = explainer.generate(img_tensor, pred)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(img.size)
    heatmap_img = heatmap_img.convert("RGB")

    blended = Image.blend(img, heatmap_img, alpha=0.5)

    buf = BytesIO()
    blended.save(buf, format="PNG")
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "label": label,
        "confidence": confidence,
        "heatmap": heatmap_base64
    }

# -----------------------------
# TEXT PREDICTION
# -----------------------------
@app.post("/predict/text")
def predict(input: TextInput, model: str = "logistic"):
    if model == "bert":
        return predict_bert(input.text)
    return predict_text(input.text, model)

@app.post("/explain/text")
def explain(input: TextInput, model: str = "logistic"):
    if model == "bert":
        return {
            "model": "bert",
            "important_words": explain_bert(input.text)
        }

    return {
        "model": model,
        "important_words": explain_text(input.text, model)
    }
