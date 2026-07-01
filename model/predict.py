"""Inference module for cat vs bread classifier"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cat_or_bread.pth"
CLASSES = ["cat", "bread"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    _model = model
    return _model

def predict(image_bytes_or_path):
    model = load_model()

    if isinstance(image_bytes_or_path, (str, Path)):
        img = Image.open(image_bytes_or_path).convert("RGB")
    else:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes_or_path)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)[0]
        cat_logit, bread_logit = logits[0].item(), logits[1].item()
        logit_diff = abs(cat_logit - bread_logit)
        max_logit = max(cat_logit, bread_logit)
        probs = torch.nn.functional.softmax(logits, dim=0)
        cat_prob = probs[0].item()
        bread_prob = probs[1].item()
        max_prob = max(cat_prob, bread_prob)
        predicted = 0 if cat_prob > bread_prob else 1

    if max_logit < 0.5 or logit_diff < 0.5:
        return {
            "prediction": "other",
            "label": "Другое",
            "description": "ни кот, ни хлеб",
            "confidence": round(max_prob * 100, 2),
            "probabilities": {
                "cat": round(cat_prob * 100, 2),
                "bread": round(bread_prob * 100, 2),
            },
        }

    return {
        "prediction": CLASSES[predicted],
        "label": "Кот" if predicted == 0 else "Хлеб",
        "description": "пушистый котик" if predicted == 0 else "аппетитная выпечка",
        "confidence": round(max_prob * 100, 2),
        "probabilities": {
            "cat": round(cat_prob * 100, 2),
            "bread": round(bread_prob * 100, 2),
        },
    }

if __name__ == "__main__":
    model = load_model()
    print("Model loaded")
    test_path = list((Path(__file__).parent.parent / "data" / "cat").glob("*.jpg"))
    if test_path:
        result = predict(str(test_path[0]))
        print(result)
