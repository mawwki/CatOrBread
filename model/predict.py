"""Inference module for cat / bread / other classifier"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cat_or_bread.pth"
CLASSES = ["cat", "bread", "other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

LABELS = {
    "cat": {"label": "Кот", "desc": "пушистый котик"},
    "bread": {"label": "Хлеб", "desc": "аппетитная выпечка"},
    "other": {"label": "Другое", "desc": "ни кот, ни хлеб"},
}

def load_model():
    global _model
    if _model is not None:
        return _model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3),
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
        probs = torch.nn.functional.softmax(logits, dim=0)
        cat_prob, bread_prob, other_prob = probs[0].item(), probs[1].item(), probs[2].item()
        predicted = torch.argmax(probs).item()
        pred_class = CLASSES[predicted]
        confidence = round(max(probs).item() * 100, 2)

    info = LABELS[pred_class]
    return {
        "prediction": pred_class,
        "label": info["label"],
        "description": info["desc"],
        "confidence": confidence,
        "probabilities": {
            "cat": round(cat_prob * 100, 2),
            "bread": round(bread_prob * 100, 2),
            "other": round(other_prob * 100, 2),
        },
    }

if __name__ == "__main__":
    model = load_model()
    print("Model loaded")
    test_path = list((Path(__file__).parent.parent / "data" / "cat").glob("*.jpg"))
    if test_path:
        result = predict(str(test_path[0]))
        print(result)
