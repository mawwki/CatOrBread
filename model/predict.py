"""Two-stage inference: cat detector first, then bread-vs-other."""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cat_or_bread.pth"
BREAD_OTHER_PATH = Path(__file__).parent / "bread_or_other.pth"
CLASSES = ["cat", "bread", "other"]
BREAD_OTHER_CLASSES = ["bread", "other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_main_model = None
_bread_other_model = None

LABELS = {
    "cat": {"label": "Кот", "desc": "пушистый котик"},
    "bread": {"label": "Хлеб", "desc": "аппетитная выпечка"},
    "other": {"label": "Другое", "desc": "ни кот, ни хлеб"},
}


def _make_resnet(outputs):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5 if outputs == 3 else 0.45),
        nn.Linear(256, outputs),
    )
    return model


def load_model():
    global _main_model
    if _main_model is not None:
        return _main_model
    model = _make_resnet(3)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    _main_model = model
    return _main_model


def load_bread_other_model():
    global _bread_other_model
    if _bread_other_model is not None:
        return _bread_other_model
    if not BREAD_OTHER_PATH.exists():
        return None
    model = _make_resnet(2)
    model.load_state_dict(torch.load(BREAD_OTHER_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    _bread_other_model = model
    return _bread_other_model


def _prepare_image(image_bytes_or_path):
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
    return transform(img).unsqueeze(0).to(device)


def _result(pred_class, confidence, cat_prob, bread_prob, other_prob):
    info = LABELS[pred_class]
    return {
        "prediction": pred_class,
        "label": info["label"],
        "description": info["desc"],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "cat": round(cat_prob * 100, 2),
            "bread": round(bread_prob * 100, 2),
            "other": round(other_prob * 100, 2),
        },
    }


def predict(image_bytes_or_path):
    img_t = _prepare_image(image_bytes_or_path)
    main_model = load_model()

    with torch.no_grad():
        logits = main_model(img_t)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        cat_prob, main_bread_prob, main_other_prob = probs[0].item(), probs[1].item(), probs[2].item()

    # Trust cat only when the cat detector is clearly ahead.
    if cat_prob >= 0.55 and cat_prob >= max(main_bread_prob, main_other_prob) + 0.12:
        return _result("cat", cat_prob, cat_prob, main_bread_prob, main_other_prob)

    bread_other_model = load_bread_other_model()
    if bread_other_model is None:
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASSES[pred_idx]
        return _result(pred_class, max(cat_prob, main_bread_prob, main_other_prob), cat_prob, main_bread_prob, main_other_prob)

    with torch.no_grad():
        bo_logits = bread_other_model(img_t)[0]
        bo_probs = torch.nn.functional.softmax(bo_logits, dim=0)
        bread_prob, other_prob = bo_probs[0].item(), bo_probs[1].item()

    # Combine: cat from stage 1, bread/other from stage 2.
    non_cat_mass = max(0.0, 1.0 - cat_prob)
    combined_bread = bread_prob * non_cat_mass
    combined_other = other_prob * non_cat_mass

    if bread_prob >= 0.68:
        return _result("bread", bread_prob, cat_prob, combined_bread, combined_other)
    return _result("other", other_prob, cat_prob, combined_bread, combined_other)


if __name__ == "__main__":
    print("Main model loaded:", bool(load_model()))
    print("Bread/other model loaded:", bool(load_bread_other_model()))
