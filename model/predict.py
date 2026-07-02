"""Dynamic inference: checks model output size and maps accordingly."""
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cat_or_bread.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_main_model = None
_num_classes = None

LABELS = {
    "cat": {"label": "Кот", "desc": ""},
    "bread": {"label": "Хлеб", "desc": ""},
    "other": {"label": "Другое", "desc": ""},
}


def _make_resnet(outputs):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, outputs),
    )
    return model


def load_model():
    global _main_model, _num_classes
    if _main_model is not None:
        return _main_model
    if not MODEL_PATH.exists():
        return None
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    # Detect number of classes from final layer weight shape
    fc_weight = None
    for k, v in state.items():
        if "fc.2.weight" in k:
            fc_weight = v
            break
        if "fc.3.weight" in k:
            fc_weight = v
            break
    _num_classes = fc_weight.shape[0] if fc_weight is not None else 3

    model = _make_resnet(_num_classes)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    _main_model = model
    return _main_model


def get_class_map():
    n = _num_classes or 3
    if n == 2:
        return ["cat", "other"]
    return ["cat", "bread", "other"]


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


def predict(image_bytes_or_path):
    img_t = _prepare_image(image_bytes_or_path)
    main_model = load_model()
    if main_model is None:
        return {"error": "Model not found"}

    class_map = get_class_map()

    with torch.no_grad():
        logits = main_model(img_t)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    # Build probability dict
    probs_dict = {}
    for i, cls in enumerate(class_map):
        probs_dict[cls] = round(probs[i].item() * 100, 2)

    # Decision: cat >= 50% -> cat, bread >= 50% -> bread, else other
    if "cat" in probs_dict and probs_dict["cat"] >= 50:
        pred = "cat"
    elif "bread" in probs_dict and probs_dict["bread"] >= 50:
        pred = "bread"
    else:
        pred = "other"

    # Ensure all keys exist for bot display
    for key in ["cat", "bread", "other"]:
        if key not in probs_dict:
            probs_dict[key] = 0.0

    confidence = probs_dict[pred]
    info = LABELS[pred]
    return {
        "prediction": pred,
        "label": info["label"],
        "description": info["desc"],
        "confidence": confidence,
        "probabilities": probs_dict,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = predict(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Model loaded:", bool(load_model()))
        print("Classes:", get_class_map())
        print("Usage: python3 model/predict.py <image_path>")
