"""Fine-tune existing 3-class model with more epochs"""
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "cat_or_bread.pth"

CLASSES = ["cat", "bread", "other"]
CLASS_LABELS = {"cat": 0, "bread": 1, "other": 2}

class ImageFolderDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data():
    all_data = []
    for cls_name, cls_idx in CLASS_LABELS.items():
        cls_dir = DATA_DIR / cls_name
        paths = list(cls_dir.glob("*.*"))
        for p in paths:
            all_data.append((str(p), cls_idx))
        print(f"{cls_name}: {len(paths)} images")
    return all_data

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_data = load_data()
    random.shuffle(all_data)

    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolderDataset(train_data, train_transform)
    val_ds = ImageFolderDataset(val_data, val_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3),
    )
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded existing model")
    model = model.to(device)

    counts = [0, 0, 0]
    for _, lbl in all_data:
        counts[lbl] += 1
    total = sum(counts)
    weights = [total / c for c in counts]
    class_weight = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc = 0.0
    # First pass: find current best accuracy from file
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        print("Reloaded best model for evaluation")
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        best_acc = 100 * val_correct / val_total
        print(f"Current best val acc: {best_acc:.1f}%")
        model.train()
    except Exception as e:
        print(f"Could not evaluate saved model: {e}")

    for epoch in range(30):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        per_class = {c: {"correct": 0, "total": 0} for c in range(3)}
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                for i in range(len(labels)):
                    lbl = labels[i].item()
                    per_class[lbl]["total"] += 1
                    if predicted[i].item() == lbl:
                        per_class[lbl]["correct"] += 1
        val_acc = 100 * val_correct / val_total
        cls_report = " ".join(
            f"{CLASSES[c]}={100*per_class[c]['correct']/max(per_class[c]['total'],1):.0f}%"
            for c in range(3)
        )
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f} "
              f"Train={train_acc:.1f}% Val={val_acc:.1f}% [{cls_report}]")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved ({val_acc:.1f}%)")
    print(f"Best: {best_acc:.1f}%")

if __name__ == "__main__":
    train()
