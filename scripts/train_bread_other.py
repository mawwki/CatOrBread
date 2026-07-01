"""Train a dedicated bread-vs-other classifier."""
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "bread_or_other.pth"
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = ["bread", "other"]
CLASS_LABELS = {"bread": 0, "other": 1}


class TelegramCompression:
    def __call__(self, img):
        if random.random() < 0.8:
            w, h = img.size
            scale = random.choice([0.4, 0.55, 0.7, 0.85])
            small = (max(64, int(w * scale)), max(64, int(h * scale)))
            img = img.resize(small, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
        if random.random() < 0.9:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(30, 88))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if random.random() < 0.25:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.1)))
        return img


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_data():
    data = []
    for cls_name, label in CLASS_LABELS.items():
        paths = list((DATA_DIR / cls_name).glob("*.*"))
        for path in paths:
            data.append((str(path), label))
        print(f"{cls_name}: {len(paths)} images")
    random.shuffle(data)
    return data


def make_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(256, 2),
    )
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_data = load_data()
    split = int(len(all_data) * 0.82)
    train_data = all_data[:split]
    val_data = all_data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_transform = transforms.Compose([
        TelegramCompression(),
        transforms.RandomResizedCrop(224, scale=(0.68, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageDataset(train_data, train_transform)
    val_ds = ImageDataset(val_data, val_transform)

    labels = [label for _, label in train_data]
    counts = [labels.count(0), labels.count(1)]
    print(f"Train class counts: {counts}")
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, min(len(train_data), 900), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00025, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    best_acc = 0.0
    for epoch in range(14):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        scheduler.step()

        model.eval()
        val_total = 0
        val_correct = 0
        per_class = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}}
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
                for i in range(len(labels)):
                    label = labels[i].item()
                    per_class[label]["total"] += 1
                    if pred[i].item() == label:
                        per_class[label]["correct"] += 1
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        bread_acc = 100 * per_class[0]["correct"] / max(per_class[0]["total"], 1)
        other_acc = 100 * per_class[1]["correct"] / max(per_class[1]["total"], 1)
        print(f"Epoch {epoch + 1}: Loss={running_loss/len(train_loader):.4f} Train={train_acc:.1f}% Val={val_acc:.1f}% [bread={bread_acc:.0f}% other={other_acc:.0f}%]")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved ({val_acc:.1f}%)")
    print(f"Best: {best_acc:.1f}%")


if __name__ == "__main__":
    train()
