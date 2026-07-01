"""Train a cat vs bread classifier using ResNet18"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random
import shutil

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "cat_or_bread.pth"
os.makedirs(MODEL_DIR, exist_ok=True)

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

def prepare_balanced_data():
    cat_dir = DATA_DIR / "cat"
    bread_dir = DATA_DIR / "bread"
    balanced_dir = DATA_DIR / "balanced"
    balanced_cat = balanced_dir / "cat"
    balanced_bread = balanced_dir / "bread"

    if balanced_dir.exists():
        shutil.rmtree(balanced_dir)

    balanced_cat.mkdir(parents=True)
    balanced_bread.mkdir(parents=True)

    cat_images = list(cat_dir.glob("*.jpg"))
    bread_images = list(bread_dir.glob("*.jpg"))

    random.shuffle(cat_images)
    n = min(len(cat_images), len(bread_images) * 3)
    for img in cat_images[:n]:
        shutil.copy2(img, balanced_cat / img.name)
    for img in bread_images:
        shutil.copy2(img, balanced_bread / img.name)

    return balanced_cat, balanced_bread

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cat_dir, bread_dir = prepare_balanced_data()

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

    cat_data = [(str(p), 0) for p in cat_dir.glob("*.jpg")]
    bread_data = [(str(p), 1) for p in bread_dir.glob("*.jpg")]
    random.shuffle(cat_data)
    random.shuffle(bread_data)

    cat_split = int(len(cat_data) * 0.8)
    bread_split = int(len(bread_data) * 0.8)

    train_data = cat_data[:cat_split] + bread_data[:bread_split]
    val_data = cat_data[cat_split:] + bread_data[bread_split:]
    random.shuffle(train_data)
    random.shuffle(val_data)

    train_ds = ImageFolderDataset(train_data, train_transform)
    val_ds = ImageFolderDataset(val_data, val_transform)

    print(f"Train: {len(train_ds)} (cat={cat_split}, bread={bread_split})")
    print(f"Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    model = model.to(device)

    class_weight = torch.tensor([1.0, 3.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0.0
    for epoch in range(15):
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
        cc = [0, 0]
        ct = [0, 0]
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                for i in range(len(labels)):
                    ct[labels[i]] += 1
                    if predicted[i] == labels[i]:
                        cc[labels[i]] += 1
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f} "
              f"Train={train_acc:.1f}% Val={val_acc:.1f}% "
              f"[Cat={100*cc[0]/max(ct[0],1):.0f}% Bread={100*cc[1]/max(ct[1],1):.0f}%]")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved ({val_acc:.1f}%)")
    print(f"Best: {best_acc:.1f}%")

if __name__ == "__main__":
    cat_count = len(list((DATA_DIR / "cat").glob("*.jpg")))
    bread_count = len(list((DATA_DIR / "bread").glob("*.jpg")))
    print(f"Found {cat_count} cat, {bread_count} bread images")
    if cat_count < 10 or bread_count < 10:
        print("Not enough images.")
        sys.exit(1)
    train()
