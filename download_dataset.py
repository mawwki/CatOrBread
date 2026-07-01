"""Download cat and bread images for training"""
import os
import sys
import json
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import concurrent.futures

DATA_DIR = Path(__file__).parent / "data"
CAT_DIR = DATA_DIR / "cat"
BREAD_DIR = DATA_DIR / "bread"

os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(BREAD_DIR, exist_ok=True)

def download_image(url, path, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((224, 224))
        img.save(path, "JPEG", quality=85)
        return True
    except Exception as e:
        return False

def download_cats():
    existing = len(list(CAT_DIR.glob("*.jpg")))
    if existing >= 300:
        print(f"Already have {existing} cat images, skipping")
        return
    print("Downloading cat images...")
    urls = set()
    for _ in range(30):
        try:
            r = requests.get(
                "https://api.thecatapi.com/v1/images/search?limit=10",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if r.status_code == 200:
                for item in r.json():
                    urls.add(item["url"])
        except:
            pass
    urls = list(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for i, url in enumerate(urls):
            path = CAT_DIR / f"cat_{i:04d}.jpg"
            if not path.exists():
                futs.append(ex.submit(download_image, url, path))
        concurrent.futures.wait(futs)
    count = len(list(CAT_DIR.glob("*.jpg")))
    print(f"Downloaded {count} cat images")

def download_bread():
    existing = len(list(BREAD_DIR.glob("*.jpg")))
    if existing >= 300:
        print(f"Already have {existing} bread images, skipping")
        return
    print("Downloading bread images...")
    bread_keywords = ["bread loaf", "sourdough bread", "white bread", "whole wheat bread",
                      "baguette bread", "croissant bread", "bread slice"]
    urls = set()
    for kw in bread_keywords:
        for _ in range(5):
            try:
                r = requests.get(
                    f"https://api.unsplash.com/photos/random?query={kw}&count=10&client_id=demo",
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0",
                             "Accept-Version": "v1"}
                )
                if r.status_code == 200:
                    for item in r.json():
                        urls.add(item["urls"]["regular"])
            except:
                pass
    if not urls:
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Korb_mit_Br%C3%B6tchen.JPG/800px-Korb_mit_Br%C3%B6tchen.JPG")
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Bread_rolls.jpg/800px-Bread_rolls.jpg")
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Sourdough_bread_(2).jpg/800px-Sourdough_bread_(2).jpg")
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Fresh_baguette_bread.jpg/800px-Fresh_baguette_bread.jpg")
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/White_bread_loaf.jpg/800px-White_bread_loaf.jpg")
        urls.add("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Croissant_bread.jpg/800px-Croissant_bread.jpg")
    urls = list(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for i, url in enumerate(urls):
            path = BREAD_DIR / f"bread_{i:04d}.jpg"
            if not path.exists():
                futs.append(ex.submit(download_image, url, path))
        concurrent.futures.wait(futs)
    count = len(list(BREAD_DIR.glob("*.jpg")))
    print(f"Downloaded {count} bread images")

def download_from_oxford():
    try:
        from torchvision import datasets
        print("Trying Oxford Pets dataset for cats...")
        oxford = datasets.OxfordIIITPet(
            root=str(DATA_DIR / "oxford_pets"),
            split="trainval",
            download=True
        )
        cat_classes = [i for i, name in enumerate(oxford.classes)
                      if name.split("_")[0].lower() in
                      ["abyssinian", "bengal", "birman", "bombay",
                       "british", "chinese", "egyptian", "maine",
                       "persian", "ragdoll", "russian", "siamese",
                       "sphynx"]]
        import torch
        for idx in range(len(oxford)):
            img, label = oxford[idx]
            if label in cat_classes:
                path = CAT_DIR / f"oxford_cat_{idx:04d}.jpg"
                if not path.exists():
                    if isinstance(img, Image.Image):
                        img = img.resize((224, 224))
                        img.save(path, "JPEG", quality=85)
                    elif isinstance(img, torch.Tensor):
                        from torchvision.transforms import ToPILImage
                        ToPILImage()(img).resize((224, 224)).save(path, "JPEG", quality=85)
        print(f"Oxford cats done. Total cats: {len(list(CAT_DIR.glob('*.jpg')))}")
    except Exception as e:
        print(f"Oxford download skipped: {e}")

if __name__ == "__main__":
    download_cats()
    download_bread()
    download_from_oxford()
    print(f"Final counts - Cats: {len(list(CAT_DIR.glob('*.jpg')))} Bread: {len(list(BREAD_DIR.glob('*.jpg')))}")
