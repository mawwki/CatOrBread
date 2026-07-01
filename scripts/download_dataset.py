#!/usr/bin/env python3
"""Bulk download images from Google Images using raw requests"""
import os, time, re, json, urllib.parse
import requests
from pathlib import Path
import concurrent.futures

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = 700

CAT_QUERIES = ["cat face", "kitten", "tabby cat", "black cat", "white cat", "ginger cat", "cat portrait", "cat animal"]
BREAD_QUERIES = ["bread loaf", "fresh bread", "baguette", "croissant", "sourdough", "ciabatta", "rye bread", "bread bakery"]


def google_image_urls(query, max_results=100):
    urls = set()
    for start in range(0, max_results, 100):
        params = {
            "q": query,
            "tbm": "isch",
            "ijn": str(start // 100),
            "start": str(start),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        url = f"https://www.google.com/search?{urllib.parse.urlencode(params)}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            # Find image URLs in the response
            for match in re.finditer(r'"(https?://[^"]+?\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"', resp.text):
                u = match.group(1)
                if any(domain in u for domain in ['google.com', 'gstatic.com']):
                    continue
                urls.add(u)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(3)
    return list(urls)[:max_results]


def download_image(url, dest_dir, idx):
    ext = url.split(".")[-1].split("?")[0][:4].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    fname = f"img_{idx:05d}.{ext}"
    fpath = dest_dir / fname
    if fpath.exists():
        return False
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and len(resp.content) > 5000:
            with open(fpath, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def download_class(class_name, queries):
    dest = DATA_DIR / class_name
    dest.mkdir(exist_ok=True)
    existing = len(list(dest.glob("*.*")))
    if existing >= TARGET:
        print(f"[{class_name}] Already have {existing}, skipping")
        return

    all_urls = []
    for q in queries:
        if len(all_urls) >= TARGET:
            break
        print(f"[{class_name}] Searching '{q}'...")
        urls = google_image_urls(q, max_results=80)
        all_urls.extend(urls)
        print(f"  Found {len(urls)} URLs")
        time.sleep(2)

    all_urls = list(set(all_urls))[:TARGET]
    print(f"[{class_name}] Got {len(all_urls)} unique URLs, downloading...")

    downloaded = 0
    for i, url in enumerate(all_urls):
        if download_image(url, dest, i + existing):
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  {downloaded}/{len(all_urls)}")

    total = len(list(dest.glob("*.*")))
    print(f"[{class_name}] Downloaded {downloaded} new, total: {total}")


def main():
    # Clean old
    for d in [DATA_DIR / "cat", DATA_DIR / "bread"]:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    print("Starting download...")
    download_class("cat", CAT_QUERIES)
    download_class("bread", BREAD_QUERIES)

    final_cat = len(list(DATA_DIR.glob("cat/*.*")))
    final_bread = len(list(DATA_DIR.glob("bread/*.*")))
    print(f"\nFinal: cat={final_cat}, bread={final_bread}")


if __name__ == "__main__":
    main()
