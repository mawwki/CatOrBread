"""Download diverse bread images from Bing"""
import os, re, time, requests, urllib.parse
from pathlib import Path

BREAD_DIR = Path(__file__).resolve().parent.parent / "data" / "bread"
BREAD_DIR.mkdir(exist_ok=True)

QUERIES = [
    "bread loaf fresh",
    "white bread sliced",
    "whole wheat bread",
    "sourdough bread loaf",
    "rye bread",
    "baguette french bread",
    "ciabatta bread",
    "fresh baked bread",
    "homemade bread loaf",
    "bread bakery fresh",
]

def bing_image_urls(query, n=30):
    urls = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for first in range(0, n, 20):
        url = f"https://www.bing.com/images/async?q={urllib.parse.quote(query)}&first={first}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            for match in re.finditer(r'murl&quot;:&quot;(https?://[^&"]+)', resp.text):
                u = match.group(1).replace("\\", "")
                if any(ext in u.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    urls.add(u)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1.5)
    return list(urls)[:n]


def download(url, idx):
    ext = url.split(".")[-1].split("?")[0][:4].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    fname = f"bread_diverse_{idx:04d}.{ext}"
    fpath = BREAD_DIR / fname
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


def main():
    existing = len(list(BREAD_DIR.glob("*.*")))
    print(f"Existing bread: {existing}")

    all_urls = []
    for q in QUERIES:
        urls = bing_image_urls(q, 25)
        all_urls.extend(urls)
        print(f"  '{q}': {len(urls)} URLs")
        time.sleep(2)

    all_urls = list(set(all_urls))
    print(f"\nTotal unique URLs: {len(all_urls)}")

    count = existing
    for i, url in enumerate(all_urls):
        if download(url, count):
            count += 1
            if (count - existing) % 20 == 0:
                print(f"  Downloaded {count - existing}")

    final = len(list(BREAD_DIR.glob("*.*")))
    print(f"Final bread: {final} (new: {final - existing})")


if __name__ == "__main__":
    main()
