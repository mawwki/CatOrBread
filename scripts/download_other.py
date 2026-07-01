"""Download 'other' class images from picsum.photos and balance classes"""
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OTHER_DIR = DATA_DIR / "other"
OTHER_DIR.mkdir(exist_ok=True)

TARGET_OTHER = 500

def download_other(idx):
    url = f"https://picsum.photos/224/224?random={idx}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 5000:
            fpath = OTHER_DIR / f"other_{idx:04d}.jpg"
            with open(fpath, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False

def main():
    existing = len(list(OTHER_DIR.glob("*.*")))
    need = TARGET_OTHER - existing
    if need <= 0:
        print(f"Already have {existing} other images")
        return

    print(f"Downloading {need} other images from picsum.photos...")
    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_other, i): i for i in range(existing, existing + need + 20)}
        for f in as_completed(futures):
            if f.result():
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  {downloaded}/{need}")

    total = len(list(OTHER_DIR.glob("*.*")))
    print(f"Other images: {total}")


if __name__ == "__main__":
    main()
