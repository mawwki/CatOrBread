"""Download diverse 'other' images: faces, people, nature, art, objects, etc"""
import os, sys, time, requests, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image

OTHER_DIR = Path(__file__).resolve().parent.parent / "data" / "other"
OTHER_DIR.mkdir(exist_ok=True)

EXISTING = len(list(OTHER_DIR.glob("*.*")))
print(f"Existing other: {EXISTING}")

# Wikimedia Commons API - reliable, no API key needed
WIKI_KEYWORDS = [
    "portrait", "face", "people", "person", "woman", "man", "child",
    "landscape", "mountain", "forest", "ocean", "beach", "sunset", "river",
    "dog", "bird", "horse", "fish", "butterfly", "elephant", "tiger", "bear",
    "flower", "rose", "tree", "garden", "nature", "plant", "fruit", "vegetable",
    "pizza", "cake", "salad", "pasta", "burger", "fruit", "vegetables",
    "car", "truck", "bicycle", "train", "airplane", "boat", "motorcycle",
    "building", "house", "city", "bridge", "architecture", "church", "castle",
    "computer", "phone", "technology", "robot", "camera", "laptop",
    "art", "painting", "drawing", "sculpture", "abstract", "pattern",
    "game", "sport", "football", "basketball", "tennis", "swimming",
    "space", "planet", "star", "galaxy", "moon", "rocket",
    "desert", "snow", "ice", "winter", "rain", "sky", "cloud",
    "book", "chair", "table", "clock", "watch", "shoes", "hat", "glass",
    "sunrise", "sunset", "night", "cityscape", "village", "park",
]

# Also hardcoded reliable URLs from various sources that always work
FALLBACK_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Portrait_of_a_young_woman.jpg/800px-Portrait_of_a_young_woman.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Sunset_over_the_ocean.jpg/800px-Sunset_over_the_ocean.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Mountain_landscape.jpg/800px-Mountain_landscape.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/European_wolf.jpg/800px-European_wolf.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/German_Shepherd_Dog.jpg/800px-German_Shepherd_Dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Pizza_in_Italy.jpg/800px-Pizza_in_Italy.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Red_Rose_Flower.jpg/800px-Red_Rose_Flower.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Skyscrapers_of_New_York_City.jpg/800px-Skyscrapers_of_New_York_City.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Computer_keyboard.jpg/800px-Computer_keyboard.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Fruit_stall_in_Barcelona.jpg/800px-Fruit_stall_in_Barcelona.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Korb_mit_Br%C3%B6tchen.JPG/800px-Korb_mit_Br%C3%B6tchen.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/A_sunny_day_at_the_beach.jpg/800px-A_sunny_day_at_the_beach.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Winter_landscape_with_snow-covered_trees.jpg/800px-Winter_landscape_with_snow-covered_trees.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Bicycle_in_Amsterdam.jpg/800px-Bicycle_in_Amsterdam.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Sydney_Opera_House_and_Harbour_Bridge.jpg/800px-Sydney_Opera_House_and_Harbour_Bridge.jpg",
]


def get_wikimedia_urls(keyword, limit=10):
    """Get random image URLs from Wikimedia Commons for a keyword"""
    urls = set()
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 6,
                "rnlimit": limit,
            },
            timeout=10,
        )
        if r.status_code == 200:
            titles = [p["title"] for p in r.json()["query"]["random"]]
            for title in titles:
                try:
                    ir = requests.get(
                        "https://commons.wikimedia.org/w/api.php",
                        params={
                            "action": "query",
                            "format": "json",
                            "titles": title,
                            "prop": "imageinfo",
                            "iiprop": "url",
                            "iiurlwidth": 800,
                        },
                        timeout=10,
                    )
                    if ir.status_code == 200:
                        pages = ir.json()["query"]["pages"]
                        for page in pages.values():
                            if "imageinfo" in page:
                                url = page["imageinfo"][0]["url"]
                                if url.endswith((".jpg", ".jpeg", ".png")):
                                    urls.add(url)
                except:
                    pass
    except:
        pass
    return list(urls)


def download_image(args):
    url, path = args
    if path.exists():
        return False
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; CatOrBread/1.0)"})
        if r.status_code == 200 and len(r.content) > 3000:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((224, 224))
            img.save(path, "JPEG", quality=85)
            return True
    except Exception:
        pass
    return False


def main():
    existing = len(list(OTHER_DIR.glob("*.*")))
    print(f"Existing other images: {existing}")

    # Collect URLs from Wikimedia
    all_urls = []
    for kw in WIKI_KEYWORDS:
        urls = get_wikimedia_urls(kw, limit=5)
        all_urls.extend(urls)
        print(f"  {kw}: {len(urls)} URLs")
        time.sleep(0.3)  # be polite to API

    all_urls.extend(FALLBACK_URLS)
    all_urls = list(set(all_urls))
    random.shuffle(all_urls)
    print(f"Total unique URLs: {len(all_urls)}")

    new_count = 0
    args_list = []
    for i, url in enumerate(all_urls):
        path = OTHER_DIR / f"other_more_{existing + i:04d}.jpg"
        args_list.append((url, path))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(download_image, args): args for args in args_list}
        for f in as_completed(futs):
            if f.result():
                new_count += 1

    final = len(list(OTHER_DIR.glob("*.*")))
    print(f"New downloads: {new_count}")
    print(f"Total other: {final}")


if __name__ == "__main__":
    main()
