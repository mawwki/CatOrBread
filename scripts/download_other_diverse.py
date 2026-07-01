"""Download diverse 'other' images (non-cat, non-bread) from Pixabay"""
import os, time, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OTHER_DIR = Path(__file__).resolve().parent.parent / "data" / "other"
OTHER_DIR.mkdir(exist_ok=True)

# Pixabay direct image URLs by category
CATEGORIES = {
    "dog": [
        "https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/09/25/13/12/dog-2785074_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg",
        "https://cdn.pixabay.com/photo/2019/08/19/07/45/pug-4414790_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/11/17/13/13/dog-1047519_1280.jpg",
    ],
    "person": [
        "https://cdn.pixabay.com/photo/2015/07/09/00/29/woman-837156_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/03/36/woman-1867093_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/08/01/01/33/beanie-2562646_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/04/27/16/23/people-3354417_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/01/08/18/29/entrepreneur-593358_1280.jpg",
    ],
    "car": [
        "https://cdn.pixabay.com/photo/2012/05/29/19/43/car-49278_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/03/27/18/51/car-1283569_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/03/27/14/39/auto-2179220_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/05/18/19/13/auto-347332_1280.jpg",
    ],
    "building": [
        "https://cdn.pixabay.com/photo/2016/11/22/19/15/building-1850129_1280.jpg",
        "https://cdn.pixabay.com/photo/2013/04/11/19/46/building-102840_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/03/28/12/10/architecture-2181915_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/11/06/13/27/building-1028400_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/03/53/architecture-1867187_1280.jpg",
    ],
    "nature": [
        "https://cdn.pixabay.com/photo/2015/12/01/20/28/road-1072823_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/06/19/21/24/avenue-815297_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/14/04/45/elephant-1822636_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832_1280.jpg",
    ],
    "object": [
        "https://cdn.pixabay.com/photo/2016/11/18/17/09/chair-1835905_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/09/17/20/26/book-450111_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/03/31/19/56/avatar-1295429_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/08/15/11/29/ball-418538_1280.jpg",
    ],
    "food_other": [
        "https://cdn.pixabay.com/photo/2015/12/09/17/11/vegetables-1085063_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/06/02/14/31/pizza-1431079_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/04/08/13/13/food-712665_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/06/23/31/breakfast-1804457_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/05/07/08/56/pancakes-2291908_1280.jpg",
    ],
    "animal_other": [
        "https://cdn.pixabay.com/photo/2015/06/19/14/23/bird-815029_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/07/18/18/24/dove-2516641_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/12/05/11/39/horse-1883583_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/04/10/18/22/rabbit-3307998_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/01/12/23/02/duck-1976125_1280.jpg",
    ],
}

def download(url, idx):
    fpath = OTHER_DIR / f"other_diverse_{idx:04d}.jpg"
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
    existing = len(list(OTHER_DIR.glob("*.*")))
    print(f"Existing other: {existing}")

    all_urls = []
    for cat, urls in CATEGORIES.items():
        all_urls.extend(urls)
        print(f"  {cat}: {len(urls)} URLs")

    print(f"\nTotal: {len(all_urls)} new URLs")

    count = existing
    for i, url in enumerate(all_urls):
        if download(url, count):
            count += 1

    final = len(list(OTHER_DIR.glob("*.*")))
    print(f"Final other: {final} (new: {final - existing})")


if __name__ == "__main__":
    main()
