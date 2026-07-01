"""Download targeted 'other' images: faces, people, screens, games, objects"""
import requests
from pathlib import Path

OTHER_DIR = Path(__file__).resolve().parent.parent / "data" / "other"
OTHER_DIR.mkdir(exist_ok=True)

URLS = [
    # faces / people
    "https://cdn.pixabay.com/photo/2015/07/09/00/29/woman-837156_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/29/03/36/woman-1867093_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/08/01/01/33/beanie-2562646_1280.jpg",
    "https://cdn.pixabay.com/photo/2018/04/27/16/23/people-3354417_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/01/08/18/29/entrepreneur-593358_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/03/27/21/16/adult-1282329_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/21/15/46/woman-1845577_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/02/16/23/10/smile-2072907_1280.jpg",
    # dogs / animals not cats
    "https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/09/25/13/12/dog-2785074_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg",
    "https://cdn.pixabay.com/photo/2019/08/19/07/45/pug-4414790_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/06/19/14/23/bird-815029_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/12/05/11/39/horse-1883583_1280.jpg",
    # screenshots / screens / UI-like
    "https://cdn.pixabay.com/photo/2015/01/08/18/26/man-593333_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/05/31/10/55/man-791049_1280.jpg",
    "https://cdn.pixabay.com/photo/2014/05/02/21/47/home-office-336378_1280.jpg",
    "https://cdn.pixabay.com/photo/2014/08/26/21/48/office-428338_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/29/06/15/computer-1869306_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/08/10/08/47/code-2618293_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/05/15/14/47/computer-768696_1280.jpg",
    "https://cdn.pixabay.com/photo/2014/08/29/15/27/technology-431655_1280.jpg",
    # objects / cars / rooms
    "https://cdn.pixabay.com/photo/2012/05/29/19/43/car-49278_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/18/17/09/chair-1835905_1280.jpg",
    "https://cdn.pixabay.com/photo/2014/09/17/20/26/book-450111_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/29/03/53/architecture-1867187_1280.jpg",
    # food but not bread
    "https://cdn.pixabay.com/photo/2017/12/09/08/18/pizza-3007395_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/06/02/14/31/pizza-1431079_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/05/07/08/56/pancakes-2291908_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/06/23/31/breakfast-1804457_1280.jpg",
]


def main():
    existing = len(list(OTHER_DIR.glob('*.*')))
    count = existing
    print(f'Existing other: {existing}')
    for url in URLS:
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200 and len(r.content) > 5000:
                path = OTHER_DIR / f'target_other_{count:04d}.jpg'
                with open(path, 'wb') as f:
                    f.write(r.content)
                count += 1
                print(path.name)
        except Exception:
            pass
    print(f'Final other: {count}')

if __name__ == '__main__':
    main()
