import requests
from PIL import Image
from io import BytesIO
import imagehash
import os


class ImageDownloader:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.seen_hashes = set()
        os.makedirs(save_dir, exist_ok=True)

    def download(self, url):
        try:
            r = requests.get(url, timeout=10)

            if r.status_code != 200:
                return

            img = Image.open(BytesIO(r.content)).convert("RGB")

            if img.width < 200 or img.height < 200:
                return

            h = imagehash.phash(img)

            if h in self.seen_hashes:
                return

            self.seen_hashes.add(h)

            filepath = os.path.join(self.save_dir, f"{h}.jpg")
            img.save(filepath)

        except:
            pass