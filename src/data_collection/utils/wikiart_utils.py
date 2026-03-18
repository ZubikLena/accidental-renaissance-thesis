import requests
from bs4 import BeautifulSoup
import time


def get_artists(config):

    headers = config["headers"]
    artists = set()

    for url in config["movement_urls"]:

        print("Reading movement:", url)

        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        artist_links = soup.select("div.artist-name a")

        for link in artist_links:

            href = link.get("href")

            if href and href.startswith("/en/"):
                artist = href.split("/")[-1]
                artists.add(artist)

    print("Artists found:", len(artists))

    return list(artists)

def get_wikiart_items(config):

    headers = config["headers"]
    max_pages = config["max_pages_per_artist"]

    artists = get_artists(config)

    all_urls = []

    for artist in artists:

        print("Artist:", artist)
        page = 1

        while page <= max_pages:

            url = f"https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl={artist}&page={page}"

            try:
                r = requests.get(url, headers=headers, timeout=10)
                data = r.json()
            except:
                break

            if not data:
                break

            urls = [p.get("image") for p in data if p.get("image")]

            print(f"Page {page}: {len(urls)} images")

            all_urls.extend(urls)

            page += 1
            time.sleep(config.get("delay", 0))

    return all_urls

def extract_wikiart_image(item, config):
    return item