import requests
from bs4 import BeautifulSoup
import time


def normalize_url(url):

    if "preview.redd.it" in url:
        url = url.replace("preview.redd.it", "i.redd.it")

    if "imgur.com" in url and not url.endswith((".jpg", ".png", ".jpeg")):
        url = url + ".jpg"

    return url


def is_image_host(url, config):

    allowed = ["i.redd.it", "preview.redd.it", "imgur.com"]

    return any(host in url for host in allowed)


def get_reddit_items(config):

    headers = config.get("headers", {})
    max_pages = config["max_pages"]

    seen_urls = set()
    all_posts = []

    for base_url in config["base_urls"]:

        url = base_url
        page = 0

        while url and page < max_pages:

            print(f"\nScraping: {url}")

            try:
                response = requests.get(url, headers=headers, timeout=10)
            except:
                break

            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.text, "html.parser")
            posts = soup.find_all("div", class_="thing")

            for post in posts:

                img_url = post.get("data-url")

                if not img_url:
                    continue

                if img_url in seen_urls:
                    continue

                seen_urls.add(img_url)
                all_posts.append(post)

            # pagination
            next_button = soup.find("span", class_="next-button")

            if next_button and next_button.a:
                url = next_button.a["href"]
                page += 1
                time.sleep(1)
            else:
                break

    return all_posts

def extract_reddit_image(post, config):

    url = post.get("data-url")

    if not url:
        return None

    if not is_image_host(url, config):
        return None

    url = normalize_url(url)

    return url