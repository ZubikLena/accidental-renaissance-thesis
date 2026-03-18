import requests
import time


def create_session():
    session = requests.Session()

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    return session


def get_rijks_items(config):

    session = create_session()

    url = config["search_url"]
    page_token = None

    all_items = []

    while True:

        full_url = url
        if page_token:
            full_url += f"&pageToken={page_token}"

        response = session.get(full_url, timeout=10)

        if response.status_code != 200:
            print("Search failed:", response.text)
            break

        data = response.json()

        items = data.get("orderedItems", [])
        page_token = data.get("nextPageToken")

        if not items:
            break

        print(f"Fetched {len(items)} items")

        all_items.extend(items)

        if not page_token:
            break

    return all_items

def extract_rijks_image(item, config):

    session = create_session()

    try:
        humanmade_id = item["id"]

        human = session.get(humanmade_id, timeout=10).json()
        visual_id = human.get("shows", [{}])[0].get("id")

        if not visual_id:
            return None

        visual = session.get(visual_id, timeout=10).json()
        digital_id = visual.get("digitally_shown_by", [{}])[0].get("id")

        if not digital_id:
            return None

        digital = session.get(digital_id, timeout=10).json()
        image_url = digital.get("access_point", [{}])[0].get("id")

        time.sleep(config.get("delay", 0))

        return image_url

    except Exception:
        return None