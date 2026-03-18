import yaml
import os

from scraper import Scraper
from downloader import ImageDownloader
from function_registry import FUNCTION_REGISTRY


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def inject_functions(config):
    config["get_items"] = FUNCTION_REGISTRY[config["functions"]["get_items"]]
    config["extract_image"] = FUNCTION_REGISTRY[config["functions"]["extract_image"]]
    return config


def main():

    config_files = [f for f in os.listdir("configs/data") if f.endswith(".yaml")]

    for file in config_files:

        config = load_config(os.path.join("configs/data", file))
        config = inject_functions(config)

        downloader = ImageDownloader(config["save_dir"])
        scraper = Scraper(config, downloader)

        scraper.run()


if __name__ == "__main__":
    main()