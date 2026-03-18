class Scraper:

    def __init__(self, config, downloader):
        self.config = config
        self.downloader = downloader

    def run(self):
        print(f"Running scraper: {self.config['name']}")

        items = self.config["get_items"](self.config)

        for item in items:
            img_url = self.config["extract_image"](item, self.config)

            if img_url:
                self.downloader.download(img_url)