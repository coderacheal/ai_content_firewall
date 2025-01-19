import scrapy
from urllib.parse import urlparse
import os
from .website_list.websites import websites

class ContentSpider(scrapy.Spider):
    name = "content_spider"

    # Flatten the dictionary and associate each URL with its category
    start_urls = [
        {"url": url, "category": category}
        for category, urls in websites.items()
        for url in urls
    ]

    # Base directory for saving scraped data
    base_dir = "scraped_data"

    def start_requests(self):
        for entry in self.start_urls:
            url = entry["url"]
            category = entry["category"]
            yield scrapy.Request(
                url, callback=self.parse, meta={"category": category}
            )

    def parse(self, response):
        # Extract category from metadata
        category = response.meta["category"]

        # Create category folder and domain folder
        domain = urlparse(response.url).netloc
        category_folder = os.path.join(self.base_dir, category)
        domain_folder = os.path.join(category_folder, domain)
        os.makedirs(domain_folder, exist_ok=True)

        # Save text content
        page_text = response.xpath('//p//text() | //h1//text() | //h2//text() | //li//text()').getall()
        page_text = ' '.join(page_text).strip()

        if not page_text:
            page_text = "No meaningful content extracted from this page."

        text_file_path = os.path.join(domain_folder, 'content.txt')
        with open(text_file_path, 'a', encoding='utf-8') as f:
            f.write(f"URL: {response.url}\n")
            f.write(page_text + "\n\n")

        # Save image data
        image_urls = response.xpath('//img/@src').getall()
        for img_url in image_urls:
            img_url = response.urljoin(img_url)
            yield scrapy.Request(img_url, callback=self.save_image, meta={"domain_folder": domain_folder})

    def save_image(self, response):
        domain_folder = response.meta["domain_folder"]
        image_name = urlparse(response.url).path.split('/')[-1]
        if not image_name:
            image_name = "image.jpg"
        image_path = os.path.join(domain_folder, image_name)
        with open(image_path, 'wb') as f:
            f.write(response.body)

