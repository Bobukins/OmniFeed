import time
import os
import schedule
import functools
from scraping.scr_config import ScrapersConfig
from scraping.scr_net import FacebookScraper, InstagramScraper, TwitterScraper


class ParserScheduler:
    """ПЛАНИРОВЩИК"""

    def __init__(self):
        self.schedule = schedule
        self.config = ScrapersConfig()

        # Инициализация с использованием данных из конфигурации
        self.scrapers = {
            "facebook": {
                "scraper": FacebookScraper(
                    self.config.login, self.config.password, self.config.fb_url,
                    self.config.fb_posts, self.config.max_posts
                ),
                "filename": self.config.fb_posts
            },
            "instagram": {
                "scraper": InstagramScraper(
                    self.config.login, self.config.password, self.config.inst_url,
                    self.config.inst_posts, self.config.max_posts
                ),
                "filename": self.config.inst_posts
            },
            "twitter": {
                "scraper": TwitterScraper(
                    self.config.login, self.config.password, self.config.twX_url,
                    self.config.twX_posts, self.config.max_posts
                ),
                "filename": self.config.twX_posts
            }
        }

    # Проверка результатов
    def data_exists(self, filename):
        filepath = os.path.join(self.config.output_dir, filename)
        return os.path.exists(filepath)

    # Провека
    def all_data_exists(self):
        return all(self.data_exists(scraper["filename"]) for scraper in self.scrapers.values())

    # Декоратор для логирования
    @staticmethod
    def log_execution(message):

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                print(f"{message}...")
                result = func(self, *args, **kwargs)
                return result

            return wrapper

        return decorator

    @log_execution("Запуск для Facebook")
    def run_facebook_parsing(self):
        if not self.data_exists(self.config.fb_posts):
            self.scrapers["facebook"]["scraper"].scrape()
            print("Результаты FacebookScraper сохранены.")
        else:
            print("Файл Facebook уже существует. Пропуск.")

    @log_execution("Запуск для Instagram")
    def run_instagram_parsing(self):
        if not self.data_exists(self.config.inst_posts):
            self.scrapers["instagram"]["scraper"].scrape()
            print("Результаты InstagramScraper сохранены.")
        else:
            print("Файл Instagram уже существует. Пропуск.")

    @log_execution("Запуск для Twitter (X)")
    def run_twitter_parsing(self):
        if not self.data_exists(self.config.twX_posts):
            self.scrapers["twitter"]["scraper"].scrape()
            print("Результаты TwitterScraper сохранены.")
        else:
            print("Файл Twitter уже существует. Пропуск.")

    # Настройка задач
    def setup_schedule(self):
        if self.all_data_exists():
            print("Все файлы уже существуют.")
            return

        scheduled_time = self.config.scheduled_time
        tasks = {
            "facebook": self.run_facebook_parsing,
            "instagram": self.run_instagram_parsing,
            "twitter": self.run_twitter_parsing
        }

        for name, task in tasks.items():
            if not self.data_exists(self.scrapers[name]["filename"]):
                self.schedule.every().day.at(scheduled_time).do(task)
                print(f"[DEBUG] Запланирована задача {name} на {scheduled_time}")

    # Запуск цикла
    def start(self):
        print("Запуск планировщика...")
        print(f"Следующий сбор данных в {self.config.scheduled_time}.")
        print("\n/// /// /// /// ///")

        while True:
            self.schedule.run_pending()
            time.sleep(1)


# # Запуск
# if __name__ == "__main__":
#     scheduler = ParserScheduler()
#     scheduler.setup_schedule()
#     scheduler.start()
