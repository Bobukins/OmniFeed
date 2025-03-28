import os
import json


class ScrapersConfig:
    """ДИНАМИЧЕСКАЯ КОНФИГУРАЦИЯ ДЛЯ SCRAPERS"""

    CONFIG_DIR = "./scraping"
    CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")

    def __init__(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        self.load_config()

    # Импорт конфигурации
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as file:
                config_data = json.load(file)
        else:
            config_data = self.default_config()
            self.save_config(config_data)

        self.login = config_data["login"]
        self.password = config_data["password"]
        self.max_posts = config_data["max_posts"]
        self.scheduled_time = config_data["scheduled_time"]

        self.fb_url = config_data["fb_url"]
        self.fb_profiles = config_data["fb_profiles"]
        self.fb_posts = config_data["fb_posts"]

        self.inst_url = config_data["inst_url"]
        self.inst_profiles = config_data["inst_profiles"]
        self.inst_posts = config_data["inst_posts"]

        self.twX_url = config_data["twX_url"]
        self.twX_profiles = config_data["twX_profiles"]
        self.twX_posts = config_data["twX_posts"]

        self.output_dir = config_data["output_dir"]

        os.makedirs(self.output_dir, exist_ok=True)

    # Сохранение
    def save_config(self, config_data):
        with open(self.CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(config_data, file, indent=4, ensure_ascii=False)

    # Обновление конфигурации через ввод
    def set_config(self, login, password, max_posts, scheduled_time, fb_url, inst_url, twX_url):
        config_data = {
            "login": login,
            "password": password,
            "max_posts": max_posts,
            "scheduled_time": scheduled_time,

            "fb_url": fb_url,
            "fb_profiles": "fb_profiles.csv",
            "fb_posts": "fb_posts.csv",

            "inst_url": inst_url,
            "inst_profiles": "inst_profiles.csv",
            "inst_posts": "inst_posts.csv",

            "twX_url": twX_url,
            "twX_profiles": "twX_profiles.csv",
            "twX_posts": "twX_posts.csv",

            "output_dir": "./data/raw"
        }
        self.save_config(config_data)
        self.load_config()

    # По умолчанию
    @staticmethod
    def default_config():
        return {
            "login": "your_email",
            "password": "your_password",
            "max_posts": 10,
            "scheduled_time": "06:00",

            "fb_url": "https://www.facebook.com/your_page",
            "fb_profiles": "fb_profiles.csv",
            "fb_posts": "fb_posts.csv",

            "inst_url": "https://www.instagram.com/your_page",
            "inst_profiles": "inst_profiles.csv",
            "inst_posts": "inst_posts.csv",

            "twX_url": "https://www.x.com/your_page",
            "twX_profiles": "twX_profiles.csv",
            "twX_posts": "twX_posts.csv",

            "output_dir": "./data/raw"
        }

# # Использование
# from config import ScrapersConfig
#
# config = ScrapersConfig()
# config.set_config(
#     login="myemail@example.com",
#     password="mypassword",
#     max_posts=100,
#     fb_ur="https://www.facebook.com/my_page",
#     inst_url="https://www.facebook.com/my_page",
#     twX_url="https://www.facebook.com/my_page",
#     scheduled_time="08:00"
# )
