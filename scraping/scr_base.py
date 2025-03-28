import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class BaseScraper:
    """БАЗОВЫЙ КЛАСС ДЛЯ СКРАППИНГА"""

    def __init__(self, login, password, base_url, output_file, max_posts):
        self.login = login
        self.password = password
        self.base_url = base_url
        self.output_file = output_file
        self.max_posts = max_posts

    # Инициализация драйвера
    @staticmethod
    def create_driver():
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        return webdriver.Chrome(options=options)

    # Точка входа (в подкласс)
    def login_site(self, driver):
        raise NotImplementedError("login_site() должен быть реализован в подклассах")

    # Закрытие всплывающих окон
    @staticmethod
    def close_popups(driver):
        popups = ['[aria-label="Allow All Cookies"]', '[aria-label="Close"]']
        for selector in popups:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                try:
                    el.click()
                    time.sleep(1)
                except:
                    pass

    # Прокрутка страницы для загрузки новых постов
    @staticmethod
    def scroll_page(driver):
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    # Извлечение данных (в подкласс)
    def extract_posts(self, driver):
        raise NotImplementedError("extract_posts() должен быть реализован в подклассах")

    # Сохранение
    def save_to_csv(self, data):
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False, encoding="utf-16")

    # Основная функция
    def scrape(self):
        driver = self.create_driver()
        try:
            self.login_site(driver)
            driver.get(self.base_url)
            self.close_popups(driver)
            self.scroll_page(driver)

            posts_data = self.extract_posts(driver)
            self.save_to_csv(posts_data)

            print(f"Сохранено {len(posts_data)} постов в {self.output_file}")
        finally:
            driver.quit()
