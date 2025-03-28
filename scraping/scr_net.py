import time
from scraping.scr_base import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class FacebookScraper(BaseScraper):
    """ПОСТЫ С FACEBOOK"""

    def login_site(self, driver):
        driver.get("https://www.facebook.com")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "email")))

        driver.find_element(By.NAME, "email").send_keys(self.login)
        driver.find_element(By.NAME, "pass").send_keys(self.password)
        driver.find_element(By.NAME, "pass").send_keys(Keys.RETURN)
        time.sleep(5)

    def extract_posts(self, driver):
        posts_data = []
        posts = driver.find_elements(By.CSS_SELECTOR, '[role="article"]')

        for post in posts:
            try:
                post_id = post.get_attribute("data-ft")
                caption = post.find_element(By.CSS_SELECTOR, "div[data-ad-comet-preview='message']").text
                posts_data.append({"post_id": post_id, "caption": caption})
                if len(posts_data) >= self.max_posts:
                    break
            except:
                pass

        return posts_data

    def scrape(self):
        print(f"Начинаем скрапинг для {self.__class__.__name__}...")
        driver = self.create_driver()
        try:
            self.login_site(driver)
            driver.get(self.base_url)
            self.close_popups(driver)
            self.scroll_page(driver)
            posts_data = self.extract_posts(driver)
            self.save_to_csv(posts_data)
            print(f"Результаты {self.__class__.__name__} сохранены.")
        except Exception as e:
            print(f"Ошибка при скрапинге {self.__class__.__name__}: {e}")
        finally:
            driver.quit()


class InstagramScraper(BaseScraper):
    """ПОСТЫ С INSTAGRAM"""

    def login_site(self, driver):
        driver.get("https://www.instagram.com")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "username")))

        driver.find_element(By.NAME, "username").send_keys(self.login)
        driver.find_element(By.NAME, "password").send_keys(self.password)
        driver.find_element(By.NAME, "password").send_keys(Keys.RETURN)
        time.sleep(5)

    def extract_posts(self, driver):
        posts_data = []
        posts = driver.find_elements(By.CSS_SELECTOR, "article div")

        for post in posts:
            try:
                caption = post.text
                posts_data.append({"caption": caption})
                if len(posts_data) >= self.max_posts:
                    break
            except:
                pass

        return posts_data


class TwitterScraper(BaseScraper):
    """ПОСТЫ С TWITTER (X)"""

    def login_site(self, driver):
        driver.get("https://www.x.com/login")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "session[username_or_email]")))

        driver.find_element(By.NAME, "session[username_or_email]").send_keys(self.login)
        driver.find_element(By.NAME, "session[password]").send_keys(self.password)
        driver.find_element(By.NAME, "session[password]").send_keys(Keys.RETURN)
        time.sleep(5)

    def extract_posts(self, driver):
        posts_data = []
        posts = driver.find_elements(By.CSS_SELECTOR, "article div")

        for post in posts:
            try:
                caption = post.text
                posts_data.append({"caption": caption})
                if len(posts_data) >= self.max_posts:
                    break
            except:
                pass

        return posts_data
