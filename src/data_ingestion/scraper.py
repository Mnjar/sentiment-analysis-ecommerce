import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

def extract_rating(rating_element):
    aria_label = rating_element.get('aria-label', '')
    if "bintang" in aria_label:
        try:
            return int(aria_label.split(' ')[1])
        except ValueError:
            return None
    return None

def scrape_reviews(url):
    driver = webdriver.Chrome()
    driver.get(url)
    reviews = []
    try:
        category = driver.find_element(By.XPATH, "(//a[@class='css-1y6qqnj-unf-heading e1qvo2ff7'])[3]").text
        product_name = driver.find_element(By.XPATH, "//h1[@data-testid='lblPDPDetailProductName']").text
        report_section = driver.find_element(By.ID, "pdp_comp-review")
        driver.execute_script("arguments[0].scrollIntoView();", report_section)
        time.sleep(2)
        
        pagination_buttons = driver.find_elements(By.XPATH, "//button[@class='css-bugrro-unf-pagination-item']")
        total_pages = int(pagination_buttons[-1].text.strip()) if pagination_buttons else 1
        
        for page in range(1, total_pages + 1):
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            review_elements = soup.find_all('span', {'data-testid': 'lblItemUlasan'})
            rating_elements = soup.find_all('div', {'data-testid': 'icnStarRating'})
            for review_element, rating_element in zip(review_elements, rating_elements):
                review_text = review_element.text
                rating = extract_rating(rating_element)
                reviews.append((product_name, category, review_text, rating))
            
            if page < total_pages:
                next_page_button = driver.find_element(By.XPATH, f"//button[@class='css-bugrro-unf-pagination-item' and text()='{page + 1}']")
                driver.execute_script("arguments[0].click();", next_page_button)
                time.sleep(2)
    finally:
        driver.quit()

    return reviews

def gather_reviews(url_list):
    all_reviews = []
    for url in url_list:
        reviews = scrape_reviews(url)
        all_reviews.extend(reviews)
        print(f"Total ulasan terkumpul: {len(all_reviews)}")
    
    df = pd.DataFrame(all_reviews, columns=['product name', 'category', 'review', 'rating', ])
    df.to_csv('reviews.csv', index=False)
    return df
