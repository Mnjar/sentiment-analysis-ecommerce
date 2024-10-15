import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

def extract_rating(rating_element):
    """extract rating bintang ke sebuah integer

    Args:
        rating_element (html_tag): html tag element

    Returns:
        int: rating
    """
    aria_label = rating_element.get('aria-label', '')
    if "bintang" in aria_label:
        try:
            return int(aria_label.split(' ')[1])
        except ValueError:
            return None
    return None

def scrape_reviews(url):
    """Function to collect reviews from Tokopedia comments section

    Args:
        url (list: str): list of url you want to scrap

    Return:
        Dataframe: Dataframe of collected reviews


    """
    driver = webdriver.Chrome()
    driver.get(url)
    reviews = []

    try:
        try:
            # Get product category
            category_element = driver.find_element(By.XPATH, "(//a[@class='css-1y6qqnj-unf-heading e1qvo2ff7'])[3]")
            category = category_element.text
        except Exception as e:
            print(f"Error getting product category from {url}: {e}")
            driver.quit()
            return reviews
        
        try:
            # Get product name
            product_name_element = driver.find_element(By.XPATH, "//h1[@data-testid='lblPDPDetailProductName']")
            product_name = product_name_element.text
        except Exception as e:
            print(f"Error getting product name from {url}: {e}")
            driver.quit()
            return reviews
        
        # Find the report_section element to scroll to.
        report_section = driver.find_element(By.ID, "pdp_comp-review")

        # Run a JavaScript script to scroll to that element
        driver.execute_script("arguments[0].scrollIntoView();", report_section)
        time.sleep(2)  # Wait a few seconds for the page to load.

        # Find total pagination pages
        pagination_buttons = driver.find_elements(By.XPATH, "//button[@class='css-bugrro-unf-pagination-item']")
        if pagination_buttons:
            last_page_button = pagination_buttons[-1]
            total_pages = int(last_page_button.text.strip()) if last_page_button.text.strip().isdigit() else 1
        else:
            total_pages = 1

        # Iterate to retrieve reviews from each page
        for page in range(1, total_pages + 1):
            try:
                # Search reviews on current page
                html_content = driver.page_source
                soup = BeautifulSoup(html_content, 'html.parser')
                review_elements = soup.find_all('span', {'data-testid': 'lblItemUlasan'})
                rating_elements = soup.find_all('div', {'data-testid': 'icnStarRating'})
                # Add, product name, rating and reviews from current page to reviews
                for review_element, rating_element in zip(review_elements, rating_elements):
                    review_text = review_element.text
                    rating = extract_rating(rating_element)
                    reviews.append((product_name, category, review_text, rating))
                
                # If you haven't reached the last page, click the next page button.
                if page < total_pages:
                    next_page_button = driver.find_element(By.XPATH, f"//button[@class='css-bugrro-unf-pagination-item' and text()='{page + 1}']")
                    driver.execute_script("arguments[0].click();", next_page_button)
                    time.sleep(2)  # Wait a few seconds for the page to load.

            except Exception as e:
                print(f"Failed to fetch reviews from page {page} on {url}: {str(e)}")
                break
    except Exception as e:
        print(f"Error processing URL {url}: {e}")

    finally:
        driver.quit()

    return reviews

def gather_reviews(url_list):
    all_reviews = []
    for url in url_list:
        reviews = scrape_reviews(url)
        all_reviews.extend(reviews)
        print(f"Total ulasan terkumpul: {len(all_reviews)}")
    
    df = pd.DataFrame(all_reviews, columns=['product name', 'category', 'review', 'rating'])
    df.to_csv('reviews.csv', index=False)
    return df
