from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

from selenium.common.exceptions import NoSuchElementException

#In order to use this script download a WebDriver (geckodriver for Mozilla, ChromeDriver for Chrome)
# download the software from https://www.selenium.dev/downloads/
# install it and place the file in a comfortable folder (here: C:/Program Files/Mozilla Firefox/)
# add the PATH to your variable and enjoy

driver = webdriver.Firefox(executable_path='C:/Program Files/Mozilla Firefox/geckodriver.exe')
book = "LittleWomen"
base_folder = "./../Data/scraping/"
if not os.path.exists(base_folder):
    os.makedirs(os.path.dirname(base_folder))
book_folder = base_folder + book + "/"
if not os.path.exists(book_folder):
    os.makedirs(os.path.dirname(book_folder))

##       WIKIPEDIA
wiki = []
driver.get("https://en.wikipedia.org/wiki/Little_Women")
soup = BeautifulSoup(driver.page_source)
main_content = soup.find(id='mw-content-text')
for paragraph in main_content.find_all('p'):
    text = paragraph.text
    text = re.sub('\[.*?\]', "", text)
    wiki.append(text)

driver.quit()

df = pd.DataFrame(wiki)
df.to_csv(book_folder + 'wiki.csv', index=False, encoding='utf-8')



##### LIBRARYTHING
librarything = []
driver.get("https://www.librarything.com/work/18857998/reviews")
show_all_button = driver.find_element_by_id('mainreviews_reviewnav').find_elements_by_tag_name('a')[3].click()
time.sleep(2)
soup = BeautifulSoup(driver.page_source)
reviews = soup.findAll("div", {"class": "bookReview"})
for book_review in reviews:
    text = book_review.find("div", {"class": "commentText"}).text

    #text = re.sub('\[.*?\]', "", text)
    librarything.append(text)

driver.quit()

df = pd.DataFrame(librarything)
df.to_csv(book_folder + 'librarything.csv', index=False, encoding='utf-8')



#####GOODREADS
gooreads = []

driver.get("https://www.goodreads.com/book/show/1934.Little_Women")
for i in range(0, 10):
    soup = BeautifulSoup(driver.page_source)
    reviews = soup.findAll("div", {"class": "friendReviews elementListBrown"})
    for book_review in reviews:
        texts = book_review.find("div", {"class": "reviewText stacked"}).findAll('span')
        if len(texts) <= 2:
            text = texts[-1].text
        else:
            text = texts[2].text
        gooreads.append(text)
    driver.find_element_by_class_name('next_page').click()
    time.sleep(2)

driver.quit()

df = pd.DataFrame(gooreads)
df.to_csv(book_folder + '/gooreads.csv', index=False, encoding='utf-8')


#######     AMAZON
amazon = []
base_url = 'https://www.amazon.com/product-reviews/B083WMN5M6/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber='
driver.get(base_url + str(1))
soup = BeautifulSoup(driver.page_source)
pages_tag = soup.findAll("div", {"id": "filter-info-section"})
pages = int(pages_tag[0].text.split('of')[1].split('reviews')[0].strip().replace(',', ''))
for i in range(1,pages/10 + 1):
    driver.get(base_url + str(i))
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source)
    reviews = soup.findAll("div", {"class": "a-section review aok-relative"})
    for book_review in reviews:
        text = book_review.find("span", {"class": "a-size-base review-text review-text-content"}).text
        amazon.append(text)

driver.quit()

df = pd.DataFrame(amazon)
df.to_csv(book_folder + 'amazon.csv', index=False, encoding='utf-8')