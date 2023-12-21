#!/usr/bin/env python
# coding: utf-8


import settings
HOST = settings.SPLUNK_HOST
PORT = settings.SPLUNK_PORT
USERNAME = settings.SPLUNK_USERNAME
PASSWORD = settings.SPLUNK_PASSWORD
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
import time
import numpy as np
import re
import nltk
from datetime import datetime
from nltk.stem import WordNetLemmatizer 
from PIL import Image
from Screenshot import Screenshot_Clipping
import socket
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordsegment import segment, load
load()



PROXY=""
webdriver.DesiredCapabilities.CHROME['proxy'] = {
    "httpProxy": PROXY,
    "ftpProxy": PROXY,
    "sslProxy": PROXY,
    "proxyType": "MANUAL",

}

# setting up the Chrome driver
webdriver.DesiredCapabilities.CHROME['acceptSslCerts']=True
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
PATH = "/home/bniese/Analytics/chromedriver"


# This function imports splunk search table to a csv
def splunk_csv_import(query):
    import os
    import pandas as pd
    global USERNAME, PASSWORD, HOST, PORT
    print('Importing:', query)
    command = "curl -k -u {}:'{}' https://{}:{}/services/search/jobs/export --data-urlencode search='{}' -d output_mode=csv -o splunk_import_tmp.csv" .format(USERNAME, PASSWORD, HOST, PORT, query)
    #print(command)
    os.system(command)
    print('Done')
    try:
        X = pd.read_csv('splunk_import_tmp.csv')
        os.remove('splunk_import_tmp.csv')
        print('Results:', len(X))
        return X
    except:
        print('No results')
        return None


# creates a dictionary that counts the number of times a word is present in list of words
def bag_of_words(words):
    bag = {}
    for word in words:
        if word in bag:
            bag[word] +=1
        if not word in bag:
            bag[word] = 1
    return bag

# gets rid of punctuation 
def de_punc(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    no_punc = ""
    for char in text:
        if char not in punctuations:
            no_punc = no_punc + char
    return no_punc

# lemmatizes a list of words
def lemma(words):
    lemmatizer = WordNetLemmatizer()
    temp = []
    for word in words:
        temp.append(lemmatizer.lemmatize(word))
    return temp

import math

# takes 2 dictionaries of counts and gives a cosine similarity score
def cosine_dic(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    if (dena*denb) == 0:
        return 0
    return 100*(numerator/math.sqrt(dena*denb))

# given a hsv pixel value returns the color
def color(hsv):
    if hsv[1] < 24:
        if hsv[2] <=85:
            return "Black"
        if hsv[2] >=225:
            return "White"
        if hsv[2] >30 and hsv[2]<225:
            return "Gray"
    if hsv[1] != 0:
        if (hsv[0] >= 0 and hsv[0] <= 15) or (hsv[0] > 345 and hsv[0] <= 360):
            return "Red"
        if hsv[0] > 15 and hsv[0] <= 45:
            return "Orange"
        if hsv[0] > 45 and hsv[0] <= 75:
            return "Yellow"
        if hsv[0] > 75 and hsv[0] <= 105:
            return "Yellow/Green"
        if hsv[0] > 105 and hsv[0] <= 135:
            return "Green"
        if hsv[0] > 135 and hsv[0] <= 165:
            return "Green/Blue"
        if hsv[0] > 165 and hsv[0] <= 195:
            return "Teal"
        if hsv[0] > 195 and hsv[0] <= 225:
            return "Blue"
        if hsv[0] > 225 and hsv[0] <= 255:
            return "Indigo"
        if hsv[0] > 255 and hsv[0] <= 285:
            return "Violet"
        if hsv[0] > 285 and hsv[0] <= 315:
            return "Pink"
        if hsv[0] > 315 and hsv[0] <= 345:
            return "Red/Pink"

# given an image returns the number of pixels in each color range producing a palette
def color_palette(img1):
    img1 = img1.convert("HSV")
    d = img1.getdata()
    palette = {}
    for item in d:
        if not color(item) in palette:
            palette[color(item)] = 1
        else:
            palette[color(item)] += 1
    return palette



# preprocessing text from a labeled training set of parked domains
# parked domains are domains that people are trying to sell
# these domains are benign until sold
data = pd.read_csv('new_MajorVerticalMaster.csv')
def preprocess_text_new(s):
    out=''
    s_split = s.split(' ')
    for i in s_split:
        seg = segment(i)
        for j in seg:
            if len(j)<15:
                out=out+' '+ j
    return out
data['text'] = data['processed_text']

start_time = time.time()
print('Processing',len(data),'entries.')
data['processed_text'] = data['text'].transform(lambda x: preprocess_text_new(str(x)))
print('Time elapsed:', time.time() - start_time )

# list categories (parked and not parked)
cats= []
for i in range(len(data['target'].unique())):
    cats.append(data[data['target'] == i]['category'].unique()[0])


classes = cats
print(classes)

classes_df = pd.DataFrame(classes)


# this section of code trains a SVC classifier model that classifies a parked domain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], 
                                                    data['target'], 
                                                    random_state=0, stratify=data['target'], test_size=0.2)

start_time = time.time()
vect = TfidfVectorizer(min_df = 2).fit(X_train)

X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

feature_names = np.array(vect.get_feature_names())
print('elapsed:', time.time()-start_time)

from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV

svc_model = LinearSVC(multi_class='ovr', verbose = 1, C=0.25)
#clf = CalibratedClassifierCV(svc_model)
#clf.fit(X_train_vectorized, y_train)
svc_model.fit(X_train_vectorized, y_train)



# this splunk search contains newly created domains that are similar to known domains
# similarity in domains in this database is defined by a string distance
q1 = 'search  index=data-science spoofed_website=* earliest=-24h| table domain, spoofed_website, spoofed_domain, similarity_score, first_seen, description'
links = splunk_csv_import(q1)
print(str(datetime.now())[0:10])


# creates and empty data frame that has all the values needed for evaluation
# domain: the newly created domain
# ip: ip associated with the domain
# spoofed_website: website that newly created domain is similar to
# spoofed_domain: domain of spoofed website
# similarity_score: score how similar the actual domain name is to the spoofed domain
# description: type of spoofed website (bank, social media, etc)
# open: 1 if the domain opens 0 if it does not
# password: 1 if the word password of login appears in the html and 0 if not
# redirect: if the domain redirects to another domain it shows the domain and "" if not
# first_seen: timestamp of first time the domain is seen
# text_similarity: cosine similarity of the text bag of words
# color similarity: cosine similarity of color palette
# parked: if the domain is parked is 1, 0 otherwise
# image_text_sim: cosine similarity of text extracted from image from pytesseract 
empty_dic = {"domain": [],"ip": [],"spoofed_website": [], "spoofed_domain": [], "similarity_score": [], "description":[], "open": [], "password": [], "redirect": [], "first_seen": [], "date_scraped": [], "text_similarity":[], "color_similarity": [], "parked": [], "image_text_sim": []}
df = pd.DataFrame(empty_dic)
word_bags = {"domain":[], "phish":[], "spoof":[]}
word_bags_df = pd.DataFrame(word_bags)

#iterates through all the newly created newly created domains
for i in range(len(links["domain"])):
    driver = webdriver.Chrome(PATH, chrome_options = chrome_options)
    redirect = '0'
    password = 0
    ip = '0'
    # this try tests if the domain opens
    try:
        driver.set_page_load_timeout(5)
        domain = "https://" + links["domain"][i]
        driver.get(domain)
        ip = socket.gethostbyname(links["domain"][i])
        proc_domain = domain + "/"

	# finds redirect and password
        if proc_domain != driver.current_url:
            redirect = driver.current_url
        if "password"  in driver.page_source.lower():
            password = 1
        if "login" or "sign" in driver.find_element_by_xpath("/html/body").text.lower():
            password = 1
        
        # creates the bag of words 
        temp_text = de_punc(driver.find_element_by_xpath("/html/body").text)
        v = vect.transform([temp_text])
        tokens = nltk.word_tokenize(temp_text)
        lemma_tokens = lemma(tokens)
        bag = bag_of_words(lemma_tokens)
        
	# creates color palette of domains
        driver.save_screenshot("/opt/ds_shared/"+str(datetime.now())[0:10]+(links["domain"][i].replace('.','_'))+"_phish.png")
        driver.save_screenshot('temp.png')
        screenshot = Image.open('temp.png')
        colors = color_palette(screenshot)
	
	# gets words from images
        im_temp_text = de_punc(pytesseract.image_to_string(screenshot))
        im_v = vect.transform([im_temp_text])
        im_tokens = nltk.word_tokenize(im_temp_text)
        im_lemma_tokens = lemma(im_tokens)
        im_bag = bag_of_words(im_lemma_tokens)

        driver.quit()
        
	# repeats the entire process for the spoofed domain
        driver2 = webdriver.Chrome(PATH, chrome_options = chrome_options)
        driver2.set_page_load_timeout(60)
        driver2.get("https://" + links["spoofed_domain"][i])
        spoof_temp_text = de_punc(driver2.find_element_by_xpath("/html/body").text)
        spoof_tokens = nltk.word_tokenize(spoof_temp_text)
        spoof_lemma_tokens = lemma(spoof_tokens)
        spoof_bag = bag_of_words(spoof_lemma_tokens)
        #print(spoof_bag)
        driver2.save_screenshot("/opt/ds_shared/"+str(datetime.now())[0:10]+links["spoofed_website"][i]+'.png')
        driver2.save_screenshot('spoof_temp.png')
        spoof_screenshot = Image.open('spoof_temp.png')
        spoof_colors = color_palette(spoof_screenshot)
        im_spoof_temp_text = de_punc(pytesseract.image_to_string(screenshot))
        im_spoof_tokens = nltk.word_tokenize(im_spoof_temp_text)
        im_spoof_lemma_tokens = lemma(im_spoof_tokens)
        im_spoof_bag = bag_of_words(im_spoof_lemma_tokens)
        #print(spoof_colors)
        driver2.quit()
        
        df.loc[i] = [links["domain"][i], ip, links["spoofed_website"][i], links["spoofed_domain"][i], links["similarity_score"][i], links["description"][i], 1, password, redirect,links["first_seen"][i], datetime.now(), cosine_dic(bag,spoof_bag), cosine_dic(colors,spoof_colors), svc_model.predict(v)[0], cosine_dic(im_bag,im_spoof_bag)]
        
        word_bags_df.loc[i] = [links["domain"][i], bag, spoof_bag]
        #element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body")))

        
	#exeptions for unopened domains
    except WebDriverException:
        #print(1)
        word_bags_df.loc[i] = [links["domain"][i], {}, {}]
        df.loc[i] = [links["domain"][i], '0',links["spoofed_website"][i], links["spoofed_domain"][i], links["similarity_score"][i], links["description"][i], 0, 0, '0', links["first_seen"][i], datetime.now(), 0,0,1,0]
    except NoSuchElementException:
        #print(2)
        word_bags_df.loc[i] = [links["domain"][i], {}, {}]
        df.loc[i] = [links["domain"][i], '0',links["spoofed_website"][i], links["spoofed_domain"][i], links["similarity_score"][i], links["description"][i], 0, 0, '0', links["first_seen"][i], datetime.now(), 0,0,1,0]
    except TimeoutException:
        #print(3)
        word_bags_df.loc[i] = [links["domain"][i], {}, {}]
        df.loc[i] = [links["domain"][i], '0',links["spoofed_website"][i], links["spoofed_domain"][i], links["similarity_score"][i], links["description"][i], 0, 0, '0', links["first_seen"][i], datetime.now(), 0,0,1,0]
    


# label for splunk database
df["type"] = "phishing_websites_analyzed"

# saves into splunk folder
df.to_csv("/opt/splunk_logs/phish/scraped_"+str(datetime.now())[0:10] +".csv", index = False )

