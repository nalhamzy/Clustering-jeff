app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
from serpapi import GoogleSearch
import os
import csv
import time
import pandas as pd
import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
from requests import get

user_input = ''
def search(input_text):
    search = GoogleSearch({
        "q": input_text, 
        "location": "Austin,Texas",
        "api_key": app_secret
    })
    result = search.get_dict()
    #print(result)
    final_results= []
    for key in result:
        print(key)
        #print(result[key])

    print(result['organic_results'])
    df = pd.DataFrame(columns=['title','text'])
    
    n_samples = 5
    for item in result['organic_results']:
        url = item['link']
        title=item['title']
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        p_list = html_soup.find_all('p')[:10]
        text = ''
        for p in p_list:
            if len(p.text) > len(text):
                text = p.text
        final_results.append(text)
        if len(p_list) > 1 and text != '':
            df.loc[len(df)] = {'title':title,'text':text}
        if n_samples <= 0: 
            break
        n_samples -= 1
        
    st.write(df)





def main():
    user_input = st.text_input("Search Text", ) 
    if st.button("Search"):
        if user_input != None:
            results = search(user_input)
if __name__ == '__main__':
	main()
