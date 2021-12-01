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


def convert_df(df):
    return df.to_csv().encode('utf-8')
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

    #print(result['organic_results'])
    df = pd.DataFrame(columns=['title','text'])
    
    n_samples = 5
  
    print(len(result['organic_results']))
    for item in result['organic_results']:
        url = item['link']
        title=item['title']
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        p_list = html_soup.find_all('p')[:10]
        st.write("\n")
        st.write("Website title: %s :"%title) 
        text = ''
        for p in p_list:

            st.write('<p>' + p.text + '<\p>')
            if len(p.text) > len(text):
                text = p.text
        final_results.append(text)
        if len(p_list) > 1 and text != '':
            df.loc[len(df)] = {'title':title,'text':text}
        if n_samples <= 0: 
            break
        n_samples -= 1
        
    st.write(df)
    return df




def main():
    user_input = st.text_input("Search Text", ) 
    if st.button("Search"):
        if user_input != None:
            results = search(user_input)
            csv = convert_df(results)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='extracted_text.csv',
                mime='text/csv',
            ) 
if __name__ == '__main__':
	main()
