 # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import streamlit as st
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
import warnings
import os, sys
import time
import joblib
#1. Read Data
STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

df = pd.read_csv('ProductRaw.csv', header=0)
df_sub=pd.read_csv('product_ws.csv', header=0)
#--------------
# GUI
st.title("Data Science Project 2")
st.write("## Recommender System")
# Upload file
# uploaded_file = st.file_uploader("Choose a file", type=['csv'])
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file, encoding='latin-1')
#     data.to_csv("ProductRaw.csv", index = False)

menu = ["Business Objective", "Content-based filtering", "Collaborative filtering", "Recommendation_System"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### A content-based recommender system is a sophisticated technology used to enhance user experiences. It suggests items by analyzing user preferences and matching them with item attributes. This personalized approach ensures that users receive recommendations tailored to their unique tastes and interests.
    """)  
    st.write("""###### => Problem/ Requirement: In this project, data is collected from an e-commerce website, assuming that the website does not have a recommender system. The objective is to build a Recommendation System to suggest and recommend products to users/customers. The goal is to create recommendation models, including Content-based filtering and Collaborative filtering, for one or multiple product categories on A.vn, providing personalized choices for users.""")
    st.image("Content_filtering.png")
    st.image("Collaborative filtering.png")

elif choice == 'Content-based filtering':
    st.subheader("Build Project Content-based filtering")
    st.write("##### 1. Product Overview")
    st.dataframe(df[["item_id", "name","price","rating","description","brand","group"]].head(3))
    st.write("###### Data before cleaning")
    st.dataframe(df[["item_id", "description"]].head(3))
    st.write("###### Data after cleaning and tokenize with Underthese ")
    #st.dataframe(df_sub[["item_id", "description_ws"]].tail(3))  
    st.write("##### 2. Some Pictures of  Products")
    st.write("##### 3. Build Content-based filtering")
    st.write("##### 4. Show result example")
    # st.code("Khách hàng click vào sản phẩm có ID 48102821 thì gợi ý các sản phẩm như: " + result_id)
    # st.code("Khách hàng tìm kiếm sản phẩm có tên Tai nghe Bluetooth thì gợi ý: " + result_text)



