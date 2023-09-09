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

#--------------

# # 2. Data pre-processing
# # lọc lại những cột cần thiết để phân tích
df_sub = df[["item_id", "name","description","brand","group"]]
df_pics=df[[ "name","image"]].head(5)
stop_words_extra=["THÔNG TIN CHI TIẾT","rất" "Thương hiệu","Xuất xứ","Ngoài ra","được","cho","Chức năng","MÔ TẢ SẢN PHẨM","Ngoài ra","thương hiệu", "với", "có","các", "Lưu ý", "sẽ", "dc","Hãy", "giúp", "bạn","được","hoặc", "Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành", "Tuy nhiên tuỳ vào từng loại sản phẩm hoặc phương thức", "địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển, phụ phí hàng cồng kềnh","đảm bảo",""]     
st="Tuy nhiên tuỳ vào từng loại sản phẩm hoặc phương thức, địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển, phụ phí hàng cồng kềnh"
# xóa cột null
df_sub=df_sub.dropna()
#Gộp thông tin các cột name, description, brand, group vào 1 cột
df_sub['description_combine'] = df_sub[['name', 'description','brand','group']].agg('-'.join, axis=1)
# hàm clean_text
def text_clean(df, column_input):
    # Remove specific words
    for word in stop_words_extra:
        df[column_input] = df[column_input].replace(word, "", regex=True)
    df[column_input] = df[column_input].str.replace(st, "")

    # Remove line breaks
    df[column_input] = df[column_input].str.replace(r'\n', ' ')

    # Convert text to lowercase
    df[column_input] = df[column_input].str.lower()

    # Remove special characters, punctuation, and symbols
    def remove_special_characters(text):
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove non-word and non-space characters
        return cleaned_text

    df[column_input] = df[column_input].apply(remove_special_characters)

    return df

df_sub=text_clean(df_sub,'description_combine')
df_sub['description_combine'] = df_sub['description_combine'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df_sub["description_ws"]=df_sub["description_combine"].apply(lambda x: word_tokenize(x,format="text"))
df_sub.to_csv('product_ws.csv', index=False)
df_clean=df_sub


