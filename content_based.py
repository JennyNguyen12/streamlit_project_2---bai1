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
# GUI
st.title("Data Science Project 2")
st.write("## Recommender System")
# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin-1')
    data.to_csv("ProductRaw.csv", index = False)





# 2. Data pre-processing
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

df_sub = pd.read_csv('product_ws.csv', header=0)


products_gem=[[text for text in x.split()] for x in df_sub.description_ws]
# Obtain the number of features based on dictionary: Use corpora.Dictionary
dictionary = corpora.Dictionary(products_gem)
# List of features in dictionary
dictionary.token2id
# Numbers of features (word) in dictionary
feature_cnt = len(dictionary.token2id)
# Obtain corpus based on dictionary (dense matrix)
corpus = [dictionary.doc2bow(text) for text in products_gem]
print(corpus[0]) # id, so lan xuat hien cua token trong van ban/ san pham
# Use TF-IDF Model to process corpus, obtaining index
tfidf = models.TfidfModel(corpus)
# tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                            num_features = feature_cnt)

# 3. Modeling - Gensim
# tìm kiếm theo ID

def recommender_id(input_id, dictionary, tfidf_model, index):
      # Convert search words into sparse vector
    selected_id = df_sub.loc[df_sub['item_id'] == input_id, 'description_ws']
    combined_text = ' '.join(selected_id)
    view_product = combined_text.split()
    kw_vector = kw_vector = dictionary.doc2bow(view_product)
    print("view product's vector",kw_vector)

  # Similarity calculation
    sim = index[tfidf[kw_vector]]

  # print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])

    df_result=pd.DataFrame({'id': list_id,'score': list_score})

  # Find five highest scores
    five_highest_score = df_result.sort_values(by='score',ascending=False).head(6)
    print('Five highest scores: ')
    print(five_highest_score)
  # Ids to list
    idToList=list(five_highest_score['id'])
    print('idToList',idToList)

    products_find=df[df.index.isin(idToList)]
    results=products_find[['item_id','name']]
    print('Recommender: ')
    results=pd.concat([results,five_highest_score],axis=1).sort_values(by='score',ascending=False)
    return results



#TH2: Khi khách hàng nhập chữ trên thanh công cụ tìm kiếm
def text_clean_2(text):
    # Convert text to lowercase
    cleaned_text = text.lower()
    # Remove special characters, punctuation, and symbols
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    #tokenzie
    cleaned_text = word_tokenize(cleaned_text)
    # remove stop_words
    def remove_stopwords(cleaned_text, stop_words):
        cleaned_text_split = cleaned_text.split()
        cleaned_text = [word for word in cleaned_text_split if word not in stop_words]
        return ' '.join(cleaned_text)

    return cleaned_text

def recommender_text(input_text, dictionary, tfidf_model, index, stop_words):
    # Preprocess the input text
    #processed_input = input_text.split()

    # Clean the input text
    cleaned_text = text_clean_2_model(input_text)

    # Convert the processed input into a sparse vector
    kw_vector = dictionary.doc2bow(cleaned_text)

    # Similarity calculation
    sim = index[tfidf_model[kw_vector]]

    # Create a DataFrame with scores
    df_result = pd.DataFrame({'id': range(len(sim)), 'score': sim})
    # Find five highest scores
    five_highest_score = df_result.sort_values(by='score', ascending=False).head(6)

    # Get the corresponding product IDs
    idToList = list(five_highest_score['id'])

    # Filter products based on IDs
    products_find = df_sub[df_sub.index.isin(idToList)]

    # Create the results DataFrame
    results = products_find[['item_id', 'name']].copy()
    results['score'] = five_highest_score['score'].values
    results = results.sort_values(by='score', ascending=False)

    return results

#SAVE FUNCTIONs
#Save the function to a file
joblib.dump(recommender_id, 'recommender_id.joblib')
joblib.dump(recommender_text, 'recommender_text.joblib')
joblib.dump(text_clean_2, 'text_clean_2.joblib') # dùng để làm sạch text
#Load the function from the file

recommender_id_model = joblib.load('recommender_id.joblib')
recommender_text_model = joblib.load('recommender_text.joblib')
text_clean_2_model=joblib.load('text_clean_2.joblib')

input_id=48102821
result_id=recommender_id_model(input_id, dictionary, tfidf, index)


input_text = "Tai nghe Bluetooth"
result_text = recommender_text_model(input_text, dictionary, tfidf, index, stop_words)

# GUI

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