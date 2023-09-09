import streamlit as st
from config  import stop_words, df,df_clean
from Functions import text_clean_2_model,dictionary, tfidf, index,recommender_id_model,recommender_text_model,result_id,result_text,result_id_3,result_id_2,result_text_2,result_text_3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import re
import warnings
import os, sys
import joblib




def run_recommender_app(choice):
    if choice == 'Recommendation_System':
        st.title("Product Recommendation")
    
        # Create a text input for users to enter a product ID
        product_id = st.text_input("Enter a Product ID")

        if product_id:
            if product_id.isdigit():
                product_id = int(product_id)  # Convert to integer
                info_id_search = df[df['item_id'] == product_id][['item_id', 'name']]
                result_id_search = recommender_id_model(product_id, dictionary, tfidf, index)

                if not result_id_search.empty:
                    st.subheader("Your search product:")
                    st.write(info_id_search)
                    st.subheader("Search Results:")
                    st.write(result_id_search)
                else:
                    st.info("No search results available for the selected product ID.")
            else:
                st.warning("Please enter a valid numeric Product ID.")

        st.subheader("How to Use:")
        st.write("Enter a product ID to perform a search and view search results.")

        product_name = st.text_input("Enter the name of a product")

        if product_name:
            st.subheader("Your search query:")
            st.write(product_name)

            # Implement content-based recommendation
            
            content_based_results = recommender_text_model(product_name, dictionary, tfidf,index, stop_words)
            if not content_based_results.empty:
                st.subheader("Recommended Products (Content-Based):")
                st.write(content_based_results)
            else:
                st.info("No content-based recommendations available for the entered product name.")

        st.subheader("How to Use:")
        st.write("Enter the name of a product to find relevant recommendations.")




# Run the Streamlit app
if __name__ == "__main__":
    st.title("Data Science Project 2")
    st.write("## Recommender System")
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='latin-1')
        data.to_csv("ProductRaw.csv", index = False)
    
    menu = ["Business Objective", "Content-based filtering", "Recommendation_System"]
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
        st.write("###### Data before cleaning includes the combination of name, description, brand, and group")
        st.dataframe(df[["item_id", "description"]].head(3))
        st.write("###### Data after cleaning and tokenization with 'Underthesea' ")
        st.dataframe(df_clean[["item_id", "description_ws"]].head(3))  
        st.write("##### 2. Some Pictures of  Products")
        image_urls = [
        'https://salt.tikicdn.com/cache/280x280/ts/product/9e/af/79/39855aad21aaa6ed4459909c7c0aea4e.jpg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/0e/03/6e/1e82e11419bd4aae424b10a5457eb932.jpeg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/28/12/73/3f373c0bd557df40f6c8c404622d16a2.jpg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/0a/07/39/b9050cc9f02a8d01cd45466a9e21b9d4.jpg'
        ]
        st.write("<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)

        # Display each image side by side with reduced width
        for image_url in image_urls:
            st.write(f"<img src='{image_url}' style='width: 45%; margin-right: 5px;'>", unsafe_allow_html=True)
        # Close the container
        st.write("</div>", unsafe_allow_html=True)

        st.write("##### 3. Build Content-based filtering")
        st.write("This is an explanation of how our product recommendation system works.")

        # Step 1: Cleaning and Tokenizing Text
        st.write("Step 1: Cleaning and Tokenizing Text with underthesea")
        st.write("Product descriptions are cleaned and divided into individual words.")

        # Step 2: Creating a Dictionary and Corpus
        st.write("Step 2: Creating a Dictionary and Corpus")
        st.write("We create a dictionary of unique words and count their occurrences in each description.")

        # Step 3: TF-IDF Transformation
        st.write("Step 3: TF-IDF Transformation")
        st.write("We use TF-IDF to identify important words in each description.")

        # Step 4: Calculating Similarities
        st.write("Step 4: Calculating Similarities")
        st.write("We calculate how similar each product is to others using cosine similarity.")

        # Step 5: Finding Recommendations
        st.write("Step 5: Finding Recommendations")
        st.write("For each product, we find the top 5 most similar products as recommendations.")

        # Step 6: Displaying Recommendations
        st.write("Step 6: Displaying Recommendations")
        st.write("We display the top recommended products for each product in the library.")

        # Conclusion
        st.write("This system acts like a helpful librarian, suggesting products based on their descriptions, making your shopping experience better.")

        st.write("##### 4. Show result example")
        result_id_1 = f"The customer clicks on the product with ID 916784, then suggests similar products: {result_id}"
        st.code(result_id_1)
        result_id_2 = f"The customer clicks on the product with ID 48102821, then suggests similar products: {result_id_2}"
        st.code(result_id_2)
        result_id_3 = f"The customer clicks on the product with ID 2860621, then suggests similar products: {result_id_3}"
        st.code(result_id_3)
        result_text_1 = f"When customers search for Bluetooth headphones, recommend products such as: {result_text}"
        st.code(result_text_1)
        result_text_2 = f"When customers search for lOA , recommend products such as: {result_text_2}"
        st.code(result_text_2)
        result_text_3 = f"When customers search for PIN SAC, recommend products such as: {result_text_3}"
        st.code(result_text_3)

    
    elif choice == 'Recommendation_System':
        run_recommender_app(choice)


        

