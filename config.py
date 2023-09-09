 # -*- coding: utf-8 -*-

import pandas as pd

STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

df = pd.read_csv('ProductRaw.csv', header=0)

df_clean=pd.read_csv('product_ws.csv', header=0)




