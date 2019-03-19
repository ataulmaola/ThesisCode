#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:44:33 2019

@author: ataul
"""
#Import all fundamental  neccessary pakages 
import os
import re
import string
import pandas as pd
import numpy as np

#Import all nltk pakages for text preprocessing 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize

#Import all fundamental  sklearn modules for text vectorization  
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#Import kmeans clustering module for text clustering 
from sklearn.cluster import KMeans



#Define all dataset paths and list all data set

path='/home/ataul/Anas/Thesis/Final/en_dataset/'
datalist= os.listdir('/home/ataul/Anas/Thesis/Final/en_dataset')

#define all variables and objects
stop_words = set(stopwords.words('english')) 
exclude = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
vectorizer_tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
vectorizer_count = CountVectorizer(stop_words='english',ngram_range=(1,2))
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)


#define clean funtion for single documents
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y


#define function for read all text data into list of strings

def text_data():
    raw_corpus=[]
    for i in datalist:
       file = open(os.path.join(path+ i), 'r')
       raw_corpus.append(file.read())
    return raw_corpus

       
    
#define function clean all text documnts
def raw_to_processed(*str_list):
    processed_copus=[]
    for text in str_list:
        text=str(text)
        cleaned_test = clean(text)
        cleaned = ' '.join(cleaned_test)
        cleaned = re.sub(r"\d+","",cleaned)
        processed_copus.append(cleaned)
    return processed_copus


#define count vectorizer funtion 
def countvector(processed):
    X_count = vectorizer_count.fit_transform(processed)
    X_count=X_count.toarray()
    return X_count

#define tfidf vectorizer funtion 
def tfidfvector(processed):
    X_tfidf = vectorizer_tfidf.fit_transform(processed)
    X_tfidf=X_tfidf.toarray()
    return X_tfidf




#define kmeans cluster for tfidf vectorizer  
def tf_kmeans(X_tf):
    y_kmeansTf = kmeans.fit_predict(X_tf)
    return y_kmeansTf

#define kmeans cluster for count vectorizer 

def co_kmeans(X_c):
    y_kmeansCo = kmeans.fit_predict(X_c)
    return y_kmeansCo






#for testing direct sentiment analysis 
#from textblob import TextBlob

#polarity=[]
#for text in processed_copus:
#    blob=TextBlob(text)
#    pl=blob.sentiment.polarity
#    if (pl > 0.1 and pl < 1):
#        polarity.append(1)
#    if ( pl > -0.1 and pl < 0.1):
#        polarity.append(0)
#    elif(pl > -1 and pl < -.1):
#        polarity.append(2)
#        
#        

        


