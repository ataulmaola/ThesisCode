#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:20:30 2019

@author: ataul
"""

# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from preprocessor import raw_to_processed,countvector,text_data ,tfidfvector,tf_kmeans,co_kmeans


raw_data=text_data()
str_lists =list(filter(None, raw_data))     
processed_corpus=raw_to_processed(*str_lists)

#define all Independant variables
X_c=pd.DataFrame(countvector(processed_corpus))
X_tf=pd.DataFrame(tfidfvector(processed_corpus))


#define all dependent variable with kmeans clustering
y_kmeansCo=co_kmeans(X_c)
y_kmeansTf=tf_kmeans(X_tf)

np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
#This is for count vectorzizer
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_kmeansCo, test_size = 0.2, random_state = 0)

#this is for tfidf vectorizer
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tf, y_kmeansTf, test_size = 0.2, random_state = 0)
# fix random seed for reproducibility


# truncate and pad input sequences
max_review_length = 10000



print(X_train_co.shape[1])

X_train = sequence.pad_sequences(X_train_co, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test_co, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train_c, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test_c, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))