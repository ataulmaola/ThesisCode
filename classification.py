#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:16:59 2019

@author: ataul
"""
#Import all fundamental  neccessary pakages
import numpy as np
import pandas as pd

#Import preprocessor modules 
from preprocessor import raw_to_processed,countvector,text_data ,tfidfvector,tf_kmeans,co_kmeans
from sklearn.model_selection import train_test_split

#Import classification  models  
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC

#import all test modules
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#call preprecessor functons 
raw_data=text_data()
str_lists =list(filter(None, raw_data))     
processed_corpus=raw_to_processed(*str_lists)

#define all Independant variables
X_c=pd.DataFrame(countvector(processed_corpus))
X_tf=pd.DataFrame(tfidfvector(processed_corpus))

#define all dependent variable with kmeans clustering
y_kmeansCo=co_kmeans(X_c)
y_kmeansTf=tf_kmeans(X_tf)


#Split dataset  into train and test data set

#This is for count vectorzizer
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_kmeansCo, test_size = 0.2, random_state = 0)

#this is for tfidf vectorizer
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tf, y_kmeansTf, test_size = 0.2, random_state = 0)


#define Gaussian and Multinomial naive bayes classifiers
 
classifierG = GaussianNB()
classifierM=MultinomialNB()

#This is for count vectorzizer and Gaussian classifier
classifierG.fit(X_train_c, y_train_c)
y_pred_c_g = classifierG.predict(X_test_c)


#This is for tfidf vectorzizer and Gaussian classifier
classifierG.fit(X_train_tf, y_train_tf) #fit
y_pred_tf_g = classifierG.predict(X_test_tf) #predict



#This is for count vectorzizer and Multinomial classifier
classifierM.fit(X_train_c, y_train_c)
y_pred_c_m = classifierM.predict(X_test_c)


#This is for tfidf vectorzizer and Multinomial classifier
classifierM.fit(X_train_tf, y_train_tf)
y_pred_tf_m = classifierM.predict(X_test_tf)



#define  svmclassifier for ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
svm_classifier = SVC(kernel = 'linear', random_state = 0)

#This is for count vectorzizer 
svm_classifier.fit(X_train_c, y_train_c) #fit
y_pred_c = svm_classifier.predict(X_test_c) #predict

#This is for tfidf vectorzizer 
svm_classifier.fit(X_train_tf, y_train_tf)  #fit
y_pred_t = svm_classifier.predict(X_test_tf) #predict



#cm = confusion_matrix(y_test_c, y_pred_c)
#acc=accuracy_score(y_test_c, y_pred_c)
#crpt=classification_report(y_test_c, y_pred_c)