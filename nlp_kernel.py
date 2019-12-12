# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:09:39 2019

@author: sunge
"""
from sklearn.pipeline import Pipeline
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from tqdm import tqdm
from numpy import transpose as T
from scipy.stats import stats
from scipy.stats import mode
from sklearn.model_selection import cross_validate

from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize         

import re


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import models
from sklearn.svm import LinearSVC
def encode_subreddit(argument):
    switch = {         
            "europe":0,
            "canada":1,                
            }
    return switch.get(argument,2)
def averageAcc(cv_results,fold):
    average = 0
    for number in cv_results:
        average+=number
    average /= fold   
    print("Cross-validate",fold,"folds accuracy is:",average)
    return average

def accuracy(predicted,true_outcome,num):
    accuracy = 0
    index = 0
    for result in predicted:
        if result == true_outcome[index]:
            accuracy+=1
        index+=1
    print("-----Accuracy:", accuracy/num)
    
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in re.split('\d|\\\|\s|[,.;:?!]|[/()]|\*',articles)]
    
start_time = time.time()
#load file
#------------------------------------------------------------------------------
canada_df = pd.read_csv(r'../data/parsed_data/canada.csv')
europe_df = pd.read_csv(r'../data/parsed_data/europe.csv')
training_df = canada_df.append(europe_df)
finish_time = time.time()
print("-----File Loaded in {} sec".format(finish_time - start_time))
encode = []
for subreddit in training_df['subreddits']:
    encode.append(encode_subreddit(subreddit))
training_df['subreddit_encoding'] = encode
#training_df.to_csv(r'../data/encoded_reddit_train.csv',',')

# 6.1 SVM
#------------------------------------------------------------------------------
svm_train_clf= Pipeline([
        ('vect',CountVectorizer(binary = True)),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC(C = 0.2)),
        ])


#Cross-validation
#------------------------------------------------------------------------------ 
svm_cv_results = cross_validate(svm_train_clf,training_df['comments'],training_df['subreddit_encoding'],cv = 7)
sorted(svm_cv_results.keys())
svm_cv_results['fit_time']
svm_cv_results['test_score']
print("SVM")
averageAcc(svm_cv_results['test_score'],7)  
    
    
    
    
    
    
    
    
    