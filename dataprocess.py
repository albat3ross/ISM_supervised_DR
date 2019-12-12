# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:10:41 2019

@author: zhouh
"""

import pandas as pd
import numpy as np
import math
import csv
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

get_name = 'iris.csv'
gen_name = 'iris_raw'

dp = np.genfromtxt('./raw data/'+get_name,delimiter=',')
test_size = 20
xtrain  = pd.DataFrame(dp[test_size:    ,:-1])
xtest   = pd.DataFrame(dp[:test_size    ,:-1])
ytrain  = pd.DataFrame(dp[test_size:    ,-1 ])
ytest   = pd.DataFrame(dp[:test_size    ,-1 ])
xtrain  .to_csv('./raw data/'+gen_name+'.csv'            ,sep=',',index=False,header=False)
xtest   .to_csv('./raw data/'+gen_name+'_test.csv'       ,sep=',',index=False,header=False)
ytrain  .to_csv('./raw data/'+gen_name+'_label.csv'      ,sep=',',index=False,header=False)
ytest   .to_csv('./raw data/'+gen_name+'_label_test.csv' ,sep=',',index=False,header=False)









def acc(y_test,y_pred):
    return "-----Accuracy: "+str((y_test==y_pred).sum()/len(y_test))

def runsvmpred(data_name):
    X_train = np.genfromtxt('./raw data/'+data_name+'.csv'           ,delimiter=',')
    y_train = np.genfromtxt('./raw data/'+data_name+'_label.csv'     ,delimiter=',')
    X_test  = np.genfromtxt('./raw data/'+data_name+'_test.csv'      ,delimiter=',')
    y_test  = np.genfromtxt('./raw data/'+data_name+'_label_test.csv',delimiter=',')
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    #print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(acc(y_test,y_pred))

runsvmpred('iris_raw')