# -*- coding: utf-8 -*-
import os
os.chdir('D:\Kaggle\Shelter Animal Outcomes')
import pandas as pd
import numpy as np
from importlib import reload
import data_preprocessing as dp
import prepare_for_model as pfm
import other_functions as of
import explanatory_analysis as ea
import sklearn.linear_model as lr
from sklearn.metrics import log_loss, confusion_matrix
from sklearn import cross_validation as cv
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
reload(of)
reload(pfm)
reload(ea)
reload(dp)
data = pd.read_csv('train.csv')
dp.preprocess(data, is_prediction = False, path = 'values_to_keep')
data2 = pfm.prepare(data,drop_first = True)
y = data.OutcomeType
## some columns should be numeric instead of dummies
## because they have natural numeric interpretation and I used a trick to code them in logistic regression
pfm.prepare_2_RF(data2,data)

## prepare general model ######################################################
# data 
X_train, X_test, y_train, y_test = cv.train_test_split( 
    data2,y,test_size=0.1, random_state=13)

## test if separatemodels for dogs/cats will be better
X_train_dog, X_test_dog, y_train_dog, y_test_dog = cv.train_test_split( 
    data2[data2.AnimalType_Dog == 1],y[data2.AnimalType_Dog == 1],test_size=0.1, random_state=13)

X_train_cat, X_test_cat, y_train_cat, y_test_cat = cv.train_test_split( 
    data2[data2.AnimalType_Dog == 0],y[data2.AnimalType_Dog == 0],test_size=0.1, random_state=13)                                
## models with the best parameters - RF was the best in cross valdation 
## prediction of test data
dog_RF = RandomForestClassifier(n_estimators = 500, max_features = 150, max_depth = 11, n_jobs = -1 )
cat_RF = RandomForestClassifier(n_estimators = 500, max_features = 150, max_depth = 11, n_jobs = -1 )
RF = RandomForestClassifier(n_estimators = 500, max_features = 120, max_depth = 13, n_jobs = -1 )

dog_RF.fit(X_train_dog, y_train_dog)
cat_RF.fit(X_train_cat, y_train_cat)
RF.fit(X_train,y_train)
test_pro_cat = cat_RF.predict_proba(X_test_cat)
test_pro_dog = dog_RF.predict_proba(X_test_dog)
test_pro = RF.predict_proba(X_test)
lc = log_loss(y_test_cat,test_pro_cat)
ld = log_loss(y_test_dog,test_pro_dog)
(lc * y_test_cat.shape[0] + ld * y_test.shape[0])/(y_test_cat.shape[0] +  y_test.shape[0])
## RF is indeed the best:
l - log_loss(y_test,test_pro)
############################################################################################
eval_data = pd.read_csv('test.csv')
dp.preprocess(eval_data, is_prediction = True, path = 'values_to_keep')
eval_data2 = pfm.prepare(eval_data, model_dummies_names = data2.columns,drop_first = False )
pfm.prepare_2_RF(eval_data2,data)
eval_pro2 = pd.DataFrame(RF.predict_proba(eval_data2), columns = RF.classes_)
eval_pro.insert(0,'ID',eval_data.loc[:, 'ID'].values)
eval_pro.to_csv('submissionRF.csv', index = False)
