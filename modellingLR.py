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

reload(of)
reload(pfm)
reload(ea)
reload(dp)
data = pd.read_csv('train.csv')
dp.preprocess(data, is_prediction = False, path = 'values_to_keep')
data2 = pfm.prepare(data)
y = data.OutcomeType
timeInc = True
model_col = [c for c in data2.columns if not 'AnimalType_' in c \
            and (timeInc or (not 'Year_' in c and 
                             not 'Month_' in c and 
                             not 'Day_' in c and 
                             not 'DayOfWeek_' in c and
                             not 'Hour_' in c)) ]
## prepare model for dogs ###########################################
X_train, X_test, y_train, y_test = cv.train_test_split( 
    data2.loc[data2.AnimalType_Dog == 1,model_col], 
    y[data2.AnimalType_Dog == 1], 
    test_size=0.0, random_state=13)
# test of C parameters
Cs = np.power(10,np.linspace(-10, 3, 10))
scores = of.cv_and_plot(X_train,y_train,Cs,'plots' )
# test parameters closer to minimum
Cs_2 = np.linspace(0.03, 2, 10)
scores_2 = of.cv_and_plot(X_train,y_train,Cs_2,'plots' )
# fitting model to whole training dataset
model_dog = lr.LogisticRegression(penalty = 'l1', C = 0.25, n_jobs = 4)
model_dog.fit(X_train,y_train)
# validation model
test_pre = model_dog.predict(X_test)
test_pro = model_dog.predict_proba(X_test)
confusion_matrix(y_test, test_pre)
ld = log_loss(y_test,test_pro)
# plotting coefficients
coefs = pd.DataFrame(model_dog.coef_, columns = X_train.columns) 
for c in data.columns:
    ea.plot(c,coefs, model_dog.classes_, 'results')

## prepare model for cats #############################################
# test of C parameters
X_train_cat, X_test_cat, y_train_cat, y_test_cat = cv.train_test_split( 
    data2.loc[data2.AnimalType_Dog == 0,model_col], 
    y[data2.AnimalType_Dog == 0], 
    test_size=0, random_state=13)
# first test of parameters    
Cs = np.power(10,np.linspace(-10, 3, 10))
scores = of.cv_and_plot(X_train_cat,y_train_cat,Cs,'plots' )
# second test of parameters
Cs_2 = np.linspace(0.05, 1, 10)
scores_2 = of.cv_and_plot(X_train_cat,y_train_cat,Cs_2,'plots' )

Cs_3 = np.linspace(0.10, 0.5, 10)
scores_3 = of.cv_and_plot(X_train_cat,y_train_cat,Cs_3,'plots')
# model
model_cat = lr.LogisticRegression(penalty = 'l1', C = 0.25, n_jobs = 4)
model_cat.fit(X_train_cat,y_train_cat)
# coefficients
coefs_cat = pd.DataFrame(model_cat.coef_, columns = X_train_cat.columns) 
for c in data.columns:
    ea.plot(c,coefs_cat, model_cat.classes_, 'results_cat')
    
test_pre_cat = model_cat.predict(X_test_cat)
test_pro_cat = model_cat.predict_proba(X_test_cat)
confusion_matrix(y_test_cat, test_pre_cat)
lc = log_loss(y_test_cat,test_pro_cat)
(lc * y_test_cat.shape[0] + ld * y_test.shape[0])/(y_test_cat.shape[0] +  y_test.shape[0])

# kaggle test data
eval_data = pd.read_csv('test.csv')
dp.preprocess(eval_data, is_prediction = True, path = 'values_to_keep')
eval_data2 = pfm.prepare(eval_data, model_dummies_names = data2.columns)
eval_pro_dog = model_dog.predict_proba(eval_data2.loc[eval_data2.AnimalType_Dog == 1, model_col])
eval_pro_cat = model_cat.predict_proba(eval_data2.loc[eval_data2.AnimalType_Dog == 0, model_col])
eval_pro_dog = pd.DataFrame(eval_pro_dog, columns = model_dog.classes_)
eval_pro_cat = pd.DataFrame(eval_pro_cat, columns = model_cat.classes_)

eval_pro_dog.insert(0,'ID',eval_data.loc[eval_data2.AnimalType_Dog == 1, 'ID'].values)
eval_pro_cat.insert(0,'ID',eval_data.loc[eval_data2.AnimalType_Dog == 0, 'ID'].values)

result = pd.concat([eval_pro_dog, eval_pro_cat]).sort_values(by = 'ID')
result.to_csv('submission1.csv', index = False)