# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.linear_model as lr
from sklearn import cross_validation as cv
import os.path
from sklearn.ensemble import RandomForestClassifier
import time
##########################################################################################################
## function plotting scores fr=or each value of C
def plot_scores(scores, path):
    sn.set_style("whitegrid") 
    plt.plot(scores.index, scores['mean'], 'g',marker='o')
    plt.plot(scores.index, scores['max'], 'g--')
    plt.plot(scores.index, scores['min'], 'g--')
    x_min = np.argmin(scores['mean'])
    y_min = scores.loc[x_min,'mean']
    label = 'Min: {0:.1e}, {1:.3f}'.format(x_min,y_min )
    plt.text(x_min,y_min,label)
    if path is not None:
        file_name = '{0:.4e}_{1:.4e}_{2}'.format(min(scores.index),max(scores.index),scores.shape[0])+'.png'
        plt.savefig(os.path.join(path,file_name))    
###########################################################################################################    
## testing different values of C and plotting results
def cv_and_plot(X,y, Cs, path = None):
    scores = pd.DataFrame({'min': np.zeros(len(Cs)),
                           'mean': np.zeros(len(Cs)),
                           'max': np.zeros(len(Cs))},index = Cs)
    for i,C in enumerate(Cs):
        print('model:', i, flush = True)
        model = lr.LogisticRegression(penalty = 'l1', C = C, n_jobs = 1 )
        score = cv.cross_val_score(model, X, y, scoring='log_loss', cv=5, n_jobs=4)
        scores.loc[C] = (-score.max(), -score.mean(), -score.min())    
    plot_scores(scores, path)
    return scores  
############################################################################################################    
## function plotting scores fr=or each value of C
def plot_scores_RF(scores, path):
    sn.set_style("whitegrid") 
    sn.factorplot(x="n_estimators", y="mean", hue="max_depths", col="max_features", data=scores,
                   palette="BuGn_r", col_wrap = 3)
    if path is not None:
        file_name =  'max_features_{:.0f}-{:.0f}_max_depts_{:.0f}-{:.0f}_n_estimators_{:.0f}-{:.0f}'.format(
         scores.max_features.min(),
         scores.max_features.max(),
         scores.max_depths.min(),
         scores.max_depths.max(),
         scores.n_estimators.min(),
         scores.n_estimators.max())+'.png'
        plt.savefig(os.path.join(path,file_name)) 
############################################################################################################
## testing different values of parameters for random forests and plotting results
def cv_and_plot_RF(X,y, max_features,max_depths, n_estimators, path = None):
    n = len(max_features) * len(max_depths) * len(n_estimators)
    print(n)
    scores = pd.DataFrame({'max_features': np.zeros(n),
                           'max_depths': np.zeros(n),
                           'n_estimators': np.zeros(n),
                           'min': np.zeros(n),
                           'mean': np.zeros(n),
                           'max': np.zeros(n)})
    scores = scores.loc[:, ['max_features', 'max_depths', 'n_estimators', 'min','mean','max' ]]
    i = 0                   
    for f in max_features:
        for d in max_depths:
            for e in n_estimators:
                print('max_features: {:.0f}, max_depts: {}, n_estimators: {}'.format(f, d,e))
                model = RandomForestClassifier(n_estimators = e, max_features = f, max_depth = d, n_jobs = 2 )
                score = cv.cross_val_score(model, X, y, scoring='log_loss', cv=5, n_jobs=-1)
                scores.loc[i] = (f, d, e, -score.max(), -score.mean(), -score.min())
                print(scores.ix[i,'mean'])
                i+=1 
    plot_scores_RF(scores, path)            
    return scores  
   