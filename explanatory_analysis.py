# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import os
from cycler import cycler
# plotting results ####################################################################
def plot(var_name,coefs, classes, path = None):
    sb.set_style('whitegrid')
    coefs_var = coefs.filter(regex = '^'+var_name+'_')
    if coefs_var[coefs_var != 0].shape[1]<1:
        return
    # if there is to many levels do not plot them
    if coefs_var.shape[1]>10 and var_name != 'AgeuponOutcome':
        abs_sum = -coefs_var.abs().sum(axis = 0)
        abs_sum_i = abs_sum.argsort()[:10]
        coefs_var = coefs_var.iloc[:,abs_sum_i]
  
    if var_name == 'AgeuponOutcome':
    # reverse additional efect on AgeuponOutcome
        AgeuponOutcomCol = [c for c in coefs_var.columns if not 'nan' in c]
        for i in range(1,len(AgeuponOutcomCol)):
            coefs_var.iloc[:,i] = coefs_var.iloc[:,i-1:i+1].sum(axis = 1)
    coefs_var = np.exp(coefs_var)
    plt.figure()
    plt.gca().set_prop_cycle(cycler('color',['darkgreen', 'black', 'sienna', 'steelblue','grey']))
    labels = [c.split('_')[1] for c in coefs_var.columns]   
    plt.plot(coefs_var.values.T)
    plt.xlim([-0.5,len(labels)-0.5])
    plt.xticks(range(coefs_var.shape[1]), labels, rotation=90)
    plt.legend(classes)
    if path is not None:
        plt.savefig(os.path.join(path,var_name+'.png')) 
