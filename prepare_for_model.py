# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
## convert numerical column to ordered category ################################
def num_to_ord(col):
    categories = np.sort(col.unique())
    f = np.vectorize(lambda x: str(x).replace('.0',''))
    categories =  f(categories)
    col = f(col)
    col = pd.Categorical(col, categories = list(categories), ordered = True) 
    return col 
    
## prepare data for modelling - mostly for linear regression ###################
def prepare(data, model_dummies_names = None, drop_first = False):
    columns_cat = ['Name', 'AnimalType', 'SexuponOutcome','Breed', 'Color',
                   'NameLength','Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 
                   'Sex','Neutered', 'withWhite', 'isDomestic', 'isMix', 'Hair']
    #columns_ord = ['AgeuponOutcome']
    data2 = data[columns_cat].apply(lambda x: x.astype('category'),axis = 0)
    data2['AgeuponOutcome'] = num_to_ord(data['AgeuponOutcome'])
   
    # get dummies variables
    data2 = pd.get_dummies(data2, prefix = data2.columns, sparse = True, drop_first = drop_first)
    if model_dummies_names is not None:
        columns_to_add = set(model_dummies_names).difference(set(data2.columns))
        for c in columns_to_add:
            data2[c] = 0
        data2 = data2.loc[:,model_dummies_names]
    #code AgeuponOutcome
    AgeuponOutcomCol = [c for c in data2.columns if 'AgeuponOutcom' in c and not 'nan' in c]
    i0 = data2.columns.get_loc(AgeuponOutcomCol[0])
    for i,col in enumerate(AgeuponOutcomCol):
       data2.loc[:,col] = data2.iloc[:,i0+i:].max(axis = 1)
    
    return data2  
    
## additional processing for random forest ####################################
def prepare_2_RF(data2,data):
    to_drop_for_rf = [c for c in data2.columns if ('AgeuponOutcome' in c or 
                                                    'NameLength' in c or 
                                                    'Year' in c or 
                                                    'Month' in c or 
                                                    'Day' in c or 
                                                    'Hour' in c or
                                                    'Name' in c)] # removes name namefirstletter and namelength - but it's ok
    data2.drop(to_drop_for_rf, inplace = True, axis = 1)
    for c in ['AgeuponOutcome', 'NameLength', 'Year', 'Month', 'Day', 'Hour']:
        data2[c] = data[c]
    data2.fillna(-1,inplace = True)