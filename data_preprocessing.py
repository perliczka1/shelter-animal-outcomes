# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
## preprocessing ##############################################################
def preprocess_AgeuponOutcome(val):
    # split numeric values and describtion
    tmp = str(val).split()
    x = 1
    if tmp[0] == 'nan':
        return np.nan
    elif 'year' in tmp[1]:
        x = 365
    elif 'month' in tmp[1]:
        x = 30
    elif 'week' in tmp[1]:
        x = 7
    return int(tmp[0])*x
    
def get_NameLength(name):
    if pd.isnull(name):
       return 0
    return len(name)
    
def get_Sex(val):
    if pd.isnull(val):
       return 'Unknown'
    elif 'Female' in val:
        return 'Female'
    elif 'Male' in val:
        return 'Male'
    return 'Unknown'
def get_Neutered(val):
    if pd.isnull(val):
        return 'Unknown'
    elif 'Intact' in val:
        return 'No'
    elif 'Neutered' in val or 'Spayed' in val:
        return 'Yes'
    return 'Unknown'
## remove text and unnecessary white characters ###############################     
def remove_and_clean(text,to_remove):
    text = text.replace(to_remove,'')
    text = re.sub('\s+',' ', text)
    text = text.strip()
    return text

## function that checks if text is in column col1 and if so assigns label to col2 
# and removes it from col1 ##
def get_and_remove(data,text,col1, label,col2):
    #checking the value
    index = data[col1].apply(lambda x: text in x)
    data.loc[index,col2] = label
    #removing it from data
    data[col1] = data[col1].apply(lambda x: remove_and_clean(x,text))
    
## show distribution of a variable ############################################    
def show_variable_info(col, quiet = False):
    if not quiet:
        notnull = col.notnull()
        missing_cnt = len(col)-sum(notnull)
        print(col.name, "| Missing values: {0} ({1:0.2f} %)".format(missing_cnt,missing_cnt/len(col)))
        if col.dtype == 'float64':        
            sns.distplot(col[notnull])
        elif len(col.unique()) < 16:
            sns.countplot(col[notnull])
        else:
            print('Showing only first 16 levels from', len(col.unique()))
            col_cut = col.value_counts()[:16]
            ax = plt.axes()  
            sns.barplot(x = col_cut.index, y =  col_cut, ax = ax) 
            ax.set_ylabel('Count')   
            
## group values with occurence less or equal x into one level #################
def lq_to_other(data,column, x, is_prediction, path):
    if is_prediction:
        to_keep = np.load(os.path.join(path,column+'.npy'))
    else:
        to_keep_tmp = data[column].value_counts() > x
        to_keep = to_keep_tmp[to_keep_tmp].index
        np.save(os.path.join(path,column),to_keep)
    index0 =  data[column].isin(to_keep)
    data.loc[~index0, column] = 'other'  
          
## data preprocessing #########################################################
def preprocess(data, is_prediction, path):

    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(preprocess_AgeuponOutcome).sort_values()
    ## time
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Year'] = data['DateTime'].apply(lambda x: x.year)
    data['Month'] = data['DateTime'].apply(lambda x: x.month)
    data['Day'] =data['DateTime'].apply(lambda x: x.day)
    data['DayOfWeek']=data['DateTime'].apply(lambda x: x.weekday())
    data['Hour'] = data['DateTime'].apply(lambda x: x.hour)
    ## names
    data['NameLength'] = data['Name'].apply(get_NameLength)
    data['Name'] = data['Name'].apply(lambda x: str(x).lower())
    data['NameFirstLetter'] = data['Name'].apply(lambda x: x[0])
    lq_to_other(data, 'Name', 10, is_prediction, path)
    ## colors
    data['Color'] = data['Color'].apply(lambda x: str(x).lower())
    data['withWhite'] = data['Color'].apply(lambda x: 'white' in x)
    lq_to_other(data, 'Color', 10, is_prediction, path)
    ## breeds
    data.Breed = data.Breed.apply(lambda x: x.lower())
    data['isDomestic'] =  data['Breed'].apply(lambda x: 'domestic' in x)
    data['isMix'] =  data['Breed'].apply(lambda x: 'mix' in x or '/' in x) 
    lq_to_other(data, 'Breed', 10, is_prediction, path)
    ## hairs length
    data['Hair'] = 'Unknown'
    get_and_remove(data,'shorthair','Breed', 'short','Hair')
    get_and_remove(data,'medium hair','Breed', 'medium','Hair')
    get_and_remove(data,'longhair','Breed', 'long','Hair')
    ## removing mix from Breed
    data['Breed'] = data['Breed'].apply(lambda x: remove_and_clean(x, 'mix'))
    ## other
    data['Neutered'] = data['SexuponOutcome'].apply(get_Neutered)
    data['Sex'] = data['SexuponOutcome'].apply(get_Sex)
