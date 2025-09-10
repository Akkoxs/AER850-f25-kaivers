# -*- coding: utf-8 -*-
""" Created on Wed Sep 10 10:15:32 2025
@author: kai-s 
Lecture 3"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder #transform data to binary one-hot encoding
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
 
""""Lec 2"""

#import data
data = pd.read_csv("data/housing.csv") #read in data, returns dataframe obj.

print(data.head()) #print few entries for all columns
print(data.columns) #print out column names
print(data['ocean_proximity']) #print the data of this column
data['ocean_proximity'].hist() #create histogram of ocean_prox
data.hist()

"""Lec 3 - binary one hot encoding"""
enc = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']]) #'learns' the data, you feed encoder data so it knows
encoded_data = enc.transform(data[['ocean_proximity']]) #applies one hot encoding 
category_names = enc.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns=category_names) #new data
data = pd.concat([data, encoded_data_df], axis=1) #add new columns onto data.csv
data = data.drop(columns = 'ocean_proximity') #delete ocean_prox, no longer need

#what we just did was basically transform the ocean_proximity variable into 3 separate
#categories which are all true or false, instead of a single variable with a non-numerical string.

"""Data splitting"""
#basic method for data splitting
#X_train, X_test, y_train, y_test = tran_test_split(X, y, test_size = 0.2, random_state = 42)

#use of stratified samplng strongly recommended
#the strata are the categories possible for ocean_proximity
#from each of the strata, we take a cut of the data for test & for training randomly 
data["income_categories"] = pd.cut(data["median_income"], #pd.cut used for binning median_income into categories
                                bins =[0, 2, 4, 6, np.inf],
                                labels=[1, 2, 3, 4])
#a random shuffling within a given strata to grab data, random_state=42 preserves a single 'random' scenario
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(data, data["income_categories"]):
    #reset index to 0 if you want a data point to be your first point in a different data set
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["income_categories"], axis = 1)
strat_data_test = strat_data_test.drop(columns=["income_categories"], axis = 1)


print(data.shape)
print(strat_data_train.shape)
print(strat_data_test.shape)
    

"""Correlation coeff"""
#quickly w 2 lines of code, we can viz the correlation b/w all of the variables 
corr_matrix = strat_data_train.corr()
sns.heatmap(np.abs(corr_matrix))

# we need to remove linearly correlated variables, why? Reza will say next week.
# CHECK GITHUB
# How to get this as sep plot?

