#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:37:22 2020

@author: franklinvelasquezfuentes


If someone buy some Cereal, probably he will buy Milk, if we put them together, probably they are going 
to buy two things.

This store is traing to improve it sales.
"""


# Data Preprocessing
import numpy as np # Mathematical tools
import matplotlib.pyplot as plt
import pandas as  pd # Import and Manage Data Sets


dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)


"""
We do not need a matrix, we need a list of list.

A list wich each item is a list of items bought.
=
A list of transactions, each transaction has a list of products bought

So we need to create that data structure
"""

transactions = []

# i -> rows
# j -> columns
# IMPORTANT : range(x,y) x-> included , y-> excluted

 
for i in range(0,7501):

    transactions.append( [ str(dataset.values[i,j]) for j in range(0,20) ])
    


# Training Apriori on the dataset
    

from apyori import apriori
    

# The rest of arguments of apriori will depend of our business model
# The minimun support, confidence and lift is not going to be the same if in the
# the data set we have 100, or 1000 transactions. 
# minimum_length is not in the Info, but is necessary to set the number of items that we are trying to associate

"""
 min_support :  we'll take as creteria that we are trying to find the items that are buy at least 3 times at day
 total of transactions -> 7500
 3 (day) *7 (week) =   21  / 7500 = 0.0028 = 0.003
 

 min_confidence : if we put a high value, we'll get some obvious relations, not because two objects are related,
 just because both of them are one of the most selling items, not for the correct reason.
 
 
 min_lift : we want lift higher than 3, the normal value is 3-6
 
 The model is experimental, we have to experiment many times to get better values for parameters
 
"""

rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3, min_legth=2 )


results = list(rules)










































