#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:24:56 2020

@author: franklinvelasquezfuentes


Thompson Sampling 
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing Thompson Sampling

N = 10000
d = 10

numbers_of_rewards_1 = [0]*d;
numbers_of_rewards_0 = [0]*d;


ads_selected = []
total_reward = 0

for n in range (0, N):   # number of round
    

    ad_selected = 0       #ad selected to being show to the user
    max_random = 0
    
    for i in range (0,d):  # number of ad
        
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1 , numbers_of_rewards_0[i] + 1)
        
        if random_beta > max_random :
            max_random = random_beta
            ad_selected = i
            
            

    ads_selected.append(ad_selected)
    # watching if the user clicked the ad or not :
    reward = dataset.values[n,ad_selected] 

    if reward == 1 :
        numbers_of_rewards_1[ad_selected] = numbers_of_rewards_1[ad_selected] + 1
    else:
        numbers_of_rewards_0[ad_selected] = numbers_of_rewards_0[0] + 1
    
    # To test performance
    total_reward = total_reward + reward
            
    
    
    
"""

Thompson Sampling  total_reward = 2081, 2685, 2689  (this value will change in each execution, because of the random factor)

Upper Confidence Bound  total_reward = 2178

We can see that the Thompson Sampling Algorithm has a better performance

"""

# Visualising the results in a Histogram ! 


plt.hist(ads_selected)
plt.title("Histograma de Anuncios Selecionados")
plt.xlabel("Anuncio")
plt.ylabel("Número de Selecciones")
plt.show()
    

"""

We can see that we get more reward because we explote more Ad No 5 that is the best one

"""
    
    
    
    
    
    
    
    
    
    
#
