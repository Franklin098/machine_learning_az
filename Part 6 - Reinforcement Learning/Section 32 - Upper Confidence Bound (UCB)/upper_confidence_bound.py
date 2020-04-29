#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:21:02 2020

@author: franklinvelasquezfuentes


Reinforcement Learning

Upper Confidence Bound
"""
    
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

"""
Notes by Franklin:
    
# We want to quickly find wich is the best Ad , Click Through Ratio (CTR)
# In real life we do not have any data, so here we are going to image that we do not have any dataset,
    this dataset is just for simulation of clicks, and only God knows the results.
    
Each time a user connects we are going to show him one Ad, and we are goint too see his response, 
clicked - 1 , not clicked - 0, but we are going to use a strategy to show them a reward, base on the
past rounds results => Upper confidence bound. Reinforcment Learning = In Real Time Learning.


If we  show the Ads ramdomly we get this results: total_reward = 1246 in the future we will compare it
In the histogram of this random method all heights are almos the same, we do not see any Ad winner.
"""


# Implementing UCB from Scratch, without any package.

N = 10000
d = 10

#  Ni(n)  , vector of size "d" with only 0s
numbers_of_selections = [0] * d  
# Ri(n) 
sums_of_rewards = [0] * d        


# huge vector of the ad that was selected at each round, at the end N (10000) ads selected
ads_selected = []

total_reward = 0

for n in range (0, N):   # number of round
    
    max_upper_bound = 0
    ad_selected = 0       #ad selected to being show to the user
    
    for i in range (0,d):  # number of ad
        
        if (numbers_of_selections [i] > 0): # to use the algorithm after the first 10 rounds
             
            avarage_reward = sums_of_rewards[i] / numbers_of_selections[i]
            
            # n+1 -> indexes i python starts at 0
            delta = math.sqrt((3/2)* math.log(n+1) / numbers_of_selections[i])
            
            # we only need the upper confidence bound, not the lower
            upper_bound = avarage_reward  + delta
        
        else : # to use the algorithm after the first 10 rounds
            upper_bound = 1e400  # we give a very large value 10^4000 
            
            
        if upper_bound > max_upper_bound : 
            max_upper_bound = upper_bound
            ad_selected = i
           
        
    numbers_of_selections[ad_selected] = numbers_of_selections[ad_selected] + 1
    ads_selected.append(ad_selected)
    # watching if the user clicked the ad or not :
    reward = dataset.values[n,ad_selected] 
    sums_of_rewards[ad_selected] = sums_of_rewards[ad_selected] + reward
    total_reward = total_reward + reward
            
        
"""
Deal with initial conditions, what happends at round 0 ? , at the beginning with the first 10 rounds
we do not have enough information about their reward, wheather they earnd reward = 1 or reward = 0, if
they were selected or not, symply because they have not beeing selected yet.

We have too choose wich ads will be selected at 10 first rounds, we will use the algorithm as soon
as we have the information of the 10 ads,  we will start using our algorithm.

Simply, we are going to choose ad number i , at the first i rounds.

So, at round 11 the number of selectios will be one for each of the 10 ads. To do that we write:
    
    if (numbers_of_selections [i] > 0):
        .......
    else : 
          upper_bound = 1e400  # we give a very large value 10^4000 
        

"""
        

"""
Total Reward = 2178

It is the double of just doing it randomly  (1246)

In a casino we would win the double of money !

In the ads_selected vector we can see that in the 10 first round we just selected each i ad.

At the last round, we can see that the (Ad at index number 4 = Add No 5)  is the best Ad.

"""



# Visualising the results in a Histogram ! 



plt.hist(ads_selected)
plt.title("Histograma de Anuncios Selecionados")
plt.xlabel("Anuncio")
plt.ylabel("Número de Selecciones")
plt.show()

























