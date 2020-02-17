#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:54:50 2020

@author: franklinvelasquezfuentes
"""

#%reset -f

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset with pandas

dataset = pd.read_csv("Mall_Customers.csv")

# X -> anual income, spending score
X = dataset.iloc[:,[3,4]].values 

# We don't know what we are looking for, so we have to find out the optimal number of clusters

# Using the Elbow method to find the optimal number of clusters


from sklearn.cluster import KMeans

# 10 diferentres Sum of Squares for 10 diferent numer of clusters

wcss = []

# 1 to 10
for i in range (1,11):
    
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=10,random_state= 0)
    
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
    

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
    

# NOw we know that the optimal number is 5
    


# Applying k-means to the mail dataset
    
    
    
    
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300, n_init=10,random_state= 0)

# Returns for each point, what cluster it belongs to 
y_kmeans = kmeans.fit_predict(X)
    

#Plot a chart with our 5 clusters represented


# we want first and second column of X where the cluster is equal to number 0
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s = 100 , c = 'red' , label = 'Cluster 1 -> Careful Clients')

# we do the same for the rest of clusters
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s = 100 , c = 'blue' , label = 'Cluster 2 -> Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s = 100 , c = 'green' , label = 'Cluster 3 -> Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s = 100 , c = 'cyan' , label = 'Cluster 4  -> Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s = 100 , c = 'magenta' , label = 'Cluster 5 -> Sensible')


#plotting the centroids

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300 , c = 'yellow' , label = 'Centroids')

plt.title("Clusters de Clientes")
plt.xlabel("Ingreso Anual (k$) ")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    