#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:19:54 2020

@author: franklinvelasquezfuentes


Herarchical Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the mall dataset with pandas

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

# Using the dendogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch

# linkage is the algorithm of hierarchy, 'ward' is a method that try to minimize the variance between clusters
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian distances")
plt.show()




""" 
    Based on the picture we can see that the optimal number of clusters is 5
"""

# Fitting hierarchical clustering to the mall dataset. We will use agglomerative clustering

from sklearn.cluster import AgglomerativeClustering

ch = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# this array show to wich number of cluster belongs each row in X
y_ch = ch.fit_predict(X)




# Visualising the clusters


# we want first and second column of X where the cluster is equal to number 0
plt.scatter(X[y_ch==0,0],X[y_ch==0,1], s = 100 , c = 'red' , label = 'Cluster 1 -> Careful Clients')
# we do the same for the rest of clusters
plt.scatter(X[y_ch==1,0],X[y_ch==1,1], s = 100 , c = 'blue' , label = 'Cluster 2 -> Standard')
plt.scatter(X[y_ch==2,0],X[y_ch==2,1], s = 100 , c = 'green' , label = 'Cluster 3 -> Target')
plt.scatter(X[y_ch==3,0],X[y_ch==3,1], s = 100 , c = 'cyan' , label = 'Cluster 4  -> Careless')
plt.scatter(X[y_ch==4,0],X[y_ch==4,1], s = 100 , c = 'magenta' , label = 'Cluster 5 -> Sensible')


plt.title("Clusters de Clientes - (Herarchical Clustering)")
plt.xlabel("Ingreso Anual (k$) ")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()
    
    
























