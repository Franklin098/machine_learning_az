#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:12:39 2020

@author: franklinvelasquezfuentes

Natural Language Processing

building a model to predict if a review of a restaurant is positive or negative, but we can apply it to :
    - Get the genre of a book
    - Analyse articles and predict its category
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
CSV vs TSV, which one we should choose?
TSV is better, because probably the text is going to have some comas, and also double quotes,
but probably if the user press Tab button, it will change to other graphic component.

This dataset was taken from a paper.
"""

#Importing the dataset
# quoting = 3 -> ingnore double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


# Cleaning the texts
"""
We are going to take all the relevant words, not words like: "the,and,puntuation"
We are going to take the root of some words, doing Stemming: loved,loving -> love
Convert all to lower text
Bag of words with tokenization
"""

"""

# removing numbers, puntuations and replacing them with an empty space, we use only letters
import re
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
#to lower case
review = review.lower()

# remove non significan words : 'that, that, then in' all the articles, all the prepositions.
# all te words that is not going to help us to know if the review is good or bad

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# we need to split the string to get and array and loop to each word and compare it with stopwords

review = review.split()

#shurcut to update a list with a for loop. It is faster to search in a set that in a list
#review = [word for word in review if not word in set(stopwords.words('english'))]

# Stemming = skeeping only the root of each word

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

review = ' '.join(review)

"""



import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# List of reviews. Corpus : collection of text
corpus =[]

# Cleaning all reviews, one by one

for i in range (0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus.append(review)
    
    

# Creating the Bag of Words model
"""
We are going to take all the words of all the reviews and create a matrix, a table
columns -> all words
rows -> all reviews
cells -> the number of times each word appears in each review

sparce matrix, with a lot of ceros, with sparce -> for that we use  X = CountVectorizer.fit()

At the end we'll predict if a review is positive or negative, and the model needs to be train
wich all the reviews that we already know if are positive or negatives, the model will create relations
between each word and the result if is positive or negative, we are doing a -> Binary Classification !

With the matrix we have to reduce the size of our sparce matrix, reduce the number of variables
in the classification model.

# max_features = 1500 only keep the 1500 most frequent words (before 1565)
# we can use also (leater in the course) reduction of dimentions
"""
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(max_features=1500) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# We are going to copy and paste our models in Part3 - Classification Models !
""" 
We can test each model and test the confusion metrix, and select one.
Naibe Bayes and Random Forest are the most used models in NLP
"""




# Splitting the Dataset into the Trainging Set and Test Set --------------------------

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0 )  

# Fitting classif¡er Regression
# Create your classifier here ! :


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)



"""
from sklearn.ensemble import RandomForestClassifier
# Always try to detect Overfeating !
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
# criterion -> etropy. We will have 10 trees voting
classifier.fit(X_train, Y_train)
"""

"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
"""

"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train, Y_train)
"""

"""
from sklearn.svm import SVC
classifier = SVC(kernel="rbf",random_state=0)
classifier.fit(X_train, Y_train)
"""





# Predicting the test set results

Y_pred = classifier.predict(X_test)

"""
Evaluating Performance of Predictions with Confusion Matrix
"""

from sklearn.metrics import confusion_matrix

# Params :
# y_true -> values in real life
# y_pred -> vector of predictions

cm = confusion_matrix(Y_test,Y_pred)

"""
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 * Precision * Recall / (Precision + Recall)
"""

accuracy = (cm[0,0] + cm[1,1]) / Y_test.size
precision = cm[1,1] / ( cm[1,1] + cm[0,1] )
recall = cm[1,1] / (cm[1,1] + cm[1,0])
F1_score = 2*precision* recall / (precision + recall)

print(accuracy," , ",precision, " , " , recall, " , " , F1_score)


"""

Model                   accuracy    ,    precision    ,    recall    ,    F1_score

Naive Bayes             0.73  ,  0.6842105263157895  ,  0.883495145631068  ,  0.7711864406779663

Random Forest           0.72  ,  0.8507462686567164  ,  0.5533980582524272  ,  0.6705882352941177

Logistic Regresion      0.71  ,  0.7586206896551724  ,  0.6407766990291263  ,  0.6947368421052632

K-Nearest Neighbors     0.61  ,  0.676056338028169  ,  0.46601941747572817  ,  0.5517241379310345

Support Vector Machine  0.485  ,  nan  ,  0.0  ,  nan


"""
















