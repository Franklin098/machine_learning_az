#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:26:36 2020

@author: Franklin Velásquez

Convolutional Neural Networks CNN

Image classification problem: Cats and Dogs

But we can change the images, and train a CNN to predict if a image contains a Tummor or not,
it can help to acelerate Cancer research.
"""

# PART 1 - Data Preprocessing = was done manually

"""
How can we prepare a data set where our input are not a tables, insted are images, we are analysing pixels
in 3 different matrixes.

A efficient way to import our images comes with Keras. We can a special and simple structure.

10,000 images in total , 8000 training set, 2000 test set 
"""

# PART 2 - Building the CNN

""""
Importing the Keras libraries and packages
Each package correspond to each our steps in CNN
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten


"""
Initialising the CNN
"""
classifier = Sequential()

"""
Adding the layers, our first layer is going to be the Convolutional Layer
"""
# STEEP 1 - Convolution

"""
We do a convolution between our input image and a feature detector and as result we get a feature map,
it is necessary to create many feature maps


Convolution2D( number_of_feature_detectors,  number_of_rows_of_filter , number_of_columns_of_filter , input_shape [], activation_function)

Most of CNN starts with 32 filters, and because we are working at CPU.

32 filtesrs of 3x3, our CNN will have 32 features maps

input_shape: to have all the images in the same format, order at Tensorflow: ( 64 pixels, 64 pixels, number_of_chanels)

The rectifier funtion help us to break linearity, because images are not linear.

"""

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))

# STEP 2 - Pooling

"""
Reducing the size of our feature maps, taking the max value of a little matrix, we get a pooled features map.
The size of the original feature map is reducing by a half when we use max_pooling with stride of 2

pool_size = (2,2) in general, most of the time we use a 2x2 matrix to preserve the info but resume
"""

classifier.add(MaxPool2D(pool_size=(2,2)))


# Ading Another Convoluitonal Layer !
"""
The input are not going to be now the images, is going to be the polling output, we can erase the input_shape

A common practice is to increase X2 the number of feature detectors, is good for best results.
"""


classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


# STEP 3 - Flattening

"""
We are going to take all of our features maps, and we are going to put them all together in a array,
that is going to be our input layer of our NN.

- Why  don't we lose spatial structure by flattening?

Because by creating our Feature Maps we resume the spatial info by getting the high numbers in the Feature maps
that are asociated with the space structure

In the pooling step we take the highest numbers, so we are still keeping spatial info.

- Why didn't we take all the pixels of the original image and aply flattening, without the other steps?

If we do that, each node will represent an input, and we don't have any spatial information, we don't know
how pixels are organize in space, making features. No info of spatial structure of pixels.


By doing Convolution and Pooling  we get info about features, each input of our NN has info about the spatial structure.

Flatten() is just goign to flatten the previous layer
"""

classifier.add(Flatten())

# STEP 4 - Full Connection - Creating a Classic NN

"""
Adding the first hidden layer

How many nodes in our input layer?

There is not a rule, it's Art, but we can take a common practice, taking the average,
here by experimentation we'll take 128 nodes.
"""

classifier.add(Dense(output_dim= 128, activation='relu' ))

"""
Final Layer
We use the sigmoid funtion to get probabilities , because it's a binary output, if not we would use -> Softmax
And because it's binary, we just need one output to get the final predition.
"""

classifier.add(Dense(output_dim= 1, activation='sigmoid' ))


# Compiling CNN
"""
adam = a stochastic gradient descent algorithm
loss = the logarithmic loss function like logistic regression, and because our output is binary, if not categorical_cross_entropy
"""

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# PART 3 - Fitting the CNN to the images

"""
Importat: Preprocess our images to prevent overfitting, if not we will get a lot of accuracy on the training set
but poor results in our training set.

How the ImageDataGenerator will prevent overfitting?

One couse of overfitting is taking to few examples to train our model, the model only find few corelations in the traing set,
but fails to find corelations our patterns in the pixels in our test mdoel.

8,000 images in our training set is really not to much, we need more, thats were Data Augmentation comes to play

It will create many baches of our images and it automatically creates some transformations in our images,
linke rotating images, changing its size, there are random transformations, we augment our images,
we enrich our data set to prevent over fitting.

The keras function will also fit our CNN to the training set and test its performance with the Test Set
"""

from keras.preprocessing.image import ImageDataGenerator

"""
rescale=1./255, = all our pixel values will be between 0-1
shear_range, zoom, and horizontal_flip are transformation to create new images in our baches
"""

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)

"""
target_size = size of images spected in our CNN model
"""

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')



test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

"""
steps_per_echo = size of our training set

validation_steps = size of our test set
"""


classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

"""
accuracy in test_set = 0.7510
accuracy in traingint set = 0.8452

This difference in the accuracy of each set is something that we need to fix.

How can we do that?

We have to do a deeper Neural Network, a deeper convolutional model. How can we do that?

1) Add another convolutional layer
2) Add another hidden layer

In this time, we'll choose adding a second convolutional layer, and we are going to see how our
accuraci in the test_set can improve
"""

"""
After adding a new Convolutional layer

accuracy in test_set = 0.8180
accuracy in traingint set = 0.8516

We reduce the difference between sets, that is something good.

For improving the accuracy:

Adding more convolutional layer and : choose a higger target_size to get more information of our
pixel patterns
"""




















#