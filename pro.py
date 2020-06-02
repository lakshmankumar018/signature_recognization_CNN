# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:46:28 2020

@author: rockn
"""


from keras.models import Sequential
from keras.layers import Convolution2D

from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
#Step 1: Convolution
#Input shape means the pic is black and white or colour if it is colour then no of channels =3 b&W = 2
#beause black and white image is converted in to 2s array while colour one as 3d aray
#convo2d uses first argument is no of filters follwed by no of rows and columns in the convolutional kernal(feature detector)
classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
#Step 2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#adding a second cnn layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#adding a third cnn layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#adding a fourth cnn layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding a fifth convolutional layer
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.summary()
#step 3:  Flattening
classifier.add(Flatten())
#Step 4 Full Connection
#output_dim is no of nodes in hidden layer
#In the hidden layer it is good to choose nodes between number of input nodes& output nodes
#hidden layer
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#output layer
#sigmoid function because of binary oucome
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling a CNN
#for binary outcome we use logarthmic loss function as binary cross entropy if it is more than 2 outcomes
#then we use categorical cross entropy   
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Image pre-processing  and fitting CNN to images
#Image augmentation 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip= False)

test_datagen = ImageDataGenerator(rescale= 1./255)
#target size is the input we gave to our cnn i.e 256x256 
training_dataset = train_datagen.flow_from_directory('handwritten-signatures/training_set',target_size=(256,256), batch_size = 64, class_mode = 'binary')

test_dataset = test_datagen.flow_from_directory('handwritten-signatures/test_set',target_size=(256,256), batch_size = 64, class_mode = 'binary')
#samples per epoch is total no of images to train themodel
#nb_epoch is no of images to train at a time
# nb_val_samples is no of images in test set
classifier.fit_generator(training_dataset,samples_per_epoch=300, nb_epoch=10, validation_data=test_dataset, nb_val_samples=156)
scores = classifier.evaluate(training_dataset,test_dataset, verbose =0)
print(scores[1]*100)
