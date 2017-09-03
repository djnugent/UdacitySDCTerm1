import os
import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, BatchNormalization, ELU
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

from functions import uniform_data, gen


# Parse CSV
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# Split samples into training/test set
train_samples, valid_samples = train_test_split(samples, test_size=0.1)

# Make data more uniform(bins data and cuts all bins to average bin length)
# Dont use anymore. Data is made uniform in generator
'''
print("Length before uniform:" + str(len(train_samples)))
train_samples = uniform_data(train_samples,bin_num=15)
print("Length after uniform:" + str(len(train_samples)))
'''


##### MODEL ####
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation and crop
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(90,320,3)))
# Convolution layers with ELU activation and batch normalization
model.add(Convolution2D(24, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
model.add(BatchNormalization())
model.add(Convolution2D(36, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 5,5, activation='elu', subsample=(2, 2), border_mode='valid'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3,3, activation='elu', border_mode='valid'))
model.add(BatchNormalization())
# Fully Connected layers with ELU activation and dropout layers
model.add(Flatten())
model.add(Dropout(0.50))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


#### TRAIN ####
# Training parameters
batch_size = 128
augmentation_factor = 4 #cameras(3) * flip(2) * 70% drop rate
epochs = 5
# Create generators
train_generator = gen(train_samples, batch_size=batch_size)
validation_generator = gen(valid_samples, batch_size=batch_size)
# fit model
training_size = len(train_samples) * augmentation_factor
valid_size = len(valid_samples) * augmentation_factor
model.fit_generator(train_generator, samples_per_epoch= training_size,\
            validation_data=validation_generator, \
            nb_val_samples=valid_size, nb_epoch=epochs)

model.save('model.h5')
