# ----------------------------------------------------------------------------------------
# 
# Neural network to recognize handwritten digits effectively using data from MNIST dataset
# 
# ----------------------------------------------------------------------------------------

# Import important bits from keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import flask

# import wandb
# from wandb.keras import WandbCallback

# --- log code ---
# run = wandb.init()
# config = run.config

# Number of passes during which the weights are changed
# NOTE:
# More runs will increase training accuracy, without increasing accuracy in validation set
# This is a result of OVERFITTING
# The same problem may arise with too many hidden nodes
# config.epochs = 20
epochs = 10
# # adam - gradient descent function with preset learning rate
# config.optimizer = 'adam'
# config.hidden_nodes = 100

# ----------
# 
# load MNIST data
#
# ----------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# --- Normalize data ---
# Values initially between 0 and 255 (pixel brightness)
# Regularize to between 0 and 1 by dividing by 255
X_test = X_test.astype("float32") / 255
X_train = X_train.astype("float32") / 255

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# --- Encode output ---
# Encode outputs to display 1 if correct digit, 0 otherwise
# Known as 'One Hot Encoding'
# Ex:
# 0 = 1 0 0 0 0 0 0 0 0 0
# nnn4 = 0 0 0 0 1 0 0 0 0 0
# etc...
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
labels = range(10)

# Size is 10 for 10 digits
num_classes = y_train.shape[1]

# ----------
# 
# Create model
#
# ----------

# Network is defined as a series of steps
model = Sequential()
# Reshapes 28 x 28 2D dimensional array to flattened 1D array
model.add(Flatten(input_shape=(img_width, img_height)))

# --- Dropout layers ---
# Dropout layers between perceptrons will force the network
# to learn multiple paths to certain predictions by zeroing
# some perceptron weights during training
# (Very helpful for correcting over-fitting)
model.add(Dropout(0.4))

# Create hidden layer after input layer
# relu - simply reads all negative values as zero
# model.add(Dense(config.hidden_nodes, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))

model.add(Dropout(0.4))

# Adds perceptrons to network (Dense means fully connected)
# Softmax functions similarly to sigmoid function, only returns zeros between 0 and 1
model.add(Dense(num_classes, activation = 'softmax'))

# mse - mean squared error (loss function)
# CHANGING mse to categorical_crossentropy
# categorical_crossentropy - returns probabilities that each choice is correct
# metrics = 'accuracy' - tells keras to output accuracy as algorithm learns
# model.compile(loss='categorical_crossentropy', optimizer = config.optimizer, metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Print model information
model.summary()

# ----------
# 
# Fit the model
#
# ----------

# .fit gets optimum values given X_train (training data) and is_five_train (desired output)
# epochs = config.epochs - override kerras default of 1 epoch with specified value
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))#, callbacks = [WandbCallback(labels=labels, data_type = "image")])


# ----------
# 
# Debugging
#
# ----------
# Look at what model outputs on first ten images
# print(model.predict(X_test[:10,:,:])) 

# # ----------
# #
# # Output to file
# #
# # ----------
# pred = model.predict('/home/mike/Pictures/MNIST-eight-28-28.png')
# print("Prediction: ")
# print(pred)
