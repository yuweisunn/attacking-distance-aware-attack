import numpy as np 
import os 
import random
import pickle
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
import keras
from keras import losses
from numpy.random import rand
from random import randrange
from itertools import cycle
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, ReLU, InputLayer
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPool2D,Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import to_categorical


# change the dataset and the model architecture you would like to use here
dataset = "cifar10"
model_name = "simplecnn"


if dataset == "mnist":
    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)
    out_class = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
    x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


if dataset == "fashion mnist":
    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)
    out_class = 10

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
    x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)



if dataset == "cifar10":
    img_rows = 32
    img_cols = 32
    channels = 3
    img_shape = (img_rows, img_cols, channels)
    out_class = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.astype('float32')/255.0).reshape(-1,32,32,3)
    x_test = (x_test.astype('float32')/255.0).reshape(-1,32,32,3)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    

if dataset == "cifar100":
    img_rows = 32
    img_cols = 32
    channels = 3
    img_shape = (img_rows, img_cols, channels)
    out_class = 100

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = (x_train.astype('float32')/255.0).reshape(-1,32,32,3)
    x_test = (x_test.astype('float32')/255.0).reshape(-1,32,32,3)
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)



if model_name == "vgg":

    # We define a VGG19-based edge neural network
    base_model = VGG19(include_top=False,weights='imagenet',input_shape=img_shape,classes=y_train.shape[1])
    for layer in base_model.layers: layer.trainable=False

    def build_discriminator():
        model= Sequential()
        model.add(base_model) #Adds the base model (in this case vgg19 to model_1)
        model.add(Flatten())
        model.add(Dense(1024,activation=('relu')))
        model.add(Dense(512,activation=('relu'))) 
        model.add(Dense(out_class,activation=('softmax')))

        img = Input(img_shape)
        
        prob = model(img)
       
        return Model(img, prob) 


    def build_base_discriminator():
        model= Sequential()
        model.add(base_model) #Adds the base model (in this case vgg19 to model_1)
        model.add(Flatten())
        model.add(Dense(1024,activation=('relu')))
        model.add(Dense(512,activation=('relu'))) 

        img = Input(shape=img_shape)
        
        prob = model(img)

        return Model(img, prob) 


    def build_tail_discriminator():
        model = Sequential()
        model.add(InputLayer(512,))
        model.add(Dense(units=out_class, activation="softmax"))

        img = Input(512,)
        
        prob = model(img)
      
        return Model(img, prob) 




if model_name == "simplecnn":
    # We define a simple CNN edge neural network
    def build_discriminator(img_shape):
        model = Sequential()
        model.add(InputLayer(input_shape=img_shape))
        model.add(Conv2D(filters=20, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(filters=50, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=200, activation="relu"))
        model.add(Dense(units=out_class, activation="softmax"))

        img = Input(img_shape)
        
        prob = model(img)
       
        return Model(img, prob) 

    def build_base_discriminator(img_shape):
        model = Sequential()
        model.add(InputLayer(input_shape=img_shape))
        model.add(Conv2D(filters=20, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(filters=50, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=200, activation="relu"))

        img = Input(shape=img_shape)
        
        prob = model(img)

        return Model(img, prob) 

    def build_tail_discriminator():
        model = Sequential()
        model.add(InputLayer(200,))
        model.add(Dense(units=out_class, activation="softmax"))

        img = Input(200,)
        
        prob = model(img)
      
        return Model(img, prob)    



# run iterations of model training with 100 clients
discriminator_list = []
client = 100
for i in range(client):
  edge = build_discriminator()
  discriminator_list.append(edge)

lr =0.001
opt = Adam(learning_rate=lr)
base = build_base_discriminator()
tail = build_tail_discriminator()
gmodel = Sequential([base, tail])
gmodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
setSize = 500
history = []

epoch = 0
while True:
  epoch+=1
  print("epoch: %s" %epoch)
  j = 10
  action = []
  local_updates = []
  while j > 0 :
    # Get random action
    temp_action = np.random.randint(0, 100)
    if temp_action not in action:
        action.append(temp_action)
        j = j - 1
  print("================ %s =================" %action)

  for i, j in enumerate(action):
    print("Local training %s/%s" %(i+1,10))
    discriminator_list[j].set_weights(gmodel.get_weights())
    opt = Adam(learning_rate=lr)
    discriminator_list[j].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    discriminator_list[j].fit(x_train[setSize*j: setSize*(j+1)], y_train[setSize*j: setSize*(j+1)], epochs=1, batch_size=16, verbose = 0)
    local_updates.append(discriminator_list[j].get_weights())
  
  aggmodel = np.mean((local_updates), axis = 0)
  gmodel.set_weights(aggmodel)
  history.append(gmodel.evaluate(x_test, y_test))

  # Depending on the dataset you are using, the baseline test accuracy can vary.  
  if history[-1][1] > 0.3:
     break

history = np.array((history))

np.save('history', history)
np.save(f'{dataset}_{model_name}_ConvergedBase.npy', base.get_weights())
np.save(f'{dataset}_{model_name}_ConvergedTail.npy', tail.get_weights())

