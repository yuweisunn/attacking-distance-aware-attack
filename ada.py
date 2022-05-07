import os
import numpy as np 
import random
import pickle
import csv
import datetime
from numpy.random import rand
from random import randrange
from itertools import cycle
from tensorflow.keras.datasets import cifar10, mnist,fashion_mnist
import tensorflow.keras
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, ReLU, InputLayer
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from numpy import linalg as LA

np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_discriminator():
    model = Sequential()
    model.add(InputLayer(input_shape=img_shape))
    model.add(Conv2D(filters=20, kernel_size=5, strides=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=50, kernel_size=5, strides=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=200, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    img = Input(img_shape)
    
    prob = model(img)
   
    return Model(img, prob) 

def build_base_discriminator():
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
    model.add(Dense(units=10, activation="softmax"))

    img = Input(200,)
    
    prob = model(img)
  
    return Model(img, prob) 


# MNIST
# Attacking distances for the source class '2': max '5'  min '1' 
"""
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

base_weights = np.load("pretrained/mnistConvergedBase.npy", allow_pickle=True)
tail_weights = np.load("pretrained/mnistConvergedTail.npy", allow_pickle=True)
"""


# Fashion-MNIST
# Attacking distances for the source class '2': max '9'  min '4' 
"""
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

base_weights = np.load("pretrained/fashionmnistConvergedBase.npy", allow_pickle=True)
tail_weights = np.load("pretrained/fashionmnistConvergedTail.npy", allow_pickle=True)
"""

# CIFAR-10
# Attacking distances for the source class '2': max '9'  min '4'
img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = (x_train.astype('float32')/255.0).reshape(-1,32,32,3)
x_test = (x_test.astype('float32')/255.0).reshape(-1,32,32,3)
y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)

base_weights = np.load("pretrained/cifarConvergedBase.npy", allow_pickle=True)
tail_weights = np.load("pretrained/cifarConvergedTail.npy", allow_pickle=True)



main_result = []
backdoor_result = []
discriminator_list = []

client = 100
for i in range(client):
  edge = build_discriminator()
  discriminator_list.append(edge)

lr =0.001
opt = Adam(learning_rate=lr)
base = build_base_discriminator()
tail = build_tail_discriminator()
base.set_weights(base_weights)
tail.set_weights(tail_weights)
gmodel = Sequential([base, tail])
gmodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



def main(scale = False, MR = False, ADA = True, eta = 0.1):

    benign_client = int((1-eta)*client)
    num = 0
    for t in range(2,3,1): # source class: 2
        for i in range(10): # target class: {0,1,2,...,9}
            
            if ADA and i != 4: # ADA uses the optimized label as the target
                continue
                
            if i == t:
                continue
                
            num = num + 1
            
            if not ADA:
                print("Meta progress: %s/%s" %(num, 9))
            
            print("Source Class: %s   Target Class: %s" %(i, t))

            # prepare the backdoor dataset of the adversary with the source and target classes.
            source = np.array((x_train[np.isin(y_train, [i])]))
            y_backdoor = np.ones(len(source))*t
            source_class = list(range(10))
            source_class.pop(i)
            main = np.array((x_train[np.isin(y_train, source_class)]))
            y_main = np.array((y_train[np.isin(y_train, source_class)]))
            y_main = to_categorical(y_main, 10)
            y_backdoor = to_categorical(y_backdoor, 10)

            main_test = np.array((x_test[np.isin(y_test, source_class)]))
            y_main_test = np.array((y_test[np.isin(y_test, source_class)]))
            y_main_test = to_categorical(y_main_test, 10)
            source_test = np.array((x_test[np.isin(y_test, [i])]))
            y_backdoor_test = np.ones(len(source_test))*t
            y_backdoor_test = to_categorical(y_backdoor_test, 10)

            # Every round, 450 samples of a client are randomly selected to retarin a converged global model. 
            setSize = 450
            main_task_acc = [0.99] # initial MTA
            backdoor_task_acc = [0] # initial ATA


            # Adversarial training
            for i in range(50):
              print("Federated Learning Round: %s/%s" %(i+1,50))

              j = 10 # randomly select 10 clients
              action = []
              local_updates = []
              norms = []
              while j > 0 :
                # Get random action
                temp_action = np.random.randint(0, client)
                if temp_action not in action:
                    action.append(temp_action)
                    j = j - 1
              action.sort()

              for i, j in enumerate(action):
                # if a selected client is benign
                if j < benign_client:
                    g = gmodel.get_weights()
                    discriminator_list[j].set_weights(g)
                    opt = Adam(learning_rate=lr)
                    discriminator_list[j].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                    discriminator_list[j].fit(main[setSize*j: setSize*(j+1)], y_main[setSize*j: setSize*(j+1)], epochs=1, batch_size=16, verbose = 0)
                    l = discriminator_list[j].get_weights()
                    local_update = [(l[i]-g[i]) for i, w in enumerate(l)]
                    local_updates.append(local_update)

                    norms.append(LA.norm(np.concatenate([w.flatten() for w in local_update])))

                # if a selected client is malicious
                else:
                  x_sub, y_sub = zip(*random.sample(list(zip(main[int(setSize*benign_client):], y_main[int(setSize*benign_client):])), int(setSize/2)))
                  discriminator_list[0].set_weights(gmodel.get_weights())
                  opt = Adam(learning_rate=lr)
                  discriminator_list[0].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                  x_attacker = np.concatenate((np.array((random.sample(list(source), int(setSize/2)))), np.array((x_sub))), axis = 0) 
                  y_attacker = np.concatenate((y_backdoor[:int(setSize/2)], np.array((y_sub))), axis = 0) 

                  discriminator_list[0].fit(x_attacker, y_attacker, epochs=1, batch_size=16, verbose = 0, shuffle=True)
                  l = discriminator_list[0].get_weights()
                  g = gmodel.get_weights()
                
                    
                  if scale or ADA:
                        # scale the weights of the malicious update to bypass norm-based detection.
                        factor = 1
                        while True:
                          l_malicious = [(l[i]-g[i])*factor for i, w in enumerate(l)] 
                          norm = LA.norm(np.concatenate([w.flatten() for w in l_malicious])) 

                          if norm <= np.median(norms): #adaptive clipping
                             local_updates.append(l_malicious)
                             norms.append(norm)
                             break
                          factor = factor - 0.1

                  # Model Replacement (MR) is to scale up the weights by a factor of the total selected client number.
                  elif MR:
                        l_malicious = [10*(l[i] - g[i]) for i, w in enumerate(l)]
                        norm = LA.norm(np.concatenate([w.flatten() for w in l_malicious]))
                        local_updates.append(l_malicious)
                        norms.append(norm)
                  
                  # Otherwise, send the unmodified model update. 
                  else:
                        l_malicious = [(l[i]-g[i]) for i, w in enumerate(l)]
                        norm = LA.norm(np.concatenate([w.flatten() for w in l_malicious]))
                        local_updates.append(l_malicious)
                        norms.append(norm)
                  

              # update the boundary of the norm difference clipping (NDC) 
              # if the NDC defense is applied, only the clients that have a qualified norm can be accepted. 
              NDC = True
              g = gmodel.get_weights()  

              benign_li = []
              boundary = np.median(norms)
              for i, norm_s in enumerate(norms):
                  if norm_s < boundary:
                      benign_li.append(local_updates[i])
              if NDC:
                  u = np.average((benign_li), axis = 0)
              else:
                  u = np.average((local_updates), axis = 0)


              # update the global model
              aggmodel = np.array(([(g[i]+u[i]) for i, w in enumerate(g)]))
              gmodel.set_weights(aggmodel)


              # Evaluation
              main_task_acc.append(gmodel.evaluate(main_test, y_main_test, 10, verbose=0)[1])
              backdoor_task_acc.append(gmodel.evaluate(source_test, y_backdoor_test, verbose=0)[1])

              print("Attacking task accuracy: %s   Main task accuracy: %s \n" %(backdoor_task_acc[-1], main_task_acc[-1]))


            main_result.append(np.array((main_task_acc)))
            backdoor_result.append(np.array((backdoor_task_acc)))
            
            if ADA:
                print("Attacking accuracy: %s" % np.max(backdoor_result[0]))
                print("Main task accuracy: %s" % np.min(main_result[0]))
            
            print("Attacking accuracy: %s" % np.max(np.max(backdoor_result, axis = 1)))
            print("Main task accuracy: %s" % np.min(np.min(main_result, axis = 1)))
            
            filename = datetime.now().strftime("%Y%m%d-%H%M%S")
            np.save('%s_main' %filename, main_result)
            np.save('%s_backdoor' %filename, backdoor_result)

main(scale = False, MR = False, ADA = True, eta = 0.1)
