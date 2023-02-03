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
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_edge_net(img_shape):
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

def build_global_featureExtra(img_shape):
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

def build_global_discriminator():
    model = Sequential()
    model.add(InputLayer(200,))
    model.add(Dense(units=10, activation="softmax"))

    img = Input(200,)
    
    prob = model(img)
  
    return Model(img, prob)    


#  Fast LAyer gradient MEthod (FLAME)
def flame(gmodel, source, x_train, y_train):
    ad = []
    for target in range(10):
        if target == source:
            continue
        print(f"measuring the attacking distance between source class {source} and target class {target}")

        # Extract samples from the source data
        x_source = np.array((x_train[np.isin(y_train, source)]))[:500]
        measure_li = []
        for x in x_source:
            with tf.GradientTape() as tape:
                y_pred = gmodel(x.reshape(1,28,28,1))
                bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                loss = bce(to_categorical(np.ones(1)*target, 10).reshape(1,10),y_pred)
                # Backward error analysis
                grads = tape.gradient(loss, gmodel.trainable_variables)
                measure = [LA.norm(grads[i]) for i in range(2,4,1)]
                measure_li.append(measure)

        # Compute the average attacking distance for each target class
        ad_avg = np.mean(measure_li)
        print(f"Attacking distance: {ad_avg}")
        ad.append(ad_avg)

    # Obtain the target class with the minimum AD
    s_min = np.argmin(ad)
    if s_min >= source:
        s_min =  s_min + 1
    print(f"Min AD target class {s_min} for source class {source}\n")

    return s_min
    

def main():
    parser = argparse.ArgumentParser(description='Tensorflow ADA Implementation')
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='dataset (default: MNIST)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--client', type=int, default=100,
                        help='client number (default: 100)')
    parser.add_argument('--source', type=int, default=2,
                        help='source attacking class (default: 2)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='attacking frequency (default: 0.1)')
    parser.add_argument('--ada', action='store_false',
                        help='apply the ADA attack (default: True)')
    parser.add_argument('--flame', action='store_true',
                        help='regenerate the AD distribution via backward error analysis (default: False)')
    parser.add_argument('--scale', action='store_false',
                        help='apply the "Train and Scale" method (default: True)')
    parser.add_argument('--ndc', action='store_false',
                        help='apply the Norm Difference Clipping (NDC) (default: True)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Use different datasets: MNIST, Fashion-MNIST, and CIFAR-10.

    # MNIST
    # Attacking distances for the source class '2': max '5'  min '1' 
    
    if args.dataset == "MNIST":
        s_max = 5
        s_min = 1
        
        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

        base_weights = np.load("pretrained/mnistConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load("pretrained/mnistConvergedTail.npy", allow_pickle=True)
   

    # Fashion-MNIST
    # Attacking distances for the source class '2': max '9'  min '4' 

    if args.dataset == "Fashion-MNIST":
        s_max = 9
        s_min = 4

        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

        base_weights = np.load("pretrained/fashionmnistConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load("pretrained/fashionmnistConvergedTail.npy", allow_pickle=True)


    # CIFAR-10
    # Attacking distances for the source class '2': max '9'  min '4'

    if args.dataset == "CIFAR-10":
        s_max = 9
        s_min = 4
    
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


    # These are our metrics: Main Task Accuracy (MTA) and Attack Task Accuracy (ATA). 
    main_result = []
    backdoor_result = []

    # We build an individual model for each client in FL.  
    edge_nets = []
    for i in range(args.client):
      edge = build_edge_net(img_shape)
      edge_nets.append(edge)

    # We build the global model that has the same architecture with edge models.
    base = build_global_featureExtra(img_shape)
    tail = build_global_discriminator()
    base.set_weights(base_weights)
    tail.set_weights(tail_weights)
    gmodel = Sequential([base, tail])
    opt = Adam(learning_rate=args.lr)
    gmodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    # Compute the total number of benign clients based on the attacking frequency.
    benign_client = int((1-args.epsilon)*args.client)


    num = 0
    s_c = args.source

    # If True, select the optimized attacking class based on FLAME, otherwise, 
    # use the target class optimized beforehand 
    # (which should be the same with outputs from FLAME). 
    if args.flame:
        s_min = flame(gmodel,s_c,x_train, y_train)

    for i in range(10): 
        num = num + 1

        # ADA uses the optimized class as the target from all the remaining classes.
        if (args.ada and i != s_min) or (i == s_c): 
            continue
        
        if args.ada:
            print(f"ADA Attack on source class {s_c} based on target class {s_min}")
           
        else:
            print(f"Meta progress: {num}/9")
            print(f"Source Class: {s_c}; Target Class: {i}")
            

        # Prepare the backdoor dataset of the adversary with the source and target classes.
        source = np.array((x_train[np.isin(y_train, [s_c])]))

        target_class = np.ones(len(source))*i
        target_class = to_categorical(target_class, 10)
        main_class = list(range(10))
        main_class.pop(s_c)


        # Training set
        main = np.array((x_train[np.isin(y_train, main_class)]))
        y_main = np.array((y_train[np.isin(y_train, main_class)]))
        y_main = to_categorical(y_main, 10)


        # Test set
        main_test = np.array((x_test[np.isin(y_test, main_class)]))
        y_main_test = np.array((y_test[np.isin(y_test, main_class)]))
        y_main_test = to_categorical(y_main_test, 10)
        
        target_test = np.array((x_test[np.isin(y_test, [s_c])]))
        target_class_test = np.ones(len(target_test))*i
        target_class_test = to_categorical(target_class_test, 10)


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
            temp_action = np.random.randint(0, args.client)
            if temp_action not in action:
                action.append(temp_action)
                j = j - 1
          action.sort()

          for i, j in enumerate(action):
            # if a selected client is benign
            if j < benign_client:
                g = gmodel.get_weights()
                edge_nets[j].set_weights(g)
                edge_nets[j].compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=args.lr), metrics=['accuracy'])
                edge_nets[j].fit(main[setSize*j: setSize*(j+1)], y_main[setSize*j: setSize*(j+1)], epochs=1, batch_size=16, verbose = 0)
                l = edge_nets[j].get_weights()
                local_update = [(l[i]-g[i]) for i, w in enumerate(l)]
                local_updates.append(local_update)

                norms.append(LA.norm(np.concatenate([w.flatten() for w in local_update])))

            # if a selected client is malicious
            else:
              x_sub, y_sub = zip(*random.sample(list(zip(main[int(setSize*benign_client):], y_main[int(setSize*benign_client):])), int(setSize/2)))
              edge_nets[0].set_weights(gmodel.get_weights())
              edge_nets[0].compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=args.lr), metrics=['accuracy'])
              x_attacker = np.concatenate((np.array((random.sample(list(source), int(setSize/2)))), np.array((x_sub))), axis = 0) 
              y_attacker = np.concatenate((target_class[:int(setSize/2)], np.array((y_sub))), axis = 0) 

              edge_nets[0].fit(x_attacker, y_attacker, epochs=1, batch_size=16, verbose = 0, shuffle=True)
              l = edge_nets[0].get_weights()
              g = gmodel.get_weights()
            
                
              # Perform the "train and scale" strategy to adjust the scale of a malicious update. This strategy is applied by default in the ADA attack. 
              if args.ada or args.scale:
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


              # Otherwise, send the unmodified malicious model update. 
              else:
                    l_malicious = [(l[i]-g[i]) for i, w in enumerate(l)]
                    norm = LA.norm(np.concatenate([w.flatten() for w in l_malicious]))
                    local_updates.append(l_malicious)
                    norms.append(norm)
              

          # update the boundary of the norm difference clipping (NDC) 
          # if the NDC defense is applied, only the clients that have a qualified norm can be accepted. 
          g = gmodel.get_weights()  

          benign_li = []
          boundary = np.median(norms)
          for i, norm_s in enumerate(norms):
              if norm_s < boundary:
                  benign_li.append(local_updates[i])
          if args.ndc:
              u = np.average((benign_li), axis = 0)
          else:
              u = np.average((local_updates), axis = 0)


          # update the global model
          aggmodel = np.array(([(g[i]+u[i]) for i, w in enumerate(g)]))
          gmodel.set_weights(aggmodel)


          # Evaluation
          main_task_acc.append(gmodel.evaluate(main_test, y_main_test, 10, verbose=0)[1])
          backdoor_task_acc.append(gmodel.evaluate(target_test, target_class_test, verbose=0)[1])

          print("Attacking task accuracy: %s   Main task accuracy: %s \n" %(backdoor_task_acc[-1], main_task_acc[-1]))



        main_result.append(np.array((main_task_acc)))
        backdoor_result.append(np.array((backdoor_task_acc)))
        
        if args.ada:
            print("Attacking accuracy: %s" % np.max(backdoor_result[0]))
            print("Main task accuracy: %s" % np.min(main_result[0]))
        
        print("Attacking accuracy: %s" % np.max(np.max(backdoor_result, axis = 1)))
        print("Main task accuracy: %s" % np.min(np.min(main_result, axis = 1)))


if __name__ == '__main__':
    main()
