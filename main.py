import os
import numpy as np 
import random
import pickle
import csv
import datetime
from numpy.random import rand
from random import randrange
from itertools import cycle
from tensorflow.keras.datasets import cifar10, mnist,fashion_mnist, cifar100
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
from tensorflow.keras.applications import VGG16, VGG19, resnet50, mobilenet
import tensorflow_datasets as tfds
from defense import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#  Fast LAyer gradient MEthod (FLAME)
def flame(gmodel, source, x_train, y_train, img_shape, out_class):
    ad = []
    for target in range(out_class):
        if target == source:
            continue
        print(f"measuring the attacking distance between source class {source} and target class {target}")

        # Extract samples from the source data
        x_source = np.array((x_train[np.isin(y_train, source)]))[:500]
        measure_li = []
        for x in x_source:
            with tf.GradientTape() as tape:
                y_pred = gmodel(x.reshape(1,img_shape[0], img_shape[1], img_shape[2]))
                bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                loss = bce(to_categorical(np.ones(1)*target, out_class).reshape(1, out_class), y_pred)
                # Backward error analysis
                grads = tape.gradient(loss, gmodel.trainable_variables)
                measure = [LA.norm(grads[i]) for i in range(len(grads))]
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
    parser.add_argument('--defense', type=str, default="None",
                        help='defense methods: NDC, Krum, TrimmedMean, DP (default: None)')
    parser.add_argument('--model', type=str, default="simplecnn",
                        help='model (default: simplecnn)')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Use different datasets: MNIST, Fashion-MNIST, and CIFAR-10.

    # MNIST
    # Attacking distances for the source class '2': max '5'  min '1' 
    
    if args.dataset == "mnist":
        s_max = 5
        s_min = 1
        
        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)
        out_class = 10

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

        base_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedTail.npy", allow_pickle=True) 

    # Fashion-MNIST
    # Attacking distances for the source class '2': max '9'  min '4' 

    if args.dataset == "fashion-mnist":
        s_max = 9
        s_min = 4

        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)
        out_class = 10

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)

        base_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedTail.npy", allow_pickle=True)

    # CIFAR-10
    # Attacking distances for the source class '2': max '9'  min '4'

    if args.dataset == "cifar10":
        s_max = 9
        s_min = 4
    
        img_rows = 32
        img_cols = 32
        channels = 3
        img_shape = (img_rows, img_cols, channels)
        out_class = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,32,32,3)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,32,32,3)
        y_train = y_train.reshape(50000,)
        y_test = y_test.reshape(10000,)

        base_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedTail.npy", allow_pickle=True)

    # CIFAR-100
    # Attacking distances for the source class '2': max '-'  min '35'
    
    if args.dataset == "cifar100":
        #s_max = 
        s_min = 35

        img_rows = 32
        img_cols = 32
        channels = 3
        img_shape = (img_rows, img_cols, channels)
        out_class = 100

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = (x_train.astype('float32')/255.0).reshape(-1,32,32,3)
        x_test = (x_test.astype('float32')/255.0).reshape(-1,32,32,3)
        y_train = y_train.reshape(50000,)
        y_test = y_test.reshape(10000,)

        base_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedTail.npy", allow_pickle=True)


    # ImageNet
    # Attacking distances for the source class '2': max '-'  min '35'
    
    if args.dataset == "imagenet":
        #s_max = 
        s_min = 81

        img_rows = 64
        img_cols = 64
        channels = 3
        img_shape = (img_rows, img_cols, channels)
        out_class = 1000

        dataset, info = tfds.load("imagenet_resized/64x64", as_supervised = True, with_info = True, batch_size = -1)

        dataset_train, dataset_test = dataset["train"], dataset["validation"]
        x_train = dataset_train[0]
        y_train = dataset_train[1]
        x_test = dataset_test[0]
        y_test = dataset_test[1]

        x_train = (np.array(x_train).astype('float32')).reshape(-1,64,64,3)
        x_test = (np.array(x_test).astype('float32')).reshape(-1,64,64,3)

        print(x_train[0])

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        base_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedBase.npy", allow_pickle=True)
        tail_weights = np.load(f"pretrained/{args.dataset}_{args.model}_ConvergedTail.npy", allow_pickle=True)



    # Model architecture
    if args.model == "resnet":

        def build_discriminator(img_shape):
            # We define a ResNet-based edge neural network
            base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=img_shape, classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model
            model.add(Flatten())
            model.add(Dense(1024,activation=('relu')))
            model.add(Dense(512,activation=('relu')))
            model.add(Dense(out_class,activation=('softmax')))

            img = Input(img_shape)

            prob = model(img)

            return Model(img, prob)


        def build_base_discriminator(img_shape):
            # We define a ResNet-based edge neural network
            base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model
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

    

    # Model architecture
    if args.model == "mobilenet":

        def build_discriminator(img_shape):
            # We define a mobile-based edge neural network
            base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model
            model.add(Flatten())
            model.add(Dense(1024,activation=('relu')))
            model.add(Dense(512,activation=('relu')))
            model.add(Dense(out_class,activation=('softmax')))

            img = Input(img_shape)

            prob = model(img)

            return Model(img, prob)


        def build_base_discriminator(img_shape):
            # We define a mobilenet-based edge neural network
            base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model
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



    # Model architecture
    if args.model == "vgg19":

        def build_discriminator(img_shape):
            # We define a VGG19-based edge neural network
            base_model = VGG19(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model (in this case vgg19 to model_1)
            model.add(Flatten())
            model.add(Dense(1024,activation=('relu')))
            model.add(Dense(512,activation=('relu'))) 
            model.add(Dense(out_class,activation=('softmax')))

            img = Input(img_shape)
            
            prob = model(img)
           
            return Model(img, prob) 


        def build_base_discriminator(img_shape):
            # We define a VGG19-based edge neural network
            base_model = VGG19(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

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

    

    # Model architecture
    if args.model == "vgg16":

        def build_discriminator(img_shape):
            # We define a VGG16-based edge neural network
            base_model = VGG16(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

            model= Sequential()
            model.add(base_model) #Adds the base model (in this case vgg19 to model_1)
            model.add(Flatten())
            model.add(Dense(1024,activation=('relu')))
            model.add(Dense(512,activation=('relu')))
            model.add(Dense(out_class,activation=('softmax')))

            img = Input(img_shape)

            prob = model(img)

            return Model(img, prob)


        def build_base_discriminator(img_shape):
            # We define a VGG16-based edge neural network
            base_model = VGG16(include_top=False,weights='imagenet',input_shape=img_shape,classes=1000)
            for layer in base_model.layers: layer.trainable=False

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


    if args.model == "simplecnn":
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


    # These are our metrics: Main Task Accuracy (MTA) and Attack Task Accuracy (ATA). 
    main_result = []
    backdoor_result = []

    # We build an individual model for each client in FL.  
    edge_nets = []
    for i in range(args.client):
      edge = build_discriminator(img_shape)
      edge_nets.append(edge)

    # We build the global model that has the same architecture with edge models.
    base = build_base_discriminator(img_shape)
    tail = build_tail_discriminator()
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
        s_min = flame(gmodel,s_c,x_train, y_train, img_shape, out_class)
    
    ata = []
    mta = []
    for i in range(out_class): 
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
        target_class = to_categorical(target_class, out_class)
        main_class = list(range(out_class))
        main_class.pop(s_c)


        # Training set
        main = np.array((x_train[np.isin(y_train, main_class)]))
        y_main = np.array((y_train[np.isin(y_train, main_class)]))
        y_main = to_categorical(y_main, out_class)


        # Test set
        main_test = np.array((x_test[np.isin(y_test, main_class)]))
        y_main_test = np.array((y_test[np.isin(y_test, main_class)]))
        y_main_test = to_categorical(y_main_test, out_class)
        
        target_test = np.array((x_test[np.isin(y_test, [s_c])]))
        target_class_test = np.ones(len(target_test))*i
        target_class_test = to_categorical(target_class_test, out_class)


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
            

          # when applying different defense methods
          if args.defense == "NDC":
              u = np.average((NDC(local_updates)), axis = 0)

          elif args.defense == "Krum":
              u = np.average((Krum(local_updates,args.gamma)), axis = 0)

          elif args.defense == "TrimmedMean":
              u = np.average((TrimmedMean(local_updates,args.beta)), axis = 0)

          elif args.defense == "DP":
              u = np.average((DP(local_updates, args.std)), axis = 0)

          else:
              u = np.average((local_updates), axis = 0)

          # update the global model
          g = gmodel.get_weights() 
          aggmodel = np.array(([(g[i]+u[i]) for i, w in enumerate(g)]))

          if args.defense == "DP":
            for i, w in enumerate(aggmodel):
                aggmodel[i] = np.random.normal(0,args.std,np.array((w)).shape) + w
          gmodel.set_weights(aggmodel)


          # Evaluation
          main_task_acc.append(gmodel.evaluate(main_test, y_main_test, 10, verbose=0)[1])
          backdoor_task_acc.append(gmodel.evaluate(target_test, target_class_test, verbose=0)[1])

          print("Attacking task accuracy: %s   Main task accuracy: %s \n" %(backdoor_task_acc[-1], main_task_acc[-1]))

        main_result.append(np.array((main_task_acc)))
        backdoor_result.append(np.array((backdoor_task_acc)))
        
        if args.ada:
            print("Overall attacking accuracy: %s" % np.max(backdoor_result[0]))
            print("Overall main task accuracy: %s" % np.min(main_result[0]))
        
        ata.append(np.max(np.max(backdoor_result, axis = 1)))
        mta.append(np.min(np.min(main_result, axis = 1)))

    print(f"Method attacking task performance {np.mean(ata)}")
    print(f"Method main task performance {np.mean(mta)}")


if __name__ == '__main__':
    main()
