## This is a Tensorflow implementation of the paper [Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis](https://arxiv.org/abs/2203.11633) at IJCNN 2022.

## Table of Contents
* [General information](#general-information)
* [Running the systems](#running-the-systems)

## General information
Attacking Distance-aware Attack (ADA) enhances a poisoning attack by finding the optimized target class in the feature space.

<center>
<img src = "semitarget.png" width = "60%"></img>
</center>
   
This instruction describes how to mount the semi-targeted attack on three different benchmark datasets, i.e., MNIST, Fashion-MNIST, and CIFAR-10. By adjusting the attacking frequency and the participanting client number, we observe how ADA performs under different settings.

## Running the systems
### Environment
Tensorflow 2

Python 3.8

### Training
To run the algorithm with the optimized target class that was prepared beforehand:

	python main.py
	
	
In case, you would like to search for the optimized target class using FLAME via the backward analysis as described in the paper:

	python main.py --flame
