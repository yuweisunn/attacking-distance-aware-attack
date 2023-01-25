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

## Citation 
If this repository is helpful for your research or you want to refer the provided results in this work, you could cite the work using the following BibTeX entry:

```
@article{sun2022semitarget,
  author = {Yuwei Sun and Hideya Ochiai and Jun Sakuma},
  title = {Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis},
  journal = {International Joint Conference on Neural Networks (IJCNN)},
  year = {2022}
}
```

## Further Reading
[1] Journal version of the paper: [Attacking Distance-aware Attack: A Semi-targeted Poisoning Attack on Federated Learning](https://www.techrxiv.org/articles/preprint/How_the_Target_Matters_Semi-Targeted_Model_Poisoning_Attack_on_Federated_Learning/20339091)

[2] A new survey paper: [Decentralized Deep Learning for Multi-Access Edge Computing: A Survey on Communication Efficiency and Trustworthiness](https://ieeexplore.ieee.org/document/9645169), IEEE Transactions on Artificial Intelligence.
