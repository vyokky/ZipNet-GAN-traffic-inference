# ZipNet-GAN-traffic-inference

This repository contains the Python code for ZipNet-GAN introduced in the paper: ZipNet-GAN: "Inferring Fine-grained Mobile Traffic Patterns via a Generative
Adversarial Neural Network."

Current version only supports 4 MTSR instances (up-2, up-4, up-10 and mixture), please config --downscale (2, 4, 10, mix) to train/test on requried instance. Both training and testing require GPU for accleration. The code is trained/tested on a processed version of the mobile traffic dataset released here: https://dandelion.eu/datamine/open-big-data/. Please contact authors for the raw dataset.

Requirement: Python2 >= 2.7, numpy >= 1.12.0, tensorflow-gpu >= 1.0.0, tensorlayer >= 1.3.10, CUDA >= 5.1, cuDNN >= 8.0

Files:

       zipnet-training.py Training ZipNet by minimising l2 distance.

       zipnet-gan-training.py Adversirial training of ZipNet-GAN. Please initialise ZipNet by zipnet-training.py prior to running this script.
       
       inference.py Performing MTSR with a trained model.

       toolbox folder: Essential tools for zipnet-gan implementation
       architecture folder: Deep learning models used in our paper 

We will release a detailed document after official publication.

 
