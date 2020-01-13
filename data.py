#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:19:28 2020

@author: leo
"""
import numpy as np
from torchvision.datasets import MNIST


# make mnist dataset
def g_mnist_d(n):
    from keras.datasets import mnist
    # download minst
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    (num_train, num_pixel, _) = train_images.shape
    (num_test, _, _) = test_images.shape
    
    N = num_train + num_test
    
    X_train = train_images.reshape(num_train,num_pixel**2)
    X_train = X_train/255
    X_test = test_images.reshape(num_test,num_pixel**2)
    X_test = X_test/255
    
    Y_train = train_labels.reshape(1,num_train) 
    Y_test = test_labels.reshape(1,num_test) 
    
    X = np.r_[X_train,X_test]
    Y = np.c_[Y_train,Y_test]
    
    v = np.random.choice(np.arange(0,N),n,replace=False)
    
    x = X[v,:]
    y = Y[:,v]
    y = y[0]
    
    return x, y
