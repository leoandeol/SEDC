# -*- coding: utf-8 -*-


###############################################################################
###############################################################################
# The following code is Spectral Embedded Deep Clustering (SEDC). 
# This code consits of three parts (PART A, PART B and PART C).
# Part A is for definition of functions used in this SEDC code.
# Part B is for Selective Geodesic Spectral Clustering (SGSC).
# In this code, MNIST dataset is used for the demonstration. 
# When we implement with (k0, k1, n_hub, lambda1, lambda2) = (10, 10, 500, 1, 1),
# the clustering accuracy is around 0.887 where the number of unlabeled data points are 70000.  
# The most sensitive hyperparameters are k1 and n_hub, second most is k0.
# lambda1 and lambda2 is relatively robust if you set both to 1. 
# In practice, the competitive combination of hyperparameters may be chose 
# by hyperparameter transfer as shown in our paper. In case of MNIST, 
# the source is, for an example, USPS dataset.
# Reference: Yuichiro WADA , Shugo MIYAMOTO, Takumi NAKAGAWA, Leo ANDEOL, Wataru
# KUMAGAI, Takafumi KANAMORI, “Spectral Embedded Deep Clustering”, Entropy
# Journal , vol. 21, no. 8, pp. 795 810, August 2019.
# Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
###############################################################################
###############################################################################


# import needed library
import argparse
import numpy as np   
import scipy as sp   
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from data import *
from model import Net

parser = argparse.ArgumentParser()

parser.add_argument("--hubs", default=500, type=int, help="Number of hubs. It is a sensitive parameter.")
parser.add_argument("--k0", default=10, type=int, help="The number of neighbors k0 is used to construct the k-nn graph.")
parser.add_argument("--k1", default=10, type=int, help="The number of neigbhors k1 is used in Selective geodesic spectral clustering.")
parser.add_argument("--n_epochs", default=50, type=int, help="Number of epochs")
#parser.add_argument("--batch_size", default=128, type=int, help="Size of batch")
parser.add_argument("--learning_rate", default=2e-03, type=float)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=torch.cuda.is_available())
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

torch.random.manual_seed(42)


###############################################################################
# PART B; compute cluster labels on hub data points by using SGSC (spectral 
# embedded deep clustering)


# create train dataset of mnist
n_classes = 10 
n = 70000 
(x,t) = g_mnist_d(n)
print('x:', x.shape)
print('t:', t.shape)
(_, dim) = x.shape

# type change
x = x.astype('float32')
t = t.astype('int32')
t = t.reshape(1,n)

#start = time.time()

# obtain knn information. The number of neighbors k0 is used to construct graph G. 
# k0 is hyperparameter.
k0 = args.k0

# define the number of negibhor (k1) used in Selective geodesic spectral clustering. This is 
# sensitive hyperparameter
k1 = args.k1

# define the number of hub nodes This is sensitive hyperparameter.
n_hub = args.hubs

sortED, sortED_idx, _ = KNN_Info( x, k0 )
print("Finish to compute KNN information.")

#elapsed_time = time.time() - start
#print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# define undirected graph G
G = Un_Digraph(sortED, sortED_idx)

#start = time.time()

#　compute geodesic distance on G
GD = sp.sparse.csgraph.johnson(G)
print("Finish to compute the geodesic distance matrix.")

#elapsed_time = time.time() - start
#print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#　define high-degree nodes
deg = G.getnnz(axis=1)
deg = deg.reshape(1, n)
sorted_deg_idx = np.argsort(-deg)

# get hub information for evaluating clustering accuracy
t_hub, hub_idx, _ = hub_info(t, sorted_deg_idx, n_hub)
            
# define similarity matrix based the number of hub (n_hub)
W, _ = Sim(GD, hub_idx, k1)

# estimate the cluster label of hub nodes by using function SC (function SC is actually SGSC)
est_t_hub_wo_perm, _, _, est_t_hub_catego_wo_perm, _ = SC( W, n_classes )

# compute accuracy of estimete label of hub nodes
acc_hub_nodes, _, _ = ACC(est_t_hub_wo_perm, t_hub)

print("Finish SGSC.")
###############################################################################
# PART C; by using cluster labels of hub points, semi-supervised learning based on 
# deep neural network is conducted. 


print("Start semi-supervised learning.")

# hyperparameters used in loss (objective) function
lambda1 = 1
lambda2 = 1

# define cluster labeled data points
xl_train = x[hub_idx,:]
xl_train = xl_train[0]
xl_train = xl_train.astype('float32')

# define true labels on hub points (used to evaluate ACC)
tl_train = est_t_hub_catego_wo_perm.copy()
tl_train = tl_train.astype('float32')

(n_labeled, _) = xl_train.shape

# define remaining unlabeled data points
xu_train = np.delete(x, hub_idx, 0)
xu_train = xu_train.astype('float32')

# define the true labels of remaining unlabeled points (used to evaluate ACC)
tu_train = np.delete(t, hub_idx, 1)
tu_train = tu_train.astype('float32')

X = np.r_[xl_train, xu_train]
T = np.c_[t_hub, tu_train]


net = Net()
net.train()


if args.cuda:
    net.cuda()
    print("Using CUDA")
else:
    print("Using CPU")

# define optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)

# define optimization plan
nl_batchsize = n_labeled//10
nu_batchsize = (n-n_labeled)//100

# set counter for counting number of mini-batch iterations
iteration = 0

#start = time.time()
# train deep neural net
for epoch in range(args.n_epochs):
    print('epoch:'+str(epoch))
    
    # permute dataset for creating mini-batch
    order_unlabeled = np.random.permutation(range(len(xu_train)))
    
    # set counter for counting number of epoch
    count = 0
    
    # intial value of loss function
    loss_train = 0
    
    n_block = np.ceil(n_labeled/nl_batchsize).astype('int32')
    for i in range(0, len(order_unlabeled), nu_batchsize):

        # define mini-batch which is composed of both pesudo labeled and unlabeled data points
        index_u = order_unlabeled[i:i+nu_batchsize]
        xu_train_batch = xu_train[index_u,:]
        (m2,_) = xu_train_batch.shape
        
        if count%n_block == 0:
            order_labeled = np.random.permutation(range(len(xl_train)))
            
        j = nl_batchsize*(count%n_block)    
        index_l = order_labeled[j:j+nl_batchsize]
        
        xl_train_batch = xl_train[index_l,:]
        tl_train_batch = tl_train[index_l,:]
        
        (m1,_) = xl_train_batch.shape
        xlu_train_batch = np.r_[ xl_train_batch, xu_train_batch ]
        
        # compute virtual adversarial vectors
        r_adv = cal_vat_v(net, xlu_train_batch)
        
        # define input data to deep net from the mini-batch
        x_batch = np.r_[ xlu_train_batch, xlu_train_batch + r_adv ]
        x_batch = x_batch.astype('float32')
        
        # compute output by deep neural net
        y_batch = net(x_batch)
        y_batch_separable = F.split_axis( y_batch, [m1, m1+m2], axis=0 )
        y1 = y_batch_separable[0]
        y2 = y_batch_separable[1]
        y3 = y_batch_separable[2]
        
        # define VAT loss 
        target_p = np.r_[y1.data, y2.data]
        loss1 = kldiv1(target_p, y3)
        
        # define pseudo cross entropy loss
        loss2 = kldiv2(y1, tl_train_batch)
        
        # define entropy loss
        loss3 = ent(F.concat((y1,y2),axis=0))
        
        # define total loss function (which should be optimized)
        loss = loss1 + lambda1*loss2 + lambda2*loss3 
        
        #　compute gradient
        optimizer.zero_grad()
        loss.backward()
        
        # update trainable parameters in deep neural net
        optimizer.step()
        
        # update counter
        iteration += 1
        count +=1
        
        # compute value of loss function
        loss_train_i = loss.data
        loss_train = loss_train + loss_train_i
        
    print('loss_train:'+str(loss_train/count))
    
    with torch.no_grad():
        Y = net(X)

    # compute clustering accuracy (ACC)
    est_X_labels = np.argmax(F.softmax(Y).data, axis=1)
    est_X_labels = est_X_labels.reshape(1,len(X))
    clustering_acc, _, _ = ACC(est_X_labels, T)
    print('clustering_acc:'+str(clustering_acc))

#elapsed_time = time.time() - start
#print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
