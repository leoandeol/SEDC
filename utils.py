#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:19:34 2020

@author: leo
"""

import numpy as np
import scipy as sp   
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from munkres import Munkres
from sklearn.cluster import KMeans
from torch.autograd import Variable


###############################################################################
# PART A; define functions used in SEDC (Spectral Embedded Deep Clustering) algorithm.


# define Euclid distance matrix function
def EDM(X):
    
    (n,_) = X.shape
    
    D = np.zeros((n,n),)
    
    for i in np.arange(0,n-1):
        for j in np.arange(i+1,n):
            D[i,j] = np.linalg.norm(X[i,:] - X[j,:])
            
    D = D + D.T
    
    return D


# define KNN graph information function
def KNN_Info(X,k):
    
    D = EDM(X)
    
    (n,_) = D.shape
    sortD = np.zeros((n,k),)
    sortD_idx = np.zeros((n,k),dtype=int)
    
    
    for i in np.arange(0,n):
        d_vec = D[i,:]
        v = np.argsort(d_vec)
        sortD_idx[i,:] = v[1:k+1]
        sortD[i,:] = d_vec[sortD_idx[i,:]]
           
    
    deg = np.zeros((1,n),dtype=int)
    v = sortD_idx.reshape(1,n*k)
    for i in np.arange(0,n):
        (_,v0) = np.where(v == i)
        deg[:,i] = len( v0 )
    
    
    return sortD, sortD_idx, deg


# define function which create graph from adjacency matrix 
def Un_Digraph(sortD, sortD_idx):
    
    (n,_) = sortD.shape
    
    A = lil_matrix( (n,n), dtype=float )
    for i in np.arange(0,n):
        A[i, sortD_idx[i,:]] = sortD[i,:]
    
    U = lil_matrix(sp.sparse.triu(A.T - A, k=0))
    u = U.nonzero()
    u0 = u[0]
    u1 = u[1]
    l = len(u0)
    for i in np.arange(0,l):
        a = U[u0[i],u1[i]]
        if a > 0:
            A[u0[i],u1[i]] = a
        else:
            A[u1[i],u0[i]] = -a
    
    return A


# define similarity matrix for spectral clustering
def Sim(GD, idx, k):
    
    (_, h) = idx.shape
    
    D = np.zeros((h,h),dtype=float)
    for i in np.arange(0,h):
        D[i,:] = GD[idx[0,i],idx[0,:]]
    
    sortD = np.zeros((h,k),)
    sortD_idx = np.zeros((h,k),dtype=int)
    for i in np.arange(0,h):
        d_vec = D[i,:]
        v = np.argsort(d_vec)
        sortD_idx[i,:] = v[1:k+1]
        sortD[i,:] = d_vec[sortD_idx[i,:]]
        
    W = np.zeros((h,h),dtype=float)
    for i in np.arange(0,h):
        W[i, sortD_idx[i,:]] = sortD[i,:]
    
    u = W[W.nonzero()]
    u = u.reshape(1,h*k)
    inf_idx = np.where(u == np.inf)
    np.delete(u, inf_idx, 1)
    
    sigma = np.median(u)
    W = np.exp( -(W**2)/(sigma**2) )
    W = np.where(W != 1, W, 0)
    
    U = np.triu(W.T - W, k=0)
    U = np.where(U >= 0, U, 0)
    V = np.tril(W.T - W, k=0)
    V = np.where(V >= 0, V, 0)
    
    W = W + U + V
    W = (W + W.T)*0.5
    
    return W, sigma
  

# transform d into unit vector
def trans_unit_v(d):
    
    (m, dim) = d.shape
    
    dist = np.linalg.norm(d,axis=1)

    dist = np.tile(dist, (dim,1)).T
    
    unit_v = d/(dist + 1e-25)
    
    return unit_v


# define spectral clustering function with W, then define class probability with given x
def SC(W,C):
    
    (h, _) = W.shape
    
    v = ( np.dot(W, np.ones((1,h)).reshape(h,1)) )**(-0.5) 
    Di = np.diagflat(v)
    L = Di @ W @ Di
    
    ei = np.linalg.eig(L) 
    M = ei[1]
    
    x = M[:,0:C]
    x = trans_unit_v(x)
    
    km_model = KMeans(n_clusters=C, init='k-means++').fit(x) 
    est_labels = km_model.labels_
    est_labels = est_labels.reshape(1,h)
    c_cent = km_model.cluster_centers_ 

    est_class_balance = np.zeros((1,C))
    for i in np.arange(0,C):
        s = np.where(est_labels == i)
        s = s[0]
        est_class_balance[0,i] = len(s)/h
    
    alpha = 1e-60
    P = np.zeros((h,C))
    for i in np.arange(0,h):
        
        nor = 0
        
        for j in np.arange(0,C):
            nor = ( 
                    nor + ( 1 + ( ( np.linalg.norm(x[i,:] - 
                                c_cent[j,:])**2 )/alpha ) )**( -(alpha + 1)/2 )  
                            )
    
        for j in np.arange(0,C):
            P[i,j] = ( 
                    ( ( 1 + ( ( np.linalg.norm(x[i,:] - 
                        c_cent[j,:])**2 )/alpha ) )**( -(alpha + 1)/2 ) )/nor
                            )
    
    max_p = P.max(axis=1)
    
    return est_labels, c_cent, est_class_balance, P, max_p


# define function which gets ture labels of hub nodes and its class balance
def hub_info(Y, idx, h):
    
    Hub_idx = idx[0,0:h].reshape(1,h)
    
    Y_Hub = Y[0,Hub_idx]
    
    C = len(np.unique(Y))
    
    true_class_balance = np.zeros( (1,C), dtype=float )
    for i in np.arange(0,C):
        s = np.where(Y_Hub == i)
        s = s[0]
        true_class_balance[0,i] = len(s)/h
    
    true_class_balance = true_class_balance[0,:]
    
    return Y_Hub, Hub_idx, true_class_balance


# define function of clustering accuracy: ACC
def ACC(y_predict, y_true):
    
    y_pred = y_predict.copy()
    
    if len( np.unique(y_pred) ) == len( np.unique(y_true) ):
        C = len( np.unique(y_true) )
        
        cost_m = np.zeros( (C,C), dtype=float )
        for i in np.arange(0,C):
            a = np.where(y_pred == i)
            a = a[1]
            l = len(a)
            for j in np.arange(0,C):
                yj = np.ones((1,l)).reshape(1,l)
                yj = j*yj
                cost_m[i,j] = np.count_nonzero(yj - y_true[0,a])
                
                
        mk = Munkres()
        best_map = mk.compute(cost_m)
        
        (_,h) = y_pred.shape
        for i in np.arange(0,h):
            c = y_pred[0,i]
            v = best_map[c]
            v = v[1]
            y_pred[0,i] = v
        
        acc = 1 - (np.count_nonzero(y_pred - y_true)/h)
        
    else:
        acc = 0
    
    
    return acc, best_map, y_pred
            

# shannon entropy
def ent(p):
    
    h = -F.sum( F.softmax(p)*F.log_softmax(p) )/p.shape[0]
    
    return h


# KL-divergence
def kldiv1(p,q):
    
    kl = F.sum( F.softmax(p)*( F.log_softmax(p) - F.log_softmax(q) ) )/p.shape[0]

    return kl


def kldiv2(p,q):
    
    kl = F.sum( F.softmax(p)*( F.log_softmax(p) - F.log(q) ) )/p.shape[0]
    
    return kl


# compute virtual adversarial vector
def cal_vat_v(dnn, x):
    
    xi = 10
    eps = 1
    
    (m, dim) = x.shape
    
    d = np.random.randn(m,dim)
    d = trans_unit_v(d)

    r = Variable(np.array(xi*d, dtype=np.float32))
    x_in = F.concat((x,x+r), axis=0)
    y_out = dnn(x_in)
    y_out_split = F.split_axis( y_out, 2, axis=0 )
    z0 = y_out_split[0]
    z1 = y_out_split[1]
    
    tgt = z0.data
    
    loss_apro = kldiv1( tgt, z1 )
    
    dnn.cleargrads()
    loss_apro.backward()
    
    g = r.grad
    r_adv = (eps)*trans_unit_v(g)

    return r_adv