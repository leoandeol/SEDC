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
import numpy as np   
import scipy as sp   
from scipy.sparse import lil_matrix
from numba import jit
#import time
from munkres import Munkres
from sklearn.cluster import KMeans
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

###############################################################################
# PART A; define functions used in SEDC (Spectral Embedded Deep Clustering) algorithm.

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


# define Euclid distance matrix function
@jit
def EDM(X):
    
    (n,_) = X.shape
    
    D = np.zeros((n,n),)
    
    for i in np.arange(0,n-1):
        for j in np.arange(i+1,n):
            D[i,j] = np.linalg.norm(X[i,:] - X[j,:])
            
    D = D + D.T
    
    return D


# define KNN graph information function
@jit
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
@jit
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
@jit
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
@jit
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
@jit
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
@jit
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
k0 = 10 

# define the number of negibhor (k1) used in Selective geodesic spectral clustering. This is 
# sensitive hyperparameter
k1 = 10

# define the number of hub nodes This is sensitive hyperparameter.
n_hub = 500

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

# define structure of deep neural net 
class Net(chainer.Chain):

    def __init__(self, n_in=dim, n_hidden=1200, n_out=n_classes):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.bn1 = L.BatchNormalization(n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)
            self.bn3 = L.BatchNormalization(n_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.l1(x)))
        h = F.relu(self.bn2(self.l2(h)))
        h = self.bn3(self.l3(h))

        return h

net = Net()

# define optimizer
optimizer = chainer.optimizers.Adam(alpha=2e-03)
optimizer.setup(net)

# define optimization plan
n_epoch = 50
nl_batchsize = n_labeled//10
nu_batchsize = (n-n_labeled)//100

# set counter for counting number of mini-batch iterations
iteration = 0

#start = time.time()
# train deep neural net
for epoch in range(n_epoch):
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
        net.cleargrads()
        loss.backward()
        
        # update trainable parameters in deep neural net
        optimizer.update()
        
        # update counter
        iteration += 1
        count +=1
        
        # compute value of loss function
        loss_train_i = loss.data
        loss_train = loss_train + loss_train_i
        
    print('loss_train:'+str(loss_train/count))
    
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        Y = net(X)

    # compute clustering accuracy (ACC)
    est_X_labels = np.argmax(F.softmax(Y).data, axis=1)
    est_X_labels = est_X_labels.reshape(1,len(X))
    clustering_acc, _, _ = ACC(est_X_labels, T)
    print('clustering_acc:'+str(clustering_acc))

#elapsed_time = time.time() - start
#print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
