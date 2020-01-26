#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:14:28 2020

@author: leo
"""

import torch.nn as nn
import torch.nn.functional as F

# define structure of deep neural net 
class Net(nn.Module):

    def __init__(self, n_in, n_out, n_hidden=1200):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_in, n_hidden)
        self.bn1 = nn.BatchNormalization(n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNormalization(n_hidden)
        self.l3 = nn.Linear(n_hidden, n_out)
        self.bn3 = nn.BatchNormalization(n_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.l1(x)))
        h = F.relu(self.bn2(self.l2(h)))
        h = self.bn3(self.l3(h))

        return h
