#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from operations import *
import csv

## Operation layer
class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)
            
        self._out = nn.Sequential(nn.Conv2d(C*len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1,0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x)*w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))

## a Group of operation layers
class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0

## Operation-wise Attention Layer (OWAL)
class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

## entire network (the number of layers = layer_num * steps)
class Network(nn.Module):
    def __init__(self, C, layer_num, criterion, steps=4, gpuID=0):
        super(Network, self).__init__()
        self._C = C
        self._layer_num = layer_num
        self._criterion = criterion
        self._steps = steps
        self.gpuID = gpuID
        self.num_ops = len(Operations)
        
        self.kernel_size = 3
        # Feature Extraction Block
        self.FEB = nn.Sequential(nn.Conv2d(3, self._C, self.kernel_size, padding=1, bias=False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),)
 
        # a stack of operation-wise attention layers
        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]
        
        # Output layer
        self.conv2 = nn.Conv2d(self._C, 3, self.kernel_size, padding=1, bias=False)


    def forward(self, input):
        s0 = self.FEB(input)
        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(s0)
                weights = F.softmax(weights, dim=-1)
            else:
                s0 = layer(s0, weights)
        
        logits = self.conv2(s0)
        return logits