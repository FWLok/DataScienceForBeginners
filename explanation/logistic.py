#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:25:46 2017

@author: wongtszlunjohneinstein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import linear_model


data = pd.read_csv('iris.csv')
data = data[data.Species != 'Iris-virginica']
train, test = train_test_split(data, test_size=0.2, random_state=42)

train_labels = train['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1})
train_features = train.drop(['Id','Species'],axis=1).as_matrix()

test_labels = train['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1})
test_features = train.drop(['Id','Species'],axis=1).as_matrix()

X = train_features[:,2].reshape(-1,1)
y= train_labels

