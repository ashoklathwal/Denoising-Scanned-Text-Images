#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:52:39 2017

@author: Ashok Lathwal
"""
import cv2
import os
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load data
linear = joblib.load('linear.model')
clean = joblib.load('clean.val')
threshold = joblib.load('threshold.model')
medianFilter = joblib.load('MedFilter.model')

x = np.column_stack((linear,threshold,medianFilter))

# fit model to training data
model = XGBRegressor()
model.fit(x,clean)

joblib.dump(model,'xgb.model')
