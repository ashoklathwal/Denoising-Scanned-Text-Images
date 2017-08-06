#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:36:04 2017

@author: Ashok Lathwal
"""
import cv2
import os
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib

#Execute the 'driver.py' file to generate the model before executing this file.


model = joblib.load('xgb.model')
ipImg = cv2.imread(os.path.join('train',"26.png"),cv2.IMREAD_GRAYSCALE)
shape = np.shape(ipImg)
ipImg = ipImg.reshape(np.product(np.shape(ipImg)),1)
predictions = model.predict(ipImg)
predictions = predictions.reshape(shape)
predictions = np.clip(predictions,0,255)

im = np.array(predictions, dtype = np.uint8)
cv2.imshow('prediction',im)

cv2.waitKey(0)
cv2.destroyAllWindows()