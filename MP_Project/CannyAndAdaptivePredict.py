
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

model = joblib.load('final.model')
finalpredictions = np.empty([0,1])

#for filename in os.listdir('train/'):
#    ipImg = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE)
#    ipImg = ipImg.reshape(np.product(np.shape(ipImg)),1)
#    predictions1 = np.vstack((predictions1,np.clip(model.predict(ipImg),0,255)))
#
#joblib.dump(predictions1, 'finalPredictions.model')


for filename in os.listdir('train13/'):
    ipImg = cv2.imread(os.path.join('train13',filename),cv2.IMREAD_GRAYSCALE)
    shape = np.shape(ipImg)
    ipImg = ipImg.reshape(np.product(np.shape(ipImg)),1)
    predictions = model.predict(ipImg)
    predictions = predictions.reshape(shape)
    predictions = np.clip(predictions,0,255)
    finalpredictions = np.vstack((finalpredictions,predictions))

#im = np.array(predictions, dtype = np.uint8)
#cv2.imshow('prediction',im)
joblib.dump(finalpredictions, 'finalPredictions.model')

cv2.waitKey(0)
cv2.destroyAllWindows()