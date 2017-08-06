#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:49:14 2017

@author: Ashok Lathwal
"""
import cv2
import os
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib

def getThreshold(dirtyImg):
    Z = dirtyImg.reshape(np.product(np.shape(dirtyImg)),1)
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    #select threshold midway between max value in cluster 1 and min value in cluster 2
    max = 0
    min = 255
    for intensity,cluster in zip(Z,label):
        if cluster == 2 and intensity < min:
            min = intensity
        if cluster == 1 and intensity > max:
            max = intensity

    threshold = 0.5*(max+min)
    return threshold,max,min

def generateThreshodModel():
    thresholdedIntensities = np.empty([0,1])

    #The directory has the training image set
    #iterate for every file in the directory
    for filename in os.listdir('train/'):

        dirtyImg = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE)
        threshold,max,min = getThreshold(dirtyImg)
        ret,dirtyImg = cv2.threshold(dirtyImg,threshold,255,cv2.THRESH_BINARY)
        dirtyImg = dirtyImg.reshape(np.product(np.shape(dirtyImg)),1)
        thresholdedIntensities = np.vstack((thresholdedIntensities,dirtyImg))

    joblib.dump(thresholdedIntensities,'threshold.model')

generateThreshodModel()