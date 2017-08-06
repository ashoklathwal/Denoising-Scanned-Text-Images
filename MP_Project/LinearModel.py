#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:46:38 2017

@author: Ashok Lathwal
"""
import cv2
import os
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import math

def generateLinearRegressionModel():
    intensities = np.empty([0,1])
    cleanIntensities = np.empty([0,1])

    #The directory has the training image set
    #iterate for every file in the directory
    for filename in os.listdir('train/'):

        dirtyImg = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE)
        dirtyImg = dirtyImg.reshape(np.product(np.shape(dirtyImg)),1)
        intensities = np.vstack((intensities,dirtyImg))

        cleanImg = cv2.imread(os.path.join('train_cleaned',filename),cv2.IMREAD_GRAYSCALE)
        cleanImg = cleanImg.reshape(np.product(np.shape(cleanImg)),1)
        cleanIntensities = np.vstack((cleanIntensities,cleanImg))


    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(intensities,cleanIntensities)

    predictions = np.empty([0,1])

    for filename in os.listdir('train/'):
        ipImg = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE)
        ipImg = ipImg.reshape(np.product(np.shape(ipImg)),1)
        predictions = np.vstack((predictions,np.clip(regr.predict(ipImg),0,255)))

    joblib.dump(predictions, 'linear.model')
    joblib.dump(cleanIntensities, 'clean.val')

    print np.sqrt((np.mean((intensities-cleanIntensities)**2)))/255
    print np.sqrt((np.mean((predictions-cleanIntensities)**2)))/255