# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import os
import xgboost as xgb
import numpy as np
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.externals import joblib

def cannyDilated1(img):
    edges = cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations =1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    return (erosion)

def cannyDilated2(img):
    edges = cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations =1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    erosion2 = cv2.erode(erosion,kernel,iterations = 1)
    dilation2 = cv2.dilate(erosion2,kernel,iterations =1)
    return (dilation2)


intensities = np.empty([0,6])
cleanIntensities = np.empty([0,1])

for filename in os.listdir('train/'):

    img = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE);
    cleanImg = cv2.imread(os.path.join('train_cleaned',filename),cv2.IMREAD_GRAYSCALE);
    cleanImg = cleanImg.reshape(np.product(np.shape(cleanImg)),1)
    cleanIntensities = np.vstack((cleanIntensities,cleanImg))

    #cleanImg = cv2.imread(os.path.join('train_cleaned',"27.png"),cv2.IMREAD_GRAYSCALE);
    adapThres= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,9,9)
    cannyImg = cv2.Canny(img,100,200)


    cannyDil1=cannyDilated1(img)
    cannyDil2=cannyDilated2(img)

    ret,kmeansThresh1 = cv2.threshold(img,30,255,cv2.THRESH_BINARY)


    img = img.reshape(np.product(np.shape(img)),1)      #dirty image
    #cleanImg = cleanImg.reshape(np.product(np.shape(cleanImg)),1)
    kmeansThresh1 = kmeansThresh1.reshape(np.product(np.shape(kmeansThresh1)),1)
    adapThres=adapThres.reshape(np.product(np.shape(adapThres)),1)
    cannyImg = cannyImg.reshape(np.product(np.shape(cannyImg)),1)
    cannyDil1 = cannyDil1.reshape(np.product(np.shape(cannyDil1)),1)
    cannyDil2 = cannyDil2.reshape(np.product(np.shape(cannyDil2)),1)

    x = np.column_stack((img,kmeansThresh1,adapThres,cannyImg,cannyDil1,cannyDil2))
    #x1=x.reshape(np.product(np.shape(x)),1)

    intensities = np.vstack((intensities,x))

#clean = joblib.load('clean.val')

model = XGBRegressor()

model.fit(intensities,cleanIntensities)
joblib.dump(model,'final.model')



#cv2.imwrite('canny1.png',img1);
#cv2.imwrite('canny2.png',img2);
#cv2.imwrite('third.png',edges);













