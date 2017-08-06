import numpy as npll
import cv2
from scipy import signal
from PIL import Image
import os
import gzip
import csv
from sklearn.externals import joblib

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def removeBck():
    final = np.empty([0, 1])
    for f in os.listdir("train/"):
    #     print(imgid)
        # convert image into array
        #inp = np.asarray(Image.open("train/"+f).convert('L'))/255.0

        inp = cv2.imread("train/"+f,cv2.IMREAD_GRAYSCALE)
        bg = cv2.medianBlur(inp,11)

        # estimate 'background' color by a median filter
        #bg = signal.medfilt2d(inp, 11)
        #save('background.png', bg)

        # compute 'foreground' mask as anything that is significantly darker than
        # the background
        mask = inp < bg - 20
        #save('foreground_mask.png', mask)

        # return the input value for all pixels in the mask or pure white otherwise
        dirtyImg = np.where(mask, inp, 255)
        # output path of images
        #print(image.shape)

        #output_path = f
        # save image
        #save(output_path, image)
        dirtyImg = dirtyImg.reshape(np.product(np.shape(dirtyImg)),1)
        final = np.vstack((final,dirtyImg))

    final = final.astype(np.uint8)
    joblib.dump(final,'MedFilter.model')
