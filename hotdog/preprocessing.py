import cv2 as cv
import os
import numpy as np
import pickle as pk

data = [] # hold vector data per image
labels = [] # label for each image

def createVec(label,doc):
    for filename in os.listdir(doc):
        f = doc + '\\' + filename
        img = cv.imread(f,0)
        resized_image = cv.resize(img, (28, 28)) # reshape images to uniform size
        normalize = resized_image/255 # get pixel intensity values/greyscale values
        flattened = normalize.flatten() # flatten vecotr to be a a vector with 28 * 28 elements
        data.append(flattened) 
        labels.append(label)
        
createVec(0,'hot_dogs\\NOT_HOT_DOG') # do for not hot dog photos
createVec(1,'hot_dogs\\hot_dog') # do for hot dog photos

data = np.array(data) # convert to numpy array
labels = np.array(labels) # convert to numpy array
allData = (data,labels) # store in tuple

pk.dump( allData, open( "hot_dogs\\allData.p", "wb" ) ) # save as pickle file

	