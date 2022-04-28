#load model from the file
from joblib import dump, load
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from skimage.color import rgb2gray
import numpy as np

model = load('model.joblib')
# possible values: 'rgb', 'grayscale', 'rgba'
mode = 'rgb'
imageSize = (64, 64)
sampleSize = 1000
testSize = 0.2

def load_images(names):
    images = []
    for name in names:
        img = imread('images/' + name)
        img = resize(img, imageSize, anti_aliasing=True)
        #remove alpha channel
        if(mode == 'rgb'):
            img = img[:, :, :3]
        elif(mode == 'grayscale'):
            img = rgb2gray(img)

        img = img.flatten()

        images.append(img)
    return images

data = pd.read_csv('sizes.csv')

filenames = data['filename']
filenames = filenames[:sampleSize]
labels = data['label']
labels = labels[:sampleSize]

X = load_images(filenames)
y = labels

#test model
predictions = model.predict(X)

# print confusion matrix
matrix = confusion_matrix(y, predictions)
#make a report
report = classification_report(y, predictions)

#print accuracy
print(np.mean(predictions == y))

print(matrix)
print(report)