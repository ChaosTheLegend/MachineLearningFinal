# load data from csv file
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from skimage.color import rgb2gray
from joblib import dump, load
import zipfile
from sklearn.model_selection import GridSearchCV
import numpy as np

imageSize = (64, 64)
sampleSize = 1000
testSize = 0.2

# possible values: 'rgb', 'grayscale', 'rgba'
mode = 'rgb'





# load images by name from images folder

def load_images(names, imageDir='images/'):
    images = []
    for name in names:
        img = imread(imageDir + name)
        img = resize(img, imageSize, anti_aliasing=True)
        #remove alpha channel
        if(mode == 'rgb'):
            img = img[:, :, :3]
        elif(mode == 'grayscale'):
            img = rgb2gray(img)

        img = img.flatten()

        images.append(img)
    return images

def get_data():
    os.chdir(os.getcwd() + '/data2')
    # load data from csv file
    df = pd.read_csv('Train.csv')
    # get labels
    labels = df['ClassId']
    # get names
    names = df['Path']
    # get images
    x_train = load_images(names, '')
    y_train = labels

    df = pd.read_csv('Test.csv')
    labels = df['ClassId']
    # get names
    names = df['Path']
    x_test = load_images(names, '')
    y_test = labels

    return x_train, y_train, x_test, y_test

def get_data2():
    os.chdir(os.getcwd() + '/data')
    data = pd.read_csv('sizes.csv')

    filenames = data['filename']
    filenames = filenames[:sampleSize]
    labels = data['label']
    labels = labels[:sampleSize]

    X = load_images(filenames)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=testSize,
        shuffle=True,
        random_state=90
    )

    return X_train, y_train, X_test, y_test
