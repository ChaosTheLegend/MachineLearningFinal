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
import SVGmodel2 as dataLoader
import numpy as np

imageSize = (64, 64)
sampleSize = 1000
testSize = 0.2

# possible values: 'rgb', 'grayscale', 'rgba'
mode = 'rgb'



X_train, y_train, X_test, y_test = dataLoader.get_data()

# create model
model = SVC(degree=3, gamma='scale', kernel='poly')

# train model
model.fit(X_train, y_train)

#test model
predictions = model.predict(X_test)

# print confusion matrix
matrix = confusion_matrix(y_test, predictions)

#save model into file
dump(model, 'model2.joblib')


#make a report
report = classification_report(y_test, predictions)

#print accuracy
print(np.mean(predictions == y_test))

print(matrix)
print(report)