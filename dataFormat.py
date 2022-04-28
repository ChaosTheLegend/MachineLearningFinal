#make svm model with the data in the data folder and classify it in 5 classes

#importing the libraries
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import os
from xml.etree.ElementTree import parse


#importing images from the data/images folder
os.chdir(os.getcwd()+'/data')

print(os.getcwd())

#loading the dataset from the images folder
files = [f for f in listdir('images') if isfile(join('images', f))]

dataset = []

for file in files:
    img = plt.imread('images/'+file)
    dataset.append(img)

sizes = []
labels = []


#get count of all the files in the data/annotations folder

print(len(dataset))
for i in range(0,len(dataset)):
    xml = parse('annotations/road'+str(i)+'.xml')
    size = xml.find('size')
    object = xml.find('object')
    label = object.find('name').text
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    sizes.append(('road'+str(i)+'.png',label, width , height))

print(sizes)

#writes the sizes in a csv file, create a new csv file if it doesn't exist
df = pd.DataFrame(sizes, columns = ['filename','label', 'width', 'height'])
df.to_csv('sizes.csv', index = False)