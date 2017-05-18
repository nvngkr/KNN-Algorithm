# -*- coding: utf-8 -*-
"""
@author: gnaveen
"""

from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import math
import operator
from collections import Counter

#computing euclidean distance
def getEuclideanDistance(trainSetDataItem,testSetDataItem):
     distance = 0
     #print("trainSetDataItem=",len(trainSetDataItem))
     for i in range(len(testSetDataItem)):
         distance += pow((testSetDataItem[i] - trainSetDataItem[i]), 2)
     return math.sqrt(distance)
    
#Neigbouring points of a point     
def getNeighbouringPoints(trainSet,labels_train, testSetDataItem, k):
    euclideanDistance = []
    for i in range(len(trainSet)):
        #print("data", trainSet[i],"label", labels_train[i])
        d = getEuclideanDistance(trainSet[i], testSetDataItem)
        euclideanDistance.append((trainSet[i], d, labels_train[i]))
    #print("dist", euclideanDistance)
    euclideanDistance.sort(key=operator.itemgetter(1))
    #print("\n\n\ndist----",euclideanDistance)
    neighbourDataItems = []
    for i in range(k):
        neighbourDataItems.append((euclideanDistance[i][0],euclideanDistance[i][2]))
    return neighbourDataItems
  
#get the labels of computed neighbours
def getLabelsOfNeighbours(neighbourDataItems):

    predictions = []
    count = 0
    #counts = Counter(neighbourDataItems)
    #print(counts)
    for i in range(len(neighbourDataItems)):
    
        prediction = neighbourDataItems[i][1]
        predictions.append(prediction[0])
        #print("\nres=" , prediction)   
    counts = Counter(predictions)
    print("\nLabel=", counts.most_common(1)[0][0])
   
    
#main method
def main():
    collection = pd.read_csv('iris.data',sep=',', header=None)
    data_array = np.array(collection)
    
    #Separating data and labels from the dataset
    data = data_array[:,[0,1,2,3]]
    label = data_array[:,[4]]
    
    #Splitting data and label for training and testing
    data_train, data_test, labels_train, labels_test = train_test_split(data,label, test_size=.1, random_state=42)
   
    # labels_train, labels_test =
    #print("data_train=",data_train,"\n data_test=",data_test,
    #  "labels_train=", labels_train, "\nlabels_test=", labels_test)
    
    #Let the value of k (number of neighbouring points taken) be 3'''

    k = 3
    for i in range(len(data_test)):
        neighbourDataItems = getNeighbouringPoints(data_train,labels_train, data_test[i], k)
        getLabelsOfNeighbours(neighbourDataItems)
main()
