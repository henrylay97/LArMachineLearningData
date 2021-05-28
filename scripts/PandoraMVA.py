#!/usr/bin/env python
# PandoraMVA.py

from sklearn import preprocessing
from datetime import datetime

import numpy as np
import sys
import time
import pickle

def LoadData(trainingFileName, delimiter=','):
    # Use the first example to get the number of columns
    with open(trainingFileName) as file:
        ncols = len(file.readline().split(delimiter))

    # First column is a datestamp, so skip it
    trainingSet = np.genfromtxt(trainingFileName, delimiter=delimiter, usecols=range(1,ncols),
            dtype=None)

    nExamples = trainingSet.size
    nFeatures = ncols - 2 # last column is the response

    return np.array(trainingSet), nFeatures, nExamples

#--------------------------------------------------------------------------------------------------

def RemoveFeature(data,col):
    new=[]
    ncols = len(data[0])
    
    for example in data:
        features=[]
        for i in [x for x in range(0,ncols) if x not in col] :
            features.append(example[i])
            
        new.append(features)
        
    return new

#--------------------------------------------------------------------------------------------------

def GetWeights(data,col):
    weights = []
    ncols = len(data[0])

    for example in data:
        weights.append(1/example[col])

    return weights

#--------------------------------------------------------------------------------------------------

def GetVetos(data,variable=91,cut=0.3):
    vetos = []
    ncols = len(data[0])

    for example in data:
        if example[variable] < cut :
            vetos.append(1)
        else :
            vetos.append(0)

    return vetos

#--------------------------------------------------------------------------------------------------

def Filter(X,Y,Z,V):

    x = X[V!=1]
    y = Y[V!=1]
    z = Z[V!=1]
    
    return x, y, z

#--------------------------------------------------------------------------------------------------

def SplitTrainingSet(trainingSet, nFeatures):
    X=[] # features sets
    Y=[] # responses

    for example in trainingSet:
        Y.append(int(example[nFeatures])) # type of Y should be bool or int
        features = []
        for i in range(0, nFeatures):
            features.append(float(example[i])) # features in this SVM must be Python float

        X.append(features)

    return np.array(X).astype(np.float64), np.array(Y).astype(np.int)

#--------------------------------------------------------------------------------------------------

def Randomize(X, Y, setSameSeed=False):
    if setSameSeed:
        np.random.seed(0)

    order = np.random.permutation(Y.size)
    return X[order], Y[order]

#--------------------------------------------------------------------------------------------------

def Randomize(X, Y, Z, V, setSameSeed=False):
    if setSameSeed:
        np.random.seed(0)

    z = np.array(Z)
    v = np.array(V)

    order = np.random.permutation(Y.size)
    return X[order], Y[order], z[order], v[order]

#--------------------------------------------------------------------------------------------------

def Sample(X, Y, testFraction=0.1):
    trainSize = int((1.0 - testFraction) * Y.size)

    X_train = X[:trainSize]
    Y_train = Y[:trainSize]
    X_test  = X[trainSize:]
    Y_test  = Y[trainSize:]

    return X_train, Y_train, X_test, Y_test

#--------------------------------------------------------------------------------------------------

def Sample(X, Y, Z, V, testFraction=0.1):
    trainSize = int((1.0 - testFraction) * Y.size)

    X_train = X[:trainSize]
    Y_train = Y[:trainSize]
    Z_train = Z[:trainSize]
    V_train = V[:trainSize]
    X_test  = X[trainSize:]
    Y_test  = Y[trainSize:]
    Z_test  = Z[trainSize:]

    return X_train, Y_train, Z_train, V_train, X_test, Y_test, Z_test

#--------------------------------------------------------------------------------------------------

def ValidateModel(model, X_test, Y_test):
    return model.score(X_test, Y_test)

#--------------------------------------------------------------------------------------------------

def OverwriteStdout(text):
    sys.stdout.write('\x1b[2K\r' + text)
    sys.stdout.flush()

#--------------------------------------------------------------------------------------------------

def OpenXmlTag(modelFile, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>\n')
    return indentation + 4

#--------------------------------------------------------------------------------------------------

def CloseXmlTag(modelFile, tag, indentation):
    indentation = max(indentation - 4, 0)
    modelFile.write((' ' * indentation) + '</' + tag + '>\n')
    return indentation

#--------------------------------------------------------------------------------------------------

def WriteXmlFeatureVector(modelFile, featureVector, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')

    firstTime=True
    for feature in featureVector:
        if firstTime:
            modelFile.write(str(feature))
            firstTime=False
        else:
            modelFile.write(' ' + str(feature))

    modelFile.write('</' + tag + '>\n')

#--------------------------------------------------------------------------------------------------

def WriteXmlFeature(modelFile, feature, tag, indentation):
    modelFile.write((' ' * indentation) + '<' + tag + '>')
    modelFile.write(str(feature))
    modelFile.write('</' + tag + '>\n')

