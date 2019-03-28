# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:52:15 2018

@author: Christian
"""

def trainData(myM, myVtemp, data):
    data = panda.DataFrame(data)
    Ax = np.matrix([np.ones(500),data[0],data[1],
            np.power(data[0],2),data[1]*data[0],
            np.power(data[1],2)]).transpose()
    
    #it took me 2 days to change i with : ...
    for i in myM:
        if i not in myVtemp:
            Ax[i,:] = 0
            
    leftside = Ax.transpose() @ Ax
    ydata = np.matrix(data[2]).transpose()
    rightside = Ax.transpose() @ ydata
    estimator = np.linalg.solve(leftside, rightside)
    return estimator

def getError(estimators, dataval):
    mse = 0
    #remove unused variable
    for i in range (0,len(dataval)):
        temp = estimators[0] + estimators[1] * dataval[i,0]
        +estimators[2] * dataval[i,1] + estimators[3] * np.power(dataval[i,0],2)
        + estimators[4] * dataval[i,0] * dataval[i,1]
        + estimators[5] * np.power(dataval[i,1],2)
        mse += np.power(np.absolute(dataval[i,2] - temp),2)
    return mse/len(dataval)

def forwardSearch():
    #Splitting datasets
    data = panda.read_csv("tutorial4.2.dat",sep= " ", names = ["1","2","3"])
    dataval,datatrain = train_test_split(np.random.permutation(data),test_size=500)
    datatrain = panda.DataFrame(datatrain)
    dataval = np.matrix(dataval)
    #error all
    A = np.matrix([np.ones(500),datatrain[0],datatrain[1],
                   datatrain[0]**2,datatrain[1] * datatrain[0],
                   datatrain[1]**2]).transpose()
    leftside = A.transpose() @ A
    ydata = np.matrix(datatrain[2]).transpose()
    rightside = A.transpose() @ ydata
    estimators = np.linalg.solve(leftside, rightside)
    #initialization
    V = []
    errallbest = getError(estimators,dataval)
    vbest = 1
    M = [5,4,3,2,1]
    #
    while vbest != 0:
        vbest = 0
        errbest = errallbest
        for v in M:
            #it uses pointers
            Vtemp = V.copy()
            Vtemp.append(v)    
            estimators = trainData(M, Vtemp, datatrain)
            #polish estimator
            for j in M:
                if j not in Vtemp:
                    estimators[j] = 0.01
            #get error
            err = getError(estimators, dataval)
            if err < errbest:
                vbest = v
                errbest = err
                
        if errbest < errallbest:
            V.append(vbest)
            M.remove(vbest)
    
            errallbest = errbest
            print("vbest: " + str(vbest))
            print("error: " + str(errbest))

import numpy as np
import pandas as panda
from sklearn.model_selection import train_test_split


for i in range(0,5):
    print("\nRun: " + str(i))
    forwardSearch()

print("After some searches I see that 5,2, and 3 are often in the search"
      +"(I am sorry I am writing this but will also run it"
      + " in Jupyter Notebook, so solutions may differ"
      +"The resulsts are therefore not always the same time.\n"
      +"Sometimes there is no parameter with a lower error than errall.\n"   
      +"I think an improvement would be if you had more datapoints to"
      +" consider, so that DataVal and DataTrain would not differ that much.")