# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:07:53 2018

@author: Christian Salamut
"""

def getError(yhead, data2):
    
    #MSE
    mse = 0
    #need - 1 because data is splittet in 499 / 500
    for i in range(len(data2)):
        mse += np.power((data2[i,2]) - yhead[i,0],2)
    
    
    return mse / (len(data2))


def forwardSearch(data):
    dataval,datatrain = train_test_split(np.random.permutation(data),test_size=500)
    datatrain = (panda.DataFrame(datatrain))
    dataval = np.matrix(dataval)
    
    #error all
    A = np.matrix([np.ones(500),datatrain[0],datatrain[1],
                       np.power(datatrain[0],2),datatrain[0]*datatrain[1],
                       np.power(datatrain[1],2)]).transpose()
        
    leftside = A.transpose() @ A
    ydata = np.matrix(datatrain[2]).transpose()
    rightside = A.transpose() @ ydata
    estimators = np.linalg.solve(leftside, rightside)
    #yhead and starting error
    yhead = A@estimators
    errallbest = getError(yhead,dataval)
    M = [0,1,2,3,4,5]
    variables = []
    vbest=1
    
    
    while vbest != 0:
        vbest = 0
        errbest = errallbest
        
        for v in M:
            variablestemp = variables.copy() 
            variablestemp.append(v)
            #train data and get yhead
            A = np.matrix([np.ones(500),datatrain[0],datatrain[1],
                           np.power(datatrain[0],2),datatrain[0]*datatrain[1],
                           np.power(datatrain[1],2)]).transpose()
            
            leftside = A.transpose() @ A
            ydata = np.matrix(datatrain[2]).transpose()
            rightside = A.transpose() @ ydata
            estimators = np.linalg.solve(leftside, rightside)
            estitemp = np.zeros((6,1))
            estitemp[v,0] = estimators[v,0]
            estimators = estitemp
            #get yhead and errbest
            yhead = (A@estimators)
            err = getError(yhead,dataval)
            
            if err < errbest:
                #update vbest and errbest
                vbest = v
                #update error
                errbest = err
                
        if errbest < errallbest:
            #print the things
            print("vbest:" + str(vbest))
            print("Error:" + str(errbest) + "\n")
            variables.append(vbest)
            #delete used variable
            M.remove(vbest)



import numpy as np
import pandas as panda
from sklearn.model_selection import train_test_split

data = panda.read_csv("tutorial4.2.dat",sep= " ")

for i in range(0,5):
    print("\nRun " + str(i))
    forwardSearch(data)
    
    
#answering questions
print("After running it 5 times parameters 4, 2, 0 and 1 were the starting"
      +" parameters.(I am sorry I am writing this but will also run it"
      + "in Jupyter Notebook, so solutions may differ"
      +"The resulsts are therefore not always the same time.\n"
      +"Sometimes there is no parameter with a lower error than errall.\n"   
      +"I think an improvement would be if you had more datapoints to"
      +" consider, so that DataVal and DataTrain would not differ that much.")
