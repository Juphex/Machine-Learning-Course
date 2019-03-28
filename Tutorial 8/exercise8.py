# -*- coding: utf-8 -*-
"""
@author: Christian Salamut

"""

def getDistance(q, x):
    return np.abs(q - x)

#get lowest K distances
def argminK(D, K):
    D = D.sort_values("distance",ascending=True)
    retVal = D.head(K).reset_index(drop=True)
    return retVal

def getYhead(C,K):
    yhead = 0
    for i in range(K):
        yhead += C["Y"][i]
        
    return np.around(yhead / K,4)

def KNNregressionNoPlot(Dtrain, q, K):
    D = Dtrain
    D["distance"] = 0.0
    
    for i in range(len(D)):
        D["distance"][i] = getDistance(q, Dtrain["X"][i])
        
    C = argminK(D,K)
    yhead = getYhead(C,K)

    return yhead

def KNNregression(Dtrainy, q, K):

    #plotting
    plt.scatter(x = Dtrainy["X"],y = Dtrainy["Y"])
    plt.xlabel("x")
    plt.ylabel("y")
    
    D = Dtrain.copy()
    D["distance"] = 0.0
    
    for i in range(len(D)):
        D["distance"][i] = getDistance(q, Dtrainy["X"][i])
        
    C = argminK(D,K)
    
    yhead = getYhead(C,K)
    plt.scatter(x = q, y = yhead,color = "red")
    plt.legend(["data","yhead"])
    plt.show()
    return yhead

def task11(Dtrain, Dtest, K):
    for i in range (len(Dtest)):
        Dtest["Y"][i] = KNNregressionNoPlot(Dtrain, Dtest["X"][i], K)

    plt.scatter(x = Dtrain["X"],y = Dtrain["Y"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x = Dtest["X"],y = Dtest["Y"], c = "r")
    plt.legend(["y of Dtrain","yhead of Dtest"])
    plt.show()

def getMSE(D, K):
    mse = 0
    for i in range(len(D)):
        mse += np.power(KNNregressionNoPlot(D, D["X"][i], K) 
                - D["Y"][i],2)
        
    return mse/len(D)
    
def task13(Dtrain, Dtest2):
#    scoreMseDtest = 0
    mseDtrain = pd.DataFrame(np.zeros((50,1)), columns = ["MSE"])
    mseDtest = pd.DataFrame(np.zeros((50,1)), columns = ["MSE"])
    for i in range(0, 50):
        #K starting from 1 to 50
        K = i + 1
#        print("K = " + str(K))
        mseDtrain["MSE"][i] = getMSE(Dtrain, K)
        mseDtest["MSE"][i] = getMSE(Dtest2, K)

    mseDtrain = mseDtrain.sort_values("MSE",ascending=True)
    mseDtest = mseDtest.sort_values("MSE",ascending=True)
    #plot mseDtrain
    print("MSE Dtrain:")
    plt.scatter(x = mseDtrain.index, y = mseDtrain["MSE"])
    plt.xlabel("K = K + 1")
    plt.ylabel("MSE")
    plt.show()
    print("MSE Dtest:")
    plt.scatter(x = mseDtest.index, y = mseDtest["MSE"])
    plt.xlabel("K = K + 1")
    plt.ylabel("MSE")
    plt.show()
    print("The optimal K value here is 1, taking the closest point")
    
    return [mseDtrain, mseDtest]

#impressions: https://en.wikipedia.org/wiki/Levenshtein_distance
def levenshteinor(str1, lenstr1, str2, lenstr2):
    actions = 0
    #strings empty
    if (lenstr1 == 0):
        return lenstr2
    elif (lenstr2 == 0):
        return lenstr1
    
    #char at len - 1
    if (str1[lenstr1 - 1] == str2[lenstr2 - 1]):
        actions = 0
    else:
        actions = 1
    #return min of the 3 deletions
    return np.amin([levenshteinor(str1, lenstr1 - 1, str2, lenstr2 -1)+actions,
                    levenshteinor(str1, lenstr1 - 1, str2, lenstr2) +1,
                   levenshteinor(str1, lenstr1, str2, lenstr2 -1)])
    
    
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#generate 100 Datapoints
print("1.1) \n")
Dtrain = pd.DataFrame(np.zeros((100,2)), columns = list('XY'))
#random doubles
for i in range(0,100):
    Dtrain["X"][i] = round(np.random.uniform(-1,1.0001),4)
    Dtrain["Y"][i] = Dtrain["X"][i] + np.random.normal(0,0.1)

Dtest = pd.DataFrame(np.zeros((200,2)), columns = list('XY'))
for i in range(len(Dtest)):
    Dtest["X"][i] = round(np.random.uniform(-2,2.0001),4)
task11(Dtrain,Dtest, 10)
print("At the edges we have straight lines to the borders -2 and 2"
      + " with y being around -1 and 1.")

print("1.2) \n")
print("K = 50")
task11(Dtrain,Dtest, 50)
print("K = 2")
task11(Dtrain,Dtest, 1)

print("For large values for K(e.g. 50) the lines center to the middle, "
      + "so the distance between those lines lowers. This is probably"
      + " because the the mean of all y data would be around 0. "
      + "With lower values for K(e.g. 2) the lines start at the very "
      + " outer points where y is around 1 and -1. This is because "
      + " the neighbours have also a y value around 1/-1."
      + "\nAlso the points in the middle are not formed like a line"
      + " comparing to the other models.")
#
print("1.3) \n")
Dtest2 = pd.DataFrame(np.zeros((100,2)), columns = list('XY'))
#random doubles
for i in range(0,100):
    Dtest2["X"][i] = round(np.random.uniform(-1,1.0001),4)
    Dtest2["Y"][i] = Dtest2["X"][i] + np.random.normal(0,0.1)
mses = task13(Dtrain, Dtest2)
#levenshtein
print("\nTask 2.4)\n")
print("Distance between the words: "
      + str(levenshteinor("HAPPY",5, "HIPPO", 5)))




















