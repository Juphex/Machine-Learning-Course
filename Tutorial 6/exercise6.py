# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:07:45 2018

@author: Christian Salamut
"""

def BIC(X,Y,estimator, N, parameterCount):
    mse = 0
    for i in range(0,N):
        mse += np.power((Y[i] - estimator[i]),2)
    
    mse = mse/N
    retVal = (-(1/2) * N * np.log(mse)
    - (parameterCount/2) * np.log(N) + 1*N)
    
##  providing negative numbers: add + N
    print(np.around(retVal,4))


import numpy as np

X = np.matrix([0.00,0.28,0.56,0.83,1.11,1.39,1.67,1.94,2.22,2.50]).T
Y = np.matrix([1.90,2.04,1.83,1.56,1.29,2.84,3.49,3.33,5.26,5.61]).T
y1 = np.matrix([0.98,1.41,1.84,2.27,2.70,3.13,3.56,3.99,4.42,4.85]).T
y2 = np.matrix([2.02,1.76,1.67,1.75,2.01,2.44,3.04,3.82,4.76,5.88]).T
y3 = np.matrix([1.90,2.07,1.72,1.53,1.76,2.39,3.22,4.05,4.81,5.70]).T

print("y1 BIC")
BIC(X,Y,y1,10,2)
print("y2 BIC")
BIC(X,Y,y2,10,3)
print("y3 BIC")
BIC(X,Y,y3,10,6)

print("y2 has the best BIC")
