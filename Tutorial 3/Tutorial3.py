# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:15:34 2018

@author: Christian
"""

#Computing squares linear regression
def aufgabe1_1(data):
    #describe = data.describe(include=[np.number])    
    xmean = np.mean(data)[0]
    ymean = np.mean(data)[1]
    #compute ß1
    beta1 = 0
    temp = 0
    for i, row in data.iterrows():
        beta1 += ((row["x"] - xmean) * (row["y"] - ymean))
        temp += (row["x"] - xmean)**2
        
    print(beta1)
    beta1 = beta1 / temp   
    yoho = (sum((data["x"]-xmean)*(data["y"] - ymean)))
    print(yoho)
    #compute ß0
    beta0 = ymean - beta1 * xmean
        
    print("1.1.)\nleast squares linear regression:\n" 
          + "y(x) = " + str(beta0) + " + " + str(beta1) + "x")
    return [beta0, beta1]

#Exercise 1.2
#Mean Square Error
def aufgabe1_2(beta0, beta1, data):
    print("\n1.2.)")
    mse = 0
    mse2 = 0
    for i, row in data.iterrows():
        mse += (row["y"] - ((beta0 + beta1 * row["x"])))**2
        mse2 += (row["y"] - (1.4 * row["x"]))**2
        
    mse = mse / (i + 1)
    mse2 = mse2 /(i + 1)
    print ("My MSE: " + str(mse) + "\nMSE proposed model: " + str(mse2))
    #mean absolute error
    mae = 0
    mae2 = 0
    temp = 0
    for i, row in data.iterrows():
        temp = (row["y"] - ((beta0 + beta1 * row["x"])))
        if temp < 0:
            temp *= -1
        mae += temp
        
        temp = (row["y"] - (1.4 * row["x"]))
        if temp < 0: 
            temp *= -1
        mae2 += temp
        
    mae = mae / (i + 1)
    mae2 = mae2 / (i + 1)
    print ("My MAE: " + str(mae) + "\nMAE proposed model: " + str(mae2))

#MSE AND MAE WITHOUT LAST  DATAPOINTS
def aufgabe1_3(beta0, beta1, data):      
    print("\n1.3.) Without last 3 datapoints:")
    mse = 0
    mse2 = 0
    for i, row in data.iterrows():
        if i <= 46:
            mse += (row["y"] - ((beta0 + beta1 * row["x"])))**2
            mse2 += (row["y"] - (1.4 * row["x"]))**2
        
    mse = mse / (i + 1 - 3)
    mse2 = mse2 / (i + 1 - 3)
    print ("My MSE: " + str(mse) + "\nMSE proposed model: " + str(mse2))
    #mean absolute error
    mae = 0
    mae2 = 0
    temp = 0
    #computing mae and mae2 reusing and overriding temp variable
    for i, row in data.iterrows():
        if i <= 46:
            temp = (row["y"] - ((beta0 + beta1 * row["x"])))
            if temp < 0:
                temp *= -1
            mae += temp
            
            temp = (row["y"] - (1.4 * row["x"]))
            if temp < 0: 
                temp *= -1
            mae2 += temp
    mae = mae / (i + 1 - 3) 
    mae2 = mae2 / (i + 1 - 3)
    print ("My MAE : " + str(mae) + "\nMAE proposed model: " + str(mae2))
    
    print("I notice that the MSE and MAE of both models decreased")

def aufgabe1_4(data, betawerte):
    import matplotlib.pyplot as plt
    #Daten sortieren
    data = data.sort_values(by = "x", ascending = True)
    print("plotted data:")
    #PLOT DATA 2 WAYS
    #data.plot(x='x', y='y', kind="line")
   # plt.plot("x", "y", "b", data=data)
    plt.scatter("x","y", data=data)
    plt.show()
    
    print("plottet functions:")
    plt.title("function: 1.4x")
    plt.plot([1,6],[1.4 * 1, 1.4* 6], "y")
    plt.show()
    
    beta0 = betawerte[0]
    beta1 = betawerte[1]
    
    plt.title("function: " + str(beta0) + " + "
              + str(beta1) + "x")
    plt.plot([1,6],[beta0 + beta1*1,
              beta0 + beta1 * 6], "r")
    plt.show()
    
    #alles zusammen
    print("all graphs:")
    plt.title("all together")
   # plt.plot("x", "y", "b", data=data)
    plt.scatter("x","y", data=data)

    plt.plot([1,6],[1.4 * 1, 1.4* 6], "y")
    
    beta0 = betawerte[0]
    beta1 = betawerte[1]
    plt.plot([1,6],[beta0 + beta1*1,
              beta0 + beta1 * 6], "r")
    plt.show()
    
    print("I would prefer the yellow estimator(given function) because"
      +" it looks more accurate to the given data if you not"
      + " mind x values 5 to 6. Also MAE and MSE are lower.")



import pandas as panda
import numpy as np
 #Exercise 1.1
data = panda.read_csv("tutorial3.dat", sep= " ", 
                     names = ["x","y"])
data = panda.DataFrame(data)

betawerte = aufgabe1_1(data)
aufgabe1_2(betawerte[0],betawerte[1],data)
aufgabe1_3(betawerte[0],betawerte[1],data)
aufgabe1_4(data, betawerte)



















    