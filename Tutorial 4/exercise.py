# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:40:17 2018

@author: Christian Salamut
"""
#round
def roundon4(value):
    return np.around(value,4)

#aufgabe 1.2
def task12():
    #read data
    data1 = panda.read_csv("tutorial4.dat", sep = " ",
                           names= ["x", "y"])
    
    #!!ARRANGE ARRAY WITH 100x  a 1
    a = panda.DataFrame(data1["x"])
    for i,column in a.iterrows():
        a["x"] = 1
    #matrix in x^2,x,1 form
    X = np.matrix([data1["x"]**2, data1["x"], a["x"]])
    #normals the format
    X = X.transpose()
    leftside = X.transpose() * X
        
    y = np.matrix([data1["y"],data1["y"],data1["y"]]).transpose()
    rightside = X.transpose() @ y
    
    print("left A^t*Aß:\n" + str(roundon4(leftside)))
    print("\n")
    print("right A^t*y:\n" + str(roundon4(rightside)))
    estimators = np.linalg.solve(leftside, rightside)
    
    #functionmunction
    print("\nestimator:\n" + "y(x) = " + str(roundon4(estimators[0,0]))
    + "x^2 * "+ str(roundon4(estimators[1,1])) + "x * "
    + str(roundon4(estimators[2,2])))
    a = estimators[0,0]
    b = estimators[1,1]
    c = estimators[2,2]
    #mse
    mse = 0
    for i, row in data1.iterrows():
        mse += abs(row["y"] - (a * row["x"]**2+ b * row["x"] + c)) **2
        
    mse = mse/100
    print("MSE: " + str(np.around(mse, 4)))
    #plot the data
    import matplotlib.pyplot as plt
    plt.scatter("x","y",data = data1)  
    #plot the estimator, that was tricky :)
    x = np.linspace(-3,3,100)
    #or a*x+b*x+c
    y = 0.9736435030982529*(x**2) + -0.03105042980187572*x* -0.8811869592565976
    plt.plot(x,y, color = "red")
    plt.legend(["y(x)"])
    plt.show()

#aufgabe1.4
def task14():
    data2 = panda.read_csv("tutorial4_2.dat", sep = " ",
                           names = ["x", "y", "z"])

    #ax^2 + 2bxy + cy^2
    A = np.matrix([data2["x"]**2, 2*(data2["x"]*data2["y"]),
                   data2["y"]**2])
    A = A.transpose()
    
    leftside = A.transpose() * A
    #we multiply with z cause we have 3 columns now
    zdata = np.matrix(data2["z"]).transpose()
    rightside = A.transpose() * zdata
    
    estimators = np.linalg.solve(leftside, rightside)
    a = estimators[0,0]
    b = estimators[1,0]
    c = estimators[2,0]    
    #initialize mse
    mse = 0
    for i, row in data2.iterrows():
        mse +=(abs(row["z"] - (a*row["x"]**2
              + b*2*row["y"]*row["x"] + c*row["y"]**2)))**2
    
    mse = mse / 1000
    print("1.4\n" + "Optimal values: a = " + str(roundon4(a))
        + ", b = " + str(roundon4(b)) + ", c = " + str(roundon4(c)))
    print("MSE: " + str(roundon4(mse)))
    
#direction
def negativeDerivat(x):
    firstterm = np.power(x, 3) * 4
    secondterm = np.power(x, 2) * 3.9
    thirdterm = np.power(x, 1)* 3.9
    fourthterm = 4
    retVal = -(firstterm - secondterm - thirdterm + fourthterm)
    return retVal

def derivat(x):
    firstterm = np.power(x, 3) * 4
    secondterm = np.power(x, 2) * 3.9
    thirdterm = np.power(x, 1)* 3.9
    fourthterm = 4
    retVal = firstterm - secondterm - thirdterm + fourthterm
    return retVal

def functionFofx(x):
    #x^4 - 1.3x^3 - 1.95x^2 + 4x + 3.65
    firstterm = np.power(x, 4)
    secondterm = np.power(x,3) * 1.3
    thirdterm = np.power(x,2)* 1.95
    fourthterm = 4 * x
    fithterm = 3.65
    retVal = firstterm - secondterm - thirdterm + fourthterm + fithterm
    return retVal

def nextX(x, alpha):
    #you can leave "-" and use negativeDerivat(x) instead
    retVal = -alpha * derivat(x) + x
    return np.around(retVal,4)

def gradientDescent(start, alpha, iterations):
    print("\nstarting with x0 = " + str(start))
    x = start
    for i in range(1, iterations):
        x = nextX(x, alpha)
        fofx = functionFofx(x)
        if fofx == 0.0:
            print(str(np.around(x,4)))
            print("success!")
            break
        if fofx < 10**-6:
            print("Finished at iteration: " + str(i))
            break
        if i == iterations - 1:
            print("Could not find the minimum within " 
                  + str(iterations) + " iterations.\n")

def getAlpha(x, minimumsteepness):
    alpha = 1
    minsteep = minimumsteepness
    while functionFofx(x)- functionFofx(x + alpha
                      *negativeDerivat(x)) < alpha*minsteep*negativeDerivat(x)*negativeDerivat(x):
        #print(str(alpha)) debug
        alpha = alpha/2
    return alpha

def gradientDescentArmijo(start, minimumsteepness, iterations):
    print("\nstarting with x0 = " + str(start))
    x = start
    for i in range(1, iterations):
        x = nextX(x, getAlpha(x, minimumsteepness))
        fofx = functionFofx(x)
        if fofx == 0.0:
            print(str(np.around(x,4)))
            print("success!")
            break
        if fofx < 10**-6:
            print("Finished at iteration: " + str(i))
            break
        if i == iterations - 1:
            print("Could not find the minimum within " 
                  + str(iterations) + " iterations."
                  + "the last was x = " + str(x) + "\n")

#its a main method
import numpy as np
import pandas as panda

task12()
task14()
print("\n2.1")
gradientDescent(2, 0.1, 50)
print("Therefore 18 iterations are needed to drop below 10^-6")
print("\n2.2")
gradientDescent(1.5, 0.1, 50)
print("Why do you need more iterations, although the start is lower "
      +"than the first one?\n")
print("It is because the first step(s) are greater when starting with 2.\n"
      + "If we start with 2 the next step will we x = " 
      + str(nextX(2, 0.1)) + "\nIf we start with 1.5 the next step will"
      + " be x = " + str(nextX(1.5, 0.1)) 
      + "\nThis ist because of the slope of the function in that area.")
print("\n2.3")
gradientDescent(-0.5, 0.15, 100000)
print("You need a lot of iterations. Even with 100000 it will not find "
      + " the local(global) minimum.")

#gradientDescent with alpha from armijo step length
print("\n2.4\nGradient Descend with armijo step length. minimum steepness = 0.1")
gradientDescentArmijo(2, 0.1, 50)
gradientDescentArmijo(1.5, 0.1, 50)
gradientDescentArmijo(-0.5, 0.1, 50)
print("\nGradient Descend with armijo step length. minimum steepness = 0.9")

gradientDescentArmijo(2, 0.9, 1000)
gradientDescentArmijo(1.5, 0.9, 1000)
gradientDescentArmijo(-0.5, 0.9, 1000)
print("\nIf the minimum steepness is too high it cannot reach the minimum"
      + " of the function.")






















