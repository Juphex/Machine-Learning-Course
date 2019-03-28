# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:01:19 2018

@author: Christian Salamut
"""

def task11():
    print("1.1\nThe difference between classification and regression"
          + " is the output. The output of a classification is discrete"
          + ". For example true/false (binary classification). \n"
          + "The output of a regression is continuous."
          + "For example the price behaviour of something."
          + "\nA classifier prediction gives you the percentage"
          + " of the data to fit the target space.")
    
def pltData(mydata, name):
    plt.title("Data: " + name)
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in range(0,len(mydata)):
        if(mydata["y"][i] == 0):
            plt.scatter(mydata["x1"][i],mydata["x2"][i], marker = "^",
                        color = "red")
        elif(mydata["y"][i] == 1):
            plt.scatter(mydata["x1"][i],mydata["x2"][i], marker = "o",
                        color = "blue")
    
    #cite: https://stackoverflow.com/questions/47391702
    #/matplotlib-making-a-colored-markers-legend-from-scratch
    from matplotlib.legend_handler import HandlerBase
    list_color  = ["red", "blue"]
    list_mak    = ["^","o"]
    list_lab    = ['y = 0','y = 1']
    
    ax = plt.gca()
    
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]
    
    
    ax.legend(list(zip(list_color,list_mak)), list_lab, 
              handler_map={tuple:MarkerHandler()}) 
    plt.show()

def pltDataWithBoundaries(mydata, betas, name):
    plt.title("Data: " + name)
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in range(0,len(mydata)):
        if(mydata["y"][i] == 0):
            plt.scatter(mydata["x1"][i],mydata["x2"][i], marker = "^",
                        color = "red")
        elif(mydata["y"][i] == 1):
            plt.scatter(mydata["x1"][i],mydata["x2"][i], marker = "o",
                        color = "blue")
            
    #cite: https://stackoverflow.com/questions/47391702
    #/matplotlib-making-a-colored-markers-legend-from-scratch
    from matplotlib.legend_handler import HandlerBase

    list_color  = ["red", "blue"]
    list_mak    = ["^","o"]
    list_lab    = ['y = 0','y = 1']
    
    ax = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]
    
    
    ax.legend(list(zip(list_color,list_mak)), list_lab, 
              handler_map={tuple:MarkerHandler()}) 
    
    x = np.linspace(-1,1, 100)
    y = betas[0,0] + betas[1,0] * x + betas[2,0] * x
    plt.plot(x,y, color = "red")
    plt.legend(["decision boundary"])
    plt.show()

def task12():
    global dX2
    dX2 = np.matrix([[-1,-1,1,1],[-1,1,-1,1],[0,1,0,1]]).transpose()
    dX2 = panda.DataFrame(dX2, columns = ["x1","x2","y"])
    
    global dXOR
    dXOR = np.matrix([[-1,-1,1,1],[-1,1,-1,1],[0,1,1,0]]).transpose()
    dXOR = panda.DataFrame(dXOR, columns = ["x1","x2","y"])
    
    global dAND
    dAND = np.matrix([[-1,-1,1,1],[-1,1,-1,1],[0,0,0,1]]).transpose()
    dAND = panda.DataFrame(dAND, columns = ["x1","x2","y"])
    pltData(dX2, "X2")
    pltData(dXOR, "XOR")
    pltData(dAND, "AND")
    print("For Data X2 and AND you can classify the data with "
          + "a logistic regression. For XOR you cannot draw a line"
          + " because on each side there would be errors.")
    
    
def exponential(x):
    return scipy.expit(x)

def getEstimator(data, betas):
    retVal = np.zeros((4,1))
    for i in range(0,4):
        retVal[i,0] = (exponential((betas[0,0]
        - data["x1"][i] * betas[1,0] + data["x2"][i] * betas[2,0])))
        
    return retVal

def gradientDescentFirstStep(data):
    betas = np.matrix([0,1,0]).transpose()
    #logistic model
    alpha = 1
    xmatrix = np.zeros((4,3))
    xmatrix[:,0] = 1
    xmatrix[:,1] = data["x1"]
    xmatrix[:,2] = data["x2"]
    y =  np.matrix(data["y"]).transpose()
    estimator = getEstimator(dX2, betas)
    
    newbetas = betas - alpha * xmatrix.transpose() @ (y - estimator)
    print("My betas:")
    print(np.around(newbetas,4))
    return newbetas
    
def task13():
    print("\n1.3)\nGradient descent with bias = 0, ß1 = 1, ß2 = 0"
      + "and alpha = 1")
    print("X2")
    betas1 = gradientDescentFirstStep(dX2)
    print("XOR")
    betas2 = gradientDescentFirstStep(dXOR)
    print("AND")
    betas3 = gradientDescentFirstStep(dAND)   
    return [betas1,betas2,betas3]
        
    
def task14(betas):
    #draw decision boundary
    pltDataWithBoundaries(dX2,betas[0],"X2")
    pltDataWithBoundaries(dXOR,betas[1],"XOR")
    pltDataWithBoundaries(dAND,betas[2],"AND")

def task21(data):
#    plot iris data SEPAL LENGTH AND WIDTH
    for i in range(0,len(data)):
        if(str(data['itsclass'][i])  == "Iris-setosa"):
            plt.scatter(data["sepal-length"][i],data["sepal-width"][i],
                        marker = "^", color = "red")
        elif(str(data['itsclass'][i]) == "Iris-versicolor"):
            plt.scatter(data["sepal-length"][i],data["sepal-width"][i],
                        marker = "o", color = "blue")
        elif(str(data['itsclass'][i])  == "Iris-virginica"):
            plt.scatter(data["sepal-length"][i],data["sepal-width"][i],
                        marker = "s", color = "green")

    from matplotlib.legend_handler import HandlerBase

    list_color  = ["red", "blue", "green"]
    list_mak    = ["^","o", "s"]
    list_lab    = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    
    ax = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]
    
    
    ax.legend(list(zip(list_color,list_mak)), list_lab, 
              handler_map={tuple:MarkerHandler()}) 
    plt.show()

    
    #PLOT PETAL LENGTH AND WIDTH
    for i in range(0,len(data)):
        if(str(data['itsclass'][i])  == "Iris-setosa"):
            plt.scatter(data["petal-length"][i],data["petal-width"][i],
                        marker = "^", color = "red")
        elif(str(data['itsclass'][i]) == "Iris-versicolor"):
            plt.scatter(data["petal-length"][i],data["petal-width"][i],
                        marker = "o", color = "blue")
        elif(str(data['itsclass'][i])  == "Iris-virginica"):
            plt.scatter(data["petal-length"][i],data["petal-width"][i],
                        marker = "s", color = "green")

    ax = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]
    
    
    ax.legend(list(zip(list_color,list_mak)), list_lab, 
              handler_map={tuple:MarkerHandler()}) 

    plt.show()

def task22(data, isprint):
    #mean
    mean = 0
    for i in range (0, len(data)):
        temp = np.matrix([data["sepal-length"][i],
                              data["sepal-width"][i],
                              data["petal-length"][i],
                              data["petal-width"][i]]).transpose()
        if(i == 0):
            mean = temp
        else:
            mean += temp
    
    
    mean = (1/len(data)) * mean
    #covarianz
    covarianz = 0
    for i in range (0, len(data)):
        temp = np.matrix([data["sepal-length"][i],
                          data["sepal-width"][i],
                          data["petal-length"][i],
                          data["petal-width"][i]]).transpose()
        if(i == 0):
            covarianz = (temp - mean) @ (temp - mean).transpose()
        else:
            covarianz += (temp - mean) @ (temp - mean).transpose()
    
    
    covarianz = (1/len(data)) * covarianz 
    if(isprint == "print"):
        print ("\xb5" + str(np.around(mean,4)))
        print ("\u03A3" + str(np.around(covarianz,4)) + "\n")
    
    return mean,covarianz

def getLDA(data, predictorData, index):
        i = index
        values = task22(predictorData, "")
        mean = values[0]
        covarianz = values[1]
        x = np.matrix([data["sepal-length"][i],
                              data["sepal-width"][i],
                              data["petal-length"][i],
                              data["petal-width"][i]]).transpose()
        lda = getLDA
        lda = -(1/2) * np.log(np.linalg.det(covarianz))
        lda += -(1/2) * ((x- mean).transpose()) @ ((np.linalg.inv(covarianz) @ (x - mean)))
        lda += np.log((len(setosa)/150))
        return lda, x
    

def task23(irisdata, setosa, versicolor, verginica):
    predictedData = panda.DataFrame([0,0,0,0,"test"]).transpose()
    #getvalues
    for i in range (0,len(irisdata)):
        #getvalues
        ldaSet = getLDA(irisdata, setosa, i)[0]
        ldaVers = getLDA(irisdata, versicolor, i)[0]
        ldaVerg = getLDA(irisdata, verginica, i)[0]
        newRow = 0
        if(ldaSet >= ldaVers and ldaSet >= ldaVerg):
            newRow = getLDA(irisdata, setosa, i)[1].transpose()
            newRow = np.c_[newRow, 0]
        elif(ldaVers >= ldaVerg and ldaVers >= ldaSet):
            newRow = getLDA(irisdata, versicolor, i)[1].transpose()
            newRow = np.c_[newRow,1]
        elif(ldaVerg >= ldaSet and ldaVerg >= ldaVers):
            newRow = getLDA(irisdata, verginica, i)[1].transpose()
            newRow = np.c_[newRow,2]
        if (i == 0):
            predictedData = newRow
        else:
            predictedData = np.vstack([predictedData,newRow])
    print("3 data points are not predicted correctly") 
    
    #plot it
    for i in range (0, len(irisdata)):
        color = "blue"
        if(i<=49): 
            if(predictedData.item(i,4) != 0):
                color = "red"
        elif(i<=99):
            if(predictedData.item(i,4) != 1):
                color = "red"
        elif(i<=149):
            if(predictedData.item(i,4) != 2):
                color = "red"
        irisdata = panda.DataFrame(irisdata)
        plt.scatter(irisdata["sepal-length"][i],irisdata["sepal-width"][i],
                    marker = "^", color = color)
    print("red is wrong, blue is right")
    plt.show()
    
    for i in range (0, len(irisdata)):
        color = "blue"
        if(i<=49): 
            if(predictedData.item(i,4) != 0):
                color = "red"
        elif(i<=99):
            if(predictedData.item(i,4) != 1):
                color = "red"
        elif(i<=149):
            if(predictedData.item(i,4) != 2):
                color = "red"
        irisdata = panda.DataFrame(irisdata)
        plt.scatter(irisdata["petal-length"][i],irisdata["petal-width"][i],
                    marker = "^", color = color)
    print("red is wrong, blue is right")

        
import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scipy

dX2 = 0
dXOR = 0
dAND = 0
task11()
print("\n1.2")
task12()
#newbetas useless
newbetas = task13()


irisdata = panda.read_csv("iris.data", sep = ",",header = 0
                          , names = ["sepal-length","sepal-width"
                          ,"petal-length","petal-width"
                          ,"itsclass"])
print("\n2.1")
task21(irisdata)

irisdata = panda.DataFrame(irisdata)
setosa = irisdata.loc[irisdata["itsclass"] == "Iris-setosa"]
versicolor = irisdata.loc[irisdata["itsclass"] == "Iris-versicolor"]
verginica = irisdata.loc[irisdata["itsclass"] == "Iris-virginica"]
#reset indizes to 0-49
versicolor = versicolor.reset_index(level = 0)
verginica = verginica.reset_index(level = 0)

print("\n2.2\nsetosa data:")
task22(setosa, "print")
print("versicolor data:")
task22(versicolor, "print")
print("verginica data:")
task22(verginica, "print")
print("\n2.3")
predictedData = task23(irisdata,setosa,versicolor,verginica)
