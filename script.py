import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    #combining the dataset with the labels
    dataWithLabels = np.concatenate((X,y),axis = 1)
    labels = np.unique(dataWithLabels[:,(len(dataWithLabels.T)-1)]) 
    
    #Creating the matrix for storing the data class wise
    labelMat = ['class']*len(labels)   
    for j in range (0,len(labels)):
        x = np.where(dataWithLabels[:,(len(dataWithLabels.T)-1)]==labels[j])
        labelMat[j] = dataWithLabels[x[0]]

    #Creating the means matrix and identifying the means
    means = np.zeros(shape = ((len(X.T)),(len(labels))))             
    for j in range (0,len(labels)):
        classSum = (np.sum(labelMat[j], axis = 0))
        classMean = (classSum/len(labelMat[j]))
        for k in range(0,len(X.T)):
            means[k,j] = classMean[k]
    #print (means)
    #Calculating the combined mean for all the dataSet
    combinedMean = ((np.sum(X, axis = 0))/len(X)) 
    tempCov = np.subtract(X,combinedMean)
    tempCovTranspose = tempCov.T
    covSum = np.dot(tempCovTranspose,tempCovTranspose.T)
    covmat = (covSum/len(X))
    #print (covmat)                                                                                                                                                   
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    #combining the dataset with the labels
    dataWithLabels = np.concatenate((X,y),axis = 1)
    labels = np.unique(dataWithLabels[:,(len(dataWithLabels.T)-1)])
    
    #Creating the matrix for storing the data class wise
    labelMat = ['class']*len(labels)
    for j in range (0,len(labels)):
        x = np.where(dataWithLabels[:,(len(dataWithLabels.T)-1)]==labels[j])
        labelMat[j] = dataWithLabels[x[0]]
        
    #Creating the means matrix and identifying the means                    
    means = np.zeros(shape = ((len(X.T)),(len(labels))))
    for j in range (0,len(labels)):
        classSum = (np.sum(labelMat[j], axis = 0))
        classMean = (classSum/len(labelMat[j]))
        for k in range(0,len(X.T)):
            means[k,j] = classMean[k]
    
    classMatrix = ['class']*len(labels)
    covmats = ['class']*len(labels)
    for j in range (0,len(labels)):
        classMatrix[j] = np.delete(labelMat[j],np.s_[-1:],1)
        jthMean = means[:,j]
        classMatrixDiff = (classMatrix[j]-(jthMean.T))
        classMatrixDiffTranspose = classMatrixDiff.T
        covMatMultiply = np.dot(classMatrixDiffTranspose, classMatrixDiffTranspose.T)
        covmats[j] = (covMatMultiply/len(classMatrix[j]))        
                           
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    predictedlabel = np.zeros(shape = (len(ytest),1))
    for i in range(0,len(Xtest)):
        individualData = Xtest[i]
        probList = ['prob']*len(means.T)
        for j in range(0,len(means.T)):
            jthMean = means[:,j]
            jthMeanTranspose = jthMean.T
            diff = np.subtract(individualData,jthMeanTranspose)
            diffOriginal = diff.T
            covInv = np.linalg.inv(covmat)
            diffOriginalTranspose = diffOriginal.T
            partialPower = np.dot(diffOriginalTranspose,covInv)
            totalPower = np.dot(partialPower,diffOriginal)
            exponePower = ((-1/2)*totalPower)
            expValue = np.exp(exponePower)
            detCov = np.linalg.det(covmat)
            finalDetCov = np.power(detCov,0.5)
            dimVal = ((len(Xtest.T))/2)
            piEstimation = (2*np.pi)
            piVal = np.power(piEstimation,dimVal)
            denomFinal = (piVal*finalDetCov)
            firstTerm = (1/denomFinal)
            classProb = (firstTerm*expValue)
            probList[j] = classProb
        classEstimation = (np.argmax(probList)+1)
        predictedlabel[i,0] = classEstimation
    
    ypred = predictedlabel
    acc = ((100)*(np.mean((ypred == ytest).astype(float))))                                 
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    predictedlabel = np.zeros(shape = (len(ytest),1))
    for i in range(0,len(Xtest)):
        individualData = Xtest[i]
        probList = ['prob']*len(means.T)
        for j in range(0,len(means.T)):
            jthMean = means[:,j]
            jthMeanTranspose = jthMean.T
            diff = np.subtract(individualData,jthMeanTranspose)
            diffOriginal = diff.T
            covmat = covmats[j]
            covInv = np.linalg.inv(covmat)
            diffOriginalTranspose = diffOriginal.T
            partialPower = np.dot(diffOriginalTranspose,covInv)
            totalPower = np.dot(partialPower,diffOriginal)
            exponePower = ((-1/2)*totalPower)
            expValue = np.exp(exponePower)
            detCov = np.linalg.det(covmat)
            finalDetCov = np.power(detCov,0.5)
            dimVal = ((len(Xtest.T))/2)
            piEstimation = (2*np.pi)
            piVal = np.power(piEstimation,dimVal)
            denomFinal = (piVal*finalDetCov)
            firstTerm = (1/denomFinal)
            classProb = (firstTerm*expValue)
            probList[j] = classProb
        classEstimation = (np.argmax(probList)+1)
        predictedlabel[i,0] = classEstimation
    
    ypred = predictedlabel
    acc = ((100)*(np.mean((ypred == ytest).astype(float))))        
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
    Xtranspose = X.T
    firstTerm = np.dot(Xtranspose,X)
    secondTerm = np.dot(Xtranspose,y)
    w = np.linalg.solve(firstTerm, secondTerm)                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    Xtranspose = X.T
    firstTerm = np.dot(Xtranspose,X)
    Ident = np.identity(len(X.T))
    lambdTerm = (lambd*Ident)
    firstTermRidge = (firstTerm+lambdTerm)
    secondTerm = np.dot(Xtranspose,y)
    w = np.linalg.solve(firstTermRidge,secondTerm)                                                    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    XtestTranspose = Xtest.T
    wTranspose = w.T
    outputTemp = np.dot(wTranspose, XtestTranspose)
    outputFinal = outputTemp.T
    rmseFirstTermTemp = np.subtract(ytest,outputFinal)
    rmseFirstTerm = rmseFirstTermTemp.T
    rmseSecondTerm = rmseFirstTermTemp
    rmseMul = np.dot(rmseFirstTerm,rmseSecondTerm)
    rmseAvg = (rmseMul/len(Xtest))
    rmse = np.sqrt(rmseAvg)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda   
    # IMPLEMENT THIS METHOD  
    w = w.reshape(-1,1)
    XTraspose = X.T
    wTranspose = w.T
    initialOutput = np.dot(wTranspose,XTraspose)
    initialOutputFinal = initialOutput.T   
    # this is y-o
    errorFirstTermTemp = np.subtract(y,initialOutputFinal)
    #calculation of error begins here
    errorFirstTerm = errorFirstTermTemp.T
    errorSecondTerm = errorFirstTerm.T
    err1 = (np.dot(errorFirstTerm,errorSecondTerm)/2)
    
    regTemp = (np.dot(w.T,w)/2)
    regFinal = (lambd*regTemp) 
    error = ((err1+regFinal)/len(X))
    gradientWithoutReg = (np.dot(XTraspose,errorFirstTermTemp)/len(X))
    avgGradientWithoutReg = ((-1)*gradientWithoutReg)  
    regGradience = ((lambd*(w))/len(X))  
    error_grad = (avgGradientWithoutReg+regGradience)
    
    #Estimation of Gradients
                                                                                                        
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.zeros(shape = (len(x),(p+1)))
    for i in range(0,len(x)):
        for j in range(0,p+1):
            Xd[i,j] = np.power(x[i],j)
            
    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmseTraining  = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)    
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))