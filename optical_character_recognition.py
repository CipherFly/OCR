# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
import numpy.linalg as nla

# display the first 22 images of the digit 9 from MNIST testing data
def displayImages():
    # read the mnist data from file 'mnist.pickle'
    with open('mnist.pickle', 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    # display 9's    
    showImages(22,data['testing'][9],'First 22 images of 9s')

# display the first N rows of data as images arranged in a square grid    
def showImages(N,data,title):
    fig=plt.figure()
    fig.suptitle(title)
    # number of rows and columnes in grid
    M = int(np.ceil(np.sqrt(N)))
    # number of rows and columns in each image
    m = int(np.sqrt(np.size(data[0])))
    for i in range(0,N):
        # display each ith image
        x = data[i]
        y = np.reshape(x,(m,m))
        plt.subplot(M,M,i+1)
        plt.axis('off')
        plt.imshow(y, cmap='Greys', interpolation='nearest')

# shuffle the MNIST data and display the first 33 images
def shuffleImages():
    # read the mnist data file
    with open('mnist.pickle', 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    # combine and shuffle the data
    shuffle = shuffleData(data)
    # display first 33 shuffled images
    showImages(33,shuffle, 'Combined and Shuffled Images')
    
# Combine and Shuffle Images    
def shuffleData(data):
    #put data into single array
    sTrain = np.vstack(data['training'])
    sTest = np.vstack(data['testing'])
    sAll = np.vstack((sTrain, sTest))
    #shuffle data
    np.random.shuffle(sAll)
    return(sAll)
    
#Fit a multivariate normal distribution to data matrix X
def fitNormal(X):
    (M,N) = X.shape
    mean = X.sum(axis=0)/M
    Xc = X - mean
    Sig = np.dot(Xc.T,Xc)/M
    return(mean,Sig)
    
# Matrix Inversion
def MatInv(X,mean,Sig):
    [M,N] = X.shape
    P = np.zeros(M)
    SigInv = la.inv(Sig)
    a = np.sqrt((2*np.pi)**N)*(la.det(Sig))
    #compute the probability of each row vector of x
    for m in range(0,M):
        #centered data vector
        x = X[m,:] - mean 
        #compute z = x*inv(Sig)*x^T using matrix inverse of Sig
        z = np.dot(x,np.dot(SigInv,x.T))
        #compute prob of x
        P[m] = np.exp(-z/2)/a
    return(P)
    
# Train multivariate normal models for digits
def train(data):
    dig = len(data) #num of digits
    Sig = [] #covariance matricies
    mean = [] # mean vectors
    var = np.zeros(dig) #pixel variance for each digit
    num = np.zeros(dig) #num of images for each digit
    for i in range(0,dig):
        [m,S] = fitNormal(data[i]) #train a multivariate model for digits
        mean.append(m)
        Sig.append(S)
        var[i] = np.mean(np.diag(S)) 
        num[i] = (np.shape(data[i])[0])
    meanVar = np.sum(var*num)/np.sum(num)
    return(mean,Sig,meanVar)
    
# Combine data into a single data matrix and a vector of labels
def flatten(data):
    d = len(data) #num of data matricies
    X = np.vstack(data) #combined data matrix
    M = (np.shape(X)[0])
    Y = np.zeros(M,dtype='int') #space for vector of labels
    m1 = 0
    m2 = 0
    for i in range(0,d):
        m = np.shape(data[i])[0] #num of images in matrix
        m2 = m2+m
        Y[m1:m2] = i #vector of labels for matrix i
        m1 = m1+m
    return(X,Y)
    
#combine each of the complex models with the simple model
#Sig is list of complex models
#var is pixel variance of the simple model
#beta is the weight of complex models
def combine(Sig,var,beta):
    d = len(Sig)
    N = (np.shape(Sig[0])[0])
    Simp = (1-beta)*var*np.eye(N) #weighted simple model
    Comb = []
    for i in range(0,d):
        comb = beta*Sig[i] + Simp #combined model for digit i
        Comb.append(comb)
    return(Comb)
    
#Compute log prob density of each row vector in matrix X using
#multivariate normal distribution with mean and covariance
#Returns the probs as a vector, logP using Cholesky factorization
def logMVNchol(X,mean,Sig):
    [M,N] = X.shape
    logP = np.zeros(M)
    L = la.cholesky(Sig,lower=True) #cholesky on lower triangle
    logDet = nla.slogdet(Sig)[1]
    logAlpha = np.log(2*np.pi)*N/2 + logDet
    #compute log prob of each row vector of X
    for m in range(0,M):
        x = X[m,:] - mean
        #compute z = x*inv(Sig)*x^T using Cholesky factorization of Sig
        y = la.solve_triangular(L,x,lower=True,check_finite=False)
        logP[m] = -np.dot(y,y)/2 - logAlpha #solve log prob of x
    return(logP)

#generate soft predictions for each image in data matrix X
#which is a list of log probs, one of each digit.
#Sig is a list of covariance matricies, and mean is list of mean vectors
#each matrix-vector pair is a model for a digit
def predict(X,mean,Sig):
    M = (np.shape(X)[0]) #num of images
    d = len(mean) #num of digits
    logP = np.zeros((M,d))
    for i in range(0,d):
        #compute log probs of all images for digit i
        logP[:,i] = logMVNchol(X,mean[i],Sig[i])
    return(logP)
    
#evaluate soft predictions in matrix logP
#displays 36 randomly chosen misclassified images
#X is data matrix of images.
#Y is a vector of correct labels.
def evaluate(logP,X,Y):
    Yhat = np.argmax(logP,axis=1) #convert soft predicitions to hard
    right = (Y==Yhat)
    M = (np.shape(X)[0])
    acc = 100*np.sum(right)/float(M) #percent of correct predictions
    print(acc)
    #display N correctly classified images
    N = 36
    Correct = (X[right,:]) #correctly classified images
    rnd.shuffle(Correct) #shuffle the images
    showImages(N,Correct, 'Correctly classified images')
    #display N misclassfied images
    Errors = X[~right,:] #misclassified images
    rnd.shuffle(Errors) #shuffle the images
    showImages(N,Errors, 'Misclassfied images')
    
def ocr():
    #read the mnist data
    with open('mnist.pickle', 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    #training
    (mean,Sig,var) = train(data['training'])
    print('training complete')
    #prediction
    (X,Y) = flatten(data['testing'])
    beta=0.25
    SigNew = combine(Sig,var,beta)
    logP = predict(X,mean,SigNew)
    print('prediction complete')
    #evaluation
    evaluate(logP,X,Y)
    print('evaluation complete')
    
displayImages()
shuffleImages()
ocr()
            
