import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

mat = loadmat("ex5data1.mat")

X     = mat['X']
y     = mat['y']
Xtest = mat['Xtest']
ytest = mat['ytest']
Xval  = mat['Xval']
yval  = mat['yval']




def polyFeatures(X,p):
	for i in range(2,p+1):
		X = np.hstack((X,(X[:,0]**i)[:,np.newaxis]))

	return X

def costFunctionRegulize(X, y,theta, Lambda):
	m=len(y)
	predictions = X @ theta
	error = (predictions - y)**2
	cost = sum(error) /(2*m)  
	regCost = cost + Lambda/(2*m) * sum(theta[1:]**2)

	j_0 = 1/m * (X.transpose() @ (predictions - y))[0]
	j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)*theta[1:]
	grad = np.vstack((j_0[:,np.newaxis],j_1))
	return regCost[0], grad

def gradientDescent(X,y,theta,alpha,numIter,Lambda):
	m = len(y)
	costs = []
	for i in range(numIter):
		cost ,grad = costFunctionRegulize(X,y,theta,Lambda)
		theta = theta - alpha*grad
		costs.append(cost)
	return theta, costs



def learningCureve(X, y, Xval,yval, Lambda):
	m = len(y)
	n = X.shape[1]

	err_train , error_val = [],[]
	for i in range(1,m+1):
		theta = gradientDescent(X[0:i,:],y[0:i,:],np.zeros((n,1)),0.001,3000,Lambda)[0]
		err_train.append(costFunctionRegulize(X[0:i,:],y[0:i,:],theta,Lambda)[0])
		error_val.append(costFunctionRegulize(Xval,yval,theta,Lambda)[0])
	return err_train,error_val

def main():
	global X, y, Xval,yval
	p=8
	X_poly = polyFeatures(X, p)
	sc_X=StandardScaler()
	X_poly=sc_X.fit_transform(X_poly)
	X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
	X_poly_test = polyFeatures(Xtest, p)
	X_poly_test = sc_X.transform(X_poly_test)
	X_poly_test = np.hstack((np.ones((X_poly_test.shape[0],1)),X_poly_test))
	X_poly_val = polyFeatures(Xval, p)
	X_poly_val = sc_X.transform(X_poly_val)
	X_poly_val = np.hstack((np.ones((X_poly_val.shape[0],1)),X_poly_val))

if __name__ == "__main__":
	main()