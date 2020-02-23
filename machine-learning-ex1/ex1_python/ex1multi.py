# Programming Exercise 1: Linear Regression
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# Read data from file 
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,[0,1]]
Y = data[:,2]
# Features Scalling (Standard Feature Scaling) ! x = (x - mu)/sigma ;
# mu is the mean ,sigma is the standard derivation is sqrt(sum(x- mu)^2/n)
mean = np.ones(X.shape[1])
std = np.ones(X.shape[1])

for i in range(X.shape[1]):
	mean[i] = np.mean(X.transpose()[i])
	std[i] = np.std(X.transpose()[i])
	for j in range(X.shape[0]):
		X[j][i] = (X[j][i] - mean[i])/std[i]


one_column = np.ones((X.shape[0],1))
X = np.concatenate((one_column, X), axis = 1)
X = np.array(X)
Y= np.array(Y)
theta = np.zeros(X.shape[1])

def hypothesis(theta):
	global X
	h = np.ones((X.shape[0],1))
	theta = theta.reshape(1,X.shape[1]) 
	for i in range(0,X.shape[0]):
		h[i] = float(np.matmul(theta, X[i]))
	h = h.reshape(X.shape[0])
	return h

def computeCost(theta):
	global X, Y
	h = hypothesis(theta) 
	return (1/X.shape[0]) * 0.5 * sum(np.square(h-Y)) 

def gradientStep(theta,alpha):
	global X,Y
	h = hypothesis(theta)
	for i in range(X.shape[1]):
		theta[i] -= (alpha/X.shape[0]) * sum((h-Y) * X.transpose()[i])
	return theta

	

def gradientDescent():
	global X,Y
	alpha = 0.03
	nbIter = 1500
	theta = np.zeros(X.shape[1])
	costs = []
	costs.append(computeCost(theta))
	
	for i in range(nbIter):
		theta = gradientStep(theta,alpha)
		costs.append(computeCost(theta))
	
	return theta, costs


def main():
	theta ,costs = gradientDescent()
	plt.plot(range(len(costs)),costs)
	plt.xlabel('No. of iterations')
	plt.ylabel('Cost')
	plt.show()


if __name__ == '__main__':
	main()



