import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat

# Load a matlab matrix with python 
mat = loadmat("ex3data1.mat")
X = mat['X']
y = mat['y']
# plot data 
#fig, axis = plt.subplots(10,10,figsize=(8,8))
#for i in range(10):
#	for j in range(10):
#		axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape((20,20),order='F'),cmap='hot')
#		axis[i,j].axis('off')
#plt.show()

def costFunctionRegulize(theta, X, y, Lambda):
	m=len(y)
	predictions = sigmoid(X @ theta)
	error = (-y * np.log(predictions)) - ((1-y)*np.log(1 - predictions))
	cost = 1/m * sum(error)
	regCost = cost + Lambda/(2*m) * sum(theta[1:]**2) # to see 

	j_0 = 1/m * (X.transpose() @ (predictions - y))[0]
	j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)*theta[1:]
	grad = np.vstack((j_0[:,np.newaxis],j_1))
	return regCost[0], grad


def gradienDescent(X,y,theta,alpha,numIter,Lambda):
	m = len(y)
	costs = []
	for i in range(numIter):
		cost ,grad = costFunctionRegulize(theta,X,y,Lambda)
		theta = theta - (alpha*grad)
		costs.append(cost)
	return theta, costs


def oneVsAll(X,y,num_label,Lambda):
	m, n = X.shape[0], X.shape[1]
	init_theta = np.zeros((n+1,1))
	all_theta = []
	all_cost  = []
	X = np.hstack((np.ones((m,1)),X))
	for i in range(1,num_label+1):
		theta , costs = gradienDescent(X,np.where(y==i,1,0),init_theta,1,300,Lambda)
		all_theta.extend(theta)
		all_cost.extend(costs)

	return np.array(all_theta).reshape(num_label,n+1), all_cost

def predictOneVsAll(all_theta,X):
	m=X.shape[0]
	X = np.hstack((np.ones((m,1)),X))
	predictions = X @ all_theta.T
	return np.argmax(predictions,axis = 1)+1


def main():
	global X,y
	all_theta, all_J = oneVsAll(X, y, 10, 0.1)
	#plt.plot(all_J)
	#plt.xlabel("Iteration")
	#plt.ylabel("$J(\Theta)$")
	#plt.title("Cost function using Gradient Descent")
	#plt.show()
	pred = predictOneVsAll(all_theta, X)
	print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000*100,"%")

def test():
	theta_t = np.array([-2,-1,1,2]).reshape(4,1)
	X_t =np.array([np.linspace(0.1,1.5,15)]).reshape(3,5).T
	X_t = np.hstack((np.ones((5,1)), X_t))
	y_t = np.array([1,0,1,0,1]).reshape(5,1)
	J, grad = costFunctionRegulize(theta_t, X_t, y_t, 3)
	print("Cost:",J,"Expected cost: 2.534819")
	print("Gradients:\n",grad,"\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

if __name__ == '__main__':
	main()
