import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ex2data1.txt',delimiter=',',header=None)
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

def sigmoid(z):
	return 1/(1+np.exp(-1 * z))

def costFunction( X,Y,theta):
	predictions = sigmoid(np.dot(X,theta))
	err = (-Y * np.log(predictions)) - ((1-Y) * np.log(1- predictions))
	cost = 1/len(Y) * np.sum(err)
	grad = 1/len(Y) * np.dot(X.transpose(),(predictions - Y))
	return cost,grad

def featuresNormalization(X):
	mean = np.mean(X,axis = 0)
	std = np.std(X,axis = 0)
	X_norm = (X - mean)/std
	return X_norm, mean, std

def gradientDescent(X,Y,alpha,num_iter,theta):
	m = len(Y)
	costs = []
	for i in range(num_iter):
		cost,grad = costFunction(X,Y,theta)
		theta -= alpha*grad
		costs.append(cost)
	return costs

def main():
	global X,Y,theta
	X=np.array(X)
	X_train, mean , std = featuresNormalization(X)
	X_train = np.c_[np.ones((X.shape[0], 1)),X_train]
	Y = np.array(Y).reshape(X_train.shape[0],1)
	theta = np.zeros((X_train.shape[1], 1)) 
	costs = gradientDescent(X_train,Y,1,150,theta)
	#plt.plot(range(len(costs)),costs)
	#plt.show()
	pos , neg = (Y==1).reshape(100,1) , (Y==0).reshape(100,1)
	x_value= np.array([np.min(X_train[:,1]),np.max(X_train[:,1])])
	y_value=-(theta[0] +theta[1]*x_value)/theta[2]
	plt.scatter(X_train[pos[:,0],1],X_train[pos[:,0],2], s=10, label='Admitted')
	plt.scatter(X_train[neg[:,0],1],X_train[neg[:,0],2], s=10, label='Not Admitted')
	plt.plot(x_value,y_value, "r")
	plt.legend(loc = 0)
	plt.show()



if __name__ == '__main__':
	main()


