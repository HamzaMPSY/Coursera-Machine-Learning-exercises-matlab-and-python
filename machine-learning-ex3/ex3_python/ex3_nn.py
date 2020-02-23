import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat


# Load a matlab matrix with python 
mat = loadmat("ex3data1.mat")
X = mat['X']
y = mat['y']
# Load Theta1 and Theta2 
weights = loadmat("ex3weights.mat")
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(X, y, Theta1, Theta2):
	m=X.shape[0]
	X = np.hstack((np.ones((m,1)),X))
	a = sigmoid(X @ Theta1.T)
	m=a.shape[0]
	a = np.hstack((np.ones((m,1)),a))
	predictions = sigmoid(a @ Theta2.T)
	return np.argmax(predictions,axis = 1)+1

def main():
	global X,y,Theta2,Theta1
	print(X.shape,y.shape,Theta1.shape,Theta2.shape)
	pred = predict(X, y, Theta1, Theta2)
	print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000*100,"%")

if __name__ == '__main__':
	main()