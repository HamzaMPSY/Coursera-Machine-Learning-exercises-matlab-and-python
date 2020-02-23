import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat


# Load a matlab matrix with python 
mat = loadmat("ex4data1.mat")
X = mat['X']
y = mat['y']
# Load Theta1 and Theta2 
weights = loadmat("ex4weights.mat")
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    """
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer
    
    return np.argmax(a2,axis=1)+1
    

def displayData():
	global X
	# plot data 
	fig, axis = plt.subplots(10,10,figsize=(8,8))
	for i in range(10):
		for j in range(10):
			axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape((20,20),order='F'),cmap='hot')
			axis[i,j].axis('off')
	plt.show()



def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    s = sigmoid(z)
    return s *(1-s)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

	# fist we reshape back nn_params to Theta1 and Theta2
	Theta1 = nn_params[:((input_layer_size+1) *hidden_layer_size)].reshape(hidden_layer_size,input_layer_size +1)
	Theta2 = nn_params[((input_layer_size+1) *hidden_layer_size):].reshape(num_labels,hidden_layer_size +1)

	m = X.shape[0]
	J = 0
	X = np.hstack((np.ones((m,1)),X))
	y10 = np.zeros((m,num_labels))

	a1 = sigmoid(X @ Theta1.T)
	a1 = np.hstack((np.ones((m,1)),a1))
	a2 = sigmoid(a1 @ Theta2.T)

	for i in range(1,num_labels+1):
		y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)

	for j in range(num_labels):
		J = J + sum(-y10[:,j] * np.log(a2[:,j]) - (1 - y10[:,j])*np.log(1 - a2[:,j]))

	cost = 1/m * J
	reg_cost = cost + Lambda/(2*m)*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
	
	grad1 = np.zeros(Theta1.shape)
	grad2 = np.zeros(Theta2.shape)

	for i in range(m):
		xi = X[i,:]   #(1,401)
		a1i = a1[i,:] #(1, 26)
		a2i = a2[i,:] #(1, 10)
		d2 =  a2i - y10[i,:]
		d1 = Theta2.T @ d2.T * sigmoidGradient(np.hstack((1,xi@Theta1.T)))
		grad1 = grad1 + d1[1:][:,np.newaxis] @ xi[:,np.newaxis].T
		grad2 = grad2 + d2.T[:,np.newaxis] @  a1i[:,np.newaxis].T

	grad1 = grad1 / m
	grad2 = grad2 / m

	grad1_reg = grad1 + (Lambda/m) * np.hstack((np.zeros((Theta1.shape[0],1)),Theta1[:,1:]))
	grad2_reg = grad2 + (Lambda/m) * np.hstack((np.zeros((Theta2.shape[0],1)),Theta2[:,1:]))

	return cost, grad1, grad2,reg_cost, grad1_reg,grad2_reg

def randomInitialization(L_in,L_out):
	# eps = sqrt(6) / sqrt(L_in + L_out)

	eps = (6**1/2) / (L_in + L_out)**1/2
	return np.random.rand(L_out,L_in + 1) * (2*eps) - eps

def gradientDescent(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size, num_labels):

	Theta1 = initial_nn_params[:((input_layer_size+1) *hidden_layer_size)].reshape(hidden_layer_size,input_layer_size +1)
	Theta2 = initial_nn_params[((input_layer_size+1) *hidden_layer_size):].reshape(num_labels,hidden_layer_size +1)

	m = len(y)
	costs = []
	for i in range(num_iters):
		nn_params = np.append(Theta1.flatten(),Theta2.flatten())
		cost, grad1, grad2 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[3:]
		Theta1 = Theta1 - alpha*grad1
		Theta2 = Theta2 - alpha*grad2
		costs.append(cost)

	final_nn_params =np.append(Theta1.flatten(),Theta2.flatten())
	return final_nn_params,costs



def main():
	global X,y
	input_layer_size  = 400
	hidden_layer_size = 25
	num_labels = 10
	Lambda = 1
	alpha = 0.8
	num_iters = 800
	initial_Theta1 = randomInitialization(input_layer_size, hidden_layer_size)
	initial_Theta2 = randomInitialization(hidden_layer_size, num_labels)
	initial_nn_params = np.append(initial_Theta1.flatten(),initial_Theta2.flatten())

	nnTheta, costs = gradientDescent(X,y,initial_nn_params,alpha,num_iters,Lambda,input_layer_size, hidden_layer_size, num_labels)
	Theta1 = nnTheta[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
	Theta2 = nnTheta[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

	plt.plot(costs)
	plt.show()
	pred3 = predict(Theta1, Theta2, X)
	print("Training Set Accuracy:",sum(pred3[:,np.newaxis]==y)[0]/5000*100,"%")


if __name__ == '__main__':
	main()