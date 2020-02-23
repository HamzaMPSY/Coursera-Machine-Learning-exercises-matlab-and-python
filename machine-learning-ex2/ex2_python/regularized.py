import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ex2data2.txt')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values



def mapFeature(x1,x2,degree):
	out = np.ones(len(x1)).reshape(len(x1),1) 
	for i in range(degree):
		for j in range(i+1):
			terms = (x1**(i-j) * x2**j).reshape(len(x1),1)
			out = np.hstack((out,terms))
	return out

def mapFeaturePlot(x1,x2,degree):
    out = np.ones(1)
    for i in range(0,degree):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out= np.hstack((out,terms))
    return out

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def costFunctionRegulize(theta, X, y, Lambda):
	m=len(y)
	y=y[:,np.newaxis]
	print(X.shape ,y.shape)
	predictions = sigmoid(X @ theta)
	error = (-y * np.log(predictions) - ((1-y)*np.log(1 - predictions)))
	cost = 1/m * sum(error)
	regCost = cost + Lambda/(2*m) * sum(theta**2)

	j_0 = 1/m * (X.transpose() @ (predictions - y))[0]
	j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)*theta[1:]
	#print(j_0)
	grad = np.vstack((j_0[:,np.newaxis],j_1))
	return cost[0], grad


def gradienDescent(X,y,theta,alpha,numIter,Lambda):
	m = len(y)
	costs = []

	for i in range(numIter):
		cost ,grad = costFunctionRegulize(theta,X,y,Lambda)
		theta -= alpha*grad
		costs.append(cost)
	return theta, costs

def main():
	global X,y
	X_train = mapFeature(X[:,0],X[:,1],6)
	theta = np.zeros((X_train.shape[1],1))
	Lambda = 1
	theta , costs = gradienDescent(X_train,y,theta,1,1500,0.2)
	#plt.plot(costs)
	#plt.xlabel("Iteration")
	#plt.ylabel("$J(\Theta)$")
	#plt.title("Cost function using Gradient Descent")
	pos , neg = (y==1).reshape(X_train.shape[0],1),(y==0).reshape(X_train.shape[0],1)
	plt.scatter(X[pos[:,0],0],X[pos[:,0],1],s=10, label='Accepted')
	plt.scatter(X[neg[:,0],0],X[neg[:,0],1],s=10, label='Rejected')
	uvals = np.linspace(-1,1.5,50)
	vvals = np.linspace(-1,1.5,50)
	z = np.zeros((len(uvals),len(vvals)))
	for i in range(len(uvals)):
		for j in range(len(vvals)):
			a = mapFeaturePlot(uvals[i],vvals[j],6)
			#print(a.shape,theta.shape)
			z[i,j] = a @ theta

	plt.contour(uvals,vvals,z.T,0)
	plt.xlabel("Exam 1 score")
	plt.ylabel("Exam 2 score")
	plt.legend(loc=0)
	plt.show()


if __name__ == '__main__':
	main()







#pos , neg = (y==1).reshape(X.shape[0],1),(y==0).reshape(X.shape[0],1)
#plt.scatter(X[pos[:,0],0],X[pos[:,0],1],s=10, label='Accepted')
#plt.scatter(X[neg[:,0],0],X[neg[:,0],1],s=10, label='Rejected')
#plt.xlabel('Test 1')
#plt.ylabel('Test 2')
#plt.legend()
#plt.show()