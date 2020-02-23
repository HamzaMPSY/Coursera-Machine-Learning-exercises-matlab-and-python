# Programming Exercise 1: Linear Regression
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read data from file 
data = pd.read_csv('ex1data1.txt',sep = ",",header = None)
data.columns = ['population','profit']
m,n = data.shape
X = []
for i in range(0,m):
	X.append(np.array([1.,data['population'].values[i]]).reshape(-1, 1))
Y = np.array(data['profit']).reshape(-1,1)


def computeCost(theta):
	global X,Y
	totalCost = 0
	for i in range(len(X)):
		totalCost += (Y[i][0] - (theta[0]*X[i][0] + theta[1]*X[i][1])) ** 2
	return totalCost[0] / (2*len(X))


def gradientStep(theta,alpha):
	global X,Y
	a = 0
	b = 0
	m = len(X)
	for i in range(m):
		a+= ((theta[0]*X[i][0] + theta[1]*X[i][1]) - Y[i][0])*X[i][0]
		b+= ((theta[0]*X[i][0] + theta[1]*X[i][1]) - Y[i][0])*X[i][1]
	return [theta[0] - alpha*a/m , theta[1] - alpha*b/m]


def gradientDescent():
	global X,Y
	theta = [0,0]
	iteration = 2500
	alpha = 0.01
	debug =[]
	print("Starting Gradient Descent at theta0 = {0}, theta1 ={1}, cost = {2}".format(theta[0],theta[1],computeCost(theta)))
	debug.append(computeCost(theta))
	print("Running...")
	for i in range(iteration):
		theta = gradientStep(theta,alpha)
		debug.append(computeCost(theta))
	#plt.plot(range(iteration+1),debug)
	#plt.show()
	return theta


theta = gradientDescent()
# print(data) 
# Scatter Plot with seaborn 
sns.scatterplot(x='population',y='profit',data = data)
plt.xlabel('Population of City in 10,000s');
plt.ylabel('Profit in $10,000s');
line = [theta[0] + theta[1]*i for i in range(4,24)]
plt.plot(range(4,24),line)
plt.show()

sns.regplot(x='population',y='profit',data = data)
plt.show()