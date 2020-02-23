import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
mat = loadmat("ex7data2.mat")
X = mat["X"]


def findClosestCentroids(X,centroids):
	K = centroids.shape[0]
	idx = np.zeros((X.shape[0],1))
	temp = np.zeros((centroids.shape[0],1))

	for i in range(X.shape[0]):
		for j in range(K):
			dist = X[i,:] - centroids[j,:]
			length = np.sum(dist**2)
			temp[j] = length
		idx[i] = np.argmin(temp)+1
	return idx

def computeCentroids(X, idx, K):
	m, n = X.shape[0],X.shape[1]
	centroids = np.zeros((K,n))
	count = np.zeros((K,1))

	for i in range(m):
		index = int((idx[i]-1)[0])
		centroids[index,:]+=X[i,:]
		count[index]+=1

	return centroids/count


def plotKmeans(X, centroids, idx, K, num_iters):
    """
    plots the data points with colors assigned to each centroid
    """
    m,n = X.shape[0],X.shape[1]
    
    
    for i in range(num_iters):    
        # Visualisation of data
        color = "rgb"
        for k in range(1,K+1):
            grp = (idx==k).reshape(m,1)
            plt.scatter(X[grp[:,0],0],X[grp[:,0],1],c=color[k-1],s=15)

        # visualize the new centroids
        plt.scatter(centroids[:,0],centroids[:,1],s=120,marker="x",c="black",linewidth=3)
        
        # Compute the centroids mean
        centroids = computeCentroids(X, idx, K)
        
        # assign each training example to the nearest centroid
        idx = findClosestCentroids(X, centroids)
        plt.show()


def KmeansRandomInit(X,K):
	m,n = X.shape
	centroids = np.zeros((K,n))
	for i in range(K):
		centroids[i] = X[np.random.randint(0,m+1),:]

	return centroids    
    

mat2 = loadmat("bird_small.mat")
A = mat2["A"]

# preprocess and reshape the image
X2 = (A/255).reshape(128*128,3)

def runKmeans(X, initial_centroids,num_iters,K):
    
    idx = findClosestCentroids(X, initial_centroids)
    
    for i in range(num_iters):
        
        # Compute the centroids mean
        centroids = computeCentroids(X, idx, K)

        # assign each training example to the nearest centroid
        idx = findClosestCentroids(X, initial_centroids)

        
    return centroids, idx

K2 = 16
num_iters = 10
initial_centroids2 = KmeansRandomInit(X2, K2)
centroids2, idx2 = runKmeans(X2, initial_centroids2, num_iters,K2)
m2,n2 = X.shape[0],X.shape[1]
X2_recovered = X2.copy()
for i in range(1,K2+1):
    X2_recovered[(idx2==i).ravel(),:] = centroids2[i-1]

# Reshape the recovered image into proper dimensions
X2_recovered = X2_recovered.reshape(128,128,3)

# Display the image
import matplotlib.image as mpimg
fig, ax = plt.subplots(1,2)
ax[0].imshow(X2.reshape(128,128,3))
ax[1].imshow(X2_recovered)
plt.show()