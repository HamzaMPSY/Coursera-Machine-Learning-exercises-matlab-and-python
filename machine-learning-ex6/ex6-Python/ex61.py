import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat



# Load a matlab matrix with python 
mat = loadmat("ex6data1.mat")
X = mat['X']
y = mat['y']
# in this exercice do not implement SVM just use an implemented one and play with it 
from sklearn.svm import SVC
classifier = SVC(C=1000,kernel="linear")
classifier.fit(X,np.ravel(y))




pos , neg = (y==1) , (y==0)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1], s=10,)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1], s=10,)

X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)


plt.show()