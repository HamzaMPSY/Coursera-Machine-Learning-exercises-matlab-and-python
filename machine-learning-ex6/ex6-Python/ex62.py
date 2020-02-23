import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat



# Load a matlab matrix with python 
mat = loadmat("ex6data2.mat")
X = mat['X']
y = mat['y']



pos , neg = (y==1) , (y==0)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1], s=10,)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1], s=10,)
plt.show()


# tkharba9t 7it f matlab kaygolihom imlementiw dakchi o f python khouna kaykhadam ghhi lwajed ma3reftch mera akhera o n3awed hadchi 
#TODO