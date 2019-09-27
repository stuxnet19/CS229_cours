import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

"""
Locally weighted Linear regression : 
    whe have to fit the model for each prediction
"""

def local_regression(x0, X, Y, tau):
    # add bias term
    
    # fit model: normal equations with kernel
    xw = X.T * radial_kernel(x0, X, tau)
    beta = np.linalg.pinv(xw @ X) @ xw @ Y
    
    # predict value
    return x0 @ beta

def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))

file_x = open('../data/ex2x.dat','r')
file_y = open('../data/ex2y.dat','r')


X = np.array([ [float(element)] for element in file_x ])
y = np.array([ [float(element)] for element in file_y ])
X =  np.c_[ np.ones(X.size),X]


nb_iter = 1500
tho = 0.1
alpha = 0.01
y_pred = np.zeros(X.shape[0])
index = 0
for x_pred in X :
    y_pred[index] = local_regression(x_pred,X,y,tho)
    index += 1

plt.scatter([x[1] for x in X], y, alpha=.3)
plt.plot([x for x in X],[y for y in y_pred])
plt.show()
