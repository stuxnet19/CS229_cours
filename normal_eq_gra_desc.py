import numpy as np
import matplotlib.pyplot as plt

def cost_equation(X,y,thetas):
	m = y.size
	j_theta = (0.5/m)*(np.dot(X,thetas)-y).T.dot(np.dot(X,thetas)-y)
	return j_theta[0][0]

file_x = open('../data/ex2x.dat','r')
file_y = open('../data/ex2y.dat','r')


X = np.array([ [float(element)] for element in file_x ])
y = np.array([ [float(element)] for element in file_y ])

#ajouter de x0 dans la matrice des features
X =  np.c_[ np.ones(X.size),X]

# normal equation for gradient descent
theta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

