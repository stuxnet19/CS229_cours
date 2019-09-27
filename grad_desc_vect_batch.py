import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X,y,alpha,nb_iter):
	# list of arrays => for each alpha
	# EX :
	# [[theta0,theta1],[theta0,theta1],[theta0,theta1]]
	thetas = [ np.zeros((X.shape[1],1)) for i in alpha]
	# historique des theta de chaque learning rate
	# EX :
	# [[[theta0,theta1],[theta0,theta1],[theta0,theta1]] , [[theta0,theta1],[theta0,theta1],[theta0,theta1]]]
	thetas_history = [[ np.zeros((X.shape[1],1)) for i in alpha] for i in range(nb_iter)]
	# historique des couts de chaque learning rate
	j_theta_hisory = [[0 for a in alpha] for i in range(nb_iter)]
	m = y.size
	
	for index1 in range(len(alpha)) :	
		for i in range(nb_iter) :
			# calculte the hipothesis
			h = np.dot(X,thetas[index1])			
			thetas[index1] = thetas[index1] - (1/m)*alpha[index1]*(X.T.dot((h-y)))
			thetas_history[i][index1] = thetas[index1]
			j_theta_hisory[i][index1] = cost_equation(X,y,thetas[index1])

	return thetas,thetas_history,j_theta_hisory

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

nb_iter = 1500
alpha = [0.001,0.003,0.01,0.03,0.05,0.07]
(theta,thetas_history,j_theta_hisory) = gradient_descent(X,y,alpha,nb_iter)

gride_size = (1,1)

index = 0
index_plt = 1
#plt.plot([i for i in range(nb_iter)],[theta[2] for theta in j_theta_hisory ])

print(theta)
for i in range(int(len(alpha)/2)-1):
	for j in range(int(len(alpha)/2)):
		plt.subplot(3,2,index+1).title.set_text('learning rate : '+str(alpha[index]))
		plt.plot([i for i in range(nb_iter)],[j_theta[index] for j_theta in j_theta_hisory])
		plt.xlabel('iteration number')
		plt.ylabel('j(o)')
		index += 1

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.suptitle('batch GD')
plt.show()