import numpy as np
import matplotlib.pyplot as plt


def newtons_methode(x,y,nb_iter):
    theta = np.zeros((x.shape[1],1))
    num_iter = 1
    for num_iter in range(nb_iter):
        H = get_hessian(x,y,theta)
        jtheta_history[num_iter] = cost_function(x,y,theta)
        grad  = get_gradient(x,y,theta)
        theta = theta - np.linalg.inv(H).T.dot(grad)
        print(theta.shape)
    return theta,jtheta_history

def get_gradient(x,y,theta):
    h = get_sigmoid(np.dot(x, theta))
    gradient = np.dot(x.T, (h - y)) / y.size
    return gradient

def get_hessian(x,y,theta):
    m = y.shape[0]
    h = get_sigmoid(np.dot(x, theta))
    diag = np.multiply(h, (1 - h)) * np.identity(m)
    hessian = (1 / m) * np.dot(np.dot(X.T, diag), X)
    return hessian

def cost_function(X, y, theta):
    h = get_sigmoid(np.dot(X, theta))
    cost = y * np.log(h) + (1 - y) * np.log(1 - h)
    return -cost.mean()

def get_sigmoid(z) :
    # utilisation de numpy pour l'exponentielle au lieux de math
    # numpy.exp accepte les tablea comme paramettre
    s = 1/(1+np.exp(-z))
    return s

file_x = open('../data/ex4x.dat','r')
file_y = open('../data/ex4y.dat','r')

X = np.array([ [ float(col) for col in element.split(" ") if col != ""] for element in file_x ])
y = np.array([ [ float(col) for col in element.split(" ") if col != ""] for element in file_y ])
y = np.array([1 if i==1 else 0 for i in y ])

#ajouter de x0 dans la matrice des features

X =  np.c_[ np.ones(X.shape[0]),X]
nb_iter = 10
jtheta_history = [0 for i in range(nb_iter)]
theta,jtheta_history = newtons_methode(X,y,nb_iter)
print(theta.shape)
color = np.array(['b','g'])
plt.subplot(3,2,1).title.set_text('data visualisation')
ax = plt.gca()
"""
x1_val = np.array(ax.get_xlim())
x2_val = - (x1_val * theta[theta.shape[1]-1][1] + theta[theta.shape[1]-1][0]) / theta[theta.shape[1]-1][2]
"""
#plt.plot(x1_val, x2_val, color='black')
plt.scatter(X[:,1],X[:,2],s=50,c=color[y])
plt.subplot(2,2,2).title.set_text('cost function')
plt.plot([i for i in range(nb_iter)],[j_theta for j_theta in jtheta_history])
plt.xlabel('iteration number')
plt.ylabel('j(o)')
plt.show()
