import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz 
from sklearn.model_selection import KFold


balance_data = pd.read_csv('balance-scale.data', sep= ',', header= None)
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

kfold = KFold(10, True, 10)
ac=0.0
ac_score=0.0
for train, test in kfold.split(X):
	X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
	clf_entropy.fit(X_train, y_train)
	y_pred_en = clf_entropy.predict(X_test)
	ac_score=accuracy_score(y_test,y_pred_en)*100
	ac=ac+ac_score
	print ("Accuracy is ", ac_score)

ac_avg=ac/10	
print ("Average Accuracy is ", ac_avg)
