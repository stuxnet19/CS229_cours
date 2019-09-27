import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz 

Jogging_data=pd.read_csv('JoggingTitre.csv', sep=',')
y=Jogging_data['Jogging']
x=Jogging_data.drop(['Jogging'], axis=1)
x_dum=pd.get_dummies(x)
x_dum
# partiel dummies : sub_dum=pd.get_dummies(x, columns=['Temps', 'Vent'])

clf_entropy = DecisionTreeClassifier(criterion = "entropy")
outputTree=clf_entropy.fit(x_dum, y)

dot_data = tree.export_graphviz(outputTree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Td2_dum01")

dot_data = tree.export_graphviz(outputTree, out_file=None, feature_names = x_dum.columns)
graph = graphviz.Source(dot_data)
graph.render("Td2_dum01Name")
#Check and compare the files Td2_dum01.pdf and Td2_dum01Name.pdf
