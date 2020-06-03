from sklearn import tree
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd

N = 1000
Dim = 10

k_neighbors = 15

posMean = 10
n_trees = 100
np.random.seed(42)
X1 = np.random.randn(N,Dim) + posMean
np.random.seed(42)
X2 = np.random.randn(N,Dim) - posMean
X = np.concatenate((X1,X2))
Y = None

start = timeit.default_timer()
estimator = tree.UnsupervisedTree()
estimator.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start)  

print(estimator.tree_.node_count)
from sklearn import ensemble
start = timeit.default_timer()
estimator2 = ensemble.UnsupervisedRandomForest(n_estimators=n_trees,random_state=42)
estimator2.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start) 

start = timeit.default_timer()
matrix1 = estimator2.get_proximity_matrix(X, typeCalc='PathNormal')
stop = timeit.default_timer()
print('Time: ', stop - start)
print(matrix1)

start = timeit.default_timer()
matrix2 = estimator2.get_proximity_matrix(X, typeCalc='PathKNN',k_neighbors =k_neighbors)
stop = timeit.default_timer()
print('Time: ', stop - start) 

print(matrix2)



Tree =estimator2.estimators_[0]
depth=Tree.tree_.max_depth
node_count=Tree.tree_.node_count
n_classes=Tree.tree_.n_classes
left_idx=Tree.tree_.children_left
right_idx=Tree.tree_.children_right
n_leaves=Tree.tree_.n_leaves
feature=Tree.tree_.feature
threshold=Tree.tree_.threshold
impurity=Tree.tree_.impurity
parent=Tree.tree_.parent
random_factor=Tree.tree_.random_factor
n_node_samples=Tree.tree_.n_node_samples
weighted_n_node_samples=Tree.tree_.weighted_n_node_samples
value=Tree.tree_.value
depthT=Tree.tree_.depth
print('Tree_structure')
Data={'parent':parent,'left_idx':left_idx,'right_idx':right_idx,'random_factor':random_factor,
      'n_node_samples':n_node_samples, 'depth': depthT,
      'weighted_n_node_samples':weighted_n_node_samples,'impurity':impurity}
Tree_Pd=pd.DataFrame(data=Data)
#Tree_Pd.to_excel('Tree.xlsx')
print(Tree_Pd) 