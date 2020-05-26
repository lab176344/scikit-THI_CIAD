from sklearn import tree
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd

N = 500
Dim = 20

posMean = 10
n_trees = 100
X1 = np.random.randn(N,Dim) + posMean
X2 = np.random.randn(N,Dim) - posMean
X = np.concatenate((X1,X2))
Y = None

print(X)
print(X.shape)

start = timeit.default_timer()
estimator = tree.UnsupervisedTree()
estimator.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start)  

print(estimator.tree_.node_count)

from sklearn import ensemble
start = timeit.default_timer()
estimator2 = ensemble.UnsupervisedRandomForest(n_estimators=n_trees)
estimator2.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start) 

start = timeit.default_timer()
matrix = estimator2.get_proximity_matrix(X, typeCalc='PathNormal')
stop = timeit.default_timer()
print('Time: ', stop - start) 
print(matrix.mean())
print(matrix)
plt.imshow(matrix)
plt.show()


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
Data={'parent':parent,'left_idx':left_idx,'right_idx':right_idx,'random_factor':random_factor,
      'n_node_samples':n_node_samples, 'depth': depthT,
      'weighted_n_node_samples':weighted_n_node_samples,'impurity':impurity}
Tree_Pd=pd.DataFrame(data=Data)
#Tree_Pd.to_excel('Tree.xlsx')
print(Tree_Pd) 