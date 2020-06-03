from sklearn import tree
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd

import scipy
N = 1000
N2 = 50
Dim = 10

k_neighbors = 50

posMean = 10
n_trees = 100
np.random.seed(42)
X_11 = np.random.randn(N,Dim) + posMean
np.random.seed(42)
X_12 = np.random.randn(N,Dim) - posMean
X = np.concatenate((X_11,X_12))
Y = None

np.random.seed(4)
X_21 = np.random.randn(N2,Dim) + posMean
np.random.seed(4)
X_22 = np.random.randn(N2,Dim) - posMean
X2 = np.concatenate((X_21,X_22))
'''
start = timeit.default_timer()
estimator = tree.UnsupervisedTree()
estimator.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start)  

print(estimator.tree_.node_count)
'''
from sklearn import ensemble
start = timeit.default_timer()
estimator2 = ensemble.UnsupervisedRandomForest(n_estimators=n_trees,random_state=42)
estimator2.fit(X, Y)
stop = timeit.default_timer()
print('Time Train: ', stop - start) 

start = timeit.default_timer()
matrix1 = estimator2.get_proximity_matrix(X, typeCalc='PathNormal')
stop = timeit.default_timer()
print('Time Path: ', stop - start)
#print(matrix1)

start = timeit.default_timer()
matrix2 = estimator2.get_proximity_matrix(X, typeCalc='PathKNN',k_neighbors =k_neighbors)
stop = timeit.default_timer()
print('Time PathKNN: ', stop - start) 
#print(matrix2)

start = timeit.default_timer()
matrix3 = estimator2.get_proximity_matrix(X, X2= X2, typeCalc='PathNormal')
stop = timeit.default_timer()
print('Time Path2: ', stop - start) 
#print(matrix3)

start = timeit.default_timer()
matrix4 = estimator2.get_proximity_matrix(X, X2= X2, typeCalc='PathKNN',k_neighbors=k_neighbors)
stop = timeit.default_timer()
print('Time PathKNN2: ', stop - start) 

matrix5 = scipy.sparse.coo_matrix((matrix4[0],(matrix4[1],matrix4[2])), shape=(2*N,2*N2)).toarray()
#print(matrix5)
print(matrix5.shape)
print(np.count_nonzero(matrix5,axis=0))

'''
matrix3test = np.zeros((N*2,N2*2))
for i in range(2*N2):
      idx = np.argpartition(matrix3[:,i],-(k_neighbors+1))
      matrix3test[idx[-(k_neighbors+1):],i] = matrix3[idx[-(k_neighbors+1):],i]

print(np.array_equal(matrix3test,matrix5))
'''

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