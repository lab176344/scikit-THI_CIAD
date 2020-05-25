from sklearn import tree
import numpy as np
import timeit

N = 5000
Dim = 20

posMean = 10

X1 = np.random.RandomState(42).randn(N,Dim) + posMean
X2 = np.random.RandomState(42).randn(N,Dim) - posMean
X = np.concatenate((X1,X2))
Y = None

print(X.shape)

start = timeit.default_timer()
estimator = tree.UnsupervisedTree()
estimator.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start)  

from sklearn import ensemble
start = timeit.default_timer()
estimator2 = ensemble.UnsupervisedRandomForest()
print("Fit the Forest")
estimator2.fit(X, Y)
stop = timeit.default_timer()
print('Time: ', stop - start)  