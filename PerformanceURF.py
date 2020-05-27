from sklearn import tree
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble
import pickle
nTrees_vect = [50,100]#,500]
D_vect = [5,10,50,100]#,500]
N_vect = [10,10000]

posMean = 10
Y = None

forest_time = np.zeros((6,5,3,3))
prox_time = np.zeros((6,5,3,3))
results = np.zeros((6,5,3,3))

for i,N in enumerate(N_vect):
    for j,D in enumerate(D_vect):
        X1 = np.random.randn(N//2,D) + posMean
        X2 = np.random.randn(N//2,D) - posMean
        X_out = np.concatenate((X1,X2))
        for k,n_trees in enumerate(nTrees_vect):
            for l in range(3):
                X = np.copy(X_out)
                print(i,j,k,l)
                start = timeit.default_timer()
                forest = ensemble.UnsupervisedRandomForest(n_estimators=n_trees,random_state=42)
                
                forest.fit(X, Y)
                time = timeit.default_timer()-start
                forest_time[i,j,k,l] = time

                start = timeit.default_timer()
                matrix = forest.get_proximity_matrix(X, typeCalc='PathNormal')
                time = timeit.default_timer()-start
                prox_time[i,j,k,l] = time

                results[i,j,k,l] = matrix.mean()
                del forest
                del matrix,X
                with open('results2.pkl','wb') as f:
                    pickle.dump([forest_time,prox_time,results],f)
        del X1
        del X2
        del X_out,k,n_trees,l
