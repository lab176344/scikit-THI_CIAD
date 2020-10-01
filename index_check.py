# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:52:13 2020

@author: balasubramanian
"""
from sklearn.ensemble import RandomForestClassifier,RandomTreesEmbedding
from sklearn.datasets import make_blobs
import numpy as np
X, Y = make_blobs(n_samples=50000, centers=5, n_features=512, random_state=0)
import time
start = time.time()
from sklearn.metrics import pairwise_distances
import umap
estimator = RandomTreesEmbedding(n_estimators=250,max_depth=None,n_jobs=-1)
estimator.fit(X, Y)
estimator.index(type_expect=3)
rfapX = estimator.encode_rfap(X)
rfapXT = []
ntrees = 250
rfap =[]

var2 = [[''.join(map(str, rfapX[j][i, :])) for j in range(250)] for i in range(25000)]
aa = np.array((var2))
from scipy.spatial import distance
D = pairwise_distances(aa, metric="hamming")
end = time.time()

print(end - start)


U = umap.UMAP(metric='precomputed')
XY = U.fit_transform(D)
import matplotlib.pyplot as plt
plt.scatter(*XY.T, s=0.3, c=Y[:25000], cmap='Spectral', alpha=1.0)

# def hamming_distance(chaine1, chaine2):
#     return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

# for j in range(25000):
#     for k in range(25000):
#         a = var2[j]
#         b = var2[k]
#         dist = 0
#         for i in range(250):
#             dist+=hamming_distance(a[i],b[i])



# start = time.time()
# for j in range(50000):
#     ntrees = 250
#     rfapXT = []
#     start = time.time()

#     while ntrees:
#         rf = rfapX[ntrees-1]
#         rfapXT.append ( ''.join(map(str,rf[j,:])))
#         ntrees-=1
#     if j==0:
#         rfap = np.array((rfapXT))
#     else:
#         rfap = np.vstack((rfap,rfapXT))
#     end = time.time()
#     print(end - start)
    
# import multiprocessing as mp
# pool = mp.Pool(processes=mp.cpu_count())
# start = time.time()

# rfap = pool.map( func, [(i) for i in range(1,50000)])
# end = time.time()
# print(end - start)