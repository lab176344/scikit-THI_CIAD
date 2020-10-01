# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:39:18 2020

@author: balasubramanian
"""
import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double abs(double t)
ctypedef np.npy_float32 DTYPE_t          # Type of X

@cython.wraparound(False)
@cython.boundscheck(False)
def hamming_distance(r):
    cdef int i, j, c, size
    cdef np.ndarray[np.float64_t, ndim=2] ans
    size = r.shape[0]
    ans = np.zeros((size,size), dtype=np.float64)
    tree_size =r.shape[1]
    c = -1
    for i in range(r.shape[0]):
        for j in range(i, r.shape[0]):
            for k in range(tree_size):
                ans[i,j] += (sum(c1 != c2 for c1, c2 in zip(r[i,k], r[j,k])))/len(r[i,k])
            ans[i,j] =ans[i,j]/ tree_size
    return ans