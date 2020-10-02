# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:39:18 2020

@author: balasubramanian
"""
import numpy as np
cimport numpy as np
import cython


from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
def hamming_distance(r):
    cdef int i, j, c, size
    cdef np.ndarray[np.float64_t, ndim=2] ans
    size = r.shape[0]
    ans = np.zeros((size,size), dtype=np.float64)
    tree_size =r.shape[1]
    c = -1
    cdef double intersect
    for i in range(size):
        for j in range(i, size):
            for k in range(tree_size):
                intersect = 0
                for c1, c2 in zip(r[i,k], r[j,k]):
                    if c1==c2:
                        intersect += 1
                    else:
                        break
                ans[i,j] += intersect/len(r[i,k])
            ans[i,j] =ans[i,j]/ tree_size
    return ans