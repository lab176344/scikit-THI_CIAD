# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Lakshman Balasubramanian and Jonas Wurst
#
# License: BSD 3 clause

cimport cython

from libc.stdlib cimport free
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ..tree._tree cimport Node
from ..tree._tree cimport Tree
from ..tree._tree cimport DTYPE_t
from ..tree._tree cimport SIZE_t
from ..tree._tree cimport INT32_t
from ..tree._utils cimport safe_realloc

ctypedef np.int32_t int32
ctypedef np.float64_t float64
ctypedef np.uint8_t uint8

# no namespace lookup for numpy dtype and array creation
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import bool as np_bool
from numpy import float32 as np_float32
from numpy import float64 as np_float64

# Parallel cython 
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)

def node_proximity_knn(int[:,::1] nodes, Py_ssize_t n_samples, Py_ssize_t n_trees, Py_ssize_t k_neighbors_in):
    cdef Py_ssize_t k_neighbors = k_neighbors_in +1
    if k_neighbors >=  n_samples:
        k_neighbors =  n_samples


    index_1 =  np.zeros((k_neighbors*n_samples,),np.int32)
    index_2 =  np.zeros((k_neighbors*n_samples,),np.int32)
    prox_values =  np.zeros((k_neighbors*n_samples,),np.double)
    cdef double[::1] prox_values_view = prox_values
    cdef int[::1] index_1_view = index_1
    cdef int[::1] index_2_view = index_2   
    
    index_1_best_intern =  np.zeros( (n_samples,k_neighbors), np.int32)
    index_2_best_intern =  np.zeros( (n_samples,k_neighbors), np.int32)
    prox_best_intern =  np.zeros( (n_samples,k_neighbors), np.double)
    cdef double[:,::1] prox_best_intern_view = prox_best_intern
    cdef int[:,::1] index_1_best_intern_view = index_1_best_intern
    cdef int[:,::1] index_2_best_intern_view = index_2_best_intern 

    prox_zws =  np.zeros( (n_samples,), np.double)
    cdef double[::1] prox_zws_view = prox_zws
    
    cdef Py_ssize_t k
    cdef Py_ssize_t k2
    cdef Py_ssize_t k3
    cdef Py_ssize_t k4
    cdef Py_ssize_t index
    cdef Py_ssize_t input_idx1
    cdef Py_ssize_t input_idx2
    cdef Py_ssize_t index_local
    
    cdef Py_ssize_t i
    cdef int j = 0

    for input_idx1 in prange(n_samples, nogil=True):
        for input_idx2 in range(n_samples):
            prox_zws_view[input_idx1] = 0.0
            for i in range(n_trees):
                if nodes[input_idx1,i] == nodes[input_idx2,i]:
                    prox_zws_view[input_idx1] = prox_zws_view[input_idx1] + 1
                
                #if prox_zws_view[input_idx1] + <double>(n_trees -i) < prox_best_intern_view[input_idx1,k_neighbors-1]:
                #    break
            if prox_zws_view[input_idx1] >= prox_best_intern_view[input_idx1,k_neighbors-1]:
                # Needs to be included to the k best values; first find index
                index = 0
                for k2 in range(k_neighbors):
                    if prox_zws_view[input_idx1] >= prox_best_intern_view[input_idx1,k2]:
                        index = k2
                        break
                if index == k_neighbors-1:
                    prox_best_intern_view[input_idx1,index] = prox_zws_view[input_idx1]
                    index_1_best_intern_view[input_idx1,index] = input_idx1
                    index_2_best_intern_view[input_idx1,index] = input_idx2
                else:
                    # Rearrange the vectors
                    for k4 in range(k_neighbors-1,index,-1):
                        prox_best_intern_view[input_idx1,k4] = prox_best_intern_view[input_idx1,k4-1]
                        index_1_best_intern_view[input_idx1,k4] = index_1_best_intern_view[input_idx1,k4-1]
                        index_2_best_intern_view[input_idx1,k4] = index_2_best_intern_view[input_idx1,k4-1]
                    prox_best_intern_view[input_idx1,index] =prox_zws_view[input_idx1]
                    index_1_best_intern_view[input_idx1,index] = input_idx1
                    index_2_best_intern_view[input_idx1,index] = input_idx2
            
        
        # Append the internal to the out vectors 
        for k3 in range(k_neighbors):
            index_local = <Py_ssize_t>(input_idx1*k_neighbors)+k3
            prox_values_view[index_local] = prox_best_intern_view[input_idx1,k3]/n_trees
            index_1_view[index_local] = index_1_best_intern_view[input_idx1,k3]
            index_2_view[index_local] = index_2_best_intern_view[input_idx1,k3]
        
    return prox_values, index_1, index_2





