# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Jonas Wurst and Lakshman Balasubramanian
#
# License: BSD 3 clause

cimport cython

from libc.stdlib cimport free
from libc.string cimport memset
from libc.math cimport floor

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
#To do
# Path proximity binary load
def path_proximity_binary(int[:,:,::1] paths, int[:,::1] path_lengths, Py_ssize_t n_samples,Py_ssize_t n_trees):
    prox_Matrix =  np.zeros((n_samples,n_samples),np.double)
    cdef double[:,::1] prox_view = prox_Matrix
    cdef int intersectAB
    cdef int unionAB
    sub =  np.zeros((n_samples,),np.double)
    cdef double[::1] sub_view = sub
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t input_idx1
    cdef Py_ssize_t input_idx2
    cdef int path_l
    cdef uint8 L
    cdef uint8 R
    cdef uint8 m
    div_lut=np.arange((path_lengths.shape[1]+1)*2,dtype=np.int)//2
    cdef int[::1] div_lut_view=div_lut 
    for input_idx1 in range(prox_view.shape[0]-1):#prange(prox_view.shape[0]-1,nogil=True):
        for input_idx2 in range(input_idx1+1,prox_view.shape[1]):
            for i in range(n_trees):
                intersectAB = 1
                j=1
                path_l = min(path_lengths[input_idx1,i],path_lengths[input_idx2,i])
                L = 0
                R = path_l - 1
                while L <= R:
                    m = div_lut_view[L+R]
                    if paths[input_idx1,i,m] == paths[input_idx2,i,m]:
                        L = m + 1
                    else:
                        R = m - 1
                intersectAB = m + 1
                #print(intersectAB)
                prox_view[input_idx1,input_idx2] += (<double>intersectAB)/(<double>(path_lengths[input_idx1,i]+ path_lengths[input_idx2,i]-intersectAB))
            prox_view[input_idx1,input_idx2] = prox_view[input_idx1,input_idx2]/n_trees    

    return prox_Matrix



