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

def node_proximity(int[:,::1] nodes, Py_ssize_t n_samples, Py_ssize_t n_trees):
    prox_Matrix =  np.zeros((n_samples,n_samples),np.double)
    cdef double[:,::1] prox_view = prox_Matrix
    cdef Py_ssize_t input_idx1
    cdef Py_ssize_t input_idx2
    cdef Py_ssize_t i
    for input_idx1 in prange(prox_view.shape[0],nogil=True):
        for input_idx2 in range(prox_view.shape[1]):
            for i in range(n_trees):
                if nodes[input_idx1,i] == nodes[input_idx2,i]:
                    prox_view[input_idx1,input_idx2] = prox_view[input_idx1,input_idx2] + 1
            prox_view[input_idx1,input_idx2] = prox_view[input_idx1,input_idx2]/n_trees    

    return prox_Matrix





