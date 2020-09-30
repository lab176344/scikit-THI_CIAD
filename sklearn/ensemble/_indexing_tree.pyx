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

ctypedef np.int32_t int32
ctypedef np.uint64_t uint64
ctypedef np.int64_t int64
ctypedef np.float64_t float64
ctypedef np.uint8_t uint8
ctypedef signed char byte8
ctypedef unsigned char ubyte8

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


def convert_number(int[:] n, Py_ssize_t max_depth):
    '''
    convert the ternary to base 10 number
    '''
    cdef int[:] n_flip
    n_flip = n[::-1]
    cdef float64 sum_val 
    cdef Py_ssize_t i
    sum_val = 0
    for i in range(max_depth):
        sum_val = sum_val + (<float64>(n_flip[i]))*(<float64>3**<float64>i)
    return sum_val

def index_tree(int64[::1,] left_idx, int64[::1,] right_idx, int64[::1,] node_depth, Py_ssize_t n_nodes, Py_ssize_t max_depth, int rfap_type=2):
    '''
    Function to index a single 
    rfap_type = 1 for original RFAP
    rfap_type = 2 for ternary conversion
    '''
    cdef int level_count
    cdef Py_ssize_t i
    level_count = 0
    rfap_gen =  np.zeros((node_depth.shape[0],max_depth-level_count),dtype=np.int)
    cdef int depth_count = max_depth-level_count
    rfap_gen[left_idx[0],0] = int(1)
    rfap_gen[right_idx[0],0] = int(2)
    rfap_raw = np.zeros((node_depth.shape[0],),np.float64)
    rfap_raw[left_idx[0],] = (float(''.join(map(str,rfap_gen[left_idx[0],:]))))
    rfap_raw[right_idx[0],] = (float(''.join(map(str,rfap_gen[right_idx[0],:]))))
    if rfap_type == 2:
        rfap_ter = np.zeros((node_depth.shape[0],),np.float64)
        rfap_ter[left_idx[0],] = convert_number((rfap_gen[left_idx[0],:]),max_depth)
        rfap_ter[right_idx[0],] = convert_number((rfap_gen[right_idx[0],:]),max_depth)
    for i in range(1,n_nodes):
        if left_idx[i]!=-1 and right_idx[i]!=-1:
            base_index = rfap_gen[i,:]
            base_to_left = base_index.copy()
            base_to_right = base_index.copy()
            base_to_left[node_depth[i]] = int(1)
            base_to_right[node_depth[i]] = int(2)
            rfap_gen[left_idx[i],:] = base_to_left.copy()
            rfap_gen[right_idx[i],:] = base_to_right.copy()

        if rfap_type == 2:       
            rfap_ter[left_idx[i],] = convert_number((rfap_gen[left_idx[i],:]),max_depth)
            rfap_ter[right_idx[i],] = convert_number((rfap_gen[right_idx[i],:]),max_depth)
        else:
            rfap_raw[left_idx[i],] = (float(''.join(map(str,rfap_gen[left_idx[i],:]))))
            rfap_raw[right_idx[i],] = (float(''.join(map(str,rfap_gen[right_idx[i],:]))))
    
    if rfap_type == 1:
        return rfap_raw
    elif rfap_type == 2:
        return rfap_ter
    else:
        return rfap_gen, depth_count








