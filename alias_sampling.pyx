#cython: cdivision=True
#cython: overflowcheck=False

import cython
import numpy as np
cimport numpy as np
from fast_prng import uniform

def check_equal(x, y, eps=1e-8): 
    """check_equal(x, y) -> raise Assertion Error if values in x & y differ by more than double precision"""
    assert (np.abs(x - y) < eps).all(), "{:}  {:}".format(x,y)

class alias(object):
    def __init__(self, np.ndarray[np.float64_t, ndim=1] pmf):
        """Generates alias for randomly-sampling from pmf"""
        self.pmf = pmf
        cdef long L = pmf.shape[0], i, j, k
        cdef np.ndarray[np.long_t, ndim=1]    A = np.arange(L)
        cdef np.ndarray[np.long_t, ndim=1]    B = np.arange(L+1)
        cdef np.ndarray[np.float64_t, ndim=1] X = np.r_[pmf/pmf.mean(), np.longdouble(2)]         # X[L] = sentinel.
        i, j = 0, L
        while True:
            while X[B[i]]< 1:               # In ascending order, find x_(b_i) > 1.
                i += 1
            while X[B[j]] >= 1:             # In descending order, find x_(b_j) < 1.
                j -= 1
            if i >= j:                      # If ascent passes descent, end
                break
            B[i], B[j] = B[j], B[i]         # Swap b_i, b_j
        i = j
        j += 1
        # At this point, X[B][:j] is < 1 and X[B][j:] is > 1
        while i>=0:
            while X[B[j]] <= 1:             # Find x_(b_j) that needs probability mass
                j += 1
            if j > L-1:                     # Nobody needs probability mass, Done
                break
            X[B[j]] -= 1 - X[B[i]]          # Send all of x_(b_i) excess probability mass to x_(b_j)                (x_(b_i) is now done).
            A[B[i]] = B[j]
            if X[B[j]] < 1:                 # If x_(b_j) now has too much probability mass,
                B[i], B[j] = B[j], B[i]     # Swap, b_i, b_j, it becomes a donor.
                j += 1
            else:                           # Otherwise, leave it as an acceptor
                i -= 1


        self._X = X[:L-1]
        self._A = A
#        new_pmf = np.copy(self._X)
#        for a_i, pmf_i in zip(self._A, self._X): new_pmf[a_i] += 1 - pmf_i	
#        check_equal(new_pmf, self.pmf/self.pmf.mean())

    def sample(self, long N=1):
        """sample(long N=1) -> N polongs sampled from pmf"""
        from numpy.random import randint
        cdef long i, L = len(self._X)
        if N == 1:
            i = <long>uniform(0,L)
            return i if uniform() < self._X[i] else self._A[i]

        cpdef np.ndarray[np.long_t,    ndim=1] samples = uniform(0, L, N).astype(np.long)
        cdef  np.ndarray[np.float64_t, ndim=1] tosses = uniform(size=N)
        cdef double [:] X = self._X
        cdef long [:] A = self._A
        cdef long [:] samps = samples
        for i in range(N):
            if tosses[i] > X[samps[i]]: 
                    samps[i] = A[samps[i]]
        return samples

