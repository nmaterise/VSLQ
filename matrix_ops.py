#!/usr/bin/env python
"""
Provides tools to construct the matrix representations
for bosonic and fermionic creation / anihilation operators
involving N-particles
"""

import numpy as np


def aop(N):
    """
    Returns the harmonic osicallator destruction operator
    """
    
    try:
        aout = np.diag(np.sqrt(np.linspace(1, N-1, N-1)), k=1)
    except Exception as err:
        print('Caught exception: %s' % err)
        print('Attempting to cast the size, (%d) to an integer ...' % int(N))
        aout = np.diag(np.sqrt(np.linspace(1, int(N-1), int(N-1))), k=1)

    return aout


def sop(x):
    """
    Returns the Pauli matrices, x, y, z, plus/minus
    """
    
    # Orthogonal axis Pauli matrices
    sx = np.array([[0, 1],   [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0],   [0, -1]])

    # Plus / minus Pauli matrices
    sp = (sx + 1j*sy) / 2.
    sm = (sx - 1j*sy) / 2.
    
    # Define a dictionary with all of the Pauli matrices return only as needed
    sdict = {'x' : sx, 'y' : sy, 'z' : sz, 'p' : sp, 'm' : sm}
    
    return sdict[x]


def basis(N, M):
    """
    Emulates behavior of QuTip basis, returning a vector of N-length
    with a 1 in the M-th position and zeros elsewhere. Vectors are returned as
    column vectors to maintain a correspondence with ket-vectors.
    """

    try:
    
        # Initialize a vector of zeros
        vec = np.zeros([N, 1])
        vec[M] = 1

    except Exception as err:
        
        print('Exception in (basis_vec): %s' % err)
        print('Attempting to cast (%d) and (%d) to integers ...' \
                % (int(N), int(M)))
        vec = np.zeros([int(N), 1])
        vec[int(M)] = 1

    return vec


def tensor(*args):
    """
    Tensor product of variable number of operators
    """

    # Check for the format [a, b, c, ..., ]
    if len(args) == 1:
        alist = args[0]
    # Check for the instance [ a ]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        return args[0]
    # Otherwise treat as [a, b, c]
    else:
        alist = args
    for idx, a in enumerate(alist):
        if idx == 0:
            aout = a
        else:
            aout = np.kron(aout, a)    

    return aout


def comm(a, b, sign='-'):
    """
    Implements the commutator / anticommutator for two matrices of equal size
    
    Parameters:
    ----------

    a, b:   left and right matrix operands 
    sign:   '-/+' for commutator / anticommutator, following the notation of
            "Quantum Noise," Gardiner & Zoller, 2004: 

            [a, b]_{-} = ab - ba,
            [a, b]_{+} = ab + ba

    """

    # Return the commutator of a and b,k assuming a, b are matrices
    # @ is the matrix product defined in Python3
    if sign == '-':
        return a@b - b@a
    
    # Return the 
    elif sign == '+':
        return a@b + b@a


def dag(a):
    """
    Returns the complex conjugate transpose of an operator
    """

    return np.transpose(np.conj(a))
