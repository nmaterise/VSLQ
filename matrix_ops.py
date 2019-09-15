#!/usr/bin/env python
"""
Provides tools to construct the matrix representations
for bosonic and fermionic creation / anihilation operators
involving N-particles
"""

import numpy as np
import scipy.sparse as scsp
import math


def destroy(N):
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
    sp = (sx - 1j*sy).real / 2.
    sm = (sx + 1j*sy).real / 2.
    
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


def coherent(N, alpha):
    """
    Returns a coherent state approximated to N photons
    """

    # Initialize the state
    alpha_ket = np.zeros([N, 1], dtype=np.complex128)
    
    # Compute the remaining terms
    for n in range(N):
        alpha_ket += basis(N, n)*alpha**n / np.sqrt(math.factorial(n))

    alpha_ket *= np.exp(-(np.abs(alpha)**2) / 2.)

    return alpha_ket


def coherent_sup(N, alphas):
    """
    Returns an equally weighted coherent superposition
    """

    # Get the number of coherent states in the superposition state
    M = len(alphas)

    return sum([coherent(N, a) for a in alphas]) / np.sqrt(M)


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


def dag(a):
    """
    Returns the complex conjugate transpose of an operator
    """

    return np.transpose(np.conj(a))


def ket2dm(psi):
    """
    Converts a ket state vector to a density matrix by computing the outer
    product

    p = | psi > < psi |

    """

    return psi @ dag(psi)


def expect(op, rho):
    """
    Returns the expectation value of an operator given the density matrix, rho
    """

    # Convert the density matrix to a numpy array if needed
    rho = np.asarray(rho)            

    # Check dimensions
    dims = rho[0].shape

    # Check if rho is sparse, convert back to dense
    if rho[0].__class__ == scsp.csc.csc_matrix:
        rho = np.array([r.todense() for r in rho])

    # Check if rho is a density matrix, <a> = Tr[a rho]
    if dims[0] == dims[1]:
        return np.array([np.trace(op@rho[i]) for i in range(rho.shape[0])])
    
    # Check if rho is a state vector, <a> = < psi | a | psi >
    elif np.min(dims) == 1:
        exp_out = np.array([dag(rho[i])@op@rho[i] for i in range(rho.shape[0])])
        return exp_out.ravel()
    
    # Dimensions problem
    else:
        raise TypeError('Dimensions of rho (%d x %d) not square or vector' \
                         % (dims[0], dims[1]))


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
    
    # Return the anticommutator
    elif sign == '+':
        return a@b + b@a


def print_sops():
    """
    Prints the Pauli matrices as a test
    """
    
    for x in ['x', 'y', 'z', 'p', 'm']:
        print('s_{}:\n{}'.format(x, sop(x)))

    print('a:')
    print(destroy(2))
    print('a^t:')
    print(dag(destroy(2)))

    print('np.allclose(sp, at): {}'\
            .format(np.allclose(sop('p'), dag(destroy(2)))))
    print('np.allclose(sm, a): {}'\
            .format(np.allclose(sop('m'), destroy(2))))



if __name__ == '__main__':

    print_sops()






