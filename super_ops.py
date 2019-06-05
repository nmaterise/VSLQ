#!/usr/bin/env python3
"""
Provides conversion of the density matrix and other operators to and from the
superoperator notation. We use row-major ordering of the matrix elements of the
density matrix to establish a consistent notation.

"""

import matrix_ops as mops
import numpy as np


def dm2sket(rho):
    """
    Converts density matrix to a superket
    """

    # Check that rho is a matrix
    if rho.ndim != 2:
        raise TypeError('rho is not a density matrix, it has (%d) dimensions.'\
                        % rho.ndim)

    # Convert to a numpy array if necessary
    if rho.__class__ == np.ndarray:
        rho_out = rho
    else:
        rho_out = np.array(rho)

    # Get the diagonal and off diagonal elements
    rho_out = rho_out.flatten()
                # np.hstack((np.diag(rho_out), 
                # rho_out[np.where(~np.eye(rho_out.shape[0], dtype=bool))]))

    return rho_out.reshape([rho_out.shape[0], 1])


def sket2dm(rho, N):
    """
    Converts a superket back to a density matrix
    """

    return rho.reshape([N, N])


def op2sop(op, action='l'):
    """
    Converts an operator acting on the left/right to a superoperator
    """

    # Get the dimension of the operator
    N = op.shape[0]
    I = np.eye(N)

    # Compute the tensor product on the left / right depending
    # on its action on the density matrix

    ## Action on the left, A * p
    if action == 'l':
        return mops.tensor(op, I) 
    
    ## Action on the right, p * A
    elif action == 'r':
        return mops.tensor(I, op.T)

    else:
        raise TypeError('action (%s) not supported.' % action)


def issuper(op, N):
    """
    Given the original dimension of the Hilbert space, determine if the operator
    is a superoperator or not
    """

    # Get the operator dimensions
    M = op.shape[0]

    # Check if M is N*N
    if M == (N * N):
        return True
    else:
        return False


def sexpect(op, rho):
    """
    Compute the expectation value of a superoperator with rho represented as a
    superket 
    """

    # Get the dimension of the Hilbert space
    N = int(np.floor(np.sqrt(rho[0].shape[0])))

    # Check for dimensions of rho
    rho = np.asarray(rho) 
    
    # Convert rho to a density matrix
    rhodm = np.array([sket2dm(r, N) for r in rho])

    return np.array([np.trace(op @ rd) for rd in rhodm])
