#!/usr/bin/env python3
"""
Collection of utility / helper functions to generate drives
"""

import numpy as np


def get_gauss_env(x, x0, sig):
    """
    Returns a Gaussian envelope
    
    Parameters:
    ----------

    x:          abscissa, same units as tscale 
    x0:         center of the Gaussian
    sig:        standard deviation of the Gaussian

    """

    # Compute the normalization factor
    norm = 1. / (np.sqrt(2*np.pi)*sig) 
    
    # Compute the argument
    arg = (x - x0)**2 / (2*sig**2)
    
    return norm * np.exp(-arg)


def get_tanh_env(x, x1, x2, ascale=1.0, tscale=0.5):
    """
    Returns a hyperbolic tangent envelope
    
    Parameters:
    ----------

    x:          abscissa, same units as tscale 
    x1, x2:     start and stop positions of the center of the rise
    ascale:     amplitude scale
    tscale:     controls the rise time of the envelope, e.g.
                y(x) = 1/2 [tanh((x - x1)/tscale) - tanh((x - x2)/tscale) + 1 ]

    Returns:
    -------

    y:          envelope function as described above

    """

    # Modify the tscale scale if not 1/2 or less than the smallest starting
    # position in time
    if tscale <= x1 / 2:
        ttscale = tscale
    else:
        ttscale = x1 / 2

    # Compute the envelope arguments
    argx1 = (x - x1) / ttscale
    argx2 = (x - x2) / ttscale
    
    # Compute the output
    y = ascale * (np.tanh(argx1) - np.tanh(argx2)) / 2

    return y


def get_tanh_seq(x, x1, x2, N, ascale=1.0, tscale=0.5):
    """
    Returns a list of tanh envelopes of length N

    Parameters:
    ----------

    x:          abscissa, same units as tscale [array of times]
    x1, x2:     start and stop positions of the center of the rise on the last
                entry in the list -- x1 is fixed, but x2 changes for all
                entries in the list
    N:          number of entries in the list of envelopes
    ascale:     amplitude scale
    tscale:     controls the rise time of the envelope, e.g.
                y(x) = 1/2 [tanh((x - x1)/tscale) - tanh((x - x2)/tscale) + 1 ]

    Returns:
    -------

    y:          envelope function as described above
    """

    # Start by converting x into numpy array
    if x.__class__ != np.ndarray:
        x = np.asarray(x)

    # Compute the list of x2's
    x2_list = np.linspace(x1+tscale, x2, N+1)

    # Get a new envelope for each x2 in x2_list
    y = np.array([get_tanh_env(x, x1, xx, ascale, tscale) for xx in x2_list])

    return y

