#!/usr/bin/env python
"""
Post processing tools for plotting, viewing states, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import qutip as qt
import datetime


def set_axes_fonts(ax, fsize):
    """
    Set axes font sizes because it should be abstracted away
    """
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(fsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(fsize)


def get_wigner(psi, file_ext=None):
    """
    Compute, return, and save the Wigner function
    """

    # Use the same size grid to compute the Wigner function each time
    xvec = np.linspace(-5, 5, 500)
    W = qt.wigner(psi.ptrace(1), xvec, xvec)

    # Write results to binary files
    if file_ext is not None:
        W.tofile('%s_wigner.bin' % file_ext)
        xvec.tofile('%s_wigner_xvec.bin' % file_ext)

    return xvec, W


def plot_wigner(xvec, W,
                xstr=r'Re$[\alpha]$',
                ystr=r'Im$[\alpha]$',
                tstr='', file_ext=None):
    """
    Plot the Wigner function on a 2D grid, normalized to W.max()
    """

    # Setup the color map, normalizations, etc
    norm = mpl.colors.Normalize(-W.max(), W.max())

    # Fontsize
    fsize = 24; tsize = 26;

    # Setup the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt1 = ax.contourf(xvec, xvec, W, 100, cmap=cm.RdBu, norm=norm)
    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)
    ax.set_title(tstr, fontsize=tsize)

    # Set the axis tick labels to a reasonable size
    # for tick in ax.get_xticklabels():
    #     tick.set_fontsize(fsize)
    # for tick in ax.get_yticklabels():
    #     tick.set_fontsize(fsize)
    set_axes_fonts(ax, fsize)
    
    # Set the color bar
    cbar = fig.colorbar(plt1, ax=ax)
    cbar.ax.set_title(r'$\left<P\right>$', fontsize=fsize)
    cbar.ax.tick_params(labelsize=fsize)

    # Write the results to file
    if file_ext is not None:
        fig.savefig('wigner_%s.eps' % file_ext, format='eps') 
        fig.savefig('wigner_%s.png' % file_ext, format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('wigner_%s.eps' % tstamp, format='eps') 
        fig.savefig('wigner_%s.png' % tstamp, format='png') 
        


def plot_expect(tpts, op_avg, op_name='',
                tscale='ns', file_ext=None,
                plot_phase=False):
    """
    Plot the expectation value of an operator as a function of time
    """


    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fsize = 24; tsize = 26;
    set_axes_fonts(ax, fsize)
    ax.plot(tpts, np.abs(op_avg))
    
    # Set the axes labels
    xstr = 'Time (%s)' % tscale
    ystr = r'$\langle\hat{%s}\rangle$' % op_name

    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)
    plt.tight_layout()
    
    # Save the figure to file
    if file_ext is not None:
        fig.savefig('wigner_%s.eps' % file_ext, format='eps') 
        fig.savefig('wigner_%s.png' % file_ext, format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('wigner_%s.eps' % tstamp, format='eps') 
        fig.savefig('wigner_%s.png' % tstamp, format='png') 
    




