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
    set_axes_fonts(ax, fsize)
    
    # Set the color bar
    cbar = fig.colorbar(plt1, ax=ax)
    cbar.ax.set_title(r'$\left<P\right>$', fontsize=fsize)
    cbar.ax.tick_params(labelsize=fsize)

    # Write the results to file
    if file_ext is not None:
        fig.savefig('figs/wigner_%s.eps' % file_ext, format='eps') 
        fig.savefig('figs/wigner_%s.png' % file_ext, format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/wigner_%s.eps' % tstamp, format='eps') 
        fig.savefig('figs/wigner_%s.png' % tstamp, format='png') 
        


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
        fig.savefig('figs/expect_%s.eps' % file_ext, format='eps') 
        fig.savefig('figs/expect_%s.png' % file_ext, format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/expect_%s.eps' % tstamp, format='eps') 
        fig.savefig('figs/expect_%s.png' % tstamp, format='png') 
    

def plot_phase_traces(tpts, adata, nkappas, drvs, kappa, tscale='ns'):
    """
    Plots the time traces <a>(t) and the drives used to produce them
    """

    # Setup the figure
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    
    # Set the figure axes
    fsize = 24
    for axx in ax:
        set_axes_fonts(axx, fsize)
    
    # Reshape the data to more sensible dimensions
    # Nkappas x Ntpts    
    adata = adata.reshape([nkappas.size, tpts.size])

    # Plot the data for each kappa
    for ad, drv, nk in zip(adata, drvs, nkappas):
        ax[0].plot(kappa*tpts, ad.real)
        ax[1].plot(kappa*tpts, ad.imag)
        ax[2].plot(kappa*tpts, drv, label=r'$%g/\kappa$'% (nk))
        
    # Set the x, y axis labels
    ax[0].set_ylabel(r'$\Re\{\hat{a}\}$', fontsize=fsize)
    ax[1].set_ylabel(r'$\Im\{\hat{a}\}$', fontsize=fsize)
    ax[2].set_xlabel(r'Time (1/$\kappa$)', fontsize=fsize)
    ax[2].set_ylabel(r'Drive Amplitude ($g_x$)', fontsize=fsize)
    
    # Get and set the legends
    hdls2, legs2 = ax[2].get_legend_handles_labels()
    ax[2].legend(hdls2, legs2, loc='best')
    

    # Fix the layout
    plt.tight_layout()
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/traces_phase_diagram_%s.eps' % tstamp, format='eps')
    fig.savefig('figs/traces_phase_diagram_%s.png' % tstamp, format='png')


def plot_phase_ss(adata, tpts, nkappas):
    """
    Plots the steady state values of <a>(t) in a quadrature plot
    """

    # Setup the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Set the figure axes
    fsize = 24
    set_axes_fonts(ax, fsize)
    
    # Reshape the data
    adata = adata.reshape([nkappas.size, tpts.size])

    # Take the steady state values, the last values
    # in the time domain simulation
    a0 = np.array([ad[-1] for ad in adata])

    # Plot the data for each kappa
    ax.plot(a0.real, a0.imag, 'x-')
        
    # Set the x, y axis labels
    ax.set_ylabel(r'$\Im\{\hat{a}\}$', fontsize=fsize)
    ax.set_xlabel(r'$\Re\{\hat{a}\}$', fontsize=fsize)
    
    # Get and set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best')
    

    # Fix the layout
    plt.tight_layout()
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/ss_phase_diagram_%s.eps' % tstamp, format='eps')
    fig.savefig('figs/ss_phase_diagram_%s.png' % tstamp, format='png')


def plot_phase_diagram(ag0, ae0, kappa):
    pass



