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


def get_expect(op, rho, man_tr=False):
    """
    Manually computes the expectation value of an operator
    given the density matrix, rho
    """
    
    # Compute the trace by qutip or manually
    if man_tr:
    
        print('Computing expectation values with manual trace ...')

        # Compute the new matrix-matrix product
        if (op.__class__ == qt.Qobj) and (rho.__class__ == qt.Qobj):
            
            A = rho * op
            return np.trace(A.data.todense())

        # Handle the vector output of rho
        elif ((rho.__class__ == np.ndarray) or (rho.__class__ == list)):

            print('Computing trace of list(rho) ...')
        
            # Convert the density matrix to a numpy array if needed
            rho = np.asarray(rho)            

            return np.array([np.trace(rho[i,:,:] @ op) \
                    for i in range(np.shape(rho)[0]) ])

    else:    
    
        # Compute the new matrix if qt.Qobjs
        if (op.__class__ == qt.Qobj) and (rho.__class__ == qt.Qobj):

            # Compute the expectation value with qutip
            return qt.expect(op, rho)

        elif ((rho.__class__ == np.ndarray) or (rho.__class__ == list)):

            print('Computing trace of list(rho) ...')

            # Compute the expectation value with qutip
            return np.array([qt.expect(op, qt.Qobj(rho[:,:,i])) \
                    for i in range(rho.shape[-1]) ])


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
    fig, ax = plt.subplots(3, 1, figsize=(8, 12), tight_layout=True)
    
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
    ax[2].set_ylabel(r'Drive Ampl ($g_x$)', fontsize=fsize)
    
    # Get and set the legends
    hdls2, legs2 = ax[2].get_legend_handles_labels()
    ax[2].legend(hdls2, legs2, loc='best')
    

    # Fix the layout
    plt.tight_layout()
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/traces_phase_diagram_%s.eps' % tstamp, format='eps')
    fig.savefig('figs/traces_phase_diagram_%s.png' % tstamp, format='png')


def plot_phase_ss(adata, tpts, nkappas, kappa, g, fext=''):
    """
    Plots the steady state values of <a>(t) in a quadrature plot
    """

    # Setup the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    
    # Set the figure axes
    fsize = 24
    set_axes_fonts(ax, fsize)
    
    # Reshape the data
    adata = adata.reshape([nkappas.size, tpts.size])

    # Take the steady state values, the last values
    # in the time domain simulation
    a0 = np.array([ad[-1] for ad in adata])

    # Plot the data for each kappa
    gk = g / kappa
    ax.plot(a0.real / gk, a0.imag / gk, 'x-')
        
    # Set the x, y axis labels
    ax.set_ylabel(r'$\Im\{\hat{a}\} / (g/\kappa)$', fontsize=fsize)
    ax.set_xlabel(r'$\Re\{\hat{a}\} / (g/\kappa)$ ', fontsize=fsize)
    
    # Get and set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best')
    
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/%s_ss_phase_diagram_%s.eps' % (fext, tstamp),
            format='eps')
    fig.savefig('figs/%s_ss_phase_diagram_%s.png' % (fext, tstamp),
            format='png')


def plot_io_a(tpts, a0, ae, g, kappa, fext=''):
    """
    Plot the input/output theory calculations of Im<a> vs. Re<a> for
    <sigma_z> = +/- hbar / 2
    
    Parameters:
    ----------

    tpts:       times that a0, ae are evaluated
    a0, ae:     values of <a> evaluated for sigma_z = -/+ hbar/2
    g:          coupling strength (chi in dispersive case)
    kappa:      cavity decay rate

    """
       
    # Setup the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    
    # Set the figure axes
    fsize = 24
    set_axes_fonts(ax, fsize)
     
    # Plot the real and imaginary parts of the ground and excited states
    gk = g / kappa

    # Plot discrete points
    ax.plot(a0.real / gk, a0.imag / gk, 'b--')
    ax.plot(ae.real / gk, ae.imag / gk, 'r--')

    # Plot the interpolated function
    tpts_interp = np.logspace(-2, 3, 6, base=2) / kappa
    tpts_interp = np.hstack((0, tpts_interp))
    a0re = np.interp(tpts_interp, tpts, a0.real)
    a0im = np.interp(tpts_interp, tpts, a0.imag)
    aere = np.interp(tpts_interp, tpts, ae.real)
    aeim = np.interp(tpts_interp, tpts, ae.imag)
    ax.plot(a0re / gk, a0im / gk, 'bo')
    ax.plot(aere / gk, aeim / gk, 'ro')

    # Set the x, y limits to agree with Didier et al.
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-0.2, 1.3])

    # Set the x, y axis labels
    ax.set_ylabel(r'$\Im\{\hat{a}\} / (g/\kappa)$', fontsize=fsize)
    ax.set_xlabel(r'$\Re\{\hat{a}\} / (g/\kappa)$ ', fontsize=fsize)
    
    # Get and set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best')
    
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/%s_ssfull_phase_diagram_%s.eps' % (fext, tstamp),
            format='eps')
    fig.savefig('figs/%s_ssfull_phase_diagram_%s.png' % (fext, tstamp),
            format='png')


def plot_io_a_full(tpts, a0_d, ae_d, a0_l, ae_l, g, chi, kappa, fext=''):
    """
    Plot the input/output theory calculations of Im<a> vs. Re<a> for
    <sigma_z> = +/- hbar / 2
    
    Parameters:
    ----------

    tpts:       times that a0, ae are evaluated
    a0_d, ae_d: values of <a> evaluated for sigma_z = -/+ hbar/2, dispersive
    a0_l, ae_l: values of <a> evaluated for sigma_z = -/+ hbar/2, longitudinal 
    g:          coupling strength (chi in dispersive case)
    kappa:      cavity decay rate

    """

    # Setup the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    
    # Set the figure axes
    fsize = 24
    set_axes_fonts(ax, fsize)
     
    # Plot the real and imaginary parts of the ground and excited states
    gk = g / kappa
    chik = chi / kappa

    # Plot discrete points
    ax.plot(a0_d.real / chik, a0_d.imag / chik, 'b--')
    ax.plot(ae_d.real / chik, ae_d.imag / chik, 'r--')
    ax.plot(a0_l.real / gk, a0_l.imag / gk, 'b--')
    ax.plot(ae_l.real / gk, ae_l.imag / gk, 'r--')

    # Plot the interpolated function
    tpts_interp = np.logspace(-2, 3, 6, base=2) / kappa
    tpts_interp = np.hstack((0, tpts_interp))
    
    # Interpolate the dispersive data
    a0d = np.interp(tpts_interp, tpts, a0_d)
    aed = np.interp(tpts_interp, tpts, ae_d)

    # Interpolate the longitudinal data
    a0l = np.interp(tpts_interp, tpts, a0_l)
    ael = np.interp(tpts_interp, tpts, ae_l)

    # Replot the circled points
    ax.plot(a0d.real / chik, a0d.imag / chik, 'bo')
    ax.plot(aed.real / chik, aed.imag / chik, 'ro')
    ax.plot(a0l.real / gk, a0l.imag / gk, 'bo')
    ax.plot(ael.real / gk, ael.imag / gk, 'ro')

    # Set the x, y limits to agree with Didier et al.
    ax.set_ylim([-1.25, 1.25])
    ax.set_xlim([-0.2, 1.25])

    # Set the x, y axis labels
    ax.set_ylabel(r'$\Im\{\hat{a}\} / (g/\kappa)$', fontsize=fsize)
    ax.set_xlabel(r'$\Re\{\hat{a}\} / (g/\kappa)$ ', fontsize=fsize)
    
    # Configure matplotlib to use color
    # ax.annotate(r'$\left|0\right>$', xy=(1.1, 0.2), fontsize=fsize)
    # ax.annotate(r'$\left|1\right>$', xy=(1.1, -0.2), fontsize=fsize)
    ax.text(0, 1.1, 'longitudinal', fontsize=fsize)
    ax.text(0.75, 1.1, 'dispersive', fontsize=fsize)
    ax.text(1.1, 0.2, r'$\left|0\right>$', fontsize=fsize, color='blue')
    ax.text(1.1, -0.2, r'$\left|1\right>$',  fontsize=fsize, color='red')
    
    # Get and set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best')
    
    
    # Save the result to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
    fig.savefig('figs/%s_ssfull_phase_diagram_%s.eps' % (fext, tstamp),
            format='eps')
    fig.savefig('figs/%s_ssfull_phase_diagram_%s.png' % (fext, tstamp),
            format='png')



def plot_phase_diagram(ag0, ae0, kappa):
    pass
