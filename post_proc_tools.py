#!/usr/bin/env python
"""
Post processing tools for plotting, viewing states, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
# import qutip as qt
import datetime
import pickle as pk


def set_axes_fonts(ax, fsize):
    """
    Set axes font sizes because it should be abstracted away
    """
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(fsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(fsize)


# def get_wigner(psi, file_ext=None):
#     """
#     Compute, return, and save the Wigner function
#     """
# 
#     # Use the same size grid to compute the Wigner function each time
#     xvec = np.linspace(-5, 5, 500)
#     W = qt.wigner(psi.ptrace(1), xvec, xvec)
# 
#     # Write results to binary files
#     if file_ext is not None:
#         W.tofile('%s_wigner.bin' % file_ext)
#         xvec.tofile('%s_wigner_xvec.bin' % file_ext)
# 
#     return xvec, W


def get_expect(op, rho):
    """
    Manually computes the expectation value of an operator
    given the density matrix, rho
    """

    # Convert the density matrix to a numpy array if needed
    rho = np.asarray(rho)            

    return np.array([np.trace(rho[i,:,:] @ op) \
            for i in range(np.shape(rho)[0]) ])


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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    fsize = 24; tsize = 26;
    set_axes_fonts(ax, fsize)
    ax.plot(tpts, np.abs(op_avg))
    
    # Set the axes labels
    xstr = 'Time (%s)' % tscale
    ystr = r'$\langle{%s}\rangle$' % op_name

    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)
    
    # Save the figure to file
    if file_ext is not None:
        fig.savefig('figs/expect_%s.eps' % file_ext, format='eps') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/expect_%s.eps' % tstamp, format='eps') 

def plot_expect_phase_ab(tpts, op_a, op_b, 
                            opname, snames, 
                            fext=None, scale=1):
    """
    Generates the quadrature plot (Im<op> vs. Re<op>) in states |a>, |b>
    
    Parameters:
    ----------

    op_a, op_b:     two operators' expectation values as functions of time 
                    for states a, b 
    opnames:        operator name 
    snames:         names of states corresponding to op_a, op_b
    scale:          amount to divide real and imaginary components by

    
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
            tight_layout=True)
    fsize = 24; tsize = 26; lw = 1.5; lsize=20
    set_axes_fonts(ax, fsize)
    ax.plot(tpts, np.unwrap(np.angle(op_a)),
            'ro-', linewidth=lw,
            label=r'$\left|%s\right>$' % snames[0])
    ax.plot(tpts, np.unwrap(np.angle(op_b)),
            'bo-', linewidth=lw,
            label=r'$\left|%s\right>$' % snames[1])

    # Set the x/y limits
    amax = op_a.max(); bmax = op_b.max()
    ymax = np.abs(amax) if amax > 0 else np.nabs(amx)
    ylim = [-1.2, 1.2]
    xlim = ylim 
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Set the axes labels
    xstr = r'$\Re\langle{%s}\rangle$' % opname
    ystr = r'$\Im\langle{%s}\rangle$' % opname
    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)

    # Add annotation arrows
    # ax.annotate(r'$t$', xy=(0.1, 0.5), xytext=(0.1, 0.1),
    #         arrowprops=dict(arrowstyle='->'))
    # ax.annotate(r'$t$', xy=(0.1, -0.5), xytext=(0.1, -0.1),
    #         arrowprops=dict(arrowstyle='->'))

    # Set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best', fontsize=lsize)
    
    # Save the figure to file
    if fext is not None:
        fig.savefig('figs/%s_expect_phase_%s.eps' \
                % (opname, fext), format='eps') 
        fig.savefig('figs/%s_expect_phase_%s.png' \
                % (opname, fext), format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/%s_expect_phase_%s_%s.eps' % (opname, fext, tstamp),
                format='eps') 
        fig.savefig('figs/%s_expect_phase_%s_%s.png' % (opname, fext, tstamp),
                format='png') 

def plot_expect_complex_ab(op_a, op_b, 
                            opname, snames, 
                            fext=None, scale=1):
    """
    Generates the quadrature plot (Im<op> vs. Re<op>) in states |a>, |b>
    
    Parameters:
    ----------

    op_a, op_b:     two operators' expectation values as functions of time 
                    for states a, b 
    opnames:        operator name 
    snames:         names of states corresponding to op_a, op_b
    scale:          amount to divide real and imaginary components by

    
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
            tight_layout=True)
    fsize = 24; tsize = 26; lw = 1.5; lsize=20
    set_axes_fonts(ax, fsize)
    ax.plot(op_a.real/scale, op_a.imag/scale,
            'ro-', linewidth=lw,
            label=r'$\left|{%s}\right>$' % snames[0])
    ax.plot(op_b.real/scale, op_b.imag/scale,
            'bo-', linewidth=lw,
            label=r'$\left|{%s}\right>$' % snames[1])

    # Set the x/y limits
    amax = op_a.max(); bmax = op_b.max()
    ymax = np.abs(amax) if amax > 0 else np.abs(amax)
    ylim = [-1.2, 1.2]
    xlim = ylim 
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Set the axes labels
    xstr = r'$\Re\langle{%s}\rangle$' % opname
    ystr = r'$\Im\langle{%s}\rangle$' % opname
    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)

    # Add annotation arrows
    # ax.annotate(r'$t$', xy=(0.1, 0.5), xytext=(0.1, 0.1),
    #         arrowprops=dict(arrowstyle='->'))
    # ax.annotate(r'$t$', xy=(0.1, -0.5), xytext=(0.1, -0.1),
    #         arrowprops=dict(arrowstyle='->'))

    # Set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best', fontsize=lsize)
    
    # Save the figure to file
    if fext is not None:
        fig.savefig('figs/%s_expect_%s.eps' % (opname, fext), format='eps') 
        fig.savefig('figs/%s_expect_%s.png' % (opname, fext), format='png') 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/%s_expect_%s_%s.eps' % (opname, fext, tstamp),
                format='eps') 
        fig.savefig('figs/%s_expect_%s_%s.png' % (opname, fext, tstamp),
                format='png') 
    

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


def plot_phase_ss(adata, tpts, nkappas, kappa, g, fext='', use_tseries=False):
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
    if use_tseries:
        a0 = adata
    else:
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


def plot_io_a_full(tpts, a0_d, ae_d, a0_l, ae_l,
                   g, chi, kappa, fext='', use_interp=True):
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
    gk = a0_l.imag.max() # g / kappa
    chik = a0_d.imag.max() # chi / kappa

    # Plot discrete points
    ax.plot(a0_d.real / chik, a0_d.imag / chik, 'b--')
    ax.plot(ae_d.real / chik, ae_d.imag / chik, 'r--')
    ax.plot(a0_l.real / gk, a0_l.imag / gk, 'b--')
    ax.plot(ae_l.real / gk, ae_l.imag / gk, 'r--')

    ## Interpolate if requested for the points
    if use_interp: 

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
    ax.set_ylabel(r'$\Im\{a\} / (g/\kappa)$', fontsize=fsize)
    ax.set_xlabel(r'$\Re\{a\} / (g/\kappa)$ ', fontsize=fsize)
    
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
