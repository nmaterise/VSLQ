#!/usr/bin/env python3
"""
Results for the VSLQ Readout, APS March Meeting
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle as pk
import time
from vslq import vslq_mops_readout
import post_proc_tools as ppt
import matrix_ops as mops
import re as regex
from prof_tools import tstamp

def parfor_expect(opp, vslq_obj, rho, pre0, pre1):
    """
    Parallel for kernel function to compute and write results to file
    """

    # Get the object with the name in the string list
    op = getattr(vslq_obj, opp)
    print('Writing <%d> to file from density matrix (%s)' % opp)
    
    # Compute the expectation value
    op_exp = mops.expect(op, rho)
    
    # Write the result to file
    op_fname = '%s/%s_%s.bin' % (pre0, pre1, opp)
    with open(op_fname, 'wb') as fid:
        pk.dump(op_exp, fid)
    fid.close()


def write_expect(rho_fname, Ns, Np, Nc, ops=['']):
    """
    Write the expectation values to file given the file name for the density
    matrix
    
    Parameters:
    ----------

    rho_fname:      filename of the density matrix pickle object
    Ns, Np, Nc:     number of levels in the shadow, primary, and cavity objects
    ops:            list of operators to compute as strings 

    """

    # Load the data from file
    with open(rho_fname, 'rb') as fid:
        rho = pk.load(fid)
    fid.close()
    
    # Get the filename prefix
    fsplit = rho_fname.split('/')
    prefix0 = fsplit[0]
    prefix1 = fsplit[-1]
    prefix2 = '.'.join(prefix1.split('.')[0:-1])

    # Create an instance of the class
    vslq_obj = vslq_mops_readout(Ns, Np, Nc, np.zeros(10), 0, 0,
                                0, 0, 0, 0, 0)

    # Loop over the operators in the list
    for opp in ops:
    
        # Get the object with the name in the string list
        op = getattr(vslq_obj, opp)
        print('Writing <%s> to file from density matrix (%s)'\
                % (opp, rho_fname))
    
        # Compute the expectation value
        op_exp = mops.expect(op, rho)
        
        # Write the result to file
        op_fname = '%s/%s_%s.bin' % (prefix0, prefix2, opp)
        with open(op_fname, 'wb') as fid:
            pk.dump(op_exp, fid)
        fid.close() 


def write_expect_driver(fname, args):
    """
    Run the above code with fixed inputs corresponding previous runs
    """

    # VSLQ Hilbert space
    Np, Ns, Nc = args

    # Operators to average
    # ops = ['ac', 'P13', 'P04', 'P24', 'Xl', 'Xr']
    ops = ['P%d' % i for i in range(0, Np)]
    ops.append('ac')
    ops.append('PXlXr') 
    # ops = ['ac']
    write_expect(fname, Ns, Np, Nc, ops)


def test_write_exp_drv(fname, Np, Ns, Nc):
    """
    Run code above on a single file, then all the files in parallel
    """
    
    # Use the arguments from the input parameters
    args = (Np, Ns, Nc)

    print('Computing expectation values for (%s) data ...' % fname)
    write_expect_driver('%s.bin' % fname, args)


def plot_ac(tpts, fnames, snames, fext, dfac=10):
    """
    Plot the cavity operator quadratures
    """

    # Get the data from the 0 and 1 states
    with open('%s_ac.bin' % fnames[0], 'rb') as fid:
        a0 = pk.load(fid)
    with open('%s_ac.bin' % fnames[1], 'rb') as fid:
        a1 = pk.load(fid)
    
    print('Decimation factor and total number of points: %d, %d' %
            ( dfac, int(tpts.size/dfac) ) )

    # Plot the results
    ppt.plot_expect_complex_ab(a0[0::dfac], a1[0::dfac], 
            'a_c', snames, fext, scale=0.5)


def test_plot_all_expect(sname, fprefix, tpts, Np,
                         use_logical=True, is_lossy=False):
    """
    Plot the expectation values vs. time
    """

    print('|%s> from (%s.bin)' % (sname, fprefix))

    ## Setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                        tight_layout=True)
    fsize = 24; lsize = 20; lw = 1.5;
    ppt.set_axes_fonts(ax, fsize)

    # Read in the expecatation values from file
    if use_logical:
        oplist = ['P%d' % i for i in range(0, Np)]
        oplist.append('PXlXr') 
        fstr = 'full_lossy' if is_lossy else 'full_lossless'
    else:
        oplist = ['P%d' % i for i in range(0, Np)]
        fstr = 'zoom_lossy' if is_lossy else 'zoom_lossless'

    # Iterate over all of the operators whose expectation
    # values we have calculated
    for op in oplist:
        fname = '%s_%s.bin' % (fprefix, op)
        with open(fname, 'rb') as fid:
            opdata = pk.load(fid)
        
        # Set the label for the legends
        plabel = regex.findall('\d+', op)[-1] if regex.findall('\d+', op) != []\
                else op
        if op == 'PXlXr':
            plabel = r'L_0'
        ax.plot(tpts, opdata.real, label=r'$\left|{%s}\right>$' % plabel,
                linewidth=lw)

    # Set the axes labels
    ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
    ax.set_ylabel('Expectation Value', fontsize=fsize)
    ax.set_title(r'Initial State ($\left|{%s}\right>$)' % sname, fontsize=fsize)

    # Set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best', fontsize=fsize)

    # Save the figure to file
    print('Writing figure to figs/logical_expect_leakage_%s_%s.pdf' \
            % (sname, fstr) )
    fig.savefig('figs/logical_expect_leakage_%s_%s.pdf' % (sname, fstr),
            format='pdf') 


def vslq_readout_dump_expect(tpts, Np, Ns, Nc, snames,
                             fnames, Ntout=25, plot_write='wp',
                             is_lossy=False):
    """
    Writes and plots the expectation values for all operators
    
    Parameters:
    ----------

    tpts:           time points for the results, should match input data 
    Np,Ns,Nc:       number of primary, shadow, and readout levels
    snames:         list of strings of the states to get the expectation values
                    of various projection operators; these names are used in
                    the plot legends and can contain latex
    fnames:         filenames of the density matrices for each state in snames
    Ntout:          number of times to decimate down to
    plot_write:     plot, write, or write and plot
    is_lossy:       use lossy terms in the VSLQ Hamiltonian or not 

    """

    # Start the timing clock
    ts = tstamp()

    # Compute a reasonable decimation factor to get ~20 points every time
    Nt = tpts.size
    Ntt = Nt if Nt <= Ntout else Ntout

    # Decimation factor between 1 and Ntt // Ntout
    dfac = Nt // Ntt

    # Construction the lossy file extension string
    fext_lossy = 'lossy' if is_lossy else 'lossless'

    # Skip straigh to the plotds
    if plot_write == 'p':
        print('\nPlotting expectation values ...\n')
        ts.set_timer('test_plot_all_expect')
        for ss, ff in zip(snames, fnames):
            test_plot_all_expect(ss, ff, tpts, Np, True, is_lossy)
            test_plot_all_expect(ss, ff, tpts, Np, False, is_lossy)
        ts.get_timer()
        
        # Plot the phase diagram for the readout cavity state
        print('\nGenerating phase diagrams ...\n')
        ts.set_timer('plot_ac')
        plot_ac(tpts, fnames, snames, 'L0L1_%s' % fext_lossy, dfac=dfac)
        ts.get_timer()

    # Compute, then plot
    elif plot_write == 'wp':
        # Get the expectation values files
        print('\nWriting expectation values ...\n')
        ts.set_timer('test_write_exp_drv')
        for ss, ff in zip(snames, fnames):
            test_write_exp_drv(ff, Np, Ns, Nc)
        ts.get_timer()

        # Plot the results
        print('\nPlotting expectation values ...\n')
        ts.set_timer('test_plot_all_expect')
        for ss, ff in zip(snames, fnames):
            test_plot_all_expect(ss, ff, tpts, Np, True, is_lossy)
            test_plot_all_expect(ss, ff, tpts, Np, False, is_lossy)
        ts.get_timer()

        # Plot the phase diagram for the readout cavity state
        print('\nGenerating phase diagrams ...\n')
        ts.set_timer('plot_ac')
        plot_ac(tpts, fnames, snames, 'L1L0', dfac=dfac)
        ts.get_timer()

    # Just compute
    elif plot_write == 'w':
        # Get the expectation values files
        print('\nWriting expectation values ...\n')
        ts.set_timer('test_write_exp_drv')
        for ss, ff in zip(snames, fnames):
            test_write_exp_drv(ff, Np, Ns, Nc)
        ts.get_timer()
        
    else:
        raise ValueError('(%s) is not a valid plot_write type' % plot_write)


if __name__ == '__main__':

    # Iterate over all the files and pass in labels
    snames = ['L_0', 'L_1', '\widetilde{L1}']
    fnames = ['data/rho_vslq_L0_0.11_us', 'data/rho_vslq_L1_0.11_us',
              'data/rho_vslq_l1L1_0.11_us']
    print('__main__ does nothing here ...')
