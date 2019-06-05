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


def write_expect_driver(fname):
    """
    Run the above code with fixed inputs corresponding previous runs
    """

    # VSLQ Hilbert space 
    Np = 5; Ns = 2; Nc = 5;

    # Operators to average
    # ops = ['ac', 'P13', 'P04', 'P24', 'Xl', 'Xr']
    ops = ['P%d' % i for i in range(0, Np)]
    ops.append('ac')
    ops.append('PXlXr') 
    # ops = ['ac']
    write_expect(fname, Ns, Np, Nc, ops)


def test_write_exp_drv(fname):
    """
    Run code above on a single file, then all the files in parallel
    """
    
    # Test above code with fname = 'data/rho_vslq_L0_1.3942us.bin' 
    # fnames = ['data/rho_vslq_L0_1.3942us.bin','data/rho_vslq_L1_1.3942us.bin' ]
    # fnames = ['data/rho_vslq_l1L0_1.9_us.bin','data/rho_vslq_l1L1_1.9_us.bin' ]
    print('Computing expectation values for (%s) data ...' % fname)
    write_expect_driver('%s.bin' % fname)


def plot_ac(tpts, fnames, snames, fext):
    """
    Plot the cavity operator quadratures
    """

    # Get the data from the 0 and 1 states
    with open(fnames[0], 'rb') as fid:
        a0 = pk.load(fid)
    with open(fnames[1], 'rb') as fid:
        a1 = pk.load(fid)
    
    # Plot the results
    ppt.plot_expect_complex_ab(a0[0::100], a1[0::100], 'a_c', snames, fext,
            scale=0.5)


def test_plot_ac(ttt):
    """
    Plot the cavity field for different logical states
    """
    # VSLQ Hilbert space 
    Np = 5; Ns = 2; Nc = 5;

    # Use the time points from the original simulation
    W = 35*2*np.pi; delta = 350*2*np.pi; Om = 13.52;
    gammap = 0; gammas = 0; #9.2;

    # Set the time array
    ## Characteristic time of the shadow resonators
    TOm = 2*np.pi / Om
    tmax = 3*TOm
    
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 20

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    tpts = np.linspace(0, tmax, Ntpts)

    # First plot the logical state, then the photon loss states
    fnames = ['data/rho_vslq_L0_%d_us_ac.bin' % ttt,
              'data/rho_vslq_L1_%d_us_ac.bin' % ttt ]
    snames = ['L_0', 'L_1']
    plot_ac(tpts, fnames, snames, 'L0L1')

    # # First plot the logical state, then the photon loss states
    # fnames = ['data/rho_vslq_l1L0_%d_us_ac.bin' % ttt,
    #           'data/rho_vslq_l1L1_%df_us_ac.bin' % ttt]
    # snames = ['\widetilde{L}_0', '\widetilde{L}_1']
    # plot_ac(tpts, fnames, snames, 'l1L0l1L1')


def test_plot_all_expect(sname, fprefix, use_logical=True):
    """
    Plot the expectation values vs. time
    """

    # VSLQ Hilbert space 
    Np = 5; Ns = 2; Nc = 5;

    # Some example settings
    Np = 5; Ns = 2; Nc = 5;
    W = 2*np.pi*70; delta = 2*np.pi*700; Om = 5.5;
    # T1p = 20 us, T1s = 109 ns
    gammap = 0.05; gammas = 9.2;

    # Set the time array
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 20

    ## Decay time of transmons
    tmax = (0.05 / gammap)

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    print('Using multiprocessing version ...')
    print('Running t=0 to %.2g us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / tpts.size

    ## Setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                        tight_layout=True)
    fsize = 24; lsize = 20; lw = 1.5;
    ppt.set_axes_fonts(ax, fsize)

    # Read in the expecatation values from file
    if use_logical:
        oplist = ['P%d' % i for i in range(0, Np)]
        oplist.append('PXlXr') 
        fstr = 'full'
    else:
        oplist = ['P%d' % i for i in range(0, Np)]
        fstr = 'zoom'

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
    fig.savefig('figs/logical_expect_leakage_%s_%s.eps' % (sname, fstr),
            format='eps') 
    fig.savefig('figs/logical_expect_leakage_%s_%s.png' % (sname, fstr),
            format='png') 


if __name__ == '__main__':
    
    # Iterate over all the files and pass in labels
    # snames = ['\widetilde{L}_0', '\widetilde{L}_1', 
    #           'L_0', 'L_1']
    snames = ['L_0', 'L_1']
    snames = ['L_1']
    # snames = ['L_0L_0', 'L_1L_1']
    # fprefix = ['data/rho_vslq_L0_1_us', 'data/rho_vslq_L1_1_us',
    #            'data/rho_vslq_l1L1_1_us']#,
    fprefix = ['data/rho_vslq_L1_1_us']

    # for ss, ff in zip(snames, fprefix):
    #    # test_write_exp_drv(ff)
    #    test_plot_all_expect(ss, ff, True)
    #    test_plot_all_expect(ss, ff, False)
    
    # Run the above code on the following test case
    test_plot_ac(1)

