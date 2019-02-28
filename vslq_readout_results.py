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

    # Code to send a thread pool to attack the expectation values
    """
    pool = mp.Pool(2)
    nsize = len(ops)
    res = pool.starmap_async(parfor_expect,
            zip(ops, [my_vslq_obj]*nsize, [rho]*nsize,
                [prefix0]*nsize, [prefix2]*nsize
                ))
    """

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
    ops = ['ac', 'P13', 'P04', 'P24', 'Xl', 'Xr']
    # ops = ['ac']
    write_expect(fname, Ns, Np, Nc, ops)


def test_write_exp_drv():
    """
    Run code above on a single file, then all the files in parallel
    """
    
    # Test above code with fname = 'data/rho_vslq_L0_1.3942us.bin' 
    # fnames = ['data/rho_vslq_L0_1.3942us.bin','data/rho_vslq_L1_1.3942us.bin' ]
    fnames = ['data/rho_vslq_l1L0_1.4_us.bin','data/rho_vslq_l1L1_1.4_us.bin' ]
    for fname in fnames:
        print('Computing expectation values for (%s) data ...' % fname)
        write_expect_driver(fname)


def plot_ac(fnames, snames, fext):
    """
    Plot the cavity operator quadratures
    """

    # Get the data from the 0 and 1 states
    with open(fnames[0], 'rb') as fid:
        a0 = pk.load(fid)
    with open(fnames[1], 'rb') as fid:
        a1 = pk.load(fid)
    
    # Plot the results
    ppt.plot_expect_complex_ab(a0, a1, 'a_c', snames, fext)


def test_plot_ac():
    """
    Plot the cavity field for different logical states
    """

    # First plot the logical state, then the photon loss states
    fnames = ['data/rho_vslq_L0_1.3942us_ac.bin',
              'data/rho_vslq_L1_1.3942us_ac.bin']
    snames = ['L_0', 'L1']
    plot_ac(fnames, snames, 'L0L1')

    # First plot the logical state, then the photon loss states
    fnames = ['data/rho_vslq_l1L0_1.4_us_ac.bin',
              'data/rho_vslq_l1L1_1.4_us_ac.bin']
    snames = ['1_l, L_0', '1_l, L1']
    plot_ac(fnames, snames, 'l1L0l1L1')


def test_plot_all_expect(sname, fprefix):
    """
    Plot the expectation values vs. time
    """

    # Use the time points from the original simulation
    W = 35*2*np.pi; delta = 350*2*np.pi; Om = 13.52;
    gammap = 0; gammas = 0; #9.2;

    # Set the time array
    ## Characteristic time of the shadow resonators
    TOm = 2*np.pi / Om
    tmax = 3*TOm
    
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 10

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    tpts = np.linspace(0, tmax, Ntpts)

    ## Setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                        tight_layout=True)
    fsize = 24; lsize = 20; lw = 1.5;
    ppt.set_axes_fonts(ax, fsize)

    # Read in the expecatation values from file
    oplist = ['P13', 'P04', 'P24', 'Xl', 'Xr']
    for op in oplist:
        fname = '%s_%s.bin' % (fprefix, op)
        with open(fname, 'rb') as fid:
            opdata = pk.load(fid)
        ax.plot(tpts, opdata.real, label=r'$\langle{%s}\rangle$' % op,
                linewidth=lw)

    # Set the axes labels
    ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
    ax.set_ylabel('Expectation Value', fontsize=fsize)
    ax.set_title(r'Initial State ($\left|%s\right>$)' % sname, fontsize=fsize)

    # Set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best', fontsize=lsize)

    # Save the figure to file
    fig.savefig('figs/logical_expect_leakage_%s.eps' % sname , format='eps') 
    fig.savefig('figs/logical_expect_leakage_%s.png' % sname , format='png') 



if __name__ == '__main__':
    
    # Run the above code on the following test case
    # test_write_exp_drv()
    test_plot_ac()

    # Iterate over all the files and pass in labels
    snames = ['1_lL_0', '1_lL_1', 'L_0L_0',  'L_1L_1']
    fprefix = ['data/rho_vslq_l1L0_1.4_us', 'data/rho_vslq_l1L1_1.4_us',
               'data/rho_vslq_L0_1.3942us',   'data/rho_vslq_L1_1.3942us']
    for ss, ff in zip(snames, fprefix):
        test_plot_all_expect(ss, ff)
    
