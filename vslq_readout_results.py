#!/usr/bin/env python3
"""
Results for the VSLQ Readout, APS March Meeting
"""

import os
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
import h5py as hdf


def write_expect(rho_fname, Ns, Np, Nc, 
                 ops=[''], readout_mode='single',
                 use_sparse=True, use_hdf5=True):
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
    ## Load from HDF5
    if use_hdf5:
        
        ## Check that the file was written
        if os.path.exists(rho_fname):
            print('Reading from filename (%s) ...' % rho_fname)
            fid = hdf.File(name=rho_fname, mode='r+')
            
            ## Check that the data was actually written to file
            if 'rho' in fid.keys():
                rho = fid['rho'][()]
            else:
                raise NameError('key (rho) not found.')
        else:
            raise NameError('Filename (%s) does not exist, cannot open file.' \
                        % rho_fname)
    
    ## Load from pickle
    else:
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
                                0, 0, 0, 0, 0, 
                                use_sparse=use_sparse,
                                readout_mode=readout_mode)

    # Loop over the operators in the list
    for opp in ops:
    
        # Get the object with the name in the string list
        op = getattr(vslq_obj, opp)
        print('Writing <%s> to file from (%s)'\
                % (opp, rho_fname))
    
        # Compute the expectation value
        op_exp = mops.expect(op, rho)
        
        # Write the result to file
        op_fname = '%s/%s_%s.bin' % (prefix0, prefix2, opp)

        # Check for HDF5 option
        if use_hdf5:
            
            ## If the name exists, write to it, else create a new dataset
            if opp in fid.keys():
                ## Overwrite the existing data
                fid[opp][()] = op_exp
            else:
                ## Create a new data set and write to it
                fid.create_dataset(name=opp, data=op_exp)
        
        ## Write directly to separate pickle files
        else:
            with open(op_fname, 'wb') as fid:
                pk.dump(op_exp, fid)
            fid.close() 

    # Close the file
    if use_hdf5:
        fid.close()


def write_expect_driver(fname, args):
    """
    Run the above code with fixed inputs corresponding previous runs
    """

    # VSLQ Hilbert space
    Np, Ns, Nc, readout_mode, use_hdf5, use_sparse = args

    # Operators to average
    # ops = ['ac', 'P13', 'P04', 'P24', 'Xl', 'Xr']
    ops = ['P%d' % i for i in range(0, Np)]
    if readout_mode == 'single':
        print('Writing single mode data ...')
        ops.append('ac')
    elif readout_mode == 'dual':
        print('Writing dual mode data ...')
        ops.append('acl')
        ops.append('acr')
    ops.append('PXlXr') 
    write_expect(fname, Ns, Np, Nc, ops,
                 readout_mode=readout_mode,
                 use_hdf5=use_hdf5, use_sparse=use_sparse)


def test_write_exp_drv(fname, Np, Ns, Nc,
                       readout_mode='single',
                       use_hdf5=True,
                       use_sparse=True):
    """
    Run code above on a single file, then all the files in parallel
    """
    
    # Use the arguments from the input parameters
    args = (Np, Ns, Nc, readout_mode, use_hdf5, use_sparse)

    print('Computing expectation values for (%s) data ...' % fname)
    print('Using readout mode (%s) ...' % readout_mode)

    # Change the file extension from bin to hdf5 to denote the difference
    # between the pickle and h5py outputs
    if use_hdf5:
        write_expect_driver('%s.hdf5' % fname, args)
    else:
        write_expect_driver('%s.bin' % fname, args)


def plot_ac(tpts, fnames, snames, fext, dfac=10,
            readout_mode='single', use_hdf5=True):
    """
    Plot the cavity operator quadratures
    """

    # Check for None decimation factor
    if dfac == None:
        ddfac = 1
    else:
        ddfac = dfac

    # Check for the readout mode
    print('Decimation factor and total number of points: %d, %d' %
                ( ddfac, int(tpts.size/ddfac) ) )
    print('Using readout_mode (%s) ...' % readout_mode)

    # Open the hdf5 file if 

    ## Single mode settings
    if readout_mode == 'single':

        # Use the hdf5 readout approach
        if use_hdf5:
            ## Get the L0 logical state data
            fname0 = '%s.hdf5' % fnames[0]
            print('Reading file (%s) ...' % fname0)
            fid0 = hdf.File(fname0, 'r')
            a0 = fid0['ac'][()]
            fid0.close()

            ## Get the L1 logical state data
            fname1 = '%s.hdf5' % fnames[1]
            print('Reading file (%s) ...' % fname1)
            fid1 = hdf.File(fname1, 'r')
            a1 = fid1['ac'][()]
            fid1.close()

        # Use pickle to get the data
        else:
            # Get the data from the 0 and 1 states
            with open('%s_ac.bin' % fnames[0], 'rb') as fid:
                a0 = pk.load(fid)
            with open('%s_ac.bin' % fnames[1], 'rb') as fid:
                a1 = pk.load(fid)

        # Plot the results
        ppt.plot_expect_complex_ab(a0[0::ddfac], a1[0::ddfac], 
                'a_c', snames, fext, scale=1.0)

    ## Dual mode settings
    if readout_mode == 'dual':

        # Use HDFf to get the data
        if use_hdf5:
            ## Get the L0 logical state data
            fid0 = hdf.File('%s.hdf5' % fnames[0], 'r')
            al0 = fid0['acl'][()]
            ar0 = fid0['acr'][()]
            fid0.close()

            ## Get the L1 logical state data
            fid1 = hdf.File('%s.hdf5' % fnames[1], 'r')
            al1 = fid1['acl'][()]
            ar1 = fid1['acr'][()]
            fid1.close()
    
        # Use pickle to get the data
        else:
            # Get the data from the 0 and 1 states
            with open('%s_acl.bin' % fnames[0], 'rb') as fid:
                al0 = pk.load(fid)
            with open('%s_acl.bin' % fnames[1], 'rb') as fid:
                al1 = pk.load(fid)
            with open('%s_acr.bin' % fnames[0], 'rb') as fid:
                ar0 = pk.load(fid)
            with open('%s_acr.bin' % fnames[1], 'rb') as fid:
                ar1 = pk.load(fid)

        # Plot the results
        ppt.plot_expect_complex_ab(al0[0::ddfac], al1[0::ddfac], 
                'a_{cl}', snames, fext, scale=1.0)
        ppt.plot_expect_complex_ab(ar0[0::ddfac], ar1[0::ddfac], 
                'a_{cr}', snames, fext, scale=1.0)


def test_plot_all_expect_sep(sname, fprefix, tpts, Np,
                         use_logical=True, is_lossy=False,
                         use_hdf5=True, readout_mode='single'):
    """
    Plot the expectation values vs. time, generate separate files for each
    of the state occupations
    """

    # Get the correct file name using the file extension passed in
    if use_hdf5:
        print('|%s> from (%s.hdf5)' % (sname, fprefix))
    else:
        print('|%s> from (%s.bin)' % (sname, fprefix))

    # Read in the expecatation values from file
    if use_logical:
        oplist = ['P%d' % i for i in range(0, Np)]
        oplist.append('PXlXr') 
        fstr = 'full_lossy_%s' % readout_mode if is_lossy \
               else 'full_lossless_%s' % readout_mode
    else:
        oplist = ['P%d' % i for i in range(0, Np)]
        fstr = 'zoom_lossy_%s' % readout_mode if is_lossy \
               else 'zoom_lossless_%s' % readout_mode

    # Iterate over all of the operators whose expectation
    # values we have calculated
    ## Use h5py
    if use_hdf5:
        ## Open the hdf5 file
        fid = hdf.File('%s.hdf5' % fprefix, 'r')
        for op in oplist:

            ## Setup the plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6),
                                tight_layout=True)
            fsize = 24; lsize = 20; lw = 1.5;
            ppt.set_axes_fonts(ax, fsize)

            # Get the data from the data set in fid
            opdata = fid[op][()]
            
            # Set the label for the legends
            plabel = regex.findall('\d+', op)[-1] if \
                     regex.findall('\d+', op) != []\
                     else op
            if op == 'PXlXr':
                plabel = r'L_0'
            ax.plot(tpts, opdata.real, linewidth=lw)

            # Set the axes labels
            ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
            ax.set_ylabel(r'Population of the $\left|{%s}\right>$ state' \
                    % plabel, fontsize=fsize)
            ax.set_title(r'Initial State ($\left|{%s}\right>$)' \
                    % sname, fontsize=fsize)

            # Save the figure to file
            print('Writing figure to figs/logical_expect_leakage_%s_%s_%s.pdf' \
                    % (sname, fstr, plabel) )
            fig.savefig('figs/logical_expect_leakage_%s_%s_%s.pdf' \
                    % (sname, fstr, plabel),
                    format='pdf') 
            plt.close()

    ## Use pickle
    else:
        for op in oplist:
            fname = '%s_%s.bin' % (fprefix, op)
            with open(fname, 'rb') as fid:
                opdata = pk.load(fid)
            
            # Set the label for the legends
            plabel = regex.findall('\d+', op)[-1] if \
                     regex.findall('\d+', op) != []\
                     else op
            if op == 'PXlXr':
                plabel = r'L_0'
            ax.plot(tpts, opdata.real, linewidth=lw)

            # Set the axes labels
            ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
            ax.set_ylabel(r'Population of the $\left|{%s}\right>$ state' \
                    % plabel, fontsize=fsize)
            ax.set_title(r'Initial State ($\left|{%s}\right>$)' \
                    % sname, fontsize=fsize)

            # Save the figure to file
            print('Writing figure to figs/logical_expect_leakage_%s_%s_%s.pdf' \
                    % (sname, fstr, plabel) )
            fig.savefig('figs/logical_expect_leakage_%s_%s_%s.pdf' \
                    % (sname, fstr, plabel),
                    format='pdf') 
            plt.close()


def test_plot_all_expect(sname, fprefix, tpts, Np,
                         use_logical=True, is_lossy=False,
                         use_hdf5=True, readout_mode='single'):
    """
    Plot the expectation values vs. time
    """

    if use_hdf5:
        print('|%s> from (%s.hdf5)' % (sname, fprefix))
    else:
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
        fstr = 'full_lossy_%s' % readout_mode if is_lossy \
               else 'full_lossless_%s' % readout_mode
    else:
        oplist = ['P%d' % i for i in range(0, Np)]
        fstr = 'zoom_lossy_%s' % readout_mode if is_lossy \
               else 'zoom_lossless_%s' % readout_mode

    # Iterate over all of the operators whose expectation
    # values we have calculated
    ## Use h5py
    if use_hdf5:
        ## Open the hdf5 file
        fid = hdf.File('%s.hdf5' % fprefix, 'r')
        for op in oplist:
            # Get the data from the data set in fid
            opdata = fid[op][()]
            
            # Set the label for the legends
            plabel = regex.findall('\d+', op)[-1] if \
                     regex.findall('\d+', op) != []\
                     else op
            if op == 'PXlXr':
                plabel = r'L_0'
            ax.plot(tpts, opdata.real, label=r'$\left|{%s}\right>$' % plabel,
                    linewidth=lw)
    
    ## Use pickle
    else:
        for op in oplist:
            fname = '%s_%s.bin' % (fprefix, op)
            with open(fname, 'rb') as fid:
                opdata = pk.load(fid)
            
            # Set the label for the legends
            plabel = regex.findall('\d+', op)[-1] if \
                     regex.findall('\d+', op) != []\
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
                             is_lossy=False,
                             readout_mode='single',
                             use_sparse=True):
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
    readout_mode:   'single' or 'dual' mode readout scheme
    use_sparse:     use sparse matrices to carry out Lindblad evolution

    """

    # Start the timing clock
    ts = tstamp()

    # Compute a reasonable decimation factor to get ~20 points every time
    Nt = tpts.size
    Ntt = Nt if Nt <= Ntout else Ntout

    # Decimation factor between 1 and Ntt // Ntout
    dfac = 1 # Nt // Ntt

    # Construction the lossy file extension string
    fext_lossy = 'lossy_%s' % readout_mode if is_lossy \
                 else 'lossless_%s' % readout_mode

    # Skip straigh to the plotds
    if plot_write == 'p':
        print('\nPlotting expectation values ...\n')
        ts.set_timer('test_plot_all_expect_sep')
        if snames.__class__ == list:
            for ss, ff in zip(snames, fnames):
                test_plot_all_expect_sep(ss, ff, tpts, Np, use_logical=True,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
                test_plot_all_expect_sep(ss, ff, tpts, Np, use_logical=False,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
        else:
            test_plot_all_expect_sep(snames, fnames, tpts, Np, use_logical=True,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
            test_plot_all_expect_sep(snames, fnames, tpts, Np, use_logical=False,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
        ts.get_timer()
        
        # Plot the phase diagram for the readout cavity state
        if snames.__class__ == list:
            print('\nGenerating phase diagrams ...\n')
            ts.set_timer('plot_ac')
            plot_ac(tpts, fnames, snames, 'L0L1_%s' % fext_lossy,
                    dfac=dfac, readout_mode=readout_mode)
            ts.get_timer()

    # Compute, then plot
    elif plot_write == 'wp' or plot_write == 'pwp':
        # Get the expectation values files
        if snames.__class__ == list:
            print('\nWriting expectation values ...\n')
            ts.set_timer('test_write_exp_drv')
            for ss, ff in zip(snames, fnames):
                test_write_exp_drv(ff, Np, Ns, Nc,
                                   readout_mode=readout_mode,
                                   use_sparse=use_sparse)
            ts.get_timer()

            # Plot the results
            print('\nPlotting expectation values ...\n')
            ts.set_timer('test_plot_all_expect_sep')
            for ss, ff in zip(snames, fnames):
                test_plot_all_expect_sep(ss, ff, tpts, Np, use_logical=False,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
                test_plot_all_expect_sep(ss, ff, tpts, Np, use_logical=True,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
            ts.get_timer()
        else:
            print('\nWriting expectation values ...\n')
            ts.set_timer('test_write_exp_drv')
            test_write_exp_drv(fnames, Np, Ns, Nc,
                               readout_mode=readout_mode,
                               use_sparse=use_sparse)
            ts.get_timer()
            print('\nPlotting expectation values ...\n')
            ts.set_timer('test_plot_all_expect_sep')
            test_plot_all_expect_sep(snames, fnames, tpts, Np, use_logical=True,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
            test_plot_all_expect_sep(snames, fnames, tpts, Np, use_logical=False,
                        is_lossy=is_lossy, use_hdf5=True,
                        readout_mode=readout_mode)
            ts.get_timer()

        # Plot the phase diagram for the readout cavity state
        if snames.__class__ == list:
            print('\nGenerating phase diagrams ...\n')
            print('>> Readout mode: (%s)')
            ts.set_timer('plot_ac')
            plot_ac(tpts, fnames, snames, 'L0L1',
                    dfac=dfac, readout_mode=readout_mode)
            ts.get_timer()

    # Just compute
    elif plot_write == 'w':
        # Get the expectation values files
        print('\nWriting expectation values ...\n')
        ts.set_timer('test_write_exp_drv')
        if snames.__class__ == list:
            for ss, ff in zip(snames, fnames):
                test_write_exp_drv(ff, Np, Ns, Nc,
                                   readout_mode=readout_mode,
                                   use_sparse=use_sparse)
        else:
            test_write_exp_drv(fnames, Np, Ns, Nc,
                               readout_mode=readout_mode,
                               use_sparse=use_sparse)

        ts.get_timer()
        
    else:
        raise ValueError('(%s) is not a valid plot_write type' % plot_write)


if __name__ == '__main__':

    # Iterate over all the files and pass in labels
    snames = ['L_0', 'L_1', '\widetilde{L}_1']
    fnames = ['data/rho_vslq_L0_0.11_us', 'data/rho_vslq_L1_0.11_us',
              'data/rho_vslq_l1L1_0.11_us']
    print('__main__ does nothing here ...')
