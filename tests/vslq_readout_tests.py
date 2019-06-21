#!/usr/bin/env python3
"""
Tests of the readout + VSLQ 

"""

# Add the VSLQ path 
from test_utils import set_path
set_path()

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import h5py as hdf
import multiprocessing as mp
from vslq import vslq_mops, vslq_mops_readout
import prof_tools as pts
from vslq_readout_results import vslq_readout_dump_expect
import argparse as ap


def parfor_vslq_dynamics(Np, Ns, Nc, W, delta,
                         Om, gammap, gammas,
                         gl, gr,
                         init_state, tpts, dt,
                         fext,
                         use_hdf5=False,
                         use_sparse=False):
    """
    Parallel for loop kernel function for computing vslq dynamics

    Parameters:
    ----------

    Np, Ns, Nc:         number of primary, shadow, readout cavity levels
    W, delta, Om:       energy scales in the VSLQ
    gl, gr:             coupling strengths between the readout cavity and the
                        Xl, Xr primary qubit operators
    gammap, gammas:     loss rates for the primary and shadow objects
    tpts, dt:           times to evaluate the density matrix and rk4 timestep
    fext:               filename extensions
    
    kwargs:
    ------

    use_hdf5:           file pointer to the HDF5 output file
    use_sparse:         convert the Hamiltonian and all other operators
                        to sparse matrices
    """

    # Set default args (deprecated external drive functionality)
    args = [1, tpts.max()/2, tpts.max()/12]
    tmax = tpts.max()

    # Initialize the profiling class object
    ts = pts.tstamp()

    ## Run for | L1 > state
    my_vslq = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr, use_sparse=use_sparse)
    my_vslq.set_init_state(logical_state=init_state)
    print('Running dynamics for |%s> ...' % init_state)

    ts.set_timer('|%s> dynamics' % init_state)
    rho1 = my_vslq.run_dynamics(tpts, args, dt=dt, use_sparse=use_sparse)
    ts.get_timer()

    ## Handles adding for hdf5 without breaking multiprocess
    if use_hdf5:

        ## Write the result to file
        ## Replace pickle dump with hdf5 write
        ts.set_timer('|%s> hdf5 write' % init_state)
        fid = hdf.File('data/rho_vslq_%s_%s_%.2f_us.bin' \
                % (fext, init_state, tmax))
        ## Add a data set to the file
        fid.create_dataset(data=rho1, name=init_state, dtype=rho1.dtype)
        fid.close()
        ts.get_timer()

    ## Defaults back to pickle
    else:
        ts.set_timer('|%s> pickle write' % init_state)
        with open('data/rho_vslq_%s_%s_%.2f_us.bin' \
                % (fext, init_state, tmax), 'wb') as fid:
            pk.dump(rho1, fid)
        fid.close()
        ts.get_timer()

    print('|%s> result written to file.' % init_state)


def parfor_vslq_wrapper(Np, Ns, Nc, W, delta,
                         Om, gammap, gammas,
                         gl, gr,
                         init_states, tpts, dt, fext=''):
    """
    Creates threads for the multiprocessing module to distribute the work
    for different instances of the same job with different inputs

    Parameters:
    ----------

    Np, Ns, Nc:         number of primary, shadow, readout cavity levels
    W, delta, Om:       energy scales in the VSLQ
    gl, gr:             coupling strengths between the readout cavity and the
                        Xl, Xr primary qubit operators
    gammap, gammas:     loss rates for the primary and shadow objects
    tpts, dt:           times to evaluate the density matrix and rk4 timestep
    fext:               filename extension

    Returns:
    -------

    """

    # Create the multiprocessing pool
    pool = mp.Pool(2)
    nsize = len(init_states)
    res = pool.starmap_async(parfor_vslq_dynamics,
            zip( [Np]*nsize, [Ns]*nsize, [Nc]*nsize,[W]*nsize, [delta]*nsize,
                         [Om]*nsize, [gammap]*nsize, [gammas]*nsize,
                         [gl]*nsize, [gr]*nsize,
                         init_states, [tpts]*nsize, [dt]*nsize, [fext]*nsize))

    # Close pool and join results
    print('Releasing multiprocessing pool ...')
    pool.close()
    pool.join()


def test_mp_vslq(plot_write='wp', Np=3, is_lossy=False):
    """
    Use both CPU's to divide and conquer the problem
    """

    # Some example settings
    Ns = 2; Nc = Np;
    W = 2*np.pi*70; delta = 2*np.pi*700;
    # # T1p = 20 us, T1s = 109 ns

    # Readout strengths
    gl = W / 50; gr = gl;

    # Turn off dissipation
    if is_lossy:
        print('Turning on dissipation ...')
        gammap = 0.05; gammas = 9.2; Om = 5.5;
        fext = 'lossy'
    else:
        print('Turning off dissipation ...')
        fext = 'lossless'
        gammap = 0.; gammas = 0.; Om = 0.;

    # Set the time array
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 20

    ## Decay time of transmons
    # tmax = (0.05 / gammap)
    tmax = max(1./gl, 1./gr)

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    print('Using multiprocessing version ...')
    print('Running t=0 to %.2f us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / tpts.size

    ## Set the string and file extension names
    snames = ['L_0', 'L_1', '\widetilde{L1}']
    fnames = ['data/rho_vslq_%s_L0_%.2f_us'   % (fext, tmax),
              'data/rho_vslq_%s_L1_%.2f_us'   % (fext, tmax),
              'data/rho_vslq_%s_l1L1_%.2f_us' % (fext, tmax)]

    # Bypass the calculation and just plot
    ## Only plot
    if plot_write == 'p':
        # Call the post processing code to plot the results
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, Ntout=25,
                                 plot_write=plot_write, is_lossy=is_lossy)

    ## Compute and plot 
    elif plot_write == 'w' or plot_write == 'wp':
        # Call the multiprocess wrapper
        init_states = ['L0', 'L1', 'l1L1']
        parfor_vslq_wrapper(Np, Ns, Nc, W, delta,
                             Om, gammap, gammas,
                             gl, gr,
                             init_states, tpts, dt, fext=fext)

        # Call the post processing code to plot the results
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, 
                                 Ntout=25, plot_write=plot_write,
                                 is_lossy=is_lossy)


def test_mp_vslq_parser():
    """
    Handles command line argument parsing for the test_mp_vslq() function
    """
    
    # Get the parser
    p = ap.ArgumentParser(description='Pseudo parallel vslq readout dynamics')
    
    # Add arguments
    p.add_argument('-p', '--plot_write', dest='plot_write', type=str,
                   help='Plot the data after the data has been generated\n'\
                        '(w,p,wp) -- write, plot, write and plot')
    p.add_argument('-n', '--Np', dest='Np', type=int,
                   help='Number of levels in the primary and readout modes')
    p.add_argument('-l', '--is_lossy', dest='is_lossy', type=bool,
                    help='Turn on / off the dissipation and Hsp term')
    
    # Get the arguments
    args = p.parse_args()

    # Call the function as desired
    test_mp_vslq(args.plot_write, args.Np, args.is_lossy)


if __name__ == '__main__':

    # Use the above parser to handle argument parsing
    test_mp_vslq_parser()
