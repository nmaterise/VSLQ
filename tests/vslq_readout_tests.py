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
                         readout_mode,
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
    readout_mode:       'single' / 'dual' for different modes of readout for the
                        left and right primary qubits
    
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
                 gammap, gammas, gl, gr, use_sparse=use_sparse,
                 readout_mode=readout_mode)
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
                         init_states, tpts, dt,
                         fext='', readout_mode='single'):
    """
    Creates threads for the multiprocessing module to distribute the work
    for different instances of the same job with different inputs

    Parameters:
    ----------

    Np, Ns, Nc:         number of primary, shadow, readout cavity levels
    W, delta, Om:       energy scales in the VSLQ
    gammap, gammas:     loss rates for the primary and shadow objects
    gl, gr:             coupling strengths between the readout cavity and the
                        Xl, Xr primary qubit operators
    init_states:        if list then parallel, else serial
    tpts, dt:           times to evaluate the density matrix and rk4 timestep
    fext:               filename extension
    readout_mode:       'single' / 'dual' readout of Xl and Xr

    Returns:
    -------

    """

    # Check if the code should run as serial or parallel
    ## Run the multiprocessing version
    if init_states.__class__ == list:

        # Create the multiprocessing pool
        pool = mp.Pool(2)
        nsize = len(init_states)
        res = pool.starmap_async(parfor_vslq_dynamics,
                zip( [Np]*nsize, [Ns]*nsize, [Nc]*nsize,[W]*nsize,
                     [delta]*nsize, [Om]*nsize, [gammap]*nsize, 
                     [gammas]*nsize, [gl]*nsize, [gr]*nsize,
                     init_states, [tpts]*nsize, [dt]*nsize, 
                    [fext]*nsize, [readout_mode]*nsize))

        # Close pool and join results
        print('Releasing multiprocessing pool ...')
        pool.close()
        pool.join()

    ## Run the serial version
    else:
        res = parfor_vslq_dynamics(Np, Ns, Nc, W,
                     delta, Om, gammap, 
                     gammas, gl, gr,
                     init_states, tpts, dt, fext, readout_mode)
        


def test_mp_vslq(init_state=None, plot_write='wp',
                 Np=3, is_lossy=False, readout_mode='single'):
    """
    Use both CPU's to divide and conquer the problem
    """

    # Convert the init_state to a string if length is one, otherwise just use
    # the list as is
    if (len(init_state) < 2) and (init_state != None):
        init_state = init_state[0]
    
    # Some example settings
    Ns = 2; Nc = Np;
    W = 2*np.pi*70; delta = 2*np.pi*700;
    # # T1p = 20 us, T1s = 109 ns

    # Readout strengths
    gl = W / 60; gr = gl;

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
    tmax = max(1./gl, 1./gr)

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    print('Using multiprocessing version ...')
    print('Running t=0 to %.2f us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / tpts.size

    ## Set the string and file extension names
    sname_dict = {'L0' : 'L_0', 'L1' : 'L_1', 'l1L1' : '\widetilde{L1}'}
    
    ## Use the input state or set them manually here
    if init_state.__class__ == str:
        snames = sname_dict[init_state] if init_state != None\
             else ['L_0', 'L_1', '\widetilde{L1}']
        fnames = 'data/rho_vslq_%s_%s_%.2f_us' % (fext, init_state, tmax) \
                    if init_state != None else \
                 ['data/rho_vslq_%s_L0_%.2f_us'   % (fext, tmax),
                  'data/rho_vslq_%s_L1_%.2f_us'   % (fext, tmax),
                  'data/rho_vslq_%s_l1L1_%.2f_us' % (fext, tmax)]
    else:
        snames = [sname_dict[ss] for ss in init_state]
        fnames = ['data/rho_vslq_%s_%s_%.2f_us' % (fext, ss, tmax) \
                 for ss in init_state]

    # Bypass the calculation and just plot
    ## Only plot
    if plot_write == 'p':
        # Call the post processing code to plot the results
        Ntout = 50
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, Ntout=Ntout,
                                 plot_write=plot_write, 
                                 is_lossy=is_lossy)

    ## Compute and plot 
    elif plot_write == 'w' or plot_write == 'wp':
        # Call the multiprocess wrapper or the serial version
        init_states = init_state if init_state != None \
                                 else ['L0', 'L1', 'l1L1']
        parfor_vslq_wrapper(Np, Ns, Nc, W, delta,
                            Om, gammap, gammas,
                            gl, gr,
                            init_states, tpts, dt, fext=fext)

        # Call the post processing code to plot the results
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, 
                                 Ntout=Ntout, plot_write=plot_write,
                                 is_lossy=is_lossy)


def test_mp_vslq_parser():
    """
    Handles command line argument parsing for the test_mp_vslq() function
    """
    
    # Get the parser
    p = ap.ArgumentParser(description='Pseudo parallel vslq readout dynamics')
    
    # Add arguments
    p.add_argument('-s', '--init_state', dest='init_state',
                   type=str, nargs='+',
                   help='Initial state vector [l1L1, l1L0, L0, L1]',
                  default='L0')
    p.add_argument('-p', '--plot_write', dest='plot_write', type=str,
                   help='Plot the data after the data has been generated\n'\
                        '(w,p,wp) -- write, plot, write and plot',
                    default='wp')
    p.add_argument('-n', '--Np', dest='Np', type=int,
                   help='Number of levels in the primary and readout modes',
                    default=5)
    p.add_argument('-l', '--is_lossy', dest='is_lossy', type=int,
                    help='Turn on / off the dissipation and Hsp term',
                    default=0)
    p.add_argument('-r', '--readout_mode', dest='readout_mode', type=str,
                   help='Readout mode (single / dual)', default='single')
    
    # Get the arguments
    args = p.parse_args()
    print('Commandline arguments before function call: {}'.format(args))

    # Call the function as desired
    test_mp_vslq(args.init_state,
                 args.plot_write,
                 args.Np,
                 args.is_lossy,
                 args.readout_mode)


if __name__ == '__main__':

    # Use the above parser to handle argument parsing
    test_mp_vslq_parser()
