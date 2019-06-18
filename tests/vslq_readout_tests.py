#!/usr/bin/env python3
"""
Tests of the readout + VSLQ 

"""

vslq_path = '/home/nmaterise/mines/research/VSLQ'
import sys
if vslq_path not in sys.path:
    sys.path.append(vslq_path)
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
                         init_state, tpts, dt, fid=None, use_sparse=False):
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
    fid:                file pointer to the HDF5 output file
    use_sparse:         convert the Hamiltonian and all other operators
                        to sparse matrices

    Returns:
    -------

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
    if fid != None:

        ## Write the result to file
        ## Replace pickle dump with hdf5 write
        fid = hdf.File('data/rho_vslq_%.2g_us.bin' \
                % (tmax))
        ## Add a data set to the file
        fid.create_dataset(data=rho1, name=init_state, dtype=rho1.dtype)
        fid.close()

    ## Defaults back to pickle
    else:
        ts.set_timer('|%s> pickle write' % init_state)
        with open('data/rho_vslq_%s_%.2g_us.bin' \
                % (init_state, tmax), 'wb') as fid:
            pk.dump(rho1, fid)
        fid.close()
        ts.get_timer()

    print('|%s> result written to file.' % init_state)


def parfor_vslq_wrapper(Np, Ns, Nc, W, delta,
                         Om, gammap, gammas,
                         gl, gr,
                         init_states, tpts, dt):
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
                         init_states, [tpts]*nsize, [dt]*nsize))

    # Close pool and join results
    print('Releasing multiprocessing pool ...')
    pool.close()
    pool.join()


def test_vslq_dynamics():
    """
    Tests the dynamics of the VSLQ with no drive and initial states
    of logical 0 or 1
    """

    # Some example settings
    Np = 4; Ns = 2; Nc = 2;
    W = 70.0*np.pi; delta = 700.0*np.pi; Om = 5.5;
    gammap = 0; gammas = 0; #9.2;

    # Set the time array
    tpts = np.linspace(0, 2*np.pi / W, 1001)
    dt = tpts.max() / (10 * tpts.size)

    # Create an instance of the vslq class
    my_vslq = vslq_mops(Ns, Np, tpts, W, delta, Om, gammap, gammas)
    my_vslq.set_init_state(logical_state='L1')
    args = [1, tpts.max()/2, tpts.max()/12]
    rho_out = my_vslq.run_dynamics(tpts, args, dt=dt)

    # Get the expectation values for Xl and Xr
    Xl = mops.expect(my_vslq.Xl, rho_out)
    Xr = mops.expect(my_vslq.Xr, rho_out)

    # Plot the results
    plt.plot(tpts, Xl.real, label=r'$\Re\langle\widetilde{X}_l\rangle$')
    plt.plot(tpts, Xl.imag, label=r'$\Im\langle\widetilde{X}_l\rangle$')
    plt.plot(tpts, Xr.real, label=r'$\Re\langle\widetilde{X}_r\rangle$')
    plt.plot(tpts, Xr.imag, label=r'$\Im\langle\widetilde{X}_r\rangle$')
    plt.legend(loc='best')
    plt.xlabel(r'Time [$\mu$s]')


def test_vslq_readout_dynamics():
    """
    Tests the dynamics of the VSLQ with no drive and initial states
    of logical 0 or 1
    """

    # Some example settings
    Np = 3; Ns = 2; Nc = Np;
    W = 35*2*np.pi; delta = 350*2*np.pi; Om = 0; #13.52;
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
    print('Running t=0 to %.2g us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / (10 * tpts.size)

    # Readout strengths
    gl = W / 50; gr = gl;

    # Create an instance of the vslq class
    args = [1, tpts.max()/2, tpts.max()/12]

    ## Solve for | L0 > logical state
    my_vslq_0 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'l1L0'
    my_vslq_0.set_init_state(logical_state=lstate)
    rho0 = my_vslq_0.run_dynamics(tpts, args, dt=dt)

    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho0, fid)
    fid.close()
    print('|1> |L0> result written to file.')

    ## Run for | L1 > state
    my_vslq_1 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'l1L1'
    my_vslq_1.set_init_state(logical_state=lstate)
    rho1 = my_vslq_1.run_dynamics(tpts, args, dt=dt)

    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho1, fid)
    fid.close()
    print('|1> |L1> result written to file.')

    ## Solve for | L0 > logical state
    my_vslq_0 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'L0'
    my_vslq_0.set_init_state(logical_state=lstate)
    rho0 = my_vslq_0.run_dynamics(tpts, args, dt=dt)

    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho0, fid)
    fid.close()
    print('|L0> result written to file.')

    ## Run for | L1 > state
    my_vslq_1 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'L1'
    my_vslq_1.set_init_state(logical_state=lstate)
    rho1 = my_vslq_1.run_dynamics(tpts, args, dt=dt)

    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho1, fid)
    fid.close()
    print('|L1> result written to file.')


def test_mp_vslq(plot_write=False):
    """
    Use both CPU's to divide and conquer the problem
    """

    # Some example settings
    Np = 3; Ns = 2; Nc = Np;
    W = 2*np.pi*70; delta = 2*np.pi*700; # Om = 5.5;
    # # T1p = 20 us, T1s = 109 ns
    # gammap = 0.05; gammas = 9.2;

    # Readout strengths
    gl = W / 50; gr = gl;

    # Turn off dissipation
    gammap = 0.; gammas = 0.;

    # Turn off the coupling term
    Om = 0.;

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
    print('Running t=0 to %.2g us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / tpts.size

    ## Set the string and file extension names
    snames = ['L_0', 'L_1', '\widetilde{L1}']
    fnames = ['data/rho_vslq_L0_%.2f_us' % tmax,
              'data/rho_vslq_L1_%.2f_us' % tmax,
              'data/rho_vslq_l1L1_%.2f_us' % tmax]

    # Bypass the calculation and just plot
    ## Only plot
    if plot_write == 'p':
        # Call the post processing code to plot the results
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, Ntout=25,
                                 plot_write=plot_write)

    ## Compute and plot 
    elif plot_write == 'w' or plot_write == 'wp':
        # Call the multiprocess wrapper
        init_states = ['L0', 'L1', 'l1L1']
        parfor_vslq_wrapper(Np, Ns, Nc, W, delta,
                             Om, gammap, gammas,
                             gl, gr,
                             init_states, tpts, dt)

        # Call the post processing code to plot the results
        print('\nPost processing dynamics ...\n')
        vslq_readout_dump_expect(tpts, Np, Ns, Nc,
                                 snames, fnames, 
                                 Ntout=25, plot_write=plot_write)


def test_proj():
    """
    Initialize a vslq object to check if the new projectors are set correectly
    """

    # Some example settings
    Np = 5; Ns = 2; Nc = 5;
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
    dt = tpts.max() / (10 * tpts.size)

    # Readout strengths
    gl = W / 50; gr = gl;
    my_vslq_1 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)


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
    
    # Get the arguments
    args = p.parse_args()

    # Call the function as desired
    test_mp_vslq(args.plot_write)


if __name__ == '__main__':

    # Test the dynamics of the vslq in different logical states
    # test_vslq_dynamics()
    # test_vslq_readout_dynamics()
    # test_mp_vslq(True)
    # test_proj()

    # Use the above parser to handle argument parsing
    test_mp_vslq_parser()
