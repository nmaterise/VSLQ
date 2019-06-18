#!/usr/bin/env python3
"""
Tests of the VSLQ with loss terms included

"""
# Add the VSLQ path 
from test_utils import set_path
set_path()

import numpy as np
import matplotlib.pyplot as plt

from vslq import vslq_mops
from scipy.optimize import curve_fit
import matrix_ops as mops
import super_ops as sops
import prof_tools as pts
import post_proc_tools as ppt
import multiprocessing as mp
import pickle as pk


def parfor_get_gamma(Np, Ns, W, delta, Om, 
                     gammap, gammas, use_sparse=False):
    """
    Parfor kernel function
    """

    # Set the time array
    ## Characteristic time, T1 of the primary qubits
    tmax = 2 / gammap
    
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 20

    ## Number of points as N = tmax / dt + 1
    Ntpts = max(int(np.ceil(tmax / dt0)) + 1, 5001)
    tpts = np.linspace(0, tmax, Ntpts)
    args_list = [1, tpts.max()/2, tpts.max()/12]

    print('Ntpts: %d, ' % Ntpts)

    # VSLQ Dynamics
    ## Initialize the VSLQ object
    my_vslq = vslq_mops(Ns, Np, tpts, W, delta, Om,
                 gammap, gammas)
    
    ## Run the dynamics, get the density matrix
    rho = my_vslq.run_dynamics(tpts, args_list,
                               dt=tpts.max()/tpts.size,
                               use_sparse=use_sparse)

    # Compute the expectation value and get T1L
    p0 = mops.expect(my_vslq.pL, rho)
    with open('data/p0_gamma_%d.bin' % int(1./gammap), 'wb') as fid:
        pk.dump(np.vstack((tpts, p0)), fid)
    fid.close()

    ## Curve fit for T1L
    def fit_fun(x, a, b, c):
        return a * np.exp(-x*b) + c

    ## Return the covariance matrix and optimal values
    popt, pcov = curve_fit(fit_fun, tpts, np.abs(p0), maxfev=10000,
                            bounds=([0.1, 0, -1], [1, 1000, 1]))
    
    ## Extract the T1L time
    T1L = 1. / popt[1]
    dT1L = np.sqrt(np.abs(np.diag(pcov)[1]))
    print('gammap: %g MHz, T1L: %g +/- %g us, T1L/T1p: %g'\
            % (Ntpts, gammap, T1L, dT1L, (T1L*gammap)))

    return T1L


def parfor_gamma_wrapper(Np, Ns, W, delta, Om,
                         gammap, gammas, use_sparse=False):
    """
    Wrapper on the parfor run with multiprocessing pool
    """
    
    # Create the pool and submit the job as a multithreaded dispatch
    pool = mp.Pool(2)

    ## Use the asynchronous star map to map the gammap's
    nsize = gammap.size
    res = pool.starmap_async(parfor_get_gamma, 
            zip([Np]*nsize, [Ns]*nsize, [W]*nsize, [delta]*nsize,
            [Om]*nsize, gammap, [gammas]*nsize, [use_sparse]*nsize))

    ## Close pool and join results
    pool.close()
    pool.join()

    ## Get the results
    T1L = np.array(res.get())

    return T1L


def test_parfor_get_gamma():
    """
    Test the get_gamma function before using parfor
    """

    # Initialize the profiling class object
    ts = pts.tstamp()

    # VSLQ Hilbert space 
    Np = 3; Ns = 2;

    # Use the time points from the original simulation
    W = 2*np.pi*35; delta = 2*np.pi*350; Om = 5.5;
    T1p_min = 5.; T1p_max = 80.;  gammas = 9.2;

    # Set the gammap list
    # gammap = 1./np.linspace(T1p_min, T1p_max, 16)
    use_sparse = True
    sdict = {0: 'Dense', 1 : 'Sparse'}
    gammap = 1./np.array([5., 80.])

    # Call the parfor wrapper function
    print('Running gammap (%s): %g MHz to %g MHz ...'\
            % (sdict[use_sparse], gammap.min(), gammap.max()))
    ts.set_timer('parfor_gamma_wrapper')
    T1L = parfor_gamma_wrapper(Np, Ns, W, delta,
                               Om, gammap, gammas, use_sparse)
    ts.get_timer()
    
    # Plot the results
    T1p = 1. / gammap
    
    ## Plot T1p vs. 
    plt.figure(1)
    plt.plot(T1p, T1L/T1p)
    plt.xlabel(r'$T_{1P} (\mu\mathrm{s})$')
    plt.ylabel(r'$T_{1L}/T_{1P}$')
    plt.savefig('figs/t1L_t1p.pdf', format='pdf')


def test_parfor_get_gamma_sparse():
    """
    Test the get_gamma function before using parfor and sparse matrix operations
    """

    # Initialize the profiling class object
    ts = pts.tstamp()

    # VSLQ Hilbert space 
    Np = 3; Ns = 2;

    # Use the time points from the original simulation
    W = 2*np.pi*35; delta = 2*np.pi*350; Om = 5.5;
    T1p_min = 5.; T1p_max = 80.;  gammas = 9.2;

    # Set the gammap list
    # gammap = 1./np.linspace(T1p_min, T1p_max, 16)
    gammap = 1./np.array([5., 80.])

    # Call the parfor wrapper function
    print('Running gammap: %g MHz to %g MHz ...'\
            % (gammap.min(), gammap.max()))
    ts.set_timer('parfor_gamma_wrapper')
    T1L = parfor_gamma_wrapper(Np, Ns, W, delta, Om, gammap, gammas)
    ts.get_timer()
    
    # Plot the results
    T1p = 1. / gammap
    
    ## Plot T1p vs. 
    plt.figure(1)
    plt.plot(T1p, T1L/T1p)
    plt.xlabel(r'$T_{1P} (\mu\mathrm{s})$')
    plt.ylabel(r'$T_{1L}/T_{1P}$')
    plt.savefig('figs/t1L_t1p.pdf', format='pdf')


def plot_exp_data():
    """
    Plot the aggregated results from above using post_proc_tools
    """

    # Use the same set of gammap's as above
    gammap = np.linspace(5., 80., 16)
    # T1p = np.array([5, 15, 25, 35, 45, 55, 65, 75])
    T1p = np.array([5, 80])
    # T1p = np.array([15, 25, 35, 45, 55, 65])

    # Call the plotting function
    # ppt.plot_gammap_sweep_exp(T1p)
    ppt.post_fit_exp(T1p)


def plot_vslq_cmaps():
    """
    Generates a color map of the VSLQ Hamiltonian and Liouvillian  
    """

    # Parameters used for VLSQ optimal operation
    Np = 3; Ns = 2;
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
    tpts = np.linspace(0, tmax, Ntpts)

    # Instantiate the VSLQ object
    print('Setting up VSLQ Hamiltonian ...')
    my_vslq = vslq_mops(Ns, Np, tpts, W, delta, Om,
                 gammap, gammas)
    my_vslq.set_H([], [])

    # Plot the Hamiltonian as a color map
    print('Plotting Hamiltonian color map ...')
    ppt.plot_hamiltonian_map(my_vslq.H, fext='vslq_sparse', use_sparse=True)
    ppt.plot_hamiltonian_map(my_vslq.H, fext='vslq', use_sparse=False)

    # Get the Liouvillian
    print('Computing Liouvillian ...')
    L = sops.liouvillian(my_vslq.H, my_vslq.cops)

    # Plot the Liouvillian as a color map
    print('Plotting Liouvillian color map ...')
    ppt.plot_liouvillian_map(L, fext='vslq_sparse', use_sparse=True)
    ppt.plot_liouvillian_map(np.abs(L), fext='vslq', use_sparse=False)



if __name__ == '__main__':
    
    # test_vslq_dissipative()
    test_parfor_get_gamma()
    # plot_exp_data()
    # plot_vslq_cmaps()

