#!/usr/bin/env python3
"""
Tests of the VSLQ with loss terms included

"""

import numpy as np
import matplotlib.pyplot as plt
from vslq import vslq_mops
from scipy.optimize import curve_fit
import matrix_ops as mops
import prof_tools as pts
import multiprocessing as mp


def parfor_get_gamma(Np, Ns, W, delta, Om, gammap, gammas):
    """
    Parfor kernel function
    """

    # Set the time array
    ## Characteristic time, T1 of the primary qubits
    tmax = 2 / gammap
    
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 300

    ## Number of points as N = tmax / dt + 1
    Ntpts = max(int(np.ceil(tmax / dt0)) + 1, 10001)
    tpts = np.linspace(0, tmax, Ntpts)
    args_list = [1, tpts.max()/2, tpts.max()/12]

    # VSLQ Dynamics
    ## Initialize the VSLQ object
    my_vslq = vslq_mops(Ns, Np, tpts, W, delta, Om,
                 gammap, gammas)
    
    ## Run the dynamics, get the density matrix
    rho = my_vslq.run_dynamics(tpts, args_list, dt=tpts.max()/tpts.size)

    # Compute the expectation value and get T1L
    p0 = mops.expect(my_vslq.psi0, rho)

    ## Curve fit for T1L
    def fit_fun(x, a, b, c):
        return a * np.exp(-x*b) + c

    ## Return the covariance matrix and optimal values
    popt, pcov = curve_fit(fit_fun, tpts, np.abs(p0),
                           bounds=(0, [1.1, 1000/gammap, 1.1]))
    
    ## Extract the T1L time
    T1L = 1. / popt[1]
    dT1L = np.sqrt(np.abs(np.diag(pcov)[1]))
    print('Ntpts: %d, gammap: %g MHz, T1L: %g +/- %g us'\
            % (Ntpts, gammap, T1L, dT1L))

    return T1L


def parfor_gamma_wrapper(Np, Ns, W, delta, Om, gammap, gammas):
    """
    Wrapper on the parfor run with multiprocessing pool
    """
    
    # Create the pool and submit the job as a multithreaded dispatch
    pool = mp.Pool(2)

    ## Use the asynchronous star map to map the gammap's
    nsize = gammap.size
    res = pool.starmap_async(parfor_get_gamma, 
            zip([Np]*nsize, [Ns]*nsize, [W]*nsize, [delta]*nsize,
            [Om]*nsize, gammap, [gammas]*nsize))

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
    gammap_min = 5.; gammap_max = 80.;  gammas = 9.2;

    # Set the gammap list
    gammap = np.linspace(gammap_min, gammap_max, 16)

    # Call the parfor function
    # T1L = parfor_get_gamma(Np, Ns, W, delta, Om, gammap, gammas)

    # Call the parfor wrapper function
    print('Running gammap: %g MHz to %g MHz ...'\
            % (gammap.min(), gammap.max()))
    ts.set_timer('parfor_gamma_wrapper')
    T1L = parfor_gamma_wrapper(Np, Ns, W, delta, Om, gammap, gammas)
    ts.get_timer()
    
    # Plot the results
    T1p = 1. / gammap
    plt.plot(T1p, T1L/T1p)
    plt.xlabel(r'$T_{1P} (\mu\mathrm{s})$')
    plt.ylabel(r'$T_{1L}/T_{1P}$')
    plt.savefig('figs/t1L_t1p.pdf', format='pdf')


if __name__ == '__main__':
    
    # test_vslq_dissipative()
    test_parfor_get_gamma()
