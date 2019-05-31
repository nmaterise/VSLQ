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
import post_proc_tools as ppt
import multiprocessing as mp
import pickle as pk


def parfor_get_gamma(Np, Ns, W, delta, Om, gammap, gammas, plot=False):
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
    rho = my_vslq.run_dynamics(tpts, args_list, dt=tpts.max()/tpts.size)

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

    if plot:
        plt.plot(tpts, np.abs(p0), '*')
        plt.plot(tpts, fit_fun(tpts, *popt), '--', 
                label=r'%2f $e^{-t/%2f}$+%2f' \
                % (popt[0], (1./popt[1]), popt[2]))
        plt.legend(loc='best')
        plt.show()

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
    T1p_min = 5.; T1p_max = 80.;  gammas = 9.2;

    # Set the gammap list
    gammap = 1./np.linspace(T1p_min, T1p_max, 16)

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
    T1p = np.array([5, 15, 25, 35, 45, 55, 65, 75])
    T1p = np.array([15, 25, 35, 45, 55])

    # Call the plotting function
    # ppt.plot_gammap_sweep_exp(T1p)
    ppt.post_fit_exp(T1p)

if __name__ == '__main__':
    
    # test_vslq_dissipative()
    # test_parfor_get_gamma()
    plot_exp_data()


