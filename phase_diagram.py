#!/usr/bin/env python
"""
Generate the data needed for a phase diagram
depicting the spread of the |0> and |1> state
as projected on the cavity state
"""

import numpy as np
import qutip as qt
import multiprocessing as mp
from transmon import transmon_disp, transmon_disp_mops, transmon_long, \
     transmon_long_mops
import post_proc_tools as ppt
import matrix_ops as mops
import drive_tools as dts
import datetime


def get_transmon_pdiag(tmon, tpts, kappa, nkappas,
                       gamma1=0, fext='', write_ttraces=False, g=None):
    """
    Wraps the transmon_disp class object to compute <a> with different drive
    durations on the cavity for measurement
    
    Parameters:
    ----------

    tmon:               instance of the transmon_disp class
    tpts:               array of times to compute the density matrix on
    kappa:              cavity linewidth, sets the cavity decay rate
    nkappas:            list of multiples of kappa to compute <a>
    gamma1:             1/T1 for the qubit, zero by default for now
    fext:               added text to the filename
    write_ttraces:      save the time traces for <a>(t)
    g:                  coupling strength between qubit and cavity

    """

    # Save the results for a_avg
    a_avg = [] # np.zeros(nkappas.size, dtype=np.complex128)

    # Save the time traces
    if write_ttraces:
        ttraces = [] # np.zeros(nkappas.size * tpts.size, dtype=np.complex128)

    # Compute the time spacing for the tpts
    dt = tpts.max() / tpts.size

    # Get the traces data
    if write_ttraces:

        # Get the full <a> (t) time traces
        ttraces = qt.parallel.parfor(parfor_update_traces,
                [tpts]*nkappas.size,
                [tmon]*nkappas.size,
                nkappas, 
                [kappa]*nkappas.size,
                [g]*nkappas.size)

        # Convert the results to numpy arrays
        ttraces = np.asarray(ttraces)

        # Write the real and imaginary components separately
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
        freal = 'data/areal_traces_%s_%s.bin' % (fext, tstamp)
        fimag = 'data/aimag_traces_%s_%s.bin' % (fext, tstamp)

        filenames = [freal, fimag]
        ttraces.real.tofile(freal)
        ttraces.imag.tofile(fimag)

        return ttraces, filenames


    else:

        # Compute the averages, only return steady state values
        a_avg = qt.parallel.parfor(parfor_update,
                [tpts]*nkappas.size,
                [tmon]*nkappas.size,
                nkappas, 
                [kappa]*nkappas.size)

        # Change the data to a numpy array
        a_avg = np.asarray(a_avg, dtype=np.complex128)
        a_avg = a_avg.flatten()

        # Write the results to file
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
        freal = 'data/areal_%s_%s.bin' % (fext, tstamp)
        fimag = 'data/aimag_%s_%s.bin' % (fext, tstamp)
        filenames = [freal, fimag]
        a_avg.real.tofile(freal) 
        a_avg.imag.tofile(fimag)

        
        return a_avg, filenames


def parfor_update(tpts, tmon, nk, kappa):
    """
    TODO: fix this function to handle steady state identification
    """

    # Setup the transmon inputs
    t0 = 3*nk / (2*kappa)
    sig = nk / (2*kappa);
    
    print('Running measurement from %g to %g ns ...'%(t0-sig/2, t0+sig/2))

    args = tmon.get_cy_window_dict(t0, sig, 0, 0.01)
    res  = tmon.run_dynamics(tpts, args)
    aavg = tmon.get_a_expect(res)


    return aavg[-1]


def parfor_update_traces(tpts, tmon, nk, kappa, g=None):
    """
    Returns the time traces for the <a> (t) measurements 
    """

    # Setup the transmon inputs
    # t0 = tpts.max() - 3*nk / (2*kappa)
    t0 = 3*nk / (2*kappa)
    sig = nk / (2*kappa)
    
    print('Running measurement from %g to %g ns ...'%(t0-sig/2, t0+sig/2))

    args = tmon.get_cy_window_dict(t0, sig, 0, 0.01)
    if g is not None:
        args['A'] = g
    res  = tmon.run_dynamics(tpts, args)
    aavg = tmon.get_a_expect(res)

    # Store the results
    return aavg


def get_transmon_pdiag_mops(tmon, tpts, kappa, nkappas,
                       gamma1=0, fext='', write_ttraces=False, g=None, dt=None):
    """
    Wraps the transmon_disp_mops class object to compute <a> with different drive
    durations on the cavity for measurement
    
    Parameters:
    ----------

    tmon:               instance of the transmon_disp class
    tpts:               array of times to compute the density matrix on
    kappa:              cavity linewidth, sets the cavity decay rate
    nkappas:            list of multiples of kappa to compute <a>
    gamma1:             1/T1 for the qubit, zero by default for now
    fext:               added text to the filename
    write_ttraces:      save the time traces for <a>(t)
    g:                  coupling strength between qubit and cavity
    dt:                 Runge-Kutta time step

    """

    # Save the results for a_avg
    a_avg = [] # np.zeros(nkappas.size, dtype=np.complex128)

    # Save the time traces
    if write_ttraces:
        ttraces = []

    # Get the traces data
    if write_ttraces:

        # Get the full <a> (t) time traces
        # Create a pool first
        nsize = nkappas.size
        print('Running with (%d) kappa values ...' % nsize)
        nthreads = mp.cpu_count() // 4
        pool = mp.Pool(2)
        res = pool.starmap_async(parfor_update_traces_mops,
                zip([tpts]*nsize, [tmon]*nsize, 
                    nkappas, [kappa]*nsize, [g]*nsize, [dt]*nsize))

        # Close pool and join results
        pool.close()
        pool.join()

        # Convert the results to numpy arrays
        ttraces = np.asarray(res.get())

        # Write the real and imaginary components separately
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
        freal = 'data/areal_traces_%s_%s.bin' % (fext, tstamp)
        fimag = 'data/aimag_traces_%s_%s.bin' % (fext, tstamp)

        filenames = [freal, fimag]
        ttraces.real.tofile(freal)
        ttraces.imag.tofile(fimag)

        return ttraces, filenames

    else:

        # Create a pool of processes to consume the inputs
        nsize = nkappas.size
        nthreads = mp.cpu_count() // 4 
        pool = mp.Pool(2)
        res = pool.starmap(parfor_update_mops,
                zip([tpts]*nsize, [tmon]*nsize, 
                    nkappas, [kappa]*nsize, [g]*nsize, [dt]*nsize))

        # Close pool and join results
        pool.close()
        pool.join()

        # Change the data to a numpy array
        a_avg = np.asarray(res.get(), dtype=np.complex128)
        a_avg = a_avg.flatten()

        # Write the results to file
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
        freal = 'data/areal_%s_%s.bin' % (fext, tstamp)
        fimag = 'data/aimag_%s_%s.bin' % (fext, tstamp)
        filenames = [freal, fimag]
        a_avg.real.tofile(freal) 
        a_avg.imag.tofile(fimag)

        
        return a_avg, filenames


def parfor_update_traces_mops(tpts, tmon, nk, kappa, g=None, dt=None):
    """
    Returns the time traces for the <a> (t) measurements 
    """

    # Setup the transmon inputs
    # t0 = tpts.max() - 3*nk / (2*kappa)
    t0 = 3*nk / (2*kappa)
    sig = nk / (2*kappa)
    
    print('Running trace measurement from %g to %g ns ...'\
            % (t0-sig/2, t0+sig/2))

    args = [1, t0, sig]
    if g is not None:
        args[0] = g
    if dt is not None:
        res = tmon.run_dynamics(tpts, args, dt=dt)
    else:
        res = tmon.run_dynamics(tpts, args, dt=tpts.max()/(10*tpts.size))
    
    # Compute the expectation value and return the time series
    aavg = tmon.get_a_expect(res)

    # Store the results
    return aavg    


def parfor_update_mops(tpts, tmon, nk, kappa, g=None, dt=None):
    """
    TODO: fix this function to handle steady state identification
    """

    # Setup the transmon inputs
    t0 = 3*nk / (2*kappa)
    sig = nk / (2*kappa);
    
    print('Running single measurement from %g to %g ns ...' \
            % (t0-sig/2, t0+sig/2))

    args = [1, t0, sig]
    if g is not None:
        args[0] = g
    if dt is not None:
        res  = tmon.run_dynamics(tpts, args, dt=dt)
    else:
        res  = tmon.run_dynamics(tpts, args)
    
    # Compute the expectation value and return the last entry
    aavg = tmon.get_a_expect(res)

    return aavg[-1]


def test_get_transmon_pdiag():
    """
    Test the phase diagram code with a simple transmon coupled to a cavity
    dispersively
    """
    # 16 levels in the cavity, 3 in the transmon
    Nc = 16; Nq = 3;

    # 250 MHz anharmonicity, 50 MHz self-Kerr
    alpha = 0.250*2*np.pi;      self_kerr = 0.05*2*np.pi;
    chi = np.sqrt(2*alpha*self_kerr)

    # Compute the coupling factor g from chi = g^2 / delta
    # Use a delta of 2.5 GHz
    wq = 5*2*np.pi; wc = 6*2*np.pi;
    
    # Set the cavity linewidth and the transmon T1
    # T1 = 40 us, kappa = 125 kHz
    T1 = 40e3; gamma1 = 1./T1; kappa = 0.000125;
    T1 = 0.; gamma1 = 0.;
    kappa = 0.1; # chi / 2.#0.1;
    alpha = 0.250*2*np.pi;
    self_kerr = 2*kappa**2/alpha;
    Delta = 1;
    g = np.sqrt(Delta * kappa / 2.)
    
    # Set the initial state
    # a = qt.tensor(qt.destroy(Nq), qt.qeye(Nc))
    psi_g0 = qt.tensor(qt.basis(Nq, 0), qt.basis(Nc, 0))
    psi_e0 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nc, 0))
    psi0 = (psi_g0 - psi_e0).unit()

    # Set the time of the simulation in ns
    tpts = np.linspace(0, 10/kappa, 3001)

    # Run the phase diagram code here
    nkappas = np.linspace(0.5, 3, 6)
    
    # Create the drive signals here
    # t0 = np.array([tpts.max() - 3*nk / (2*kappa) \
    #         for nk in nkappas])
    t0 = np.array([3*nk / (2*kappa) \
            for nk in nkappas])
    sig = np.array([nk / (2*kappa) for nk in nkappas])

    # Drives used to generate the phase diagram
    drvs = np.array([np.sin(wc*tpts)*np.exp(-((tpts - t00)**2)/(2*sigg**2))\
            for t00, sigg in zip(t0, sig)])

    ## Run once with the |00> state
    print('Running simulation with g = %g MHz ...\n\n' % (g/1e-3))
    # tmon = transmon_long(alpha, self_kerr, wq, Nq, Nc,
    #         psi_g0, g=g*1j, gamma1=gamma1, kappa=kappa)#*np.exp(1j*np.pi/4))
    # adata_g, _ = get_transmon_pdiag(tmon, tpts, kappa, nkappas,
    #              gamma1, fext='0g', write_ttraces=True, g=g)
    # tmon = transmon_long(alpha, self_kerr, wq, Nq, Nc,
    #         psi_e0, g=g*1j, gamma1=gamma1, kappa=kappa) #*np.exp(1j*np.pi/4))
    # adata_e, _ = get_transmon_pdiag(tmon, tpts, kappa, nkappas,
    #             gamma1, fext='0e', write_ttraces=True, g=g)

    tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc,
            psi_g0, g=g, wc=wc)
    adata_g, _ = get_transmon_pdiag(tmon, tpts, kappa, nkappas,
                gamma1, fext='0g', write_ttraces=True, g=g)

    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi_e0,
    #         g=g*np.exp(1j*np.pi/4))
    # adata_e, _ = get_transmon_pdiag(tmon, tpts, kappa, nkappas,
    #             gamma1, fext='0e',
    #             write_ttraces=True, g=g)

    # Plot the results of the traces
    ppt.plot_phase_traces(tpts, adata_g, nkappas, drvs, kappa)
    # ppt.plot_phase_ss(adata_g, tpts, nkappas)
    # ppt.plot_phase_traces(tpts, adata_e, nkappas, drvs, kappa)
    # ppt.plot_phase_ss(adata_e, tpts, nkappas)

    ## Run again with |01> state
    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi_e0, g)
    # get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0e',
    #                      write_ttraces=False)

    # Try with the Bell state (|g0> - |e0>) / sqrt(2)
    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi0)
    # get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0e0g',
    #                     write_ttraces=False)


def test_get_transmon_pdiag_mops():
    """
    Test the phase diagram code with a simple transmon coupled to a cavity
    dispersively
    """

    # 16 levels in the cavity, 3 in the transmon
    Nc = 16; Nq = 2;
    
    # Set the cavity linewidth and the transmon T1
    # T1 = 40 us, kappa = 125 kHz
    kappa = 0.1; chi  = kappa / 2.
    Delta = 1;
    g = np.sqrt(Delta * chi)
    
    # Set the initial state
    # a = qt.tensor(qt.destroy(Nq), qt.qeye(Nc))
    psi_g0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0))) 
    psi_e0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 1), mops.basis(Nc, 0)))

    # Set the time of the simulation in ns
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))

    # Run the phase diagram code here
    nkappas = np.linspace(0.5, 4, 8) # np.logspace(-2, 3, 6, base=2)
    
    # Create the drive signals here
    t0 = np.array([3*nk / (2*kappa) for nk in nkappas])
    sig = np.array([nk / (2*kappa) for nk in nkappas])

    # Drives used to generate the phase diagram
    drvs = np.array([np.exp(-((tpts - t00)**2)/(2*sigg**2))\
            for t00, sigg in zip(t0, sig)])

    # drvs = dts.get_tanh_seq(tpts, 0.1*tpts.max(), 0.9*tpts.max(), nkappas.size)

    print('Running simulation with g = %g MHz ...\n\n' % (g/1e-3))

    # ## Run once with the |g0> state
    # print('Running Transmon with dispersive coupling ...')
    # tmon = transmon_disp_mops(Nq, Nc, tpts,
    #         psi0=psi_g0, gamma1=0, kappa=kappa, g=chi)
    # addata_g, _ = get_transmon_pdiag_mops(tmon, tpts, kappa, nkappas,
    #             gamma1=0, fext='0g', write_ttraces=True, g=g)
    # adg = np.array([ad[-1] for ad in addata_g])
    # 
    # ## Run once with the |e0> state
    # dt = tpts.max() / (10 * tpts.size)
    # tmon = transmon_disp_mops(Nq, Nc, tpts,
    #         psi0=psi_e0, gamma1=0, kappa=kappa, g=g)
    # addata_e, _ = get_transmon_pdiag_mops(tmon, tpts, kappa, nkappas,
    #             gamma1=0, fext='0e', write_ttraces=True, g=g, dt=dt)
    # ade = np.array([ad[-1] for ad in addata_e])

    # ppt.plot_phase_ss(addata_e, tpts, nkappas, kappa, g)
    # ppt.plot_phase_traces(tpts, addata_e, nkappas, drvs, kappa)

    ## Run once with the |g0> state
    # print('Running Transmon with longitudinal coupling ...')
    dt = tpts.max() / (10 * tpts.size)
    # tmon = transmon_long_mops(Nq, Nc, tpts,
    #         psi0=psi_g0, gamma1=0, kappa=kappa, g=g)
    # aldata_g, _ = get_transmon_pdiag_mops(tmon, tpts, kappa, nkappas,
    #             gamma1=0, fext='0g', write_ttraces=True, g=g, dt=dt)
    # alg = np.array([ad[-1] for ad in aldata_g])

    ## Run once with the |e0> state
    tmon = transmon_long_mops(Nq, Nc, tpts,
            psi0=psi_e0, gamma1=0, kappa=kappa, g=g)
    aldata_e, _ = get_transmon_pdiag_mops(tmon, tpts, kappa, nkappas,
                gamma1=0, fext='0e', write_ttraces=True, g=g)
    ale = np.array([ad[-1] for ad in aldata_e])

    # ppt.plot_phase_traces(tpts, aldata_g, nkappas, drvs, kappa)
    ppt.plot_phase_ss(aldata_e, tpts, nkappas, kappa, g, use_tseries=True)

    # ppt.plot_io_a_full(nkappas/kappa, adg, ade, alg, ale,
    #                   g, 20*chi, kappa, fext='mops')



if __name__ == '__main__':

    # Run the above test code
    # test_get_transmon_pdiag()
    test_get_transmon_pdiag_mops()
