#!/usr/bin/env python
"""
Generate the data needed for a phase diagram
depicting the spread of the |0> and |1> state
as projected on the cavity state
"""

import numpy as np
import qutip as qt
from transmon import transmon_disp
import datetime


def get_transmon_pdiag(tmon, tpts, kappa, nkappas,
                       gamma1=0, fext='', write_ttraces=False):
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

    """

    # Save the results for a_avg
    a_avg = [] # np.zeros(nkappas.size, dtype=np.complex128)

    # Save the time traces
    if write_ttraces:
        ttraces = [] # np.zeros(nkappas.size * tpts.size, dtype=np.complex128)

    # Compute the time spacing for the tpts
    dt = tpts.max() / tpts.size
    
    # Create a loop over kappa runs

    # Test the parfor version of the function
    a_avg = qt.parallel.parfor(parfor_update, [tpts]*nkappas.size,
                                [tmon]*nkappas.size, nkappas, 
                                [kappa]*nkappas.size, [gamma1]*nkappas.size)

    # for idx, nk in enumerate(nkappas):

    #     # Setup the transmon inputs
    #     t0 = tpts.max() - nk / (2*kappa); sig = nk / (6*kappa);
    #     
    #     print('Running measurement from %g to %g ns ...'%(t0-sig/2, t0+sig/2))

    #     args = tmon.get_cy_window_dict(t0, sig, 0, 0.01)
    #     res  = tmon.run_dynamics(tpts, gamma1, kappa, args)
    #     aavg = tmon.get_a_expect(res)
    #     if write_ttraces:
    #         ttraces.append(aavg)
    #         
    #     # Store the results
    #     idx = ((tpts < t0+dt) & (tpts > t0-dt))
    #     a_avg.append(aavg[idx][np.argmax(np.abs(aavg[idx]))])


    a_avg = np.asarray(a_avg)
    a_avg = a_avg.flatten()

    # Write the results to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
    a_avg.real.tofile('data/areal_%s_%s.bin' % (fext, tstamp)) 
    a_avg.imag.tofile('data/aimag_%s_%s.bin' % (fext, tstamp))
    if write_ttraces:
        ttraces = np.asarray(ttraces)
        ttraces.real.tofile('data/areal_traces_%s_%s.bin' % (fext, tstamp))
        ttraces.imag.tofile('data/aimag_traces_%s_%s.bin' % (fext, tstamp))
    

def parfor_update(tpts, tmon, nk, kappa, gamma1):
    """
    TODO: Test this function with qt.parallel.parfor
    """

    # Setup the transmon inputs
    t0 = tpts.max() - nk / (2*kappa); sig = nk / (6*kappa);
    dt = tpts.max() / tpts.size
    
    print('Running measurement from %g to %g ns ...'%(t0-sig/2, t0+sig/2))

    args = tmon.get_cy_window_dict(t0, sig, 0, 0.01)
    res  = tmon.run_dynamics(tpts, gamma1, kappa, args)
    aavg = tmon.get_a_expect(res)
    

    # Store the results
    idx = ((tpts < t0+2*dt) & (tpts > t0-2*dt))
    
    if aavg[idx].size > 1:
        aavg_out = aavg[idx][np.argmax(np.abs(aavg[idx]))]
        return aavg_out

    else:
        print('aavg[idx]: {}'.format(aavg[idx]))
        return aavg[idx]
    



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
    wq = 5*2*np.pi; 
    
    # Set the cavity linewidth and the transmon T1
    # T1 = 40 us, kappa = 125 kHz
    T1 = 40e3; gamma1 = 1./T1; kappa = 0.000125;
    T1 = 0.; gamma1 = 0.; kappa = 0.1; # chi / 2.#0.1;
    alpha = 0.250*2*np.pi;      self_kerr = 2*kappa**2/alpha;
    
    # Set the initial state
    psi_g0 = qt.tensor(qt.basis(Nq, 0), qt.basis(Nc, 0))
    psi_e0 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nc, 0))
    psi0 = (psi_g0 - psi_e0).unit()
    # psi0 = psi_g0 
    
    # Create an instance of the class
    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi_g0)

    # Set the time of the simulation in ns
    tpts = np.linspace(0, 100/kappa, 3001)

    # Run the phase diagram code here
    nkappas = np.linspace(1, 100, 10)

    ## Run once with the |00> state
    tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi_g0)
    get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0g',
            write_ttraces=False)

    ## Run again with |01> state
    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi_e0)
    # get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0e',
    #                      write_ttraces=False)

    # Try with the Bell state (|g0> - |e0>) / sqrt(2)
    # tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi0)
    # get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0e0g',
    #                     write_ttraces=False)


if __name__ == '__main__':

    # Run the above test code
    test_get_transmon_pdiag()



