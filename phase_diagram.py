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
    a_avg = np.zeros(nkappas.size, dtype=np.complex128)

    # Save the time traces
    if write_ttraces:
        ttraces = np.zeros(nkappas.size * tpts.size, dtype=np.complex128)

    # Create a loop over kappa runs
    for idx, nk in enumerate(nkappas):

        # Setup the transmon inputs
        t1 = nk / kappa; t2 = tpts.max()
        args = tmon.get_cy_window_dict(t1, t2, 0, 0.01)
        res  = tmon.run_dynamics(tpts, gamma1, kappa, args)
        aavg = tmon.get_a_expect(res)
        if write_ttraces:
            ttraces[idx*tpts.size:idx*tpts.size+tpts.size] = aavg
            
    
        # Store the results
        a_avg[idx] = np.average(aavg)

    # Write the results to file
    tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S') 
    a_avg.real.tofile('data/areal_%s_%s.bin' % (fext, tstamp)) 
    a_avg.imag.tofile('data/aimag_%s_%s.bin' % (fext, tstamp))
    if write_ttraces:
        ttraces.real.tofile('data/areal_traces_%s_%s.bin' % (fext, tstamp))
        ttraces.imag.tofile('data/aimag_traces_%s_%s.bin' % (fext, tstamp))
    

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
    T1 = 0.; gamma1 = 0.; kappa = 0.1;
    
    # Set the initial state
    psi00 = qt.tensor(qt.basis(Nq, 0), qt.basis(Nc, 0))
    psi01 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nc, 0))
    psi0 = (psi00 - psi01).unit()
    psi0 = psi00 
    
    # Create an instance of the class
    tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi0)

    # Set the time of the simulation in ns
    tpts = np.linspace(0, 1000, 3001)

    # Run the phase diagram code here
    nkappas = np.logspace(-2, 6, 9, base=2)
    ## Run once with the |00> state
    get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0g',
            write_ttraces=True)

    ## Run again with |01> state
    # tmon.set_init_state(psi01) 
    # get_transmon_pdiag(tmon, tpts, kappa, nkappas, gamma1, fext='0e')


if __name__ == '__main__':

    # Run the above test code
    test_get_transmon_pdiag()



