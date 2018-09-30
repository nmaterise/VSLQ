#!/usr/bin/env python
"""
Dispersive readout of a transmon qubit Hamiltonian class
"""

import qutip as qt
import numpy as np
import post_proc_tools as ppt


class transmon_disp:
    """
    Implements the cavity-transmon interaction in the dispersive regime
    """

    def __init__(self, alpha, self_kerr, wq, Nq, Nc, psi0=None):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        # self-Kerr, and cross-Kerr (chi)
        self.alpha = alpha; self.self_kerr = self_kerr;
        self.Nq    = Nq;    self.Nc        = Nc;
        self.chi   = np.sqrt(2 * self_kerr * alpha)
        self.wq    = wq 
        
        # Initialize the collapse operators as None
        self.cops  = None
        self.set_ops()
        self.set_init_state(psi0)
        self.set_H()

    
    def __del__(self):
        """
        Class destructor
        """
        pass


    @staticmethod
    def get_cy_window_dict(t1, t2, w, beta, A=1, ph=0, dc=0):
        """
        Computes the windowed sine function with start and stop
        times t1, t2, at frequency w and rise time of the window
        set by beta. The amplitude of the signal is set by A, and
        the phase and dc offset are ph and dc
        """

        # Arguments dictionary
        args = {'w'  : w,  'a'  : beta, 'A'  : A, 't1' : t1,
                't2' : t2, 'dc' : dc,   'ph' : ph}

        return args

    
    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = qt.destroy(self.Nq)
        self.at = qt.tensor(at0, qt.qeye(self.Nc))

        # Set the cavity operators
        ac0 = qt.destroy(self.Nc)
        self.ac = qt.tensor(qt.qeye(self.Nq), ac0)
        


    def set_H(self):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity

        *** Note ***
        This function does not include the arguments for the string-based
        Cython time-dependent Hamiltonian, Hc in self.H 

        """

        # Time independent Hamiltonian
        H0 = self.wq*self.at.dag()*self.at - 0.5 * self.alpha * self.at**2 \
             - self.chi * self.ac.dag()*self.ac * self.at.dag()*self.at

        # Time dependent readout Hamiltonian
        Hc = (self.ac + self.ac.dag())
        Hc_str = 'A * exp(-(t-(t1-t2)/2)**2/((t2-t1)**2/8)) * cos(w*t-ph) + dc'

        self.H = [0.*H0, [Hc, Hc_str]]


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = qt.tensor(qt.basis(self.Nq, 0), qt.basis(self.Nc, 0))
        self.psi0 = psi0 if psi0 is not None else psi_gnd


    def set_cops(self, gamma1, kappa):
        """
        Set the collapse operators, assuming the system is shot noise limited,
        e.g. T2 > T1
        """

        # Use 1/T1 for the transmon and the line width of the cavity
        self.cops = [np.sqrt(gamma1) * self.at,
                     np.sqrt(kappa) * self.ac]



    def run_dynamics(self, tpts, gamma1, kappa, args):
        """
        Run the master equation solver and return the results object
        """

        # Set the collapse operators if they are None
        if (self.cops is None) and (gamma1 is not None)\
            and (kappa is not None):
            self.set_cops(gamma1, kappa)


        # Run the dynamics and return the results object
        psif = qt.mesolve(self.H, self.psi0, tpts, self.cops, [],
                         options=qt.Options(nsteps=1000), args=args)

        
        return psif



    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = qt.expect(self.ac, psif.states)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = qt.expect(self.at.dag()*self.at, psif.states)
        
        return n_expect


def test_transmon():
    """
    Testing function for the above class using a simple example
    of a three level transmon coupled to a cavity with 16 levels
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
    my_tmon = transmon_disp(alpha, self_kerr, wq, Nq, Nc, psi0)

    # Set the time of the simulation in ns
    tpts = np.linspace(0, 1000, 3001)

    # Set the drive parameters for the readout
    t1 = 0; t2 = tpts.max(); w = 0.; beta = 0.01;
    args = my_tmon.get_cy_window_dict(t1, t2, w, beta) 

    # Run the mesolver
    res = my_tmon.run_dynamics(tpts, gamma1, kappa, args) 
    aavg = my_tmon.get_a_expect(res)
    # navg = my_tmon.get_n_expect(res)

    ppt.plot_expect(tpts, aavg, 'a', file_ext='a')
    # ppt.plot_expect(tpts, navg, 'n', file_ext='n')


if __name__ == '__main__':
    
    # Run the test function above
    test_transmon()



