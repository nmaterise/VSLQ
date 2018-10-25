#!/usr/bin/env python
"""
Implements the parent class to derive transmon dispersive,
transmon longitudinal, VSLQ dispersive, and VSLQ longitudinal-like
"""

import numpy as np
import qutip as qt

class base_cqed:
    """
    Implements the base class for circuit QED Hamiltonians including
    time independent and time dependent Hamiltonians
    """

    def __init__(self, *args, **kwargs):
        """
        Class constructor
        """

        # Set the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize the collapse operators as None
        self.cops  = None
        self.set_ops()
        self.set_H()


    def __del__(self):
        """
        Class destructor
        """
        pass

    @staticmethod
    def get_cy_window_dict(t0, sig, w, beta, A=1, ph=0, dc=0):
        """
        Computes the windowed sine function with start and stop
        times t1, t2, at frequency w and rise time of the window
        set by beta. The amplitude of the signal is set by A, and
        the phase and dc offset are ph and dc
        """

        # Arguments dictionary
        args = {'w'  : w,  'a'  : beta, 'A'  : A, 't0' : t0,
                'sig' : sig, 'dc' : dc,   'ph' : ph}

        return args

    
    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values -- implemented by the derived class
        """
        
        pass


    def set_H(self, Hc_str=None):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity

        *** Note ***
        This function does not include the arguments for the string-based
        Cython time-dependent Hamiltonian, Hc in self.H 
        
        Implemented by the derived class

        """

        pass


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        self.psi0 = psi0


    def set_cops(self, gammas, cops):
        """
        Set the collapse operators, assuming the system is shot noise limited,
        e.g. T2 > T1 
        """
        # Use 1/T1 for the transmon and the line width of the cavity
        self.cops = [np.sqrt(g)*cop for g, cop in zip(gammas, cops)]


    def run_dynamics(self, tpts, args):
        """
        Run the master equation solver and return the results object
        """

        # Run the dynamics and return the results object
        psif = qt.mesolve(self.H, self.psi0, tpts,
                          c_ops=self.cops, e_ops=[], args=args)
        
        return psif
