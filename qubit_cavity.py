#!/usr/bin/env python
"""
Implements the parent class to derive transmon dispersive,
transmon longitudinal, VSLQ dispersive, and VSLQ longitudinal-like
"""

import numpy as np
import ode_solver as odes


class base_cqed_mops(object):
    """
    Implements the base class for circuit QED Hamiltonians including
    time independent and time dependent Hamiltonians, using the matrix_ops
    fourth order Runge-Kutta tools
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
        self.cops = None


    def __del__(self):
        """
        Class destructor
        """
        pass

    
    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values -- implemented by the derived class
        """
        
        pass


    def set_H(self, args):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity

        This drive function accepts optional arguments, args

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


    def run_dynamics(self, tpts, *args, **kwargs):
        """
        Run the master equation solver and return the results object
        """

        # Run the dynamics and return the results object
        self.set_H(tpts, args)
    
        # Get the time step
        if kwargs is not None:
            dt = kwargs['dt']
        else:
            dt = self.tpts.max() / (10 * self.tpts.size)
        me_rk4 = odes.mesolve_rk4(self.psi0, tpts, dt,
                self.H, self.cops) 
    
        # Return the density matrix
        psif = me_rk4.mesolve()
        
        return psif
