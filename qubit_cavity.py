#!/usr/bin/env python
"""
Implements the parent class to derive transmon dispersive,
transmon longitudinal, VSLQ dispersive, and VSLQ longitudinal-like
"""

import numpy as np
import ode_solver as odes
import ode_solver_super as sodes
import scipy.sparse as scsp


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
        if cops[0].__class__ == np.ndarray:
            if not np.any(gammas):
                self.cops = [np.zeros(cop.shape) for cop in cops]
            else:
                self.cops = [np.sqrt(g)*cop for g, cop in zip(gammas, cops)]

        # Trust that the matrix-scalar multiplication makes sense
        # for the scipy.sparse.csr_matrix representation
        elif cops[0].__class__ == scsp.csr.csr_matrix:
            if not np.any(gammas):
                self.cops = [scsp.csr_matrix(cop.shape, dtype=cop.dtype) \
                            for cop in cops]
            else:
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
            use_sparse = kwargs['use_sparse']
        else:
            dt = self.tpts.max() / (10 * self.tpts.size)

        # Run the master equation solver
        me_rk4 = odes.mesolve_rk4(self.psi0, tpts, dt,
                self.H, self.cops, use_sparse=use_sparse) 
    
        # Return the density matrix
        psif = me_rk4.mesolve()
        
        return psif


class base_cqed_sops(object):
    """
    Implements the base class for circuit QED Hamiltonians including
    time independent and time dependent Hamiltonians, using the matrix_ops
    and super_ops implicit midpoint superoperator master equation solver
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
        if not np.any(cops):
            self.cops = [np.zeros(cop.shape) for cop in cops]
        else:
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
            solver = kwargs['solver']
            use_sparse = kwargs['use_sparse']
        else:
            dt = self.tpts.max() / (10 * self.tpts.size)
    
        # Set whether to use implicit midpoint, RK4 or another solver
        ## Implicit midpoint
        if solver == 'implicitmdpt':
            me = sodes.mesolve_super_impmdpt(self.psi0, tpts, dt,
                self.H, self.cops, use_sparse=use_sparse) 
        
        ## Runge-Kutta 4
        elif solver == 'rk4':
            me = sodes.mesolve_super_rk4(self.psi0, tpts, dt,
                self.H, self.cops, use_sparse=use_sparse) 

        else:
            raise TypeError('Solver (%s) not supported.' % solver)

    
        # Return the density matrix
        psif = me.mesolve()
        
        return psif
