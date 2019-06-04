#!/usr/bin/env python3
"""
Superoperator class of solvers using implicit integration methods
"""

import super_ops as sops
import matrix_ops as mops
from ode_solver import implicitmdpt
import numpy as np


class mesolve_super_impmdpt(implicitmdpt):
    """
    Lindblad master equation solver using the superoperator formulation
    
    d / dt || p >>  = -L || p >>

    L = i/hbar (I x H^T - H x I) 
        - sum_j g_j ( c_j* x c_j - 1/2 (I x (c_j^t c_j) + (c_j^t c_j)^T x I) )

    """

    def __init__(self, rho0, tpts, dt, H, cops):
        """
        Class constructor of the same form as mesolve_rk4

        Parameters:
        ----------

        rho0:       initial density matrix in dense matrix form
        tpts:       array of times to compute the density matrix on
        dt:         time step used by the RK4 solver, should be equal to
                    t_n - t_n-1 
        H:          Hamiltonian in same format as qt.mesolve [H0, [Hc, eps(t)]]
        cops:       collapse operators including the coefficients as a list
                    of qt.Obj's scaled by the damping coefficients

        """

        # Check if H is list or not, e.g. if it is time-dependent or not
        if H.__class__ == list:
            is_A_const = False
        elif H.__class__ == np.ndarray:
            is_A_const = True
        else:
            raise TypeError('Hamiltonian type (%s) not supported.' \
                    % H.__class__)

        # Convert the density matrix to a superket
        rho0 = sops.dm2sket(rho0)

        # Call the implicitmdpt constructor
        implicitmdpt.__init__(self, rho0, tpts, dt, is_A_const=is_A_const, 
                              H=H, cops=cops)

        # Compute the Liouvillian
        self.set_Ld()


    def set_Ld(self):
        """
        Computes the dissipative part of the Liouvillian
        """

        self.Ld = -np.array([mops.tensor(c.conj(), c) \
        - 0.5 * (mops.tensor(mops.dag(c).conj()@c.conj(), np.eye(c.shape[0])) \
                       + mops.tensor(np.eye(c.shape[0]), (mops.dag(c)@c))) \
                    for c in self.cops])


    def get_Lu(self):
        """
        Computes the unitary part of the Liouvillian
        """

        return 1j*(sops.op2sop(self.H, 'left') 
                    - sops.op2sop(self.H, 'right'))


    def rhs_A(self, t):
        """
        Computes the Liouvillian
        """
        
        # Compute the constant time Liouvillian
        if self.is_A_const:
           return -(self.get_Lu() + self.Ld)

        # Compute the time dependent Liouvillian
        ## Extract time-independent and time-dependent components
        H0 = self.H[0]; Hp = self.H[1]

        ## Multiply and add Hp components
        Hpp = sum([h*d(t) for h, d in Hp])

        ## Sum contributions and compute the unitary part of the Liouvillian
        Htot = H0 + Hpp
        Lu = 1j*(sops.op2sop(Htot, 'left') - sops.op2sop(Htot, 'right'))


        return -(Lu + self.Ld)


    def mesolve(self):
        """
        Run the implicitmdpt solver, providing the interpolated 
        time-dependent drive terms
        """

        # Handle time-independent Hamiltonian
        if self.H.__class__ == np.ndarray:
            rho_out = self.solver()
            return rho_out
        
        # Handle the case involving drive terms
        elif self.H.__class__ == list:

            # Extract the time-indepent and time-dependent Hamiltonians
            H0 = self.H[0]
            Hp = self.H[1]

            # Transpose Hp and extract the operators
            HpT = list(map(list, zip(*Hp)))
            Hpops = HpT[0]
            Hpdrvs = HpT[1]

            # Interpolate the drive terms
            drvs = [interp(self.tpts, d) for d in Hpdrvs]
            HH = [H0, [[Hk, d] for Hk, d in zip(Hpops, drvs)]]

            # Set the class instance of the Hamiltonian to the 
            # list comprehension with the interpolants embedded
            self.H = HH

            # Call the solver with the interpolated drives
            rho_out = self.solver()

            return rho_out

        else:
            raise('H.__class__ ({}) not supported'.format(H.__class__))
        
        
        return self.solver()