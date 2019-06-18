#!/usr/bin/env python
"""
Dispersive readout of a transmon qubit Hamiltonian class
"""

import numpy as np
from qubit_cavity import base_cqed_mops
import matrix_ops as mops
import drive_tools as dts


class transmon_disp_mops(base_cqed_mops):
    """
    Implements the cavity-transmon interaction in the dispersive regime
    """

    def __init__(self, Nq, Nc, tpts, psi0=None,
                 gamma1=0., kappa=0.1, g=0.05):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed_mops.__init__(self, tpts=tpts, Nq=Nq, Nc=Nc, psi0=psi0,
                           gamma1=gamma1, kappa=kappa, g=g)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.kappa], [self.ac])
        self.set_init_state(psi0)

    def get_drive(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            A, t0, sig = args[0]
        else:
            A, t0, sig = args

        return A * np.exp(-(tpts - t0)**2 / (2*sig)**2)


    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = mops.destroy(self.Nq)

        if self.Nq > 2:
            self.at = mops.tensor(at0, np.eye(self.Nc))
        else:
            self.sz = mops.tensor(mops.sop('z'), np.eye(self.Nc))

            # Attempt to fix -0 terms
            ## Get the indices of the non-zeros
            zidx = set(list(np.flatnonzero(self.sz)))
    
            ## Get all of the indices, the take the union
            ## and subtract intersection
            allidx = set(list(range(0, self.sz.size)))
            szflat = self.sz.flatten()
            iunionidx = list(allidx.symmetric_difference(zidx))
            
            ## Overwrite the -0 values with abs(0)
            szflat[iunionidx] = np.abs(szflat[iunionidx])
            self.sz = szflat.reshape(self.sz.shape)

        # Set the cavity operators
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(np.eye(self.Nq), ac0)
        

    def set_H(self, tpts, args):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Time independent Hamiltonian
        # From Didier et al. supplemental section
        # H0 = np.zeros(self.ac.shape, dtype=np.complex128)
        # H0 = self.g * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        if self.Nq > 2:
            H0 = self.g * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        else:
            H0 = self.g * mops.dag(self.ac)@self.ac @ self.sz 

        # Time dependent readout Hamiltonian
        Hc = (self.ac + mops.dag(self.ac))
        # Hc = self.chi * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        Hd = self.get_drive(tpts, args)
        self.H = [H0, [Hc, Hd]]


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = mops.tensor(mops.basis(self.Nq, 0), mops.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = mops.expect(self.ac, psif)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = mops.expect(mops.dag(self.at) @ self.at, psif)

        return n_expect


class transmon_long_mops(base_cqed_mops):
    """
    Implements the cavity-transmon longitudinal interaction
    """

    def __init__(self, Nq, Nc, tpts, psi0=None,
                 gamma1=0., kappa=0.1, g=0.05, phi=0):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed_mops.__init__(self, tpts=tpts, Nq=Nq, Nc=Nc, psi0=psi0,
                           gamma1=gamma1, kappa=kappa, g=g, phi=phi)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.kappa], [self.ac])
        self.set_init_state(psi0)

    def get_drive(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            A, t0, sig = args[0]
        else:
            A, t0, sig = args

        return A * np.exp(-(tpts - t0)**2 / (2*sig)**2)


    def get_drive_tanh(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            x1, x2, a, b = args[0]
        else:
            x1, x2, a, b = args

        return dts.get_tanh_env(tpts, x1, x2, a, b)


    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = mops.destroy(self.Nq)
    
        if self.Nq > 2:
            self.at = mops.tensor(at0, np.eye(self.Nc))
        else:
            self.sz = mops.tensor(mops.sop('z'), np.eye(self.Nc))

            # Attempt to fix -0 terms
            ## Get the indices of the non-zeros
            zidx = set(list(np.flatnonzero(self.sz)))
    
            ## Get all of the indices, the take the union
            ## and subtract intersection
            allidx = set(list(range(0, self.sz.size)))
            szflat = self.sz.flatten()
            iunionidx = list(allidx.symmetric_difference(zidx))
            
            ## Overwrite the -0 values with abs(0)
            szflat[iunionidx] = np.abs(szflat[iunionidx])
            self.sz = szflat.reshape(self.sz.shape)

        # Set the cavity operators
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(np.eye(self.Nq), ac0)
        

    def set_H(self, tpts, args):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Set the time independent Hamiltonian based on 2-level or 3-level
        # approximation of the transmon
        if self.Nq > 2:
            H0 = self.g * self.at @ (mops.dag(self.ac) + self.ac)
        else:
            H0 = self.sz @ (mops.dag(self.ac)*self.g \
                    + self.ac*np.conj(self.g))

        # Time independent readout Hamiltonian
        self.H = H0


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system,
        if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = mops.tensor(mops.basis(self.Nq, 0),
                              mops.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = mops.expect(self.ac, psif)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = mops.expect(mops.dag(self.at) @ self.at, psif)

        return n_expect
