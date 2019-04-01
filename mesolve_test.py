#!/usr/bin/env python3
"""
Test of the updated master equation solver
using a quantum harmonic oscillator with and without dissipation

"""

import numpy as np
import matplotlib.pyplot as plt
import matrix_ops as mops
import post_proc_tools as ppt
from qubit_cavity import base_cqed_mops
from ode_solver import mesolve_rk4


class qho(base_cqed_mops):
    """
    Derived class of the base class for implementing Hamiltonians and
    non-unitary evolution for the quantum harmonic oscillator
    """

    def __init__(self, Nc, w, gamma):
        """
        Call the base class constructor with the above keyword arguments
    
        Parameters:
        ----------

        Nc:         number of levels in the oscillator
        w:          resonance frequency 
        gamma:      dissipation rate
        
        """

        # Call the base constructor
        base_cqed_mops.__init__(self, Nc=Nc, w=w, gamma=gamma)

        # Set the operators
        self.set_ops()
        self.set_H([], [])
        self.set_cops([gamma], [self.a])


    def set_ops(self):
        """
        Set the relevant operators for the Hamiltonian
        """

        # Set the destruction operator (s)
        self.a = mops.destroy(self.Nc)

        # Set the identity operator
        self.Ic = np.eye(self.Nc)

    
    def set_H(self, tpts, args):
        """
        Set the Hamiltonian
        """

        # H = w(a^t a + 1/2)
        self.H = self.w * (mops.dag(self.a)@self.a + self.Ic/2)


def test_qho_mesolve_fock_decay(N):
    """
    Test the decay of Fock state in a harmonic oscillator
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 16
    T = 10 * 2*np.pi / w
    gamma = 2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 5001)
    dt = tpts.max() / (tpts.size)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.basis(Nc, N))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho(Nc, w, gamma)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt)

    # Get the average popultion
    navg = mops.expect(mops.dag(my_qho.a)@my_qho.a, rho)

    # Plot the results
    ppt.plot_expect(tpts, navg, op_name='a^{\dagger}a',
                    file_ext='qho_n_%d' % N) 


def test_qho_mesolve_coherent_decay(alpha):
    """
    Test the decay of coherent state, track its position
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 16
    T = 10 * 2*np.pi / w
    gamma = 2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 5001)
    dt = tpts.max() / (tpts.size)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.coherent(Nc, alpha))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho(Nc, w, gamma)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt)

    # Get the average popultion
    x = np.sqrt(1. / (2*w)) * (my_qho.a + mops.dag(my_qho.a))
    xavg = mops.expect(x, rho)

    # Plot the results
    ppt.plot_expect(tpts, xavg, op_name='x',
                    file_ext='qho_alpha_{}_x'.format(alpha)) 


if __name__ == '__main__':
    
    # Run the above test by default
    # test_qho_mesolve_fock_decay(3)
    test_qho_mesolve_coherent_decay(1)
