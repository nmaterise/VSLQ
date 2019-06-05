#!/usr/bin/env python3
"""
Tests for the superoperator mesolve solver

"""

# Add the VSLQ path 
vslq_path = '/home/nmaterise/mines/research/VSLQ'
import sys
if vslq_path not in sys.path:
    sys.path.append(vslq_path)
from qubit_cavity import base_cqed_sops
import matrix_ops as mops
import super_ops as sops
from ode_solver_super import mesolve_super_impmdpt
import post_proc_tools as ppt
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk


class qho_super(base_cqed_sops):
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
        base_cqed_sops.__init__(self, Nc=Nc, w=w, gamma=gamma)

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


def test_qho_mesolve_fock_decay_super(N):
    """
    Test the decay of Fock state in a harmonic oscillator
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 2
    T = 10 * 2*np.pi / w
    gamma = 2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 51)
    dt = tpts.max() / (tpts.size)

    print('dt: %g' % dt)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.basis(Nc, N))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho_super(Nc, w, gamma)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt, solver='implicitmdpt')

    # Get the average population
    navg = sops.sexpect(mops.dag(my_qho.a)@my_qho.a, rho)

    # Plot the results
    ppt.plot_expect(tpts, navg, op_name='a^{\dagger}a',
                   file_ext='qho_super_n_%d' % N, ms='x')


def test_qho_mesolve_coherent_decay_super(alpha):
    """
    Test the decay of coherent state, track its position
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 16
    T = 10 * 2*np.pi / w
    gamma = 2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 201)
    dt = tpts.max() / (tpts.size)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.coherent(Nc, alpha))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho_super(Nc, w, gamma)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt, solver='implicitmdpt')

    # Get the average population
    x = np.sqrt(1. / (2*w)) * (my_qho.a + mops.dag(my_qho.a))
    xavg = sops.sexpect(x, rho)

    # Plot the results
    ppt.plot_expect(tpts, xavg, op_name='x',
            file_ext='qho_super_alpha_{}_x'.format(alpha)) 


if __name__ == '__main__':
    
    # Run these tests by default
    # test_qho_mesolve_fock_decay_super(1)
    test_qho_mesolve_coherent_decay_super(1)
