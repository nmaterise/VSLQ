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
import drive_tools as dts

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


class qho2(base_cqed_mops):
    """
    Two bilinearly coupled harmonic oscillators
    """

    def __init__(self, tpts, N1, N2, w1, w2,
                 g, gamma1, gamma2, use_Ht=False):
        """
        Call the base class constructor with the above keyword arguments
    
        Parameters:
        ----------

        tpts:       time points to evaluate the H(t) term
        N1, N2:     number of levels in oscillators 1 and 2
        w1, w2:     resonance frequencies of 1 and 2
        g:          coupling between oscillators 1 and 2
        gamma1/2:   dissipation rates of 1 and 2
        use_Ht:     use the time-dependent Hamiltonian
        
        """

        # Call the base constructor
        base_cqed_mops.__init__(self, tpts=tpts, N1=N1, N2=N2, w1=w1, w2=w2,
                                gamma1=gamma1, gamma2=gamma2, g=g)

        # Set the operators
        self.set_ops()
        self.set_H(tpts, use_Ht)
        self.set_cops([gamma1, gamma2],\
                      [self.a1, self.a2])


    def set_ops(self):
        """
        Set the relevant operators for the Hamiltonian
        """

        # Set the identity operators
        I1 = np.eye(self.N1)
        I2 = np.eye(self.N2)

        # Set the destruction operator (s)
        a01 = mops.destroy(self.N1)
        a02 = mops.destroy(self.N2)

        # Tensor the identity and destruction operators
        self.a1 = mops.tensor(a01, I2)
        self.a2 = mops.tensor(I1, a02)

        # Number operators
        self.n1 = mops.dag(self.a1) @ self.a1
        self.n2 = mops.dag(self.a2) @ self.a2


    
    def set_H(self, tpts, args):
        """
        Set the Hamiltonian
        """

        # Extract args
        use_Ht = args

        # H = w1 a1^t a1 + w2 a2^t a2 + g(a1 + a1^t) (a2 + a2^t) 
        #       + sqrt(2) cos(w2 t) (a1 + a1^t)
        # Define the time independent and time dependent Hamiltonians
        H0 = self.w1 * self.n1 + self.w2 * self.n2 \
               + self.g * (mops.dag(self.a1)@self.a2 \
                       + self.a1@mops.dag(self.a2))

        # Combine both components as a single list
        if use_Ht:
            print('Using time-dependent Hamiltonian ...')
            Hp = self.a1 + mops.dag(self.a1)
            delta = abs(self.w1 - self.w2)
            wR = np.sqrt(4*self.g**2 + delta**2)
            self.H = [H0, [[Hp, np.sqrt(2) * np.cos(wR*tpts)]]]
        else:
            self.H = H0


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

    # Get the average population
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

    # Get the average population
    x = np.sqrt(1. / (2*w)) * (my_qho.a + mops.dag(my_qho.a))
    xavg = mops.expect(x, rho)

    # Plot the results
    ppt.plot_expect(tpts, xavg, op_name='x',
                    file_ext='qho_alpha_{}_x'.format(alpha)) 


def test_qho2_mesolve_fock_decay(N):
    """
    Test the population transfer of two oscillators and decay
    """

    # Choose physical parameters
    delta = 2*np.pi*0.1
    w1 = 2*np.pi*1; w2 = (w1-delta)
    N1 = 15; N2 = 2;
    g = max(w1, w2) / 20
    T = 4*2*np.pi / (2*g) 
    gamma1 = g / (2*np.pi * 5); gamma2 = gamma1 / 10

    # Set the times and time step
    tpts = np.linspace(0, T, 101)
    dt = tpts.max() / (tpts.size)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.tensor(mops.basis(N1, 0), mops.basis(N2, N)))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho2(tpts, N1, N2, w1, w2, g, gamma1, gamma2)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt)

    # Get the average popultion, n1
    n1 = mops.expect(my_qho.n1, rho)

    # Get the average popultion, n2
    n2 = mops.expect(my_qho.n2, rho)

    plt.plot(tpts, np.abs(n1), label=r'$\langle{a_1^{\dagger}a_1}\rangle$')
    plt.plot(tpts, np.abs(n2), label=r'$\langle{a_2^{\dagger}a_2}\rangle$')
    
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/qho2_n1n2_detuned_%d.eps' % N, format='eps')


def test_qho2_mesolve_driven(N):
    """
    Test driven population transfer of two oscillators with decay
    """

    # Choose physical parameters
    delta = 2*np.pi*0.1
    w1 = 2*np.pi*1; w2 = (w1-delta)
    N1 = 15; N2 = 2;
    g = max(w1, w2) / 20
    T = 5*2*np.pi / np.sqrt(((2*g)**2 + delta**2))
    gamma1 = g / (2*np.pi * 2); gamma2 = gamma1 / 4

    # Set the times and time step
    tpts = np.linspace(0, T, 4001)
    dt   = tpts.max() / (tpts.size)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.tensor(mops.basis(N1, 0), mops.basis(N2, N)))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho2(tpts, N1, N2, w1, w2, g, gamma1, gamma2, use_Ht=True)
    my_qho.set_init_state(rho0)
    print('Running driven dynamics for %g ns ...' % T)
    rho = my_qho.run_dynamics(tpts, [], dt=dt)

    # Get the average popultion, n1
    n1 = mops.expect(my_qho.n1, rho)

    # Get the average popultion, n2
    n2 = mops.expect(my_qho.n2, rho)

    plt.plot(tpts, np.abs(n1), label=r'$\langle{a_1^{\dagger}a_1}\rangle$')
    plt.plot(tpts, np.abs(n2), label=r'$\langle{a_2^{\dagger}a_2}\rangle$')
    
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/qho2_n1n2_detuned_driven_%d.pdf' % N, format='pdf')


if __name__ == '__main__':
    
    # This is a test of the population transfer between a harmonic
    # oscillator and a qubit with loss applied to both the qubit
    # and oscillator.
    # There is a similar example in the QuTip documentation that I used
    # to validate this test related to the Jaynes-Cummings model
    # test_qho2_mesolve_fock_decay(1)
    test_qho2_mesolve_driven(0)
