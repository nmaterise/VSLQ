#!/usr/bin/env python3
"""
Test of the updated master equation solver
using a quantum harmonic oscillator with and without dissipation

"""

# Add the VSLQ path 
from test_utils import set_path
set_path()

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
    Nc = 2
    T = 10 * 2*np.pi / w
    gamma = 2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 51)
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
                    file_ext='qho_n_%d' % N, ms='o') 


def test_qho_mesolve_coherent_decay(alpha):
    """
    Test the decay of coherent state, track its position
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 16
    T = 10 * 2*np.pi / w
    gamma = 0 #2*np.pi / T

    # Set the times and time step
    tpts = np.linspace(0, T, 301)
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


def test_mesolve():
    """
    Tests the mesolve_rk4() class
    """

    # Setup a basic cavity system
    Nc = 2;
    Nq = 3;
    a = mops.tensor(mops.qeye(Nq), mops.destroy(Nc))
    b = mops.tensor(mops.destroy(Nq), mops.qeye(Nc))
    wc = 5;
    kappa = 0.1
    dt = (1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))

    # Time independent Hamiltonian
    H0 = wc*a.dag()*a
    
    # Time dependent Hamiltonian
    Hc = (a + a.dag())
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [kappa * a]
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0)))

    # Setup the master equation solver instance
    me_rk4 = mesolve_rk4(rho0, tpts, 4*tpts.max()/tpts.size, H, cops) 
    rho_out = me_rk4.mesolve()

    # Compute the expectation value of a^t a
    a_avg = mops.expect(a, rho_out)

    # Plot the results
    plt.plot(tpts, a_avg.real, label=r'$\Re \langle a\rangle$')
    plt.plot(tpts, a_avg.imag, label=r'$\Im \langle a\rangle$')
    plt.legend(loc='best')
    

def test_mesolve_mops():
    """
    Tests the mesolve_rk4() class using the matrix_ops module
    """

    # Setup a basic cavity system
    Nc = 16;
    Nq = 3;
    a = mops.tensor(np.eye(Nq), mops.destroy(Nc))
    b = mops.destroy(Nq); bd = mops.dag(b)
    sz = mops.tensor(bd@b, np.eye(Nc))
    wc = 5; wq = 6;
    kappa = 0.1; chi = kappa / 2.; g = np.sqrt(chi)
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))
    # tpts_d = np.linspace(0, 10/kappa, 4*tpts.size)

    # Time independent Hamiltonian
    # H0 = wc*mops.dag(a)@a + wq*sz/2. + chi*mops.dag(a)@a@sz
    # In rotating frame
    H0 = g*(mops.dag(a) + a) @ sz
    
    # Time dependent Hamiltonian
    Hc = (a + mops.dag(a))
    # Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2) \
    # * np.sin(wc*tpts)
    # In rotating frame
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [np.sqrt(kappa) * a]
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0)))
    print('rho0:\n{}'.format(rho0))
    print('Time = [%g, %g] ns' % (tpts.min(), tpts.max()))

    # Setup the master equation solver instance
    me_rk4 = mesolve_rk4(rho0, tpts, tpts.max()/(10*tpts.size), H, cops) 
    rho_out = me_rk4.mesolve()

    # Compute the expectation value of a^t a
    a_avg = mops.expect(a, rho_out)
    print('{}'.format(a_avg.real))

    # Plot the results
    plt.plot(kappa*tpts, a_avg.real,label=r'$\Re \langle a\rangle$')
    plt.plot(kappa*tpts, a_avg.imag,label=r'$\Im \langle a\rangle$')
    plt.xlabel(r'Time (1/$\kappa$)')
    plt.legend(loc='best')


if __name__ == '__main__':
    
    # This is a test of the population transfer between a harmonic
    # oscillator and a qubit with loss applied to both the qubit
    # and oscillator.
    # There is a similar example in the QuTip documentation that I used
    # to validate this test related to the Jaynes-Cummings model
    # test_qho_mesolve_fock_decay(1)
    # test_qho_mesolve_coherent_decay(1)
    # test_qho2_mesolve_fock_decay(1)
    test_qho2_mesolve_driven(0)
