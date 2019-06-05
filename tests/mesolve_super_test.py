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

        print('Is H Hermitian? %r' % np.allclose(self.H, self.H.T.conj()))


def test_qho_mesolve_fock_decay_super(N):
    """
    Test the decay of Fock state in a harmonic oscillator
    """

    # Choose physical parameters
    w = 2*np.pi*1
    Nc = 16
    T = 10 * 2*np.pi / w
    gamma = 1. / T

    # Set the times and time step
    tpts = np.linspace(0, T, 501)
    dt = tpts.max() / (tpts.size)

    print('dt: %g' % dt)

    # Set the initial density matrix and solve for the new rho
    rho0 = mops.ket2dm(mops.coherent(Nc, 1))
    
    # Initialize the class object and run_dynamics()
    my_qho = qho_super(Nc, w, gamma)
    my_qho.set_init_state(rho0)
    rho = my_qho.run_dynamics(tpts, [], dt=dt)
    
    # Get the average population
    navg = sops.sexpect(mops.dag(my_qho.a)@my_qho.a, rho)
    # navg = sops.sexpect(np.eye(Nc), rho)
    # unit_tr = np.ones(navg.size)

    # print('sqrt(sum((Tr[p] - 1)^2)): %g' % \
    #         (np.sqrt(sum((unit_tr - np.abs(navg))**2))))
    # print('sqrt(sum((Tr[Re p] - 1)^2)): %g' % \
    #         (np.sqrt(sum((unit_tr - navg.real)**2))))
    # print('sqrt(sum((Tr[Im p] - 1)^2)): %g' % \
    #         (np.sqrt(sum((unit_tr - navg.imag)**2))))

    # print('rho[-1].shape: {}'.format(rho[-1].shape))

    with open('data/rho_imp.bin', 'wb') as fid:
        pk.dump(rho, fid)
    fid.close()

    # Plot the results
    ppt.plot_expect(tpts, navg.real, op_name='a^{\dagger}a',
                   file_ext='qho_super_n_%d' % N) 

    plt.figure(2)
    q = sops.sexpect(np.sqrt(0.5/w)*(my_qho.a + mops.dag(my_qho.a)), rho)
    p = sops.sexpect(-1j*np.sqrt(0.5*w)*(my_qho.a - mops.dag(my_qho.a)), rho)
    plt.plot(np.abs(q), np.abs(p))
    plt.xlabel(r'$\Re q$'); plt.ylabel(r'$\Re p $')
    plt.savefig('figs/phase_space_coherent_sket.pdf', format='pdf')


if __name__ == '__main__':
    
    # Run these tests by default
    test_qho_mesolve_fock_decay_super(1)
