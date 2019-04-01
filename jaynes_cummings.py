"""
Author: Nick Materise
Filename: jaynes_commings.py
Description: This class includes the basic features of single mode
             Jaynes-Cummings Hamiltonian
Created: 190330
"""

import numpy as np
import matrix_ops as mops
from qubit_cavity import base_cqed_mops
import matplotlib.pyplot as plt
import pickle as pk
import multiprocessing as mp
import post_proc_tools as ppt


class jaynes_cummings(base_cqed_mops):
    """
    Class for the Jaynes-Cummings Hamiltonian derived from the base circuit QED
    class using the matrix_mops modules. Returns the density matrix after 
    running an mesolve calculation

    """

    def __init__(self, Nc, Nq, wc, wq, g, gammac, gammaq, use_rwa=True):
        """
        Constructor for the class

        Parameters:
        ----------

        Nc, Nq:         number of cavity, qubit levels
        wc, wq:         cavity and qubit bare frequencies
        g:              coupling constant between cavity and qubit
        gammac/q:       loss rates for the cavity and qubit
        use_rwa:        use rotating wave approximation if true
    
        """

        # Set the class members here
        base_cqed_mops.__init__(self, Nc=Nc, Nq=Nq, wc=wc, wq=wq,
                                g=g, gammac=gammac, gammaq=gammaq,
                                use_rwa=use_rwa)

        # Set the states and the operators for the class
        self.set_ops()
        
        # Collapse operators only including dissipation, no dephasing
        self.set_cops([self.gammac, self.gammaq], [self.ac, self.sm])

    
    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Ic = np.eye(self.Nc)
        self.Iq = np.eye(self.Nq)

        # Destruction operators
        ## Cavity operators
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(ac0, self.Iq)
        
        # Qubit operators
        sz0 = mops.sop('z')
        sm0 = mops.sop('m')
        sp0 = mops.sop('p')
        self.sz = mops.tensor(self.Ic, sz0)
        self.sp = mops.tensor(self.Ic, sp0)
        self.sm = mops.tensor(self.Ic, sm0)


    def set_H(self, tpts, args):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Set the total Hamiltonian
        if self.use_rwa:
            print('Setting RWA Jaynes-Cummings Hamiltonian ...')
            self.H = self.wc*mops.dag(self.ac)@self.ac \
                     + self.wq*self.sp@self.sm \
                    -1j*self.g*(self.sp@self.ac - self.sm@mops.dag(self.ac))

        else:
            self.H = self.wc*mops.dag(self.ac)@self.ac + self.wq*self.sz / 2 \
                + self.g*(self.sp + self.sm)@(self.ac - mops.dag(self.ac))


def test_jc():
    """
    Test the above Jaynes-Cummings system using the Lindblad solver
    """

    # Physical constants
    delta = 0
    wc = 2*np.pi*1; wq = (wc-delta)
    Nc = 15; Nq = 2;
    g = max(wc, wq) / 20
    T = 8*np.pi / (2*g) 
    gammac = g / (2*np.pi * 2); gammaq = gammac / 10

    # Set the times and time step
    tpts = np.linspace(0, T, 1001)
    dt = tpts.max() / (tpts.size)


    # Initialize the Jaynes-Cummings object
    my_jc = jaynes_cummings(Nc, Nq, wc, wq,
                            g, gammac, gammaq,
                            use_rwa=True)
    
    # Set the initial density matrix and solve for the new rho
    init_state = '0e'
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nc, 0), mops.basis(Nq, 1)))
    my_jc.set_init_state(rho0)

    # Run the dynamics
    args = []
    rho = my_jc.run_dynamics(tpts, args, dt=dt)

    # Get the expected values of sigma_z
    sz = mops.expect(my_jc.sp@my_jc.sm, rho)
    n1 = mops.expect(mops.dag(my_jc.ac)@my_jc.ac, rho)

    # Plot the results
    plt.plot(tpts, sz.real, label=r'$\langle\sigma_z\rangle$')
    plt.plot(tpts, n1.real, label=r'$\langle{a_1^{\dagger}a_1}\rangle$')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('figs/expect_n1sz_%s.eps' % init_state, format='eps')


if __name__ == '__main__':

    # Run the default test of the class
    test_jc()

