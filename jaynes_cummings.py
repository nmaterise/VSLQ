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
        self.set_init_state()
        

    def set_init_state(self, psi0=None, dressed_state='10'):
        """
        Set the initial state
        """

        # Set a particular product state, dressed state of the Jaynes-Cummings
        # Hamiltonian as |e/g, n >
        ## |e, 0 >
        if dressed_state == '10':
            self.psi0 = mops.ket2dm(mops.tensor(mops.basis(self.Nq, 1),
                        mops.basis(self.Nc, 0)))
        ## |g, 1 >
        elif dressed_state == '01':
            self.psi0 = mops.ket2dm(mops.tensor(mops.basis(self.Nq, 0),
                        mops.basis(self.Nc, 1)))
        ## |e/g, n > and all other valid states 
        else:
            self.psi0 = psi0

    
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
        self.ac = mops.tensor(self.Iq, ac0)
        
        # Qubit operators
        sz0 = mops.sop('z')
        sm0 = mops.sop('m')
        sp0 = mops.sop('p')
        self.sz = mops.tensor(sz0, self.Ic)
        self.sp = mops.tensor(sp0, self.Ic)
        self.sm = mops.tensor(sm0, self.Ic)


    def set_H(self, tpts, args):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Set the total Hamiltonian
        if self.use_rwa:
            self.H = self.wc*mops.dag(self.ac)@self.ac \
                     + self.wq*self.sp@self.sm / 2 \
                     + self.g * (self.sp@self.ac + self.sm@mops.dag(self.ac))

        else:
            self.H = self.wc*mops.dag(self.ac)@self.ac + self.wq*self.sz / 2 \
                + self.g*(self.sp + self.sm)@(self.ac - mops.dag(self.ac))


def test_jc():
    """
    Test the above Jaynes-Cummings system using the Lindblad solver
    """

    # Choose a large detuning between qubit and cavity 
    wc = 2*np.pi*1; wq = wc; # wq = 2*np.pi*5; 

    # Choose number of photons for cavity and number of levels for the qubit
    Nc = 16; Nq = 2;

    # Compute g from dispersive limit, chi = g^2 / delta
    chi = 2*np.pi*20e-3; delta = abs(wc - wq);
    g = 2*np.pi*0.05; # np.sqrt(chi * delta)

    # Decay rates small compared to g
    gammac = 0*2*np.pi*0.005; gammaq = 0*2*np.pi*0.05

    # Choose a time scale that is ~10 periods of the Rabi frequency
    TR = 2*np.pi/(2*g); NR = 50; tmax = NR * TR;
    # dt = 2*np.pi / max(wc, wq);
    dt = 2*np.pi / (10*wc);
    Ntmp = int(np.ceil(TR / dt))
    Nt = Ntmp + 1 if not (Ntmp % 2) else Ntmp
    print('Running Jaynes-Cummings for (%g) us with (%d) time steps ...' \
            % (tmax, Nt))
    tpts = np.linspace(0, tmax, Nt)

    # Initialize the Jaynes-Cummings object
    my_jc = jaynes_cummings(Nc, Nq, wc, wq,
                            g, gammac, gammaq,
                            use_rwa=True)
    
    # Set the intial state to a simple dressed state
    init_state = '10'
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.coherent(Nc, 1)))
    my_jc.set_init_state(dressed_state=init_state)

    # Run the dynamics
    args = [1, tpts.max()/2, tpts.max()/12]
    rho = my_jc.run_dynamics(tpts, args, dt=dt/10)

    # Get the expected values of sigma_z
    sz = mops.expect(mops.dag(my_jc.ac)@my_jc.ac, rho)
    # sz = mops.expect(my_jc.sz, rho)
    # sz = mops.expect(my_jc.ac + mops.dag(my_jc.ac), rho)

    # Plot the results
    ppt.plot_expect(tpts, sz, op_name='\sigma_z',
                    file_ext='sz_%s' % init_state, tscale='us') 

if __name__ == '__main__':

    # Run the default test of the class
    test_jc()

