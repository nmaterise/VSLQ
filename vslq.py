"""
Author: Nick Materise
Filename: vslq.py
Description: This class includes the basic features of a generalized VSLQ for
             N-logical circuits with Nq transmon levels and Ns shadow levels

Created: 180921
"""

import numpy as np
import qutip as qt

class vslq:
    """
    Very Small Logical Qubit (VSLQ) Class
    
    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np):
        """
        Constructor for the class

        Parameters:
        ----------

        Ns:     number of levels in the shadow resonators
        Np:     number of levels in the primary qubits
    
        """

        # Set the class members here
        self.Ns = Ns; self.Np = Np;

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()


    def __del__(self):
        pass
    

    def set_states(self):
        """
        Sets the basis states for the class
        """
    
        # States for the qubit / shadow degrees of freedom
        s0  = qt.basis(self.Np, 0);
        s1  = qt.basis(self.Np, 1); 
        s2  = qt.basis(self.Np, 2)
        ss0 = qt.basis(self.Ns, 0);
        ss1 = qt.basis(self.Ns, 1)
        
        # Compute the density matrices corresponding to the states
        self.s1dm  = qt.ket2dm(s1); 
        self.ss0dm = qt.ket2dm(ss0);
        self.ss1dm = qt.ket2dm(ss1)

        # Define the logical states
        self.L0 = qt.ket2dm((s2 + s0).unit())
        self.L1 = qt.ket2dm((s2 - s1).unit())

        # Initial density matrix
        # psi0 = |L0> x |L0> x |0s> x | 0s>
        self.psi0 = qt.tensor(self.L0, self.L0, self.ss0dm, self.ss0dm)


    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Is = qt.qeye(self.Ns)
        self.Ip = qt.qeye(self.Np)

        # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
        self.Pl1 = qt.tensor(self.s1dm, self.Ip, self.Is, self.Is)
        self.Pr1 = qt.tensor(self.Ip, self.s1dm, self.Is, self.Is)

        # Destruction operators
        ## Primary qubits
        ap0 = qt.destroy(self.Np)
        self.apl = qt.tensor(ap0, self.Ip, self.Is, self.Is)
        self.apr = qt.tensor(self.Ip, ap0, self.Is, self.Is)
        
        ## Shadow resonators
        as0 = qt.destroy(self.Ns)
        self.asl = qt.tensor(self.Ip, self.Ip, as0, self.Is)
        self.asr = qt.tensor(self.Ip, self.Ip, self.Is, as0)

        ## Two photon operators on the logical manifold
        self.Xl = (self.apl**2 + self.apl.dag()**2) / np.sqrt(2)
        self.Xr = (self.apr**2 + self.apr.dag()**2) / np.sqrt(2)


    def set_H(self, W, d, Om):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Hp = -W Xl Xr + 1/2 d (Pl1 + Pr1)
        Hp = -W * self.Xl*self.Xr + 0.5*d*(self.Pl1 + self.Pr1)
        
        # Hs = (W + d/2) (asl^t asl + asr^t asr)
        Hs = (W + d/2.) * (self.asl.dag()*self.asl + self.asr.dag()*self.asr)

        # Hps = O (apl^t asl^t + apr^t asr^t + h.c.)
        Hps = Om*(self.apl.dag()*self.asl.dag() + self.apr.dag()*self.asr.dag())
        Hps += Hps.dag()

        self.H = Hp + Hs + Hps


    def set_cops(self, gammap, gammas):
        """
        Set the collapse operators list using the relaxation rates for the
        primary and shadow lattice
        """
    
        self.cops = [np.sqrt(gammap) * self.apl, np.sqrt(gammap) * self.apr,
                    np.sqrt(gammas) * self.asl, np.sqrt(gammas) * self.asr]


    def run_dynamics(self, tpts, gammap, gammas):
        """
        Run the master equation dynamics for a given set of times, tpts
        """

        # Set the collapse operators
        self.set_cops(gammap, gammas)

        # Run the master equation solver and get the density matrix
        psif = qt.mesolve(self.H, self.psi0, tpts, self.cops, [],
                options=qt.Options(nsteps=1000))


        return psif


    def get_logical_expect(self, psif):
        """
        Computes the expectation value of the logical operator, pL
        """

        # Projection operator to compute the expectation value
        self.pL = 0.5 * self.Xl * (1. + self.Xl*self.Xr) * (1. - self.Pl1) \
                  * (1 - self.Pr1)

        # Compute the expectation value using the states from the mesolve
        # solution, psif
        pL_expect = qt.expect(self.pL, psif.states)

        return pL_expect


if __name__ == '__main__':
    
    # Some example settings
    N = 3; Ns = 2
    W = 70.0*pi; delta = 700.0*pi; Ohm = 5.5; gamma_S = 9.2
