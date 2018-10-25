"""
Author: Nick Materise
Filename: vslq.py
Description: This class includes the basic features of a generalized VSLQ for
             N-logical circuits with Nq transmon levels and Ns shadow levels

Created: 180921
"""

import numpy as np
import qutip as qt
from qubit_cavity import base_cqed

class vslq(base_cqed):
    """
    Very Small Logical Qubit (VSLQ) Class
    
    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np, W, d, Om, gammap, gammas, readout_type='disp'):
        """
        Constructor for the class

        Parameters:
        ----------

        Ns:             number of levels in the shadow resonators
        Np:             number of levels in the primary qubits
        W:              primary qubit energy
        d:              anharmonicity of the primary qubits
        Om:             primary-shadow qubit interaction strength
        gammap/s:       loss rate for the primary / shadow qubits
        readout_type:   'disp' / 'long' readout types for dispersive and
                        'longitudinal' interactions
    
        """

        # Set the class members here
        # self.Ns = Ns; self.Np = Np;
        base_cqed.__init__(self, Ns=Ns, Np=Np, W=W, d=d, Om=Om,
                gammap=gammap, gammas=gammas)

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()
        self.set_cops([self.gammas, self.gammas, self.gammap, self.gammap],
                      [self.asl, self.asr, self.apl, self.apr])
        self.set_H()
        self.set_init_state()

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


    def set_init_state(self, psi0=None):
        """
        Set the initial state
        """
    
        # Initialize to a 0-logical state in the primary and shadow lattice
        if psi0 is None:
            # Initial density matrix
            # psi0 = |L0> x |L0> x |0s> x | 0s>
            self.psi0 = qt.tensor(self.L0, self.L0, self.ss0dm, self.ss0dm)
        else:
            self.psi0 = psi0


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


    def set_H(self):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Hp = -W Xl Xr + 1/2 d (Pl1 + Pr1)
        Hp = -self.W * self.Xl*self.Xr + 0.5*self.d*(self.Pl1 + self.Pr1)
        
        # Hs = (W + d/2) (asl^t asl + asr^t asr)
        Hs = (self.W + self.d/2.) * (self.asl.dag()*self.asl \
                + self.asr.dag()*self.asr)

        # Hps = O (apl^t asl^t + apr^t asr^t + h.c.)
        Hps = self.Om*(self.apl.dag()*self.asl.dag() \
                + self.apr.dag()*self.asr.dag())
        Hps += Hps.dag()
        
        # Time independent Hamiltonian is sum of all contributions
        H0 = Hp + Hs + Hps

        # Set the drive Hamiltonian
        Hc = [(self.asl + self.asl.dag()), (self.asr + self.asr.dag())]

        # Use a Gaussian pulse for now
        Hc_str = 'A * exp(-(t - t0)**2/(2*sig**2))*cos(w*t-ph) + dc'

        self.H = [H0, [[Hc[0], Hc_str], [Hc[1], Hc_str]]]


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
