#!/usr/bin/env python
"""
Dispersive readout of a transmon qubit Hamiltonian class
"""

import qutip as qt
import numpy as np

class transmon_disp:
    """
    Implements the cavity-transmon interaction in the dispersive regime
    """

    def __init__(self, alpha, self_kerr, Nq, Nc):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        # self-Kerr, and cross-Kerr (chi)
        self.alpha = alpha; self.self_kerr = self_kerr;
        self.Nq    = Nq;    self.Nc        = Nc;
        self.chi = np.sqrt(2 * self_kerr * alpha)
    
    def __del__(self):
        """
        Class destructor
        """
        pass

    
    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """


    def set_H(self):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the transmon and cavity
        """

        self.H = 


