#!/usr/bin/env python3
"""
Tests for the Backward Euler Integrator

"""

# Add the VSLQ path 
vslq_path = '/home/nmaterise/mines/research/VSLQ'
import sys
if vslq_path not in sys.path:
    sys.path.append(vslq_path)

from ode_solver import bkeuler
import numpy as np
import matplotlib.pyplot as plt


class sho_bke(bkeuler):
    """
    Simple Harmonic Oscillator in 1D to test Backward Euler
    """
    
    def __init__(self, y0, tpts, dt, is_A_const=True, w=1, m=1):
        """
        Class constructor
        """

        # Call the base class constructor
        bkeuler.__init__(self, y0, tpts, dt, is_A_const, w=w, m=m)


    def rhs_A(self, tpts):
        """
        User defined computation of the right hand side matrix A
        """

        # Compute the rhs matrix A
        A = np.array([[0, 1/self.m], [-self.m*self.w**2, 0]])

        return A


def test_sho_bke():
    """
    Test the above class with simple initial conditions
    """

    # Set the frequency of oscillation
    w = 2*np.pi*1;
    tpts = np.linspace(0, 10, 1001)
    dt = tpts.max() / (tpts.size)

    # Initialize the x and y as 1
    yinit = np.array([[1], [1]])
    print('yinit.shape: {}'.format(yinit.shape))

    # Run the code
    my_sho_bke = sho_bke(yinit, tpts, dt, is_A_const=True, w=w, m=1)
    res = np.asarray(my_sho_bke.solver())

    print('res.shape: {}'.format(res.shape))
    
    # Plot the results
    plt.plot(tpts, res[:,0], label=r'x')
    plt.plot(tpts, res[:,1], label=r'y')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/sho_bke_demo.pdf', format='pdf')


if __name__ == '__main__':

    test_sho_bke()
