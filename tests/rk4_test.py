#!/usr/bin/env python3
"""
Test the Runge Kutta solver with the harmonic oscillator
"""

from ode_solver import rk4
import numpy as np
import matplotlib.pyplot as plt


class sho(rk4):
    """
    Simple Harmonic Oscillator with RK4
    """

    def __init__(self, tpts, dt, yinit, w):
        """
        Call the base constructor
        """

        rk4.__init__(self, yinit, tpts, dt, w=w)


    def rhs(self, y, t):
        """
        Harmonic oscillator rhs
        """

        # ydot[0] = y[1], ydot[1] = -w^2 y[0]
        rhs_data = np.array([y[1], -self.w**2 * y[0]]) 

        return rhs_data


def test_sho():
    """
    """

    # Set the frequency of oscillation
    w = 2*np.pi*1;
    tpts = np.linspace(0, 20, 8001)
    dt = tpts.max() / (tpts.size)

    # Initialize the x and y as 1
    yinit = np.array([1, 1])

    # Run the code
    my_sho = sho(tpts, dt, yinit, w)
    res = np.asarray(my_sho.solver())

    plt.plot(tpts, res[:,0], label=r'x')
    plt.plot(tpts, res[:,1], label=r'y')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('figs/sho_demo.eps', format='eps')


if __name__ == '__main__':

    test_sho()
