#!/usr/bin/env python3
"""
Tests for the Implicit Midpoint Integrator

"""

# Add the VSLQ path 
vslq_path = '/home/nmaterise/mines/research/VSLQ'
import sys
if vslq_path not in sys.path:
    sys.path.append(vslq_path)
from scipy.optimize import curve_fit
from ode_solver import implicitmdpt
import numpy as np
import matplotlib.pyplot as plt


class exp_impmp(implicitmdpt):
    """
    Exponential decay, Implicit Midpoint method
    """

    def __init__(self, y0, tpts, dt, is_A_const=True, lam=[1., 2.]):
        """
        Class constructor
        """
        
        # Call the base class constructor
        implicitmdpt.__init__(self, y0, tpts, dt, is_A_const, lam=lam)


    def rhs_A(self, tpts):
        """
        Right hand side for a diagonal (uncoupled) system
        """
        
        A = np.array([[self.lam[0], 0], [0, self.lam[1]]])

        return A


class sho_impmp(implicitmdpt):
    """
    Simple Harmonic Oscillator in 1D to test Implicit Midpoint
    """
    
    def __init__(self, y0, tpts, dt, is_A_const=True, w=1, m=1):
        """
        Class constructor
        """

        # Call the base class constructor
        implicitmdpt.__init__(self, y0, tpts, dt, is_A_const, w=w, m=m)


    def rhs_A(self, tpts):
        """
        User defined computation of the right hand side matrix A
        """

        # Compute the rhs matrix A
        A = np.array([[0, 1/self.m], [-self.m*self.w**2, 0]])

        return A


class sho_damped_impmp(implicitmdpt):
    """
    Damped harmonic oscillator in 1D to test Implicit Midpoint
    """
    
    def __init__(self, y0, tpts, dt, is_A_const=True, w=1, m=1, gamma=1):
        """
        Class constructor
        """

        # Call the base class constructor
        implicitmdpt.__init__(self, y0, tpts, dt, 
                              is_A_const, w=w, m=m, gamma=gamma)


    def rhs_A(self, tpts):
        """
        User defined computation of the right hand side matrix A
        """

        # Compute the rhs matrix A
        A = np.array([[-2*self.gamma/self.m, 1/self.m], [-self.m*self.w**2, 0]])

        return A


class sho_damped_driven_impmp(implicitmdpt):
    """
    Damped harmonic oscillator in 1D to test implicit midpoint
    """
    
    def __init__(self, y0, tpts, dt, is_A_const=False,
                 w=1, m=1, gamma=1, wd=2, f=np.sin):
        """
        Class constructor
        """

        # Call the base class constructor
        implicitmdpt.__init__(self, y0, tpts, dt, 
                              is_A_const, w=w, m=m,
                              gamma=gamma, wd=wd, f=f)


    def rhs_A(self, t):
        """
        User defined computation of the right hand side matrix A
        """

        # Compute the rhs matrix A
        A = np.array([[-2*self.gamma/self.m, 1/self.m], 
                      [-self.m*self.f(t, self.wd)*self.w**2, 0]])

        return A


def l2error(x1, x2):
    """
    Compute the l2 error between two arrays
    """
    
    return np.sqrt(sum([(a - b)**2 for a, b in zip(x1, x2)]))


def test_exp_impmp():
    """
    Test the above class with simple initial conditions
    """

    # Set the frequency of oscillation
    lam = [1., 2.]
    tpts = np.linspace(0, 10, 101)
    dt = tpts.max() / (tpts.size)

    print('Exponential decay model')

    # Initialize the x and y as 1
    yinit = np.array([[1], [1]])

    # Run the code
    my_exp_impmp = exp_impmp(yinit, tpts, dt,
                         is_A_const=True,
                         lam=lam)
    res = np.asarray(my_exp_impmp.solver())
    
    # Fit the data to an exponentially damped sinusoid
    def fit_fun(x, a, b, c):
        fout = a * np.exp(-b*x) + c
        return fout

    # Get the fitting parameters
    q = res[:, 0]; p = res[:, 1];
    qopt, qcov = curve_fit(fit_fun, tpts, q.ravel(), maxfev=10000)
    popt, pcov = curve_fit(fit_fun, tpts, p.ravel(), maxfev=10000)

    print('dt: %g' % (dt))
    print('lam - dt: {}'.format(lam - dt))
    print('Optimized parameters:\n\nqopt:\n{}\npopt:\n{}\n'\
            .format(qopt, popt))

    # Fitting error
    print('Fitting rms error: %g\n' % l2error(fit_fun(tpts, *qopt), q.ravel()))
    
    # Plot the results
    df = 10
    plt.plot(tpts[0::df], res[:,0][0::df], 'bo', label=r'$y_0$')
    plt.plot(tpts[0::df], res[:,1][0::df], 'ro', label=r'$y_1$')
    plt.plot(tpts, fit_fun(tpts, *qopt), 'b-', label=r'$y_0$-fit')
    plt.plot(tpts, fit_fun(tpts, *popt), 'r-', label=r'$y_1$-fit')
    plt.legend(loc='best')
    plt.savefig('figs/exp_impmp_demo.pdf', format='pdf')


def test_sho_impmp():
    """
    Test the above class with simple initial conditions
    """

    # Set the frequency of oscillation
    w = 1; m = 1;
    tpts = 2*np.pi*np.linspace(0, 10, 101)
    dt = tpts.max() / (tpts.size)

    # Initialize the x and y as 1
    yinit = np.array([[0], [1]])
    print('yinit.shape: {}'.format(yinit.shape))

    # Run the code
    my_sho_impmp = sho_impmp(yinit, tpts, dt, is_A_const=True, w=w, m=m)
    res = np.asarray(my_sho_impmp.solver())


    # Fit the data to an exponentially damped sinusoid
    def fit_fun(x, a, b, c, d):
        fout = a * np.sin(b*x + c) + d 
        return fout

    # Get the fitting parameters
    q = res[:, 0]; p = res[:, 1];
    qopt, qcov = curve_fit(fit_fun, tpts, q.ravel(), maxfev=10000)
    popt, pcov = curve_fit(fit_fun, tpts, p.ravel(), maxfev=10000)

    print('Optimized parameters:\n\nqopt:\n{}\npopt:\n{}\n'\
            .format(qopt, popt))

    # Fitting error
    print('Fitting rms error: %g\n' % l2error(fit_fun(tpts, *qopt), q.ravel()))
    
    # Plot the results
    df = 10
    plt.plot(tpts[0::df], res[:,0][0::df], 'bo', label=r'q')
    plt.plot(tpts[0::df], res[:,1][0::df], 'ro', label=r'p')
    plt.plot(tpts, fit_fun(tpts, *qopt), 'b-', label=r'q-fit')
    plt.plot(tpts, fit_fun(tpts, *popt), 'r-', label=r'p-fit')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/sho_impmp_demo.pdf', format='pdf')

    plt.figure(2)
    plt.plot(tpts, 0.5*m*w**2*q**2 + 0.5*p**2/m)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.savefig('figs/sho_impmp_energy_cons_demo.pdf', format='pdf')


def test_sho_damped_impmp():
    """
    Test the above class with simple initial conditions
    """

    # Set the frequency of oscillation
    w = 1; gamma = w / 100;
    tpts = 2*np.pi*np.linspace(0, 10, 10001)
    dt = tpts.max() / (tpts.size)

    print('\nDamped harmonic oscillator:\n\nw: %g\ngamma: %g\n'\
            % (w, gamma))

    # Initialize the x and y as 1
    yinit = np.array([[0], [1]])

    # Run the code
    my_sho_damped_impmp = sho_damped_impmp(yinit, tpts, dt,
                         is_A_const=True,
                         w=w, m=1, gamma=gamma)
    res = np.asarray(my_sho_damped_impmp.solver())
    
    # Fit the data to an exponentially damped sinusoid
    def fit_fun(x, a, b, c, d, e):
        fout = a * np.exp(-b*x) * np.sin(c*x + d) + e
        return fout

    # Get the fitting parameters
    qopt0 = [1., gamma+dt, w, 0., 0.]
    popt0 = [1., gamma+dt, w, np.pi/2, 0.]
    q = res[:, 0]; p = res[:, 1];
    qopt, qcov = curve_fit(fit_fun, tpts, q.ravel(), 
                            p0=qopt0, maxfev=10000)
    popt, pcov = curve_fit(fit_fun, tpts, p.ravel(),
                            p0=popt0, maxfev=10000)

    print('2*gamma: %g' % (2*gamma))
    print('dt: %g' % (dt))
    print('dt - 2*gamma: %e' % (dt - 2*gamma))

    print('Optimized parameters:\n\nqopt:\n{}\npopt:\n{}\n'\
            .format(qopt, popt))
    
    # Plot the results
    df = 10
    plt.plot(tpts[0::df], res[:,0][0::df], 'bo', label=r'q')
    plt.plot(tpts[0::df], res[:,1][0::df], 'ro', label=r'p')
    plt.plot(tpts, fit_fun(tpts, *qopt), 'b-', label=r'q-fit')
    plt.plot(tpts, fit_fun(tpts, *popt), 'r-', label=r'p-fit')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/sho_damped_impmp_demo.pdf', format='pdf')


def test_sho_damped_driven_impmp():
    """
    Test the above class with simple initial conditions
    """

    # Set the frequency of oscillation
    w = 1; gamma = w / 100; wd = 2*w; m = 1;
    tpts = 2*np.pi*np.linspace(0, 20, 2001)
    dt = tpts.max() / (tpts.size)

    print('\nDamped, driven harmonic oscillator:\n\nw: %g\ngamma: %g\n'\
            % (w, gamma))

    # Initialize the x and y as 1
    yinit = np.array([[0], [1]])

    def f(t, wdd):
        return np.sin(wdd*t)

    # Run the code
    my_sho_damped_driven_impmp = sho_damped_driven_impmp(yinit, tpts, dt,
                         is_A_const=True,
                         w=w, m=m, gamma=gamma, wd=wd, f=f)
    res = np.asarray(my_sho_damped_driven_impmp.solver())
    

    print('2*gamma: %g' % (2*gamma))
    print('dt: %g' % (dt))
    print('dt - 2*gamma: %e' % (dt - 2*gamma))

    
    # Fit the data to an exponentially damped sinusoid
    def fit_fun(x, a, b, c, d, e):
        fout = a * np.exp(-b*x) * np.sin(c*x + d) + e
        return fout

    # Get the fitting parameters
    qopt0 = [1., gamma+dt, w, 0., 0.]
    popt0 = [1., gamma+dt, w, np.pi/2, 0.]
    q = res[:, 0]; p = res[:, 1];
    qopt, qcov = curve_fit(fit_fun, tpts, q.ravel(), 
                            p0=qopt0, maxfev=10000)
    popt, pcov = curve_fit(fit_fun, tpts, p.ravel(),
                            p0=popt0, maxfev=10000)

    print('Optimized parameters:\n\nqopt:\n{}\npopt:\n{}\n'\
            .format(qopt, popt))
    
    # Plot the results
    df = 10
    plt.plot(tpts[0::df], res[:,0][0::df], 'bo', label=r'q')
    plt.plot(tpts[0::df], res[:,1][0::df], 'ro', label=r'p')
    plt.plot(tpts, fit_fun(tpts, *qopt), 'b-', label=r'q-fit')
    plt.plot(tpts, fit_fun(tpts, *popt), 'r-', label=r'p-fit')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('figs/sho_damped_driven_impmp_demo.pdf', format='pdf')



if __name__ == '__main__':

    test_sho_impmp()
    # test_sho_damped_impmp()
    # test_exp_impmp()
    # test_sho_damped_driven_impmp()
