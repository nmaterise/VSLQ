#!/usr/bin/env python3
"""
Input / output equations of motion following  these papers
Didier et al., PRL 115, 203601 (2015)
Clerk et al., arxiv:0810.4729v2 (2010) 
Gardiner & Zoller, Quantum Noise (2014)
"""

import numpy as np
import post_proc_tools as ppt


def alpha_disp_cav(t, eps, chi, kappa, sigz):
    """
    Returns the value of alpha(t) = <a(t)> for the interaction Hamiltonian

    Hint = chi a^t a b^t b,
    
    where a operates on the cavity and b operates on the qubit
    
    Parameters:
    ----------

    t:      time (scalar) 
    eps:    epsilon, drive strength
    chi:    dispersive interaction strength
    kappa:  decay rate of the cavity
    sigz:   expectation value of sigma_z, <sigma_z> (+/- 1/2) 


    """

    # Compute the mean input field
    ain = -eps / np.sqrt(kappa)
    
    # Calculate the qubit state-dependent phase shift
    phiqb = 2 * np.arctan(2*chi/kappa)

    # Compute the mean output field
    aout = eps/np.sqrt(kappa) * np.exp(-1j*phiqb*sigz) \
            * (1 - 2 * np.cos(phiqb/2) * np.exp(-(1j*chi*sigz+kappa/2)*t) \
            * np.exp(1j*phiqb*sigz/2)) 

    # Use aout = ain + sqrt(k) a to get a
    a = (aout - ain) / np.sqrt(kappa)
    
    return a


def alpha_long_cav(t, g, kappa, sigz):
    """
    Returns the value of alpha(t) = <a(t)> for the interaction Hamiltonian

    Hint = (g_z a^t g_z* a) b^t b,
    
    where a operates on the cavity and b operates on the qubit
    
    Parameters:
    ----------

    t:      time (scalar) 
    g:      longitudinal interaction strength
    kappa:  decay rate of the cavity
    sigz:   expectation value of sigma_z, <sigma_z> (+/- 1/2) 


    """


    return -1j * g * sigz * (1 - np.exp(-kappa * t / 2)) / kappa


def test_io_eoms():
    """
    Compute the values of <a> for the dispersive and longitudinal cases
    """

    # Set the parameters for the simulations
    kappa = 0.1; chi = kappa / 2; g = np.sqrt(chi); # <-- Delta = 1

    # Run the time interval t = [0, 10/kappa]
    # tpts = np.logspace(-2, 3, 6, base=2) / kappa
    # tpts = np.hstack((0, tpts))
    tpts = np.linspace(0, 8, 100) / kappa

    # Unit drive strength
    eps = 1
    
    # Compute the dispersive case first for sigma_z = -1/2
    alpha_disp_g = alpha_disp_cav(tpts, eps, chi, kappa, -1)
    alpha_disp_e = alpha_disp_cav(tpts, eps, chi, kappa, 1)

    # Plot the results
    # ppt.plot_io_a(tpts, alpha_disp_g, alpha_disp_e, 
    # 20*chi, kappa, fext='disp')

    # Compute the longitudinal case
    alpha_long_g = alpha_long_cav(tpts, g, kappa, -1)
    alpha_long_e = alpha_long_cav(tpts, g, kappa, 1)
    
    # Plot the results
    # ppt.plot_io_a(tpts, alpha_long_g, alph_long_e, g, kappa, fext='long')

    # Plot on the same figure
    ppt.plot_io_a_full(tpts, alpha_disp_g, alpha_disp_e,
                       alpha_long_g, alpha_long_e,
                       g, 20*chi, kappa, fext='')
    

if __name__ == '__main__':

    # Run the above test, including the plots
    test_io_eoms()

