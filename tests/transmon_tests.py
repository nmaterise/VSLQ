#!/usr/bin/env python3
"""
Tests of the transmon matrix_ops classes

"""

# Add the VSLQ path 
from test_utils import set_path
set_path()

import post_proc_tools as ppt
import matplotlib.pyplot as plt
import matrix_ops as mops
from transmon import transmon_disp_mops, transmon_long_mops


def test_transmon_mops():
    """
    Tests the transmon_disp_mops class
    """

    # Setup a basic cavity system
    Nc = 16;
    Nq = 2;
    wc = 5; wq = 6;
    gamma1=1/40.; kappa = 0.1; chi = kappa / 2.;
    g = np.sqrt(chi) # 10*kappa 
    gk = g / kappa
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))

    # Form the total Hamiltonian and set the collapse operators
    print('Time = [%g, %g] ns' % (tpts.min(), tpts.max()))

    # Initial density matrices as outer products of ground and excited many body
    # states of the qubit and the cavity
    psi_e0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 1), mops.basis(Nc, 0)))
    psi_g0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0))) 

    # Run the dynamics
    ## A, t0, sig
    args = [1, tpts.max()/2, tpts.max()/12]
    #         gamma1=gamma1, kappa=kappa)
    ## Play with the phase of the a, a^t operators
    phi = 0
    phi2 = 0 #np.exp(1j*2.1)
    # my_tmon = transmon_long_mops(Nq, Nc, tpts,
    #         psi0=psi_g0, gamma1=0, kappa=kappa, g=g*phi2, phi=phi)
    # rho_g = my_tmon.run_dynamics(tpts, args,
    #         dt=tpts.max()/(10*tpts.size))
    # a_g = my_tmon.get_a_expect(rho_g)

    # Compute the expectation value of a^t a
    my_tmon = transmon_long_mops(Nq, Nc, tpts,
            psi0=psi_e0, gamma1=0, kappa=kappa, g=g*phi2, phi=phi)
    rho_e = my_tmon.run_dynamics(tpts, args,
            dt=tpts.max()/(10*tpts.size))
    a_e = my_tmon.get_a_expect(rho_e)
    # n_avg = mops.expect(sz, rho_out)

    rhotr = mops.expect(np.eye(Nq*Nc), rho_e)

    plt.plot(tpts, rhotr.real, label=r'$\Re\mathrm{Tr}\rho$')
    plt.plot(tpts, rhotr.imag, label=r'$\Im\mathrm{Tr}\rho$')

    # Plot the results
    # plt.plot(kappa*tpts, a_g.real,label=r'$\Re \langle a\rangle$')
    # plt.plot(kappa*tpts, a_g.imag,label=r'$\Im \langle a\rangle$')

    # plt.plot(a_g.real / gk, a_g.imag / gk, 'b-', label=r'$\left| 0\right>$')
    # plt.plot(a_e.real / gk, a_e.imag / gk, 'r-', label=r'$\left| 1\right>$')
    # plt.xlabel(r'$\Re\left< a\right> / (g/\kappa)$', fontsize=20)
    # plt.ylabel(r'$\Im\left< a\right> / (g/\kappa)$', fontsize=20)
    plt.legend(loc='best')
    plt.tight_layout()

    # plt.plot(kappa*tpts, n_avg.real, label=r'$\Re\langle \sigma_z \rangle$')
    # plt.plot(kappa*tpts, n_avg.imag, label=r'$\Im\langle \sigma_z \rangle$')
    # plt.xlabel(r'Time (1/$\kappa$)')
    # plt.legend(loc='best')

    # plt.figure()


if __name__ == '__main__':
    
    # Run the test function above
    test_transmon_mops()
