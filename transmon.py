#!/usr/bin/env python
"""
Dispersive readout of a transmon qubit Hamiltonian class
"""

import numpy as np
import post_proc_tools as ppt
import matplotlib.pyplot as plt
from qubit_cavity import base_cqed, base_cqed_mops
import matrix_ops as mops
import drive_tools as dts


class transmon_disp_mops(base_cqed_mops):
    """
    Implements the cavity-transmon interaction in the dispersive regime
    """

    def __init__(self, Nq, Nc, tpts, psi0=None,
                 gamma1=0., kappa=0.1, g=0.05):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed_mops.__init__(self, tpts=tpts, Nq=Nq, Nc=Nc, psi0=psi0,
                           gamma1=gamma1, kappa=kappa, g=g)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.kappa], [self.ac])
        self.set_init_state(psi0)

    def get_drive(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            A, t0, sig = args[0]
        else:
            A, t0, sig = args

        return A * np.exp(-(tpts - t0)**2 / (2*sig)**2)


    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = mops.destroy(self.Nq)

        if self.Nq > 2:
            self.at = mops.tensor(at0, np.eye(self.Nc))
        else:
            self.sz = mops.tensor(mops.sop('z'), np.eye(self.Nc))

            # Attempt to fix -0 terms
            ## Get the indices of the non-zeros
            zidx = set(list(np.flatnonzero(self.sz)))
    
            ## Get all of the indices, the take the union
            ## and subtract intersection
            allidx = set(list(range(0, self.sz.size)))
            szflat = self.sz.flatten()
            iunionidx = list(allidx.symmetric_difference(zidx))
            
            ## Overwrite the -0 values with abs(0)
            szflat[iunionidx] = np.abs(szflat[iunionidx])
            self.sz = szflat.reshape(self.sz.shape)

        # Set the cavity operators
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(np.eye(self.Nq), ac0)
        

    def set_H(self, tpts, args):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Time independent Hamiltonian
        # From Didier et al. supplemental section
        # H0 = np.zeros(self.ac.shape, dtype=np.complex128)
        # H0 = self.g * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        if self.Nq > 2:
            H0 = self.g * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        else:
            H0 = self.g * mops.dag(self.ac)@self.ac @ self.sz 

        # Time dependent readout Hamiltonian
        Hc = (self.ac + mops.dag(self.ac))
        # Hc = self.chi * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at
        Hd = self.get_drive(tpts, args)
        self.H = [H0, [Hc, Hd]]


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = mops.tensor(mops.basis(self.Nq, 0), mops.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = mops.expect(self.ac, psif)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = mops.expect(mops.dag(self.at) @ self.at, psif)

        return n_expect


class transmon_long_mops(base_cqed_mops):
    """
    Implements the cavity-transmon longitudinal interaction
    """

    def __init__(self, Nq, Nc, tpts, psi0=None,
                 gamma1=0., kappa=0.1, g=0.05, phi=0):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed_mops.__init__(self, tpts=tpts, Nq=Nq, Nc=Nc, psi0=psi0,
                           gamma1=gamma1, kappa=kappa, g=g, phi=phi)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.kappa], [self.ac])
        self.set_init_state(psi0)

    def get_drive(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            A, t0, sig = args[0]
        else:
            A, t0, sig = args

        return A * np.exp(-(tpts - t0)**2 / (2*sig)**2)


    def get_drive_tanh(self, tpts, args):
        """
        Returns a Gaussian signal centered at t0, with width, sig
        """
    
        # Unpack arguments to compute the drive signal
        if len(args) == 1:
            x1, x2, a, b = args[0]
        else:
            x1, x2, a, b = args

        return dts.get_tanh_env(tpts, x1, x2, a, b)


    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = mops.destroy(self.Nq)
    
        if self.Nq > 2:
            self.at = mops.tensor(at0, np.eye(self.Nc))
        else:
            self.sz = mops.tensor(mops.sop('z'), np.eye(self.Nc))

            # Attempt to fix -0 terms
            ## Get the indices of the non-zeros
            zidx = set(list(np.flatnonzero(self.sz)))
    
            ## Get all of the indices, the take the union
            ## and subtract intersection
            allidx = set(list(range(0, self.sz.size)))
            szflat = self.sz.flatten()
            iunionidx = list(allidx.symmetric_difference(zidx))
            
            ## Overwrite the -0 values with abs(0)
            szflat[iunionidx] = np.abs(szflat[iunionidx])
            self.sz = szflat.reshape(self.sz.shape)

        # Set the cavity operators
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(np.eye(self.Nq), ac0)
        

    def set_H(self, tpts, args):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Set the time independent Hamiltonian based on 2-level or 3-level
        # approximation of the transmon
        if self.Nq > 2:
            H0 = self.g * self.at @ (mops.dag(self.ac) + self.ac)
        else:
            H0 = self.sz @ (mops.dag(self.ac)*self.g \
                    + self.ac*np.conj(self.g))

        # Time independent readout Hamiltonian
        self.H = H0


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system,
        if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = mops.tensor(mops.basis(self.Nq, 0),
                              mops.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = mops.expect(self.ac, psif)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = mops.expect(mops.dag(self.at) @ self.at, psif)

        return n_expect


def test_transmon():
    """
    Testing function for the above class using a simple example
    of a three level transmon coupled to a cavity with 16 levels
    """

    # 16 levels in the cavity, 3 in the transmon
    Nc = 16; Nq = 3;

    # 250 MHz anharmonicity, 50 MHz self-Kerr
    alpha = 0.250*2*np.pi;      self_kerr = 0.05*2*np.pi;
    chi = np.sqrt(2*alpha*self_kerr)

    # Compute the coupling factor g from chi = g^2 / delta
    # Use a delta of 2.5 GHz
    wq = 5*2*np.pi; 
    
    # Set the cavity linewidth and the transmon T1
    # T1 = 40 us, kappa = 125 kHz
    T1 = 40e3; gamma1 = 1./T1; kappa = 0.000125;
    T1 = 0.; gamma1 = 0.; kappa = 0.1;
    
    # Set the initial state
    psi00 = mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0))
    psi01 = mops.tensor(mops.basis(Nq, 1), mops.basis(Nc, 0))
    
    # Create an instance of the class
    my_tmon = transmon_disp(alpha, self_kerr, wq, Nq,
                            Nc, psi00, g=np.exp(1j*2),
                            gamma1=gamma1, kappa=kappa)

    # Set the time of the simulation in ns
    tpts = np.linspace(0, 10./kappa, 1001)

    # Set the drive parameters for the readout
    t0 = 3. / (2*kappa); 
    sig = 1. / (6*kappa); 
    w = 0.; beta = 0.01;
    args = my_tmon.get_cy_window_dict(t0, sig, w, beta) 

    # Run the mesolver
    res = my_tmon.run_dynamics(tpts, args,
            dt=tpts.max()/(10*tpts.size))
    aavg = my_tmon.get_a_expect(res)
    # xvec, Wi = ppt.get_wigner(res.states[0])
    # xvec, Wf = ppt.get_wigner(res.states[-1])

    # ppt.plot_wigner(xvec, Wi, tstr='Initial State')
    # ppt.plot_wigner(xvec, Wi-Wf, tstr='Initial-Final State')

    # navg = my_tmon.get_n_expect(res)

    # ppt.plot_expect(tpts, aavg, 'a', file_ext='a')
    # ppt.plot_expect(tpts, navg, 'n', file_ext='n')
    plt.plot(aavg.real, aavg.imag)


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
