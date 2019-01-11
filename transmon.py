#!/usr/bin/env python
"""
Dispersive readout of a transmon qubit Hamiltonian class
"""

import qutip as qt
import numpy as np
import post_proc_tools as ppt
import matplotlib.pyplot as plt
from qubit_cavity import base_cqed, base_cqed_mops
import matrix_ops as mops


class transmon_disp(base_cqed):
    """
    Implements the cavity-transmon interaction in the dispersive regime
    """

    def __init__(self, alpha, self_kerr, wq,
                 Nq, Nc, psi0=None, g=1.+0*1j,
                 gamma1=0., kappa=0.1, wc=1.):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed.__init__(self, alpha=alpha, self_kerr=self_kerr, 
                           wq=wq, Nq=Nq, Nc=Nc, psi0=psi0, g=g,
                           gamma1=gamma1, kappa=kappa,
                           chi=np.sqrt(2 * self_kerr * alpha), wc=wc)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.gamma1, self.kappa], [self.at, self.ac])
        self.set_init_state(psi0)
        self.set_H()
    

    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = qt.destroy(self.Nq)
        self.at = qt.tensor(at0, qt.qeye(self.Nc))

        # Set the cavity operators
        ac0 = qt.destroy(self.Nc)
        self.ac = qt.tensor(qt.qeye(self.Nq), ac0)
        

    def set_H(self):
        """
        Sets the dispersive Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Time independent Hamiltonian
        # From Didier et al. supplemental section
        H0 = self.wq*self.at.dag()*self.at + self.wc*self.ac.dag()*self.ac\
             + self.chi * self.ac.dag()*self.ac * self.at.dag()*self.at
        # H0 = - self.chi * self.ac.dag()*self.ac * self.at.dag()*self.at
        
        # Simply just g * (b^t a + b a^t)
        # H0 = self.g * (self.at.dag()*self.ac + self.at*self.ac.dag())
        # From Didier et al. supplemental section
        # H0 = self.g*self.ac.dag()*self.at \
        #     + self.g.conjugate()*self.ac*self.at.dag()

        # Time dependent readout Hamiltonian
        Hc = (self.ac + self.ac.dag())
        Hc_str = 'A * exp(-(t - t0)**2/(2*sig**2))*cos(w*t-ph) + dc'

        self.H = [H0, [Hc, Hc_str]]


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = qt.tensor(qt.basis(self.Nq, 0), qt.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = qt.expect(self.ac, psif.states)

        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = qt.expect(self.at.dag()*self.at, psif.states)

        return n_expect


class transmon_long(base_cqed):
    """
    Implements the cavity-transmon interaction in the longitudinal scheme 
    """

    def __init__(self, alpha, self_kerr,
                 wq, Nq, Nc, psi0=None,
                 g=1.+0*1j, gamma1=0, kappa=0.1):
        """
        Class constructor
        """

        # Initialize base class
        base_cqed.__init__(self, alpha=alpha, self_kerr=self_kerr,
                            wq=wq, Nq=Nq, Nc=Nc, psi0=psi0, g=g,
                            gamma1=gamma1, kappa=kappa)

        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.gamma1, self.kappa], [self.at, self.ac])
        self.set_init_state(psi0)
        self.set_H()
    

    def set_ops(self):
        """
        Set the operators needed to construct the Hamiltonian and
        compute expectation values
        """

        # Set the transmon operators
        at0 = qt.destroy(self.Nq)
        self.at = qt.tensor(at0, qt.qeye(self.Nc))

        # Set the cavity operators
        ac0 = qt.destroy(self.Nc)
        self.ac = qt.tensor(qt.qeye(self.Nq), ac0)
        

    def set_H(self):
        """
        Sets the intraction Hamiltonian for a transmon coupled to a cavity
        in the rotating frame of the cavity
        """

        # Time independent Hamiltonian
        # From Didier et al. supplemental section
        H0 = (self.g*self.ac.dag() \
            + self.g.conjugate()*self.ac) * self.at.dag()*self.at

        # Time dependent readout Hamiltonian
        Hc = (self.ac + self.ac.dag())
        Hc_str = 'A * exp(-(t - t0)**2/(2*sig**2))*cos(w*t-ph) + dc'

        self.H = [H0, [Hc, Hc_str]]


    def set_init_state(self, psi0=None):
        """
        Sets the initial state of the system, if None set to the ground state of
        the qubits and the cavity
        """

        # Set the state psi0
        psi_gnd = qt.tensor(qt.basis(self.Nq, 0), qt.basis(self.Nc, 0))
        self.psi0 = psi0 if (psi0 is not None) else psi_gnd


    def get_a_expect(self, psif):
        """
        Compute the expectation value of the a operator for the cavity
        """

        # Compute the expectation value and return it
        a_expect = qt.expect(self.ac, psif.states)


        return a_expect


    def get_n_expect(self, psif):
        """
        Compute the expectation value of the number operator for the transmon
        """

        # Compute the expectation value and return it
        n_expect = qt.expect(self.at.dag()*self.at, psif.states)

        
        return n_expect


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
        self.set_cops([self.gamma1, self.kappa], [self.at, self.ac])
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
        self.at = mops.tensor(at0, np.eye(self.Nc))

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
        H0 = self.g * mops.dag(self.ac)@self.ac @ mops.dag(self.at)@self.at

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
                 gamma1=0., kappa=0.1, g=0.05):
        """
        Class constructor
        """

        # Set the class members for the anharmonicity (alpha),
        base_cqed_mops.__init__(self, tpts=tpts, Nq=Nq, Nc=Nc, psi0=psi0,
                           gamma1=gamma1, kappa=kappa, g=g)
    
        # Initialize the collapse operators as None
        self.set_ops()
        self.set_cops([self.gamma1, self.kappa], [self.at, self.ac])
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
        self.at = mops.tensor(at0, np.eye(self.Nc))

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
        H0 = (np.conj(self.g) * self.ac + self.g * mops.dag(self.ac)) \
             @ mops.dag(self.at) @ self.at

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
    psi00 = qt.tensor(qt.basis(Nq, 0), qt.basis(Nc, 0))
    psi01 = qt.tensor(qt.basis(Nq, 1), qt.basis(Nc, 0))
    
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
    res = my_tmon.run_dynamics(tpts, args)
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
    Nq = 3;
    a = mops.tensor(np.eye(Nq), mops.destroy(Nc))
    b = mops.destroy(Nq); bd = mops.dag(b)
    sz = mops.tensor(bd@b, np.eye(Nc))
    wc = 5; wq = 6;
    gamma1=1/40.; kappa = 0.1; chi = kappa / 2.;
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))

    # Time independent Hamiltonian
    # H0 = wc*mops.dag(a)@a + wq*sz/2. + chi*mops.dag(a)@a@sz
    # In rotating frame
    H0 = chi*mops.dag(a) @ a @ sz
    
    # Time dependent Hamiltonian
    Hc = (a + mops.dag(a))
    # Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2) * np.sin(wc*tpts)
    # In rotating frame
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [np.sqrt(kappa) * a]
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0)))
    print('rho0:\n{}'.format(rho0))
    print('Time = [%g, %g] ns' % (tpts.min(), tpts.max()))


    # Create an instance of the class
    my_tmon = transmon_disp_mops(Nq, Nc, tpts, psi0=rho0, chi=chi,
            gamma1=gamma1, kappa=kappa)

    # Run the dynamics
    ## A, t0, sig
    args = [1, tpts.max()/2, tpts.max()/12]
    rho_out = my_tmon.run_dynamics(tpts, args)    

    # Compute the expectation value of a^t a
    a_avg = my_tmon.get_a_expect(rho_out)

    # Plot the results
    plt.plot(kappa*tpts, a_avg.real,label=r'$\Re \langle a\rangle$')
    plt.plot(kappa*tpts, a_avg.imag,label=r'$\Im \langle a\rangle$')
    plt.xlabel(r'Time (1/$\kappa$)')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(a_avg.real, a_avg.imag)


if __name__ == '__main__':
    
    # Run the test function above
    test_transmon_mops()
