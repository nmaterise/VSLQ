"""
Author: Nick Materise
Filename: vslq.py
Description: This class includes the basic features of a generalized VSLQ for
             N-logical circuits with Nq transmon levels and Ns shadow levels

Created: 180921
"""

import numpy as np
import matrix_ops as mops
from qubit_cavity import base_cqed, base_cqed_mops
import matplotlib.pyplot as plt
import pickle as pk


class vslq_mops(base_cqed_mops):
    """
    Very Small Logical Qubit (VSLQ) Class
    
    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np, tpts, W, d, Om,
                 gammap, gammas, readout_type='disp'):
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
        base_cqed_mops.__init__(self, tpts=tpts, Ns=Ns, Np=Np, W=W, d=d, Om=Om,
                gammap=gammap, gammas=gammas)

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()
        self.set_cops([self.gammas, self.gammas, self.gammap, self.gammap],
                      [self.asl, self.asr, self.apl, self.apr])
        self.set_init_state()


    def set_states(self):
        """
        Sets the basis states for the class
        """
    
        # States for the qubit / shadow degrees of freedom
        s0  = mops.basis(self.Np, 0);
        s1  = mops.basis(self.Np, 1); 
        s2  = mops.basis(self.Np, 2)
        ss0 = mops.basis(self.Ns, 0);
        ss1 = mops.basis(self.Ns, 1)
        
        # Compute the density matrices corresponding to the states
        ## These correspond to the projectors |n_k > < n_k|
        self.s1dm  = mops.ket2dm(s1); 
        self.s2dm  = mops.ket2dm(s2); 
        self.ss0dm = mops.ket2dm(ss0);
        self.ss1dm = mops.ket2dm(ss1)

        # Define the logical states
        self.L0 = mops.ket2dm((s2 + s0) / np.sqrt(2))
        self.L1 = mops.ket2dm((s2 - s0) / np.sqrt(2))


    def set_init_state(self, logical_state='L0'):
        """
        Set the initial state
        """
    
        # Initialize to a 0-logical state in the primary and shadow lattice
        if logical_state == 'L0':
            # Initial density matrix
            # psi0 = |L0> x |L0> x |0s> x | 0s>
            self.psi0 = mops.tensor(self.L0, self.L0, self.ss0dm, self.ss0dm)
        elif logical_state == 'L1':
            self.psi0 = mops.tensor(self.L1, self.L1, self.ss0dm, self.ss0dm)
        else:
            self.psi0 = psi0


    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Is = np.eye(self.Ns)
        self.Ip = np.eye(self.Np)

        # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
        self.Pl1 = mops.tensor(self.s1dm, self.Ip, self.Is, self.Is)
        self.Pr1 = mops.tensor(self.Ip, self.s1dm, self.Is, self.Is)

        # Destruction operators
        ## Primary qubits
        ap0 = mops.destroy(self.Np)
        self.apl = mops.tensor(ap0, self.Ip, self.Is, self.Is)
        self.apr = mops.tensor(self.Ip, ap0, self.Is, self.Is)
        
        ## Shadow resonators
        as0 = mops.destroy(self.Ns)
        self.asl = mops.tensor(self.Ip, self.Ip, as0, self.Is)
        self.asr = mops.tensor(self.Ip, self.Ip, self.Is, as0)

        ## Two photon operators on the logical manifold
        self.Xl = (self.apl@self.apl \
                + mops.dag(self.apl)@mops.dag(self.apl)) / np.sqrt(2)
        self.Xr = (self.apr@self.apr \
                + mops.dag(self.apr)@mops.dag(self.apr)) / np.sqrt(2)


    def set_H(self, tpts, args):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Hp = -W Xl Xr + 1/2 d (Pl1 + Pr1)
        Hp = -self.W * self.Xl@self.Xr + 0.5*self.d*(self.Pl1 + self.Pr1)
        
        # Hs = (W + d/2) (asl^t asl + asr^t asr)
        Hs = (self.W + self.d/2.) * (mops.dag(self.asl)@self.asl \
                + mops.dag(self.asr)@self.asr)

        # Hps = O (apl^t asl^t + apr^t asr^t + h.c.)
        Hps = self.Om*(mops.dag(self.apl)@mops.dag(self.asl) \
                + mops.dag(self.apr)@mops.dag(self.asr))
        Hps += mops.dag(Hps)
        
        # Time independent Hamiltonian is sum of all contributions
        self.H = Hp + Hs + Hps


    def get_logical_expect(self, psif):
        """
        Computes the expectation value of the logical operator, pL
        """

        # Projection operator to compute the expectation value
        self.pL = 0.5 * self.Xl * (1. + self.Xl*self.Xr) * (1. - self.Pl1) \
                  * (1 - self.Pr1)

        # Compute the expectation value using the states from the mesolve
        # solution, psif
        pL_expect = mops.expect(self.pL, psif.states)

        return pL_expect


class vslq_mops_readout(base_cqed_mops):
    """
    Very Small Logical Qubit (VSLQ) Class with additional DOF for readout
    
    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np, Nc, tpts, W, d, Om,
                 gammap, gammas, gl, gr):
        """
        Constructor for the class

        Parameters:
        ----------

        Ns:             number of levels in the shadow resonators
        Np:             number of levels in the primary qubits
        Nc:             number of levels in the readout cavity
        W:              primary qubit energy
        d:              anharmonicity of the primary qubits
        Om:             primary-shadow qubit interaction strength
        gammap/s:       loss rate for the primary / shadow qubits
        gl, gr:         readout strengths for left / right primary qubits
    
        """

        # Set the class members here
        base_cqed_mops.__init__(self, tpts=tpts, Ns=Ns, Np=Np, Nc=Nc,
                W=W, d=d, Om=Om, gammap=gammap, gammas=gammas,
                gl=gl, gr=gr)

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()
        self.set_cops([self.gammas, self.gammas, self.gammap, self.gammap],
                      [self.asl, self.asr, self.apl, self.apr])
        self.set_init_state()


    def set_states(self):
        """
        Sets the basis states for the class
        """
    
        # States for the qubit / shadow degrees of freedom
        self.s0  = mops.basis(self.Np, 0);
        self.s1  = mops.basis(self.Np, 1); 
        self.s2  = mops.basis(self.Np, 2);
        self.ss0 = mops.basis(self.Ns, 0);
        self.c0  = mops.basis(self.Nc, 0);
        
        # Compute the density matrices corresponding to the states
        ## These correspond to the projectors |n_k > < n_k|
        self.s1dm  = mops.ket2dm(self.s1); 
        self.s2dm  = mops.ket2dm(self.s2); 
        self.ss0dm = mops.ket2dm(self.ss0);
        self.c0dm  = mops.ket2dm(self.c0);

        # Define the logical states
        self.L0 = mops.ket2dm((self.s2 + self.s0) / np.sqrt(2))
        self.L1 = mops.ket2dm((self.s2 - self.s0) / np.sqrt(2))


    def set_init_state(self, logical_state='L0'):
        """
        Set the initial state
        """
    
        # Initialize to a 0-logical state in the primary and shadow lattice
        if logical_state == 'L0':
            # Initial density matrix
            # psi0 = |L0> x |L0> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.L0, self.L0, 
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        elif logical_state == 'L1':
            # psi0 = |L1> x |L1> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.L1, self.L1,
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        elif logical_state == 'l1L0':
            # psi0 = |1> x |L0> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.s1dm, self.L0,
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        elif logical_state == 'l1L1':
            # psi0 = |1> x |L1> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.s1dm, self.L1,
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        elif logical_state == 'r1L0':
            # psi0 = |L0> x |1> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.L0, self.s1dm,
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        elif logical_state == 'r1L1':
            # psi0 = |L1> x |1> x |0s> x |0s> |0c>
            self.psi0 = mops.tensor(self.L1, self.s1dm,
                                    self.ss0dm, self.ss0dm,
                                    self.c0dm)
        else:
            self.psi0 = psi0


    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Is = np.eye(self.Ns)
        self.Ip = np.eye(self.Np)
        self.Ic = np.eye(self.Nc)

        # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
        self.Pl1 = mops.tensor(self.s1dm, self.Ip, self.Is, self.Is, self.Ic)
        self.Pr1 = mops.tensor(self.Ip, self.s1dm, self.Is, self.Is, self.Ic)

        # Destruction operators
        ## Primary qubits
        ap0 = mops.destroy(self.Np)
        self.apl = mops.tensor(ap0, self.Ip, self.Is, self.Is, self.Ic)
        self.apr = mops.tensor(self.Ip, ap0, self.Is, self.Is, self.Ic)
        
        ## Shadow resonators
        as0 = mops.destroy(self.Ns)
        self.asl = mops.tensor(self.Ip, self.Ip, as0, self.Is, self.Ic)
        self.asr = mops.tensor(self.Ip, self.Ip, self.Is, as0, self.Ic)

        ## Cavity resonator
        ac0 = mops.destroy(self.Nc)
        self.ac = mops.tensor(self.Ip, self.Ip, self.Is, self.Is, ac0)

        ## Projectors |0><2| + |2><0| for left and right qubits
        P02 = np.outer(self.s0, self.s2) + np.outer(self.s2, self.s0)
        P02l = mops.tensor(P02, self.Ip, self.Is, self.Is, self.Ic)
        P02r = mops.tensor(self.Ip, P02, self.Is, self.Is, self.Ic)

        ## Projectors for |3> and |4> states of transmon
        P13 = np.outer(mops.basis(self.Np, 1), mops.basis(self.Np, 3))
        P04 = np.outer(mops.basis(self.Np, 0), mops.basis(self.Np, 4))
        P24 = np.outer(mops.basis(self.Np, 2), mops.basis(self.Np, 4))
        self.P13 = mops.tensor(P13, P13, self.Is, self.Is, self.Ic)
        self.P04 = mops.tensor(P04, P04, self.Is, self.Is, self.Ic)
        self.P24 = mops.tensor(P24, P24, self.Is, self.Is, self.Ic)

        ## Two photon operators on the logical manifold
        self.Xl = P02l
        self.Xr = P02r


    def set_H(self, tpts, args):
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Hp = -W Xl Xr + 1/2 d (Pl1 + Pr1)
        Hp = -self.W * self.Xl@self.Xr + 0.5*self.d*(self.Pl1 + self.Pr1)
        
        # Hs = (W + d/2) (asl^t asl + asr^t asr)
        Hs = (self.W + self.d/2.) * (mops.dag(self.asl)@self.asl \
                + mops.dag(self.asr)@self.asr)

        # Hps = O (apl^t asl^t + apr^t asr^t + h.c.)
        Hps = self.Om*(mops.dag(self.apl)@mops.dag(self.asl) \
                + mops.dag(self.apr)@mops.dag(self.asr))
        Hps += mops.dag(Hps)

        # Hpc = gl (ac + ac^t) Xl + gr (ac + ac^t) Xr
        Hpc = self.gl * (self.ac + mops.dag(self.ac)) @ self.Xl \
            + self.gr * (self.ac + mops.dag(self.ac)) @ self.Xr
        
        # Time independent Hamiltonian is sum of all contributions
        # Ignore the shadow / bath interaction for now
        self.H = Hp + Hs + Hpc


def test_vslq_dynamics():
    """
    Tests the dynamics of the VSLQ with no drive and initial states
    of logical 0 or 1
    """

    # Some example settings
    Np = 4; Ns = 2; Nc = 2;
    W = 70.0*np.pi; delta = 700.0*np.pi; Om = 5.5;
    gammap = 0; gammas = 0; #9.2;

    # Set the time array
    tpts = np.linspace(0, 2*np.pi / W, 1001)
    dt = tpts.max() / (10 * tpts.size)
    
    # Create an instance of the vslq class
    my_vslq = vslq_mops(Ns, Np, tpts, W, delta, Om, gammap, gammas)
    my_vslq.set_init_state(logical_state='L1')
    args = [1, tpts.max()/2, tpts.max()/12]
    rho_out = my_vslq.run_dynamics(tpts, args, dt=dt)

    # Get the expectation values for Xl and Xr
    Xl = mops.expect(my_vslq.Xl, rho_out)
    Xr = mops.expect(my_vslq.Xr, rho_out)

    # Plot the results
    plt.plot(tpts, Xl.real, label=r'$\Re\langle\widetilde{X}_l\rangle$')
    plt.plot(tpts, Xl.imag, label=r'$\Im\langle\widetilde{X}_l\rangle$')
    plt.plot(tpts, Xr.real, label=r'$\Re\langle\widetilde{X}_r\rangle$')
    plt.plot(tpts, Xr.imag, label=r'$\Im\langle\widetilde{X}_r\rangle$')
    plt.legend(loc='best')
    plt.xlabel(r'Time [$\mu$s]')


def test_vslq_readout_dynamics():
    """
    Tests the dynamics of the VSLQ with no drive and initial states
    of logical 0 or 1
    """

    # Some example settings
    Np = 5; Ns = 2; Nc = 5;
    W = 35*2*np.pi; delta = 350*2*np.pi; Om = 13.52;
    gammap = 0; gammas = 0; #9.2;

    # Set the time array
    ## Characteristic time of the shadow resonators
    TOm = 2*np.pi / Om
    tmax = 3*TOm 
    
    ## Time step 1/10 of largest energy scale
    Tdhalf = 4*np.pi / delta
    dt0 = Tdhalf / 10

    ## Number of points as N = tmax / dt + 1
    Ntpts = int(np.ceil(tmax / dt0)) + 1
    print('Running t=0 to %.2g us, %d points ...' % (tmax, Ntpts))
    tpts = np.linspace(0, tmax, Ntpts)
    dt = tpts.max() / (10 * tpts.size)

    # Readout strengths
    gl = W / 50; gr = gl;

    # Create an instance of the vslq class
    args = [1, tpts.max()/2, tpts.max()/12]

    ## Solve for | L0 > logical state
    my_vslq_0 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'l1L0' 
    my_vslq_0.set_init_state(logical_state=lstate)
    rho0 = my_vslq_0.run_dynamics(tpts, args, dt=dt)
    
    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho0, fid)
    fid.close()
    print('|1> |L0> result written to file.')

    ## Run for | L1 > state
    my_vslq_1 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'l1L1'
    my_vslq_1.set_init_state(logical_state=lstate)
    rho1 = my_vslq_1.run_dynamics(tpts, args, dt=dt)
    
    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho1, fid)
    fid.close()
    print('|1> |L1> result written to file.')

    ## Solve for | L0 > logical state
    my_vslq_0 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'L0' 
    my_vslq_0.set_init_state(logical_state=lstate)
    rho0 = my_vslq_0.run_dynamics(tpts, args, dt=dt)
    
    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho0, fid)
    fid.close()
    print('|L0> result written to file.')

    ## Run for | L1 > state
    my_vslq_1 = vslq_mops_readout(Ns, Np, Nc, tpts, W, delta, Om,
                 gammap, gammas, gl, gr)
    lstate = 'L1'
    my_vslq_1.set_init_state(logical_state=lstate)
    rho1 = my_vslq_1.run_dynamics(tpts, args, dt=dt)
    
    ## Write the result to file
    with open('data/rho_vslq_%s_%.2g_us.bin' \
            % (lstate, tmax), 'wb') as fid:
        pk.dump(rho1, fid)
    fid.close()
    print('|L1> result written to file.')


if __name__ == '__main__':
    
    # Test the dynamics of the vslq in different logical states
    # test_vslq_dynamics()
    test_vslq_readout_dynamics()
