"""
Author: Nick Materise
Filename: vslq.py
Description: This class includes the basic features of a generalized VSLQ for
             N-logical circuits with Nq transmon levels and Ns shadow levels

Created: 180921
"""

import numpy as np
import matrix_ops as mops
import scipy.sparse as scsp
from qubit_cavity import base_cqed_mops


class vslq_mops(base_cqed_mops):
    """
    Very Small Logical Qubit (VSLQ) Class

    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np, tpts, W, d, Om,
                 gammap, gammas):
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
        self.s0  = mops.basis(self.Np, 0);
        self.s1  = mops.basis(self.Np, 1);
        self.s2  = mops.basis(self.Np, 2)
        self.ss0 = mops.basis(self.Ns, 0);
        self.ss1 = mops.basis(self.Ns, 1)

        # Compute the density matrices corresponding to the states
        ## These correspond to the projectors |n_k > < n_k|
        self.s1dm  = mops.ket2dm(self.s1);
        self.s2dm  = mops.ket2dm(self.s2);
        self.ss0dm = mops.ket2dm(self.ss0);
        self.ss1dm = mops.ket2dm(self.ss1)

        # Define the logical states
        self.L0 = mops.ket2dm((self.s2 + self.s0) / np.sqrt(2))
        self.L1 = mops.ket2dm((self.s2 - self.s0) / np.sqrt(2))

        # Identity operators
        self.Is = np.eye(self.Ns)
        self.Ip = np.eye(self.Np)


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

        ## Set the logical state operator
        self.Ifull = mops.tensor(self.Ip, self.Ip, self.Is, self.Is)
        self.pL = 0.5 * self.Xl @ (self.Ifull + self.Xl@self.Xr) \
                  @ (self.Ifull - self.Pl1) @ (self.Ifull - self.Pr1)


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


class vslq_mops_readout(base_cqed_mops):
    """
    Very Small Logical Qubit (VSLQ) Class with additional DOF for readout

    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np, Nc, tpts, W, d, Om,
                 gammap, gammas, gl, gr,
                 use_sparse=False, readout_mode='single'):
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
        use_sparse:     converts all operators (rho, H, etc.) to sparse csc
        readout_mode:   single or dual for different numbers of cavity modes
                        used to separately readout Xl and Xr

        """

        # Set the class members here
        base_cqed_mops.__init__(self, tpts=tpts, Ns=Ns, Np=Np, Nc=Nc,
                W=W, d=d, Om=Om, gammap=gammap, gammas=gammas,
                gl=gl, gr=gr,
                use_sparse=use_sparse, readout_mode=readout_mode)

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()
        self.set_cops([self.gammas, self.gammas, self.gammap, self.gammap],
                      [self.asl, self.asr, self.apl, self.apr])
        self.set_init_state()

        # Set the projection operators
        self.set_proj_ops()


    def set_states(self):
        """
        Sets the basis states for the class
        """

        # States for the qubit / shadow degrees of freedom
        s0  = mops.basis(self.Np, 0);
        s1  = mops.basis(self.Np, 1);
        s2  = mops.basis(self.Np, 2);
        ss0 = mops.basis(self.Ns, 0);
        c0  = mops.basis(self.Nc, 0);

        # Compute the density matrices corresponding to the states
        ## These correspond to the projectors |n_k > < n_k|
        self.s0dm  = mops.ket2dm(s1);
        self.s1dm  = mops.ket2dm(s1);
        self.s2dm  = mops.ket2dm(s2);
        self.ss0dm = mops.ket2dm(ss0);
        self.c0dm  = mops.ket2dm(c0);

        # Define the logical states
        self.L0 = mops.ket2dm((s2 + s0) / np.sqrt(2))
        self.L1 = mops.ket2dm((s2 - s0) / np.sqrt(2))


    def set_init_state(self, logical_state='L0'):
        """
        Set the initial state
        """

        # Single mode states
        if self.readout_mode == 'single':

            # Initialize to a 0-logical state in the primary and shadow lattice
            if logical_state == 'L0':
                # Initial density matrix
                # psi0 = |L0> x |L0> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L0, self.L0,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            elif logical_state == 'L1':
                # psi0 = |L1> x |L1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L1, self.L1,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            elif logical_state == 'l1L0':
                # psi0 = |1> x |L0> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.s1dm, self.L0,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            elif logical_state == 'l1L1':
                # psi0 = |1> x |L1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.s1dm, self.L1,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            elif logical_state == 'r1L0':
                # psi0 = |L0> x |1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L0, self.s1dm,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            elif logical_state == 'r1L1':
                # psi0 = |L1> x |1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L1, self.s1dm,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm)
            else:
                self.psi0 = psi0
                self.state_name = ''
                return 0;

        # Dual mode states
        elif self.readout_mode == 'dual':

            # Initialize to a 0-logical state in the primary and shadow lattice
            if logical_state == 'L0':
                # Initial density matrix
                # psi0 = |L0> x |L0> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L0, self.L0,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            elif logical_state == 'L1':
                # psi0 = |L1> x |L1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L1, self.L1,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            elif logical_state == 'l1L0':
                # psi0 = |1> x |L0> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.s1dm, self.L0,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            elif logical_state == 'l1L1':
                # psi0 = |1> x |L1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.s1dm, self.L1,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            elif logical_state == 'r1L0':
                # psi0 = |L0> x |1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L0, self.s1dm,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            elif logical_state == 'r1L1':
                # psi0 = |L1> x |1> x |0s> x |0s> x |0c>
                self.psi0 = mops.tensor(self.L1, self.s1dm,
                                        self.ss0dm, self.ss0dm,
                                        self.c0dm, self.c0dm)
            else:
                self.psi0 = psi0
                self.state_name = ''
                return 0;

        # Set the state name
        self.state_name = logical_state


    def get_proj_k(self, ket, is_log_state=False, num_st_idx=4):
        """
        Get the projection operator corresponding to occupation of the
        state |k> by subtracting off the other state (s)
        """

        # Cover the logical and the error state cases
        if is_log_state:

            # Return the projection of the initial state onto to itself
            Pk = self.psi0

        # cover the error state case
        else:

            # Normalize the resulting projector
            kjldm = mops.ket2dm(mops.basis(self.Np, num_st_idx))

            ## Single and dual readout mode options
            if self.readout_mode == 'single':
                Pk = mops.tensor(kjldm, self.s0dm,
                                 self.ss0dm, self.ss0dm,
                                 self.c0dm)
            elif self.readout_mode == 'dual':
                Pk = mops.tensor(kjldm, self.s0dm,
                                 self.ss0dm, self.ss0dm,
                                 self.c0dm, self.c0dm)
            else:
                raise TypeError('(%s) readout mode not supported.' \
                                % self.readout_mode)

        # Convert the operators to sparse csr
        if self.use_sparse:
            Pk = scsp.csr_matrix(Pk)

        return Pk


    def set_proj_ops(self):
        """
        Sets projection operators for the leakage states and the logical
        states to compute the probability of occupying unwanted states
        """

        # Compute the logical states first
        self.PXlXr = self.get_proj_k(self.L0, True)

        # Set self.Pj for probability of occupying |j>
        for j in range(0, self.Np):
            setattr(self, 'P%d' % j, self.get_proj_k(mops.basis(self.Np, j),
                        False, j))

        # Convert to sparse matrices
        if self.use_sparse:
            for j in range(0, self.Np):
                setattr(self, 'P%d' % j,
                        scsp.csr_matrix(getattr(self, 'P%d' % j)))


    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Is = np.eye(self.Ns)
        self.Ip = np.eye(self.Np)
        self.Ic = np.eye(self.Nc)

        # Operators list
        ops_list = ['Pl1', 'Pr1', 'Xl', 'Xr', 'apl', 'apr', 'asl', 'asr']

        # Readout mode options
        ## Single mode
        if self.readout_mode == 'single':
        
            # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
            self.Pl1 = mops.tensor(self.s1dm, self.Ip,
                                   self.Is, self.Is, self.Ic)
            self.Pr1 = mops.tensor(self.Ip, self.s1dm,
                                   self.Is, self.Is, self.Ic)

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

            ## Add readout operators to the list
            ops_list.append('ac')

        ## Dual mode
        elif self.readout_mode == 'dual':
        
            # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
            self.Pl1 = mops.tensor(self.s1dm, self.Ip,
                                   self.Is, self.Is,
                                   self.Ic, self.Ic)
            self.Pr1 = mops.tensor(self.Ip, self.s1dm,
                                   self.Is, self.Is,
                                   self.Ic, self.Ic)

            # Destruction operators
            ## Primary qubits
            ap0 = mops.destroy(self.Np)
            self.apl = mops.tensor(ap0, self.Ip,
                                   self.Is, self.Is,
                                   self.Ic, self.Ic)
            self.apr = mops.tensor(self.Ip, ap0,
                                   self.Is, self.Is,
                                   self.Ic, self.Ic)

            ## Shadow resonators
            as0 = mops.destroy(self.Ns)
            self.asl = mops.tensor(self.Ip, self.Ip,
                                   as0, self.Is,
                                   self.Ic, self.Ic)
            self.asr = mops.tensor(self.Ip, self.Ip,
                                   self.Is, as0,
                                   self.Ic, self.Ic)

            ## Cavity resonator
            ac0 = mops.destroy(self.Nc)
            self.acl = mops.tensor(self.Ip, self.Ip,
                                   self.Is, self.Is,
                                   ac0, self.Ic)
            self.acr = mops.tensor(self.Ip, self.Ip,
                                   self.Is, self.Is,
                                   self.Ic, ac0)

            ## Add readout operators to the list
            ops_list.append('acl')
            ops_list.append('acr')

        ## Two photon operators on the logical manifold
        self.Xl = (self.apl@self.apl \
                  + mops.dag(self.apl)@mops.dag(self.apl)) / np.sqrt(2)
        self.Xr = (self.apr@self.apr \
                  + mops.dag(self.apr)@mops.dag(self.apr)) / np.sqrt(2)

        # Convert operators to sparse
        if self.use_sparse:
            for op in ops_list:
                setattr(self, op, scsp.csr_matrix(getattr(self, op)))


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
        ## Set if omega > 0
        if self.Om > 0.:
            Hps = self.Om*(mops.dag(self.apl)@mops.dag(self.asl) \
                    + mops.dag(self.apr)@mops.dag(self.asr))
            Hps += mops.dag(Hps)
        ## Otherwise, zero-out the contribution
        else:
            Hps = np.zeros(Hs.shape, dtype=Hs.dtype)

        # Hpc = gl (ac + ac^t) Xl + gr (ac + ac^t) Xr
        ## Single mode
        if self.readout_mode == 'single':
            Hpc = self.gl * (self.ac + mops.dag(self.ac)) @ self.Xl \
                + self.gr * (self.ac + mops.dag(self.ac)) @ self.Xr
        ## Dual mode
        if self.readout_mode == 'dual':
            Hpc = self.gl * (self.acl + mops.dag(self.acl)) @ self.Xl \
                + self.gr * (self.acr + mops.dag(self.acr)) @ self.Xr

        # Time independent Hamiltonian is sum of all contributions
        # Ignore the shadow / bath interaction for now
        self.H = Hp + Hs + Hps + Hpc

        # Convert to sparse
        if self.use_sparse:
            self.H = scsp.csr_matrix(self.H)
