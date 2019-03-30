#!/usr/bin/env python3
"""
Solver for ODE's using Runge-Kutta 4 and other methods
"""

import numpy as np
from scipy.interpolate import interp1d as interp
import qutip as qt
import post_proc_tools as ppt
import matplotlib.pyplot as plt
import matrix_ops as mops

# Use for catching errors
import traceback

# Set for the runtime warnings
np.seterr(all='raise')


class rk4:
    """
    Runge-Kutta 4 class
    """

    def __init__(self, rho0, tpts, dt, **kwargs):
        """
        Class constructor
    
        Parameters:
        ----------

        rho0:       initial density matrix / dependent variable to solve for
        tpts:       times to evaluate the function
        dt:         time step referred to as h in Numerical Recipes

        """

        # Set the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Store the initial values
        self.rho0 = rho0; self.tpts = tpts; self.dt = dt

    @staticmethod
    def dagger(a):
        """
        Compute the comple conjugate transpose of an operator from its matrix
        representation
        """

        return np.transpose(np.conjugate(a))


    def rhs(self, rho, t):
        """
        Compute the right hand side of the ODE
        """

        pass

    
    def solver(self):
        """
        Run the RK4 algorithm with the prescribed right hand side function
        """

        # Set rho[t=0] = rho0
        rho = [self.rho0] * self.tpts.size
        
        # Get the time step as a register here
        h = self.dt

        # Catch the exception
        try:

            # Iterate over the time steps
            for n in range(1, self.tpts.size):

                # Compute the standard kj values with function calls to rhs
                k1 = h * self.rhs(rho[n-1],          self.tpts[n-1]       )
                k2 = h * self.rhs(rho[n-1] + 0.5*k1, self.tpts[n-1]+ 0.5*h)
                k3 = h * self.rhs(rho[n-1] + 0.5*k2, self.tpts[n-1]+ 0.5*h)
                k4 = h * self.rhs(rho[n-1] + k3,     self.tpts[n-1]+     h)
                
                # Store the updated value of rho
                rho[n] = rho[n-1] + k1/6. + k2/3. + k3/3. + k4/6.

        except Exception as err:
            print('Failed on time step ({:6.4f})\nError message: {}'\
                .format(self.tpts[n-1], err))
            print(traceback.format_exc())
            # raise(Exception(locals()))
        
        return rho


class mesolve_rk4(rk4):
    """
    Lindblad master equation solver using the rk4 class options above
    """

    def __init__(self, rho0, tpts, dt, H, cops):
        """
        Class constructor
    
        Parameters:
        ----------

        rho0:       initial density matrix in dense matrix form
        tpts:       array of times to compute the density matrix on
        dt:         time step used by the RK4 solver, should be equal to
                    t_n - t_n-1 
        H:          Hamiltonian in same format as qt.mesolve [H0, [Hc, eps(t)]]
        cops:       collapse operators including the coefficients as a list
                    of qt.Obj's scaled by the damping coefficients

        """

        # Call the rk4 constructor to start
        rk4.__init__(self, rho0, tpts, dt, H=H, cops=cops)


    def rhs(self, rho, t):
        """
        Implement the right hand side of the Lindblad master equation
        """        
        
        # Readoff H and cops
        H = self.H
        cops = self.cops

        # Define the commutator and anti commutator
        ## Use the overloaded operator multiplication with Qobjs
        if rho.__class__ == qt.qobj.Qobj:
            comm  = lambda a, b : a*b - b*a
            acomm = lambda a, b : a*b + b*a
            D = lambda a, p : a*p*a.dag() - 0.5*acomm(a.dag()*a, p)

        ## Use the matrix multiplication operator in Python 3
        elif rho.__class__ == np.ndarray:
            comm  = lambda a, b : a@b - b@a
            acomm = lambda a, b : a@b + b@a
            D = lambda a, p : a @ p @ self.dagger(a) \
                - 0.5*acomm(self.dagger(a)@a, p)

        # Compute the commutator and the dissipator terms, with hbar = 1
        ## -i/hbar [H, rho] + sum_j D[a_j] rho
        ## Start with the case where H is just a qt.Qobj
        if H.__class__ == qt.qobj.Qobj and H[0].__class__ == np.ndarray:
            Hcommrho = -1j * comm(H, qt.Qobj(rho))

        ## Handle the list case, e.g. with time dependence as a numpy array
        ## specifying the drive Hamiltonian in the same format as qutip, e.g.
        ## [H0, [H1, e1(t)], [H2, e2(t)], ...]
        elif H.__class__ == list and rho.__class__ == qt.qobj.Qobj:

            ## Extract the time-independent Hamiltonian terms
            H0 = H[0]
            Hcommrho = comm(H0, qt.Qobj(rho))

            ## Readoff the remaining time-dependent Hamiltonian terms
            Hcommrho += np.sum([Hd(t)*Hk for Hk, Hd in H[1:][0]])            
            Hcommrho *= -1j

        ## Time independent case
        elif H.__class__ == np.ndarray and rho.__class__ == np.ndarray:

            # Just calculate the commutator
            # Hcommrho = -1j*comm(H, rho)
            Hcommrho = -1j * (H@rho - rho@H)

        ## Numpy array equivalent calculations of commutators
        elif H.__class__ == list and rho.__class__ == np.ndarray:

            ## Extract the time-independent Hamiltonian terms
            H0 = H[0]
            Hcommrho = np.complex128(comm(H0, rho))

            ## Readoff the remaining time-dependent Hamiltonian terms
            Hcommrho += np.sum([Hd(t)*Hk for Hk, Hd in H[1:][0]])
            Hcommrho *= -1j

        ## Compute the dissipator terms
        Dterms = np.zeros(rho.shape, dtype=np.complex128)
        for ck in cops:
            # Dterms += D(ck, rho)
            Dterms += ck @ rho @ ck - 0.5 * (mops.dag(ck)@ck@rho \
                        + rho@mops.dag(ck)@ck)

        ## Return the result as a dense matrix
        ## Use the numpy array convention to 
        if Hcommrho.__class__ == np.ndarray:
            rhs_data = Hcommrho + Dterms

        ## If relevant,  convert the qobjs to dense matrices
        elif Hcommrho.__class__ == qt.qobj.Qobj:
            rhs_data = (Hcommrho + Dterms).data.todense()
    
        ## Complain if the result is not of the expected type
        else:
            raise('Data type for Hcommrho not supported.')

        return rhs_data
    

    def mesolve(self):
        """
        Run the rk4 solver, providing the interpolated 
        time-dependent drive terms
        """

        # Handle the simple, time-independent case
        if self.H.__class__ == qt.qobj.Qobj:
            rho_out = self.solver()
            return rho_out
        
        # Handle time-independent Hamiltonian
        elif self.H.__class__ == np.ndarray:
            rho_out = self.solver()
            return rho_out
        
        # Handle the case involving drive terms
        elif self.H.__class__ == list:

            # Interpolate the drive terms
            drvs = [interp(self.tpts, self.H[1:][0][1])]
            HH = [self.H[0], [[Hk, d] for Hk, d in zip(self.H[1:][0], drvs)]]

            # Set the class instance of the Hamiltonian to the 
            # list comprehension with the interpolants embedded
            self.H = HH

            # Call the solver with the interpolated drives
            rho_out = self.solver()

            return rho_out

        else:
            raise('H.__class__ ({}) not supported'.format(H.__class__))


class langevin_rk4(rk4):
    """
    Markovian Langevin equation solver using the rk4 class options above
    """

    def __init__(self, a0, tpts, dt, ain, kappa, sz0, gz, eq_type):
        """
        Class constructor
    
        Parameters:
        ----------

        a0:         initial Heisenberg matrix in dense matrix form
        tpts:       array of times to compute the density matrix on
        dt:         time step used by the RK4 solver, should be equal to
                    t_n - t_n-1 
        ain:        input mode, e.g. form of the drive
        kappa:      damping coefficient, e.g. linewidth
        sz0:        initial state of the qubit
        gz:         coupling between the qubit and cavity
        eq_type:    equation of motion type (long / disp) for longitudinal
                    or dispersive coupling type

        """

        # Call the rk4 constructor to start
        rk4.__init__(self, a0, tpts, dt, ain=ain,
                     kappa=kappa, sz0=sz0, gz=gz,
                     eq_type=eq_type)


    def rhs(self, a, t):
        """
        Implement the right hand side of the Langevin equation of motion
        """        
        
        # Use the right hand side for the simple longitudinal case
        if self.eq_type == 'long':
            rhs_data = -1j * 0.5 * self.gz * self.sz0 - 0.5 * self.kappa * a \
                        - np.sqrt(self.kappa) * self.ain

        # Handle the dispersive case
        elif self.eq_type == 'disp':
            rhs_data = -1j * self.gz * self.sz0 * a - 0.5 * self.kappa * a \
                        - np.sqrt(self.kappa) * self.ain

        return rhs_data
    

    def langevin_solve(self):
        """
        Langevin solver wrapper for langevin_rk4()
        """

        # Compute the resultant cavity mode
        ares = self.solver()

        return np.array(ares)


def test_mesolve():
    """
    Tests the mesolve_rk4() class
    """

    # Setup a basic cavity system
    Nc = 2;
    Nq = 3;
    a = qt.tensor(qt.qeye(Nq), qt.destroy(Nc))
    b = qt.tensor(qt.destroy(Nq), qt.qeye(Nc))
    wc = 5;
    kappa = 0.1
    dt = (1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))

    # Time independent Hamiltonian
    H0 = wc*a.dag()*a
    
    # Time dependent Hamiltonian
    Hc = (a + a.dag())
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [kappa * a]
    rho0 = qt.ket2dm(qt.tensor(qt.basis(Nq, 0), qt.basis(Nc, 0)))

    # Setup the master equation solver instance
    me_rk4 = mesolve_rk4(rho0, tpts, 4*tpts.max()/tpts.size, H, cops) 
    rho_out = me_rk4.mesolve()

    # Compute the expectation value of a^t a
    a_avg = ppt.get_expect(a, rho_out, man_tr=True)

    # Plot the results
    plt.plot(tpts, a_avg.real, label=r'$\Re \langle a\rangle$')
    plt.plot(tpts, a_avg.imag, label=r'$\Im \langle a\rangle$')
    plt.legend(loc='best')
    

def test_mesolve_mops():
    """
    Tests the mesolve_rk4() class using the matrix_ops module
    """

    # Setup a basic cavity system
    Nc = 16;
    Nq = 3;
    a = mops.tensor(np.eye(Nq), mops.destroy(Nc))
    b = mops.destroy(Nq); bd = mops.dag(b)
    sz = mops.tensor(bd@b, np.eye(Nc))
    wc = 5; wq = 6;
    kappa = 0.1; chi = kappa / 2.; g = np.sqrt(chi)
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))
    # tpts_d = np.linspace(0, 10/kappa, 4*tpts.size)

    # Time independent Hamiltonian
    # H0 = wc*mops.dag(a)@a + wq*sz/2. + chi*mops.dag(a)@a@sz
    # In rotating frame
    H0 = g*(mops.dag(a) + a) @ sz
    
    # Time dependent Hamiltonian
    Hc = (a + mops.dag(a))
    # Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2) \
    # * np.sin(wc*tpts)
    # In rotating frame
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [np.sqrt(kappa) * a]
    rho0 = mops.ket2dm(mops.tensor(mops.basis(Nq, 0), mops.basis(Nc, 0)))
    print('rho0:\n{}'.format(rho0))
    print('Time = [%g, %g] ns' % (tpts.min(), tpts.max()))

    # Setup the master equation solver instance
    me_rk4 = mesolve_rk4(rho0, tpts, tpts.max()/(10*tpts.size), H, cops) 
    rho_out = me_rk4.mesolve()

    # Compute the expectation value of a^t a
    a_avg = mops.expect(a, rho_out)
    print('{}'.format(a_avg.real))

    # Plot the results
    plt.plot(kappa*tpts, a_avg.real,label=r'$\Re \langle a\rangle$')
    plt.plot(kappa*tpts, a_avg.imag,label=r'$\Im \langle a\rangle$')
    plt.xlabel(r'Time (1/$\kappa$)')
    plt.legend(loc='best')


def test_langevin_solve():
    """
    Tests the Langevin equation solver for the
    longitudinal case in Didier, 2015
    """


    # Parameters for the cavity, coupling, etc.
    kappa = 0.1; chi = kappa / 2.; gz = np.sqrt(chi)
    dt = (1./kappa) / 1e3
    tpts = np.linspace(0, 8, 101) / kappa

    # Use the vacuum as the input mode
    ain = -1 / np.sqrt(kappa) 
    
    # Initialize the state of the intracavity mode
    sz0e = 1; sz0g = -1;
    a0e = np.array([0], dtype=np.complex128)
    a0g = np.array([0], dtype=np.complex128)
    
    print('Time = [%g, %g] ns' % (tpts.min(), tpts.max()))

    # Setup the master equation solver instance
    ## Solve the dispersive case first
    le_rk4 = langevin_rk4(a0g, tpts, dt, ain, 
            kappa, sz0g, 40*chi, eq_type='disp') 
    ag_disp = le_rk4.langevin_solve()
    le_rk4 = langevin_rk4(a0e, tpts, dt, ain,
            kappa, sz0e, 40*chi, eq_type='disp') 
    ae_disp = le_rk4.langevin_solve()

    ## Solve the longitudinal case next
    le_rk4 = langevin_rk4(a0g, tpts, dt, 0*ain, kappa,
            sz0g, gz, eq_type='long') 
    ag_long = le_rk4.langevin_solve()
    le_rk4 = langevin_rk4(a0e, tpts, dt, 0*ain, kappa,
            sz0e, gz, eq_type='long') 
    ae_long = le_rk4.langevin_solve()

    ## Plot the results
    ppt.plot_io_a_full(tpts, ag_disp, ae_disp, ag_long, ae_long, gz, 40*chi,
            kappa, fext='langevin_numerical', use_interp=False)
    


if __name__ == '__main__':
    
    # Run the test example above for a single cavity driven by an applied field
    # test_mesolve()
    test_langevin_solve()
