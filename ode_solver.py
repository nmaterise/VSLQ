#!/usr/bin/env python3
"""
Solver for ODE's using Runge-Kutta 4 and other methods
"""

import numpy as np
from scipy.interpolate import interp1d as interp
import post_proc_tools as ppt
import matplotlib.pyplot as plt
import matrix_ops as mops

# Use for catching errors
import traceback

# Set for the runtime warnings
np.seterr(all='raise')


class implicitmdpt(object):
    """
    Implicit midpoint integration scheme of the vector equation
    
    dy / dt = A y
    
    y_n+1 = (I + h/2 A)^-1 (I - h/2 A) y_n

    """

    def __init__(self, y0, tpts, dt, is_A_const, **kwargs):
        """
        Implicit Midpoint constructor 
    
        Parameters:
        ----------

        y0:             solution vector at t=t0
        tpts:           times to compute y
        dt:             time step
        is_A_const:     check if A is constant in time

        """

        # Set the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Store the class members for the initial y and time step
        self.y0 = y0; self.tpts = tpts; self.dt = dt;
        self.is_A_const = is_A_const

    
    def rhs_A(self, t):
        """
        User defined computation of the right hand side matrix A
        """

        pass


    def inv1pA1mA(self, t, h):
        """
        Perform the step (I + h/2A)^-1(I - h/2A)
        """

        # Get the current value of A
        A = self.rhs_A(t + 0.5*h)
        I = np.eye(A.shape[0])
        D = (I + 0.5*h*A) @ np.linalg.inv(I - 0.5*h*A)

        return D


    def solver(self):
        """
        Run the Implicit Midpoint solver routine, given the right hand side
        operator, A
        """

        # Load the class members into "registers"
        tpts = self.tpts
        y0 = self.y0
        h = self.dt
    
        # Initialize y as a copy of initial values
        y = [y0] * tpts.size

        # Check if A in constant in time
        if self.is_A_const:

            print('Using constant right hand side ...')
            
            # Compute the rhs matrix once
            # Time-independent, just pass first time
            oneminAinv = self.inv1pA1mA(tpts[0], h)

            # Iterate over all remaining times
            for n in range(1, tpts.size):

                # y_n = (1 - hA)^-1 * y_n-1
                y[n] = oneminAinv @ y[n-1]
                # y_n = ((1 + h/2A)(1 - h/2A)^-1)^n * y_0
                # y[n] = np.linalg.matrix_power(oneminAinv, n) @ y0
        
        # Otherwise update on each time step
        else:

            # Iterate over all times
            for n in range(1, tpts.size):

                # y_n = (1 - hA)^-1 * y_n-1
                y[n] = self.inv1pA1mA(tpts[n-1], h) @ y[n-1]

        # Return the result as a numpy array
        
        return np.array(y)


class bkeuler(object):
    """
    Backward Euler integration scheme of the vector equation
    
    dy / dt = -A y
    
    y_n+1 = (I + hA)^-1 y_n

    """

    def __init__(self, y0, tpts, dt, is_A_const, **kwargs):
        """
        Backward Euler constructor 
    
        Parameters:
        ----------

        y0:             solution vector at t=t0
        tpts:           times to compute y
        dt:             time step
        is_A_const:     check if A is constant in time

        """

        # Set the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Store the class members for the initial y and time step
        self.y0 = y0; self.tpts = tpts; self.dt = dt;
        self.is_A_const = is_A_const

    
    def rhs_A(self, t):
        """
        User defined computation of the right hand side matrix A
        """

        pass


    def inv1pA(self, t, h):
        """
        Perform the step (I + hA)^-1
        """

        # Get the current value of A
        A = self.rhs_A(t+h) 
        B = np.eye(A.shape[0]) + h*A
        
        # Return (I + hA)^-1

        return np.linalg.inv(B)


    def solver(self):
        """
        Run the backward Euler solver routine, given the right hand side
        operator, A
        """

        # Load the class members into "registers"
        tpts = self.tpts
        y0 = self.y0
        h = self.dt
    
        # Initialize y as a copy of initial values
        y = [y0] * tpts.size

        # Check if A in constant in time
        if self.is_A_const:
            
            # Compute the rhs matrix once
            oneminAinv = self.inv1pA(tpts[0], h)

            # Iterate over all times
            for n in range(1, tpts.size):

                # y_n = (1 - hA)^-1 * y_n-1
                y[n] = oneminAinv @ y[n-1]
        
        # Otherwise update on each time step
        else:

            # Iterate over all times
            for n in range(1, tpts.size):

                # y_n = (1 - hA)^-1 * y_n-1
                y[n] = self.inv1pA(tpts[n-1], h, kwargs) @ y[n-1]

        # Return the result as a numpy array
        
        return np.array(y)


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
        
        # Get the time step as a register here
        h = self.dt
        
        # Update the number of points and times
        Nt = self.tpts.size if h >= self.tpts.max() / self.tpts.size\
                else int(np.ceil(self.tpts.max() / h))
        tpts = np.linspace(self.tpts.min(), self.tpts.max(), Nt)
        

        # Set rho[t=0] = rho0
        rho = [self.rho0] * Nt

        # Catch the exception
        try:

            # Iterate over the time steps
            for n in range(1, Nt):

                # Compute the standard kj values with function calls to rhs
                k1 = h * self.rhs(rho[n-1],          tpts[n-1]       )
                k2 = h * self.rhs(rho[n-1] + 0.5*k1, tpts[n-1]+ 0.5*h)
                k3 = h * self.rhs(rho[n-1] + 0.5*k2, tpts[n-1]+ 0.5*h)
                k4 = h * self.rhs(rho[n-1] + k3,     tpts[n-1]+     h)
                
                # Store the updated value of rho
                rho[n] = rho[n-1] + k1/6. + k2/3. + k3/3. + k4/6.

        except Exception as err:
            print('Failed on time step ({:6.4f})\nError message: {}'\
                .format(self.tpts[n-1], err))
            print(traceback.format_exc())
        
        # Decimate the result
        if h < self.tpts.max() / self.tpts.size:
            rho_out = rho[0::int(tpts.size // self.tpts.size)]
        else:
            rho_out = rho
    

        return rho_out


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
        
        # Lindblad equation using the expanded form for the dissipator
        ## Compute the unitary contribution
        ## Time independent case
        if self.H.__class__ == np.ndarray:
            Hcommrho = -1j*(self.H@rho - rho@self.H)

        ## Time dependent case
        elif self.H.__class__ == list:
            # Extract time-independent and time-dependent components
            H0 = self.H[0]; Hp = self.H[1]

            # Multiply and add Hp components
            Hpp = sum([h*d(t) for h, d in Hp])

            # Sum contributions and compute commutator
            Htot = H0 + Hpp
            Hcommrho = -1j*(Htot@rho - rho@Htot)

        else:
            raise TypeError('Time dependent Hamiltonian type (%s) not \
                             supported' % self.H.__class__)
        ## Compute the dissipator contribution
        Drho = np.zeros(rho.shape, dtype=np.complex128)
        for ck in self.cops:
            Drho += ck@rho@mops.dag(ck) \
                    - 0.5 * (mops.dag(ck)@ck@rho + rho@mops.dag(ck)@ck)

        ## Sum both contributions
        rhs_data = Hcommrho + Drho

        return rhs_data
    

    def mesolve(self):
        """
        Run the rk4 solver, providing the interpolated 
        time-dependent drive terms
        """

        # Handle time-independent Hamiltonian
        if self.H.__class__ == np.ndarray:
            rho_out = self.solver()
            return rho_out
        
        # Handle the case involving drive terms
        elif self.H.__class__ == list:

            # Extract the time-independent and time-dependent Hamiltonians
            H0 = self.H[0]
            Hp = self.H[1]

            # Transpose Hp and extract the operators
            HpT = list(map(list, zip(*Hp)))
            Hpops = HpT[0]
            Hpdrvs = HpT[1]

            # Interpolate the drive terms
            drvs = [interp(self.tpts, d) for d in Hpdrvs]
            HH = [H0, [[Hk, d] for Hk, d in zip(Hpops, drvs)]]

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
