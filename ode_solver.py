#!/usr/bin/env python3
"""
Solver for ODE's using Runge-Kutta 4 and other methods
"""

import numpy as np
from scipy.interpolate import interp1d as interp
import qutip as qt
import post_proc_tools as ppt
import matplotlib.pyplot as plt


class rk4:
    """
    Runge-Kutta 4 class
    """

    def __init__(self, rho0, tpts, dt, **kwargs):
        """
        Class constructor
    
        Parameters:
        ----------

        rho0:       initial density matrix 
        tpts:       times to evaluate the function
        dt:         time step referred to as h in Numerical Recipes

        """

        # Set the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Store the initial values
        self.rho0 = rho0; self.tpts = tpts; self.dt = dt


    def rhs(self, rho, t):
        """
        Compute the right hand side of the ODE
        """

        pass

    
    def solver(self):
        """
        Run the RK4 algorithm with the prescribed right hand side function
        """

        # Initialize the output
        rho = np.zeros([self.rho0.shape[0], self.rho0.shape[1],
                        self.tpts.size],
                        dtype=self.rho0.data.todense().dtype)

        # Set rho[t=0] = rho0
        rho[:,:,0] = self.rho0.data.todense()
        
        # Get the time step as a register here
        h = self.dt

        # Catch the exception
        try:

            # Iterate over the time steps
            print('Running solver() with %d time points ...' % (self.tpts.size))

            for n in range(1, self.tpts.size):

                # Compute the standard kj values with function calls to rhs
                k1 = h * self.rhs(rho[:,:,n-1], self.tpts[n-1])
                k2 = h * self.rhs(rho[:,:,n-1]+0.5*k1, self.tpts[n-1]+0.5*h)
                k3 = h * self.rhs(rho[:,:,n-1]+0.5*k2, self.tpts[n-1]+0.5*h)
                k4 = h * self.rhs(rho[:,:,n-1]+k3, self.tpts[n-1]+h)
                
                # Store the updated value of rho
                rho[:,:,n] = rho[:,:,n-1] + k1/6. + k2/3. + k3/3. + k4/6.

        except ValueError as err:
            print('Failed on time step ({:6.4f})\nError message: {}'\
                .format(self.tpts[n-1], err))

        
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
        
        # # Readoff H and cops
        H = self.H
        cops = self.cops

        # Define the commutator and anti commutator
        comm  = lambda a, b : a*b - b*a
        acomm = lambda a, b : a*b + b*a

        # Define the dissipator superoperator, D
        D = lambda a, p : a*p*a.dag() - 0.5*acomm(a.dag()*a, p)

        # Compute the commutator and the dissipator terms, with hbar = 1
        ## -i/hbar [H, rho] + sum_j D[a_j] rho
        ## Start with the case where H is just a qt.Qobj
        if H.__class__ == qt.qobj.Qobj:
            Hcommrho = -1j * comm(H, qt.Qobj(rho))

        ## Handle the list case, e.g. with time dependence as a numpy array
        ## specifying the drive Hamiltonian in the same format as qutip, e.g.
        ## [H0, [H1, e1(t)], [H2, e2(t)], ...]
        elif H.__class__ == list:
            ## Extract the time-independent Hamiltonian terms
            H0 = H[0]
            Hcommrho = comm(H0, qt.Qobj(rho))

            ## Readoff the remaining time-dependent Hamiltonian terms
            Hcommrho += np.sum([Hd(t)*Hk for Hk, Hd in H[1:][0]])            
            Hcommrho *= -1j

        ## Compute the dissipator terms
        Dterms = np.sum([D(C, rho) for C in cops])

        ## Return the result as a dense matrix
        rhs_data = (Hcommrho + Dterms).data.todense()

        return rhs_data
    

    def mesolve(self):
        """
        Run the rk4 solver, providing the interpolated time-dependent drive terms
        """

        # Handle the simple, time-independent case
        if self.H.__class__ == qt.qobj.Qobj:
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



def test_mesolve():
    """
    Tests the mesolve_rk4() class
    """

    # Setup a basic cavity system
    Nc = 2;
    a = qt.destroy(Nc)
    wc = 5;
    kappa = 0.1
    dt =(1./kappa) / 1e2
    tpts = np.linspace(0, 10/kappa, int(np.round((10/kappa)/dt)+1))
    tpts_d = np.linspace(0, 10/kappa, 4*tpts.size)

    # Time independent Hamiltonian
    H0 = wc*a.dag()*a
    
    # Time dependent Hamiltonian
    Hc = (a + a.dag())
    Hd = np.exp(-(tpts - tpts.max()/2)**2/(2*tpts.max()/6)**2)

    # Form the total Hamiltonian and set the collapse operators
    H = [H0, [Hc, Hd]]
    cops = [1./kappa * a]
    rho0 = qt.ket2dm(qt.basis(Nc, 0))

    # Setup the master equation solver instance
    me_rk4 = mesolve_rk4(rho0, tpts, 4*tpts.max()/tpts.size, H, cops) 
    rho_out = me_rk4.mesolve()
    
    print('rho_out.shape: {}'.format(rho_out.shape))
    print('tpts.size: %d' % (tpts.size))
    print('rho_out.__class__: {}'.format(rho_out.__class__))

    # Compute the expectation value of a^t a
    a_avg = ppt.get_expect(a, rho_out, man_tr=True)
    print('a_avg.shape: {}'.format(a_avg.shape))


    # Plot the results
    plt.plot(tpts, a_avg.real, label=r'$\Re \langle a\rangle$')
    plt.plot(tpts, a_avg.imag, label=r'$\Im \langle a\rangle$')
    plt.legend(loc='best')
    


if __name__ == '__main__':
    
    # Run the test example above for a single cavity driven by an applied field
    test_mesolve()
