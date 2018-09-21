"""
Author: Nick Materise
Filename: vslq.py
Description: This class includes the basic features of a generalized VSLQ for
             N-logical circuits with Nq transmon levels and Ns shadow levels

Created: 180921
"""

import sys
import numpy as np
import qutip as qt
from scipy.optimize import curve_fit
import time
import datetime

"""
======================== DEFINING THE SYSTEM =======================
N = Number of photon states in the primary qubits.
Ns = Number of photon states in teh shadow resonators
W,delta,Ohm = empirical constants which are specified by engineering
              of device (all in MHz).
              W = Ej*alpha*C02**2
"""

f = open('parameters_15.log','a')
date_time = datetime.datetime.now()
f.write('\n------------- PARAMETER OPTIMIZIATION ---------------\n' +
         'Simulation run on :: ' + str(date_time) + '\n' +
         '-----------------------------------------------------\n')

N = 3; Ns = 2
W = 70.0*pi; delta = 700.0*pi; Ohm = 5.5; gamma_S = 9.2

class vslq:
    """
    Very Small Logical Qubit (VSLQ) Class
    
    Returns the Hamiltonian, and the density matrix after running an mesolve
    calculation

    """

    def __init__(self, Ns, Np):
        """
        Constructor for the class

        Parameters:
        ----------

        Ns:     number of levels in the shadow resonators
        Np:     number of levels in the primary qubits
    
        """

        # Set the class members here
        self.Ns = Ns; self.Np = Np;

        # Set the states and the operators for the class
        self.set_states()
        self.set_ops()
    

    def __del__(self):
        pass
    

    def set_states(self):
        """
        Sets the basis states for the class
        """
    
        # States for the qubit / shadow degrees of freedom
        s0  = qt.basis(self.Np, 0);
        s1  = qt.basis(self.Np, 1); 
        s2  = qt.basis(self.Np, 2)
        ss0 = qt.basis(self.Ns, 0);
        ss1 = qt.basis(self.Ns, 1)
        
        # Compute the density matrices corresponding to the states
        self.s0dm  = qt.ket2dm(s0); 
        self.s1dm  = qt.ket2dm(s1); 
        self.s2dm  = qt.ket2dm(s2)
        self.ss0dm = qt.ket2dm(ss0);
        self.ss1dm = qt.ket2dm(ss1)

        # Define the logical states
        self.L0 = qt.ket2dm((s2 + s0).unit())
        self.L1 = qt.ket2dm((s2 - s1).unit())

    def set_ops(self):
        """
        Sets the operators for the class, e.g. the creation / annihilation
        operators for the primary and shadow degrees of freedom
        """

        # Identity operators
        self.Is = qt.qeye(Ns)
        self.Ip = qt.qeye(Np)

        # Projection operators |1Ll> <1Ll|, |1Lr> <1Lr|
        self.Pl1 = tensor(ss1dm, Ip, Is, Is)
        self.Pr1 = tensor(Ip, ss1dm, Is, Is)

        # Destruction operators
        ## Primary qubits
        ap0 = qt.destroy(Np)
        self.apl = qt.tensor(ap0, Ip, Is, Is)
        self.apr = qt.tensor(Ip, ap0, Is, Is)
        
        ## Shadow resonators
        as0 = qt.destroy(Ns)
        self.asl = qt.tensor(Ip, Ip, as0, Is)
        self.asr = qt.tensor(Ip, Ip, Is, as0)

        ## Two photon operators on the logical manifold
        self.Xl = (self.apl**2 + self.apl.dag()**2) / np.sqrt(2)
        self.Xr = (self.apr**2 + self.apr.dag()**2) / np.sqrt(2)


    def set_H(self, W, d, Om, ws)
        """
        Compute the Hamiltonian in the rotating frame of the primary qubits
        """
        
        # Hp = -W Xl Xr + 1/2 d (Pl1 + Pr1)
        Hp = -W * self.Xl*self.Xr + 0.5*d*(self.Pl1 + self.Pr1)
        
        # Hs = (W + d/2) (asl^t asl + asr^t asr)
        Hs = (W + d/2.) * (self.asl.dag()*asl + self.asr.dat()*self.asr)

        # Hps = O (apl^t asl^t + apr^t asr^t + h.c.)
        Hps = Om*(self.apl.dag()*self.asl.dag() + self.apr.dag()*self.asr.dag())
        Hps += Hps.dag()

        self.H = Hp + Hs + Hps

def exp_func(t,A,B,C):
    return A*exp(-B*t) + C

def get_Lifetime(Ohm,gamma_S,w_s,W,delta,i,t_max,t_steps):
    # Collapse operators for  Lindblad master equation
    gamma_P = 1.0/(5*i) #MHz... photon loss rate of qubits
    c_ops = [sqrt(gamma_S)*asl,sqrt(gamma_P)*al,sqrt(gamma_P)*ar,sqrt(gamma_S)*asr]
    t_tot = linspace(0,t_max,t_steps)

    # Projection operator to measure expectation value. Initialize state to logical manifold 
    pL = 0.5*Xl*(1 + Xl*Xr)*(1 - l1)*(1 - r1)
    psi = tensor(ss0dm,L0,L0,ss0dm)
    sol = mesolve(Hamiltonian(W,delta,Ohm,w_s),psi,t_tot,c_ops,pL,options=Options(nsteps=10000))
    #sol = mesolve(Hamiltonian(W,delta,Ohm,w_s),psi,t_tot,c_ops,pL,nsteps=SAMSING!?!?!?)

    x = t_tot
    y = sol.expect[0]
    coef,bleh = curve_fit(exp_func,x,y,maxfev=10000)
    return 1/coef[1]


"""
================================== RUNNING SIMULATION =====================================
# This is a very crude way to get the time to scale appropriately for the mesolve function.
# It is very inefficient, so I'm looking into truncating this to a much smaller time list
# such that an appropriate exponential curve_fit can still be applied. As it stands,
# anything larger than i = 5 takes too long. For reference, Kapit's mathematica simulator
# runs up to i = 16.
"""

i = 1; max_it = 30
tg0 = time.time()
# for i in range(1,max_i):
num_it = 0; epsilon = 0.75; threshold = 0.01
t_max = 35*i*i; t_steps = t_max*600
lifetime = get_Lifetime(Ohm,gamma_S,w_s,W,delta,i,t_max,t_steps)
opt_dat = grape_optimize(i,W,delta,num_it,max_it,epsilon,lifetime,
                         Ohm,gamma_S,w_s)
log_data(i,lifetime,opt_dat)

tg1 = time.time()

print_runtime(tg0,tg1)
f.close()
