"""
===========================================================================================
Author: David Rodriguez Perez
Filename: vslq_grape.py
Description: This program runs multiple simulations of the Very Small Logical Qubit. 
             It performs the same calculations as vslq_script.py, but here we do a 
             GRAPE algorithm to vary parameters in the vslq hamiltonian to find which 
             values optimize the qubit lifetime.
Input: None
Output: - parameters.log :: Outputs initial parameters, followed by the final optimized 
                            parameters for each different value of T_P. Also prints the
                            total runtime for the entire optimization process.
Edited: 10/21/2016
===========================================================================================
"""
import sys
sys.path.append("/lustre/project/ekapit/lib/python3.5/site-packages")
from numpy import sqrt,pi,linspace,savetxt,real,exp,floor,array,append
from qutip import basis,ket2dm,qeye,destroy,tensor,mesolve,Options
from qutip.parallel import parfor
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
w_s = W + 0.5*delta

f.write('Fixed Parameters ::     W = ' + str(W) + '\n'
        + '                    delta = ' + str(delta) + '\n'
        + 'Initial Parameters :: Ohm = ' + str(Ohm) + '\n'
        + '                  gamma_s = ' + str(gamma_S) + '\n'
        + '                      w_s = ' + str(w_s) + '\n\n')

# sn  --> photon state in qubit
# ssn --> photon state in shadow
s0 = basis(N,0); s1 = basis(N,1); s2 = basis(N,2)
s0dm = ket2dm(s0); s1dm = ket2dm(s1); s2dm = ket2dm(s2)
ss0 = basis(Ns,0); ss1 = basis(Ns,1)
ss0dm = ket2dm(ss0); ss1dm = ket2dm(ss1)

# Logical states, identities, and ladder operators
L0 = ket2dm((s2+s0).unit())
L1 = ket2dm((s2-s1).unit())
I = qeye(N); a = destroy(N)
II = qeye(Ns); a_s = destroy(Ns)

# Projectors onto pure states
l1 = tensor(II,s1dm,I,II); r1 = tensor(II,I,s1dm,II)

al = tensor(II,a,I,II); ar = tensor(II,I,a,II)
ald = al.dag(); ard = ar.dag()

asl = tensor(a_s,I,I,II); asr = tensor(II,I,I,a_s)
asld = asl.dag(); asrd = asr.dag()

Xl = (ald*ald + al*al)/sqrt(2) 
Xr = (ard*ard + ar*ar)/sqrt(2)

"""
================================= DEFINING FUNCTIONS ======================================
Note: The for loop at the bottom loops over the single qubit lifetimes in integers of 5,
      i.e. (5,10,15,20,etc.) \mu s. This is contained in the gamma_P declaration in the 
      get_Lifetime function.
"""

def Hamiltonian(W,delta,Ohm,w_s):
    Hp = -W*Xl*Xr + 0.5*delta*(l1 + r1)
    Hs = w_s*(asld*asl + asrd*asr)
    Hps = Ohm*(ald*asld + ard*asrd)
    return Hp + Hps + Hps.dag() + Hs

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

def grape_optimize(i,W,delta,num_it,max_it,epsilon,old_lifetime,Ohm,gamma_S,w_s):
    print('--------------------------------------------------------------')
    print('num_it = ' + str(num_it))
    print('params = ' + str(Ohm) + ', ' + str(gamma_S) + ', ' + str(w_s))
    print(old_lifetime)

    num_it += 1
    if(num_it > max_it): 
        return Ohm, gamma_S, w_s, old_lifetime, num_it

    Ohm_vec = [Ohm + epsilon, Ohm, Ohm]
    Gam_vec = [gamma_S, gamma_S + epsilon, gamma_S]
    w_s_vec = [w_s, w_s, w_s + epsilon]
    tmp_lifetime = parfor(get_Lifetime, Ohm_vec, Gam_vec, w_s_vec, 
                          W=W, delta=delta, i=i, t_max=t_max, t_steps=t_steps)
    dLT = tmp_lifetime - old_lifetime; dOhm = dLT[0]; dGam = dLT[1]; dw_s = dLT[2]
    new_lifetime = get_Lifetime(Ohm+dOhm,gamma_S+dGam,w_s+dw_s,W,delta,i,t_max,t_steps)

    if(old_lifetime > new_lifetime):
        epsilon = epsilon*0.5
        opt_dat = grape_optimize(i,W,delta,num_it,max_it,epsilon,old_lifetime,
                                 Ohm,gamma_S,w_s)
        return opt_dat
    if(new_lifetime - old_lifetime > threshold):
        opt_dat = grape_optimize(i,W,delta,num_it,max_it,epsilon,new_lifetime,
                                 Ohm+dOhm,gamma_S+dGam,w_s+dw_s)
        return opt_dat
    return Ohm, gamma_S, w_s, new_lifetime, num_it

def print_runtime(tg0,tg1):
    t_run = tg1 - tg0
    h = int(floor(t_run/3600))
    m = int(floor(t_run/60))%60
    s = int(t_run%60)
    f.write('\nRuntime for entire simulation:\n    '
             + str(h) + ' hr '
             + str(m) + ' min '
             + str(s) + ' sec\n')

def log_data(i,lifetime,opt_dat):
    f.write('\n---------------------------------------------\n'
            + 'T_P = ' + str(5*i) + ', T_L = ' + str(lifetime) 
            + '\n----------------------\n'
            + 'Optimized Parameters :: Ohm = ' + str(opt_dat[0]) + '\n'
            + '                    gamma_S = ' + str(opt_dat[1]) + '\n'
            + '                        w_s = ' + str(opt_dat[2]) + '\n'
            + 'num_it = ' + str(opt_dat[4]) + '\n'
            + 'T_L = ' + str(opt_dat[3]) + '\n')

"""
================================== RUNNING SIMULATION =====================================
# This is a very crude way to get the time to scale appropriately for the mesolve function.
# It is very inefficient, so I'm looking into truncating this to a much smaller time list
# such that an appropriate exponential curve_fit can still be applied. As it stands,
# anything larger than i = 5 takes too long. For reference, Kapit's mathematica simulator
# runs up to i = 16.
"""

i = 3; max_it = 30
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
