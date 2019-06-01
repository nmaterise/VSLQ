# VSLQ
## Updated as a class for ease of use

## Class inheritance structure
```bash
| -- matrix_ops (module)
|    | -- bkeuler(class)
|    | -- rk4 (class)
|         | -- langevin_rk4 (class)
|         | -- mesolve_rk4 (class)
|              | -- base_cqed_mops (class)
|                   | -- qho (class) 
|                   | -- qho2 (class)
|                   | -- transmon_disp_mops (class) 
|                   | -- transmon_long_mops (class)
|                   | -- vslq_mops (class)
|                   | -- jaynes_cummings (class)
| -- post_proc_tools (module)
```

## Types of solvers
    * rk4
        - Explicit fourth-order Runge-Kutta, fixed time step
        - Currently tested for cases below
    * bkeuler
        - Backward Euler, fixed time step
        - Uses the superoperator and vectorized density matrix approach
        - Untested as of 180531

## Tests of solvers
### Runge-Kutta Sanity checks
    * rk4_tests.py
        - Tests a classical harmonic oscillator to make sure the solver produces
          the correct dynamics for an undriven, undamped system

### Lindblad Equation tests
    * mesolve_test.py
        - Runs basic time-independent tests of the Lindblad solver that have
          been validated against QuTip examples
        - Includes basic qubit-oscillator Jaynes-Cummings dynamics including
          unitary and Lindblad loss examples
    * jaynes_cummings.py
        - Uses the base_cqed_mops class to validate QuTip examples

### Langevin Equations
    * ode_solver.py
        - Includes another class to solve linear Langevin equations related to
          the input-output theory for transmons in the dispersive and
          longitudinal coupling cases

## Driver Files
    * qubit_cavity.py
        - Contains base class (base_cqed_mops) for all Hamiltonians
    * matrix_ops.py
        - Performs basic tensor and matrix operations on operators and states
    * transmon.py
        - Implements basic transmon / cavity coupling Hamiltonians
    * vslq.py
        - Implements a simplified version of the VSLQ Hamiltonian, including the
          forms for the logical and single photon loss states
    * vslq_grape_15.py
        - David's version of the QuTip-based Grape optimization study of VSLQ
    * ode_solver.py
        - An implementation of Runge-Kutta 4
        - Includes methods for solving the Lindblad equation and Langevin
          equations in the input / output formalism
        - Also allows for general right hand sides for inherited version of the
          rk4 class
    * io_eoms.py
        - Evaluates the solutions to the input / output Langevin equations
          derived by Didier et al.
    * drive_tools.py
        - Collection of drive function generation tools for non-parametric
          drives, e.g. those applied directly to the cavity mode as a function
          of time
    * post_proc_tools.py
        - Post processing and plotting routines that support the computational
          components herein
    * phase_diagram.py
        - Simple plotting tools used to generate some of the <a_c> phase
          diagrams plots, others moved to post_proc_tools.py
    * longitudinal_coupling.py (.ipynb)
        - Deprecated Python file (notebook) to study a simple longitudinally
          coupled transmon to a single cavity mode
