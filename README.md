# PyPBEC

## A Python package that solves the nonequilibrium model of light-matter interaction in microcavities

The `PyPBEC` repository contains different approaches to solve the general theoretical model introduced by [Kirton and Keeling
in *Nonequilibrium Model of Photon Condensation*](https://doi.org/10.1103/PhysRevLett.111.100404), to study both the thermal and nonequilibrium properties of light, inside a dye-filled microcavity.

The present version of the repository includes a preliminary set of methods to obtain the steady-state, as well as the temporal dynamics of photons. The model has been used to predict and describe several interesting physical processes:

[1] [*Thermalization and breakdown of thermalization in photon condensates*](https://doi.org/10.1103/PhysRevA.91.033826), P. Kirton and J. Keeling, Phys. Rev. A **91**, 033826 (2015).

[2] [*Spatial dynamics, thermalization, and gain clamping in a photon condensate*](https://doi.org/10.1103/PhysRevA.93.013829), J. Keeling and P. Kirton, Phys. Rev. A **93**, 013829 (2016).

[3] [*Decondensation in nonequilibrium photonic condensates: when less is more*](https://doi.org/10.1103/PhysRevLett.120.040601), H. J. Hesten, R. A. Nyman, and F. Mintert, Phys. Rev. Lett. **120**, 040601 (2018).

[4] [*Driven-dissipative non-equilibrium Bose–Einstein condensation of less than ten photons*](https://doi.org/10.1038/s41567-018-0270-1) B. T. Walker, L. C. Flatten, H. J. Hesten, F. Mintert, D. Hunger, A. A. P. Trichet, J. M. Smith, and R. A. Nyman, Nat. Phys. **14**, 1173 (2018)

[5] [*Noncritical slowing down of photonic condensation*](https://doi.org/10.1103/PhysRevLett.123.203602), B, T. Walker, H, J. Hesten, H, S. Dhar, R, A. Nyman, and F, Mintert, Phys. Rev. Lett. **123**, 203602 (2019).

[6] [*Non-stationary statistics and formation jitter in transient photon condensation*](https://doi.org/10.1038/s41467-020-15154-7), B. T. Walker, J. D. Rodrigues, H. S. Dhar, R. F. Oulton, F. Mintert, and R. A. Nyman, Nat. Commun. **11**, 1390 (2020).

We are still in the process of building a more complete documentation and we will keep updating this page as we further expand the repository by adding more functionalities to `PyPBEC`.


## Installation

The code is written in Python 3 (https://www.python.org/downloads/). The following packages are recommended for installation along with Python:

*numpy*<br/>
*scipy*<br/>
*matplotlib*<br/>
*notebook*<br/>
*jupyter*

The *notebook* and *jupyter* packages are necessry to run some the examples only.

To access `PyPBEC` package, clone the repository to your local machine using Git.

```
git clone git@github.com:photonbec/PyPBEC.git
```

To install Git, [check this](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Else, you can directly download the zip of the repository from GitHub.

## Brief theory

In recent years, two-dimensional photon gas inside a dye-filled microcavity has become a useful tool for studying both near- and out-of-equlibrium physics. The photon gas can thermalize via repeated absorption and reemission events with the dye molecules, ultimately leading to the formation of a Bose-Einstein condensate (BEC).

The light-matter interaction inside the microcavity can be modelled using a nonequilibrium model. Taking into account the intrinsic losses and the absorption and emission processes, the equation of motion for the photon-molecule system is given by the Markovian Lindblad equation:

```
dp/dt =  -i[H_0,p] + sum_(l,m) { k L[a_m] + G_up L[s^+_l] +  G_down L[s^m_l] + Am L[a_m s^+_l] + Em L[a^p_m s^-_l]} p,

```

where ```p``` is the photon-molecule density matrix, ```H_0``` is the energy-conserving Hamiltonian, and ```L[x]p = 1/2{x^+ x,p} - x p x^+``` is the Lindblad operator. Here, ```x^+``` is the complex conjugate of ```x```. Moreover, ```a_m``` and ```a^p_m``` are the photon annihilation and creation operators for the cavity mode ```m```, whereas ```s^-_l``` and ```s^+_l``` are the lowering and raising Pauli operator for the ```lth``` molecule. Importantly, ```A_m``` and ```E_m``` are the abosrption and emission rates for mode ```m```, and ```G_up``` is the rate of incoherent pumping of the molecules, ```k``` is the rate of photon loss and ```G_down``` is loss rate due to fluorescence in all noncavity modes.

The above master equation can be used to derive a set of coupled equations of motion for the photon and molecular excitations:

```
dn_m/dt =  - k n_m + \sum_j {E_m g_(m,j) M f_j(n_m+1) - A_m g_(m,j) M (1-f_j) n_m}, and

df_j/dt = -{G_down + \sum_m E_m g_(m,j)(n_m+1)}f_j + {G_up + \sum_m A_m g_(m,j)(n_m)}(1-f_j)}.

```

The molecules in the cavity have been divided into spatial bins (```j```), with ```f_j``` being the fraction of molecular excitations. The photon population in mode ```m``` is given by ```n_m```. Furthermore, ```g_(m,j)``` is the coupling between the mode ```m```with molecules in the spatial bin ```l```. We note that in the above equations we assume that the mode-mode coherence is weak and can be ignored if one is only interested in studying the photon and molecular excitations.

## Outline

The `PyPBEC` package is used to solve both the steady-state and the time evolution of both the photonic, ```n_m``` and molecular populations ```f_j```.

- The steady-state populations are obtained by directly solving the algebraic equations ```dn_m / dt``` and ```df_j/dt = 0 ``` - implemented in the `PyPBEC.Solver.SteadyState` solver - or by examining the long-time solution of the above equations of motion.

- The mean-field dynamics, determined by the above equations of motion, are calculated by the `PyPBEC.Solver.ODE` solver.

- The stochastic evolution of the photonic and molecular population can be solved directly from the Lindblad master equation by adopting a stochastic solver based on the quantum trajectories method - implemented in the `PyPBEC.Solver.MonteCarlo`. This is particular relevant to obtain correlation function and beyond-mean-field effects.

All solver classes share the same calling structure. In particular, they all contain the methods `set_initial_photons()` and `set_initial_molecules()`, used to define the initial conditions. Note that this is not necessary, however, in the `SteadyState` solver.

While the `Solver` module simply implements the dynamical model, all the cavity properties are separately set using the `PyPBEC.Cavity.Cavity` class. Here, one can define the number of photonic (```M```) and molecular (```J```) modes, all the rates involved in the system, the pump details and the coupling terms ```g_(m,j)```. Please refer to the full package reference for further details.

The `Solver` class, when instantiated takes an `Cavity` object as an argument, then fully determining the system.

The `Solver` module together with the `Cavity` class allow for complete freedom in setting the cavity properties and solving its dynamics. The `PyPBEC` package contains, however, several tools that allow a higher level way to define the details of the cavity:

* The `PyPBEC.Cavity.Modes` class allows the user to define arbitrary cavity geometries, which maps directly to the external potential felt by the photons. The cavity eigenmodes are then obtained by solving the time-independent Schrodinger equation. This allows the direct of the coupling terms ```g_(m,j)```.

* The `PyPBEC.OpticalMedium.OpticalMedium` class allows the user to get the properties (emission and absorption rates) off some commonly used optical media. Currently the `PyPBEC` package contains experimental data on the following media:
  - [Rhodamine6G](https://doi.org/10.5281/zenodo.569817)



## Typical code structure

Let us look at a simple example to find the steady state solution of a 2D photon gas inside with an harmonic trapping potential. The `examples` folder contains other examples.

#### Defining the cavity mode structure

The first step is to calculate the mode structure of a cavity made up of one planar and one spherical mirror, which we know to map onto an harmonic trapping potential.

We begin by import the `PyPBEC.Cavity.Modes` class and setup a cavity of (maximum) length `L0`, in a 2D numerical grid of total size, `grid_size` (in both directions), and resolution `grid_delta`. Also, `q` is longitudinal mode number, `n` the refractive index of the optical medium and `n_modes` the number of cavity modes to be calculated:

```python
from PyPBEC.Cavity import Modes
cavity_modes = Modes(grid_size,grid_delta,L0,q,n,n_modes)
```

Notably, the ```Modes``` class allows for the design of an arbitrary cavity geometry (arbitrary trapping potential) by using the `set_geometry()` method. Some typical geometries, such as the spherical and box potential are already pre-implemented. Here, we consider a spherical geometry (one planar and one spherical mirror) with radius, `feature_RoC` and depth `feature_depth`. The `compute_cavity_modes()` method returns the wavelength and the spacial structure (squared wavefunction) of the cavity eigenmodes:

```python
cavity_modes.set_geometry_spherical(feature_RoC,feature_depth)
lambdas, modes = cavity_modes.compute_cavity_modes()
g = cavity_modes.get_coupling_matrix()
```

#### Defining the properties of the optical medium

The properties of the optical medium (emission and absorption rates) are set by calling the `set_cavity_emission_rates()` and `set_cavity_absorption_rates` of the `Cavity` class. The user may be interested however in grabbing these quantities from the list of available optical media in the `PyPBEC` package:

```python
from PyPBEC.OpticalMedium import OpticalMedium
R6G = OpticalMedium(optical_medium="Rhodamine6G")
absorption_rates, emission_rates = R6G.get_rates(lambdas,dye_concentration,n)
```

Here, `lambdas` are the wavelengths of the cavity modes computed earlier, and `dye_concentration` is the molar concentration of Rhodamine 6G. As before, `n` is the refractive index.

#### Defining the cavity parameters

All the physical parameters of the system can now be loaded onto a `Cavity` object:
```Python
from PyPBEC.Cavity import Cavity
```

- Properties of the photonic modes:

```python

cavity = Cavity(M=n_modes, J=g.shape[1])
cavity.set_cavity_loss_rates(cavity_loss_rates)
cavity.set_cavity_emission_rates(emission_rates)
cavity.set_cavity_absorption_rates(absorption_rates)
```
Here, `M` is the number of modes and `J` is the number of spatial molecular bins.

- Properties of the molecular modes:

```python
cavity.set_reservoir_decay_rates(Gamma_down*np.ones(g.shape[1]))
cavity.set_reservoir_pump_rates(np.reshape(pump, [pump.shape[0]*pump.shape[1]]))
molecular_population = np.array(sc.Avogadro*dye_concentration*(0.5*L0*grid_delta**2)*np.ones(g.shape[1]), dtype=int)
cavity.set_reservoir_population(molecular_population)
```
`Gamma_down` here is the loss due to emission in noncavity modes and `pump` is the rate of incoherent pumping in each molecular bin. Moreover, `molecular_population` is the number of molecules in each spatial bin.

- Coupling between photonic and molecular modes:

```python
cavity.set_coupling_terms(coupling_terms=g)
```
The photon-molecule coupling `g` was computed earlier by the `Modes` object.


#### Solving for the steady-state photon population and molecular excitation


In the example, we use the equations of motion to compute the steady state values of the photon number of the `M` modes and molecular excitation in each of the `J` spatial bins. This is done by importing the `Solver.Steadystate` solver class:

```python
from PyPBEC.Solver import SteadyState

solver_steadystate = SteadyState(cavity,ANNEALING=False)
solver_steadystate.set_initial_photons(initial_photons)
solver_steadystate.set_initial_excited_molecules(initial_excited_molecules)  
solved_cavity_steadystate = solver_steadystate.solve()
```

`ANNEALING` (bool):	If True, slowing increases the pump up to the user defined value, computing steady-state solutions at all cases and using them as initial guesses for the next pump value. This helps with numerical stability.

If the system is initially unexcited the `initial_photons` and `initial_excited_molecules` are simply arrays with zeros.

```python
initial_photons = np.zeros(M)
initial_excited_molecues = np.zeros(J)
```

The steady state photon number and the molecular excitation can then be retrieved as:

```python
solved_cavity_steadystate.photons
solved_cavity_steadystate.excited_molecules
```
