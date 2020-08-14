# PyBEC

## A module in Python to solve the nonequilibrium model of photon condensation

The `PyBEC` repository contains different approaches to solve the theoretical model introduced by [Kirton and Keeling
in *Nonequilibrium Model of Photon Condensation*](https://doi.org/10.1103/PhysRevLett.111.100404). 

The present version of the repository includes a preliminary set of solvers to obtain both steady-state and time-dynamics as discussed in the following papers:

[1] [*Thermalization and breakdown of thermalization in photon condensates*](https://doi.org/10.1103/PhysRevA.91.033826), P. Kirton and J. Keeling, Phys. Rev. A **91**, 033826 (2015).

[2] [*Spatial dynamics, thermalization, and gain clamping in a photon condensate*](https://doi.org/10.1103/PhysRevA.93.013829), J. Keeling and P. Kirton, Phys. Rev. A **93**, 013829 (2016).

[3] [*Decondensation in Nonequilibrium Photonic Condensates: When Less Is More*](https://doi.org/10.1103/PhysRevLett.120.040601), H. J. Hesten, R. A. Nyman, and F. Mintert, Phys. Rev. Lett. **120**, 040601 (2018).

[4] [*Non-stationary statistics and formation jitter in transient photon condensation*](https://doi.org/10.1038/s41467-020-15154-7), B. T. Walker, J. D. Rodrigues, H. S. Dhar, R. F. Oulton, F. Mintert, and R. A. Nyman, Nat. Commun. **11**, 1390 (2020).

We are still in the process of builiding a more complete documentation and we will keep updating this page as we further expand the repository by adding more solvers to `PyBEC`.


## Installation

The repository uses Python 3, which we assume is preinstalled in the users system. Else, install the latest version of [Python](https://www.python.org/downloads/). The following packages are recommended for installation along with Python:

*numpy*<br/>
*scipy*<br/>
*matplotlib*<br/>
*notebook*<br/>
*jupyter*


To access the solvers in `PyBEC`, clone the repository to your local machine using Git. 

```
git clone git@github.com:photonbec/PyPBEC.git
```

To install Git, [check this](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Else, you can directly download the zip of the repository from GitHub.

## Brief theory

In recent years, two-dimensional photon gas inside a dye-filled microcavity has become a useful tool for studying both equilibrium and nonequlibrium physics. The photon gas can thermalize via repeated absorption and reemission of photons by the dye molecules, ultimately leading to the formation of a near-equilibrium Bose-Einstein condensate (BEC). 

The light-matter interaction inside the microcavity can be modelled using a nonequilibrium model. Taking into account the intrinsic losses and the absorption and emission processes, the equation of motion for the photon-molecule system is given by a Markovian Lindblad equation. 


## Outline

The `PyBEC` module is primarily used to obtain the following 

- Both the transient and the steady state population of the cavity mode(s) and the number of excited molecules, and
- The time-dependent second-order correlation function and the photon statistics of the cavity mode(s)