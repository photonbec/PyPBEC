"""
Microlaser example
--------
This short script is intended to be a minimum working example 
of the PyPBEC package. It solves for the steady-state
population of a single-mode microlaser as a function of pump
rate, using Rhodamine 6G as the gain medium.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt

from PyPBEC.Cavity import Cavity
from PyPBEC.Solver import SteadyState

# Cavity object
cavity = Cavity(M=1, J=1)
cavity.set_cavity_loss_rates(rates=[1.0])
cavity.set_cavity_emission_rates(rates=[10.0])
cavity.set_cavity_absorption_rates(rates=[0.0])
cavity.set_reservoir_decay_rates(rates=[500.0])
cavity.set_reservoir_population(population=[1e9])
cavity.set_coupling_terms(coupling_terms=[[1.0]])

# Solver
pumps = np.geomspace(1.0, 1000, 100)
populations = list()
for pump in pumps:
	cavity.set_reservoir_pump_rates(rates=[pump])
	solver = SteadyState(cavity_obj=cavity, ANNEALING=True)
	solver.set_initial_photons(initial_photons=[0])
	solver.set_initial_excited_molecules(initial_excited_molecules=[0])  
	solution = solver.solve()
	populations.append(solution.photons[0])

# Plot
plt.plot(pumps, populations)
plt.xlabel("Pump (arb. units)")
plt.ylabel("Photon number")
plt.xscale("log")
plt.yscale("log")
plt.show()

