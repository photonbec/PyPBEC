"""

--------
This code implements the stochastic dynamics of a couple of degenerate lasing modes inside a 
dye-filled microcavity, using the 'MonteCarlo' solver in the PyPBEC package.

The results of this code were discussed in the conference proceeding:

"Phase transitions of light in a dye-lled microcavity: observations and simulations"
-- R.A. Nyman, H.S. Dhar, J.D. Rodrigues, R.F. Oulton and F. Mintert

--------

"""

import os
import sys
sys.path.insert(0, os.path.abspath('../conf_plots_hsd/'))
import numpy as np

from PyPBEC.Cavity import Cavity
cavity = Cavity(M=2, J=1, cavity_loss_rates=[1,1], cavity_emission_rates=[50,50], \
    cavity_absorption_rates=[10, 10], reservoir_decay_rates=[1], \
        reservoir_population=[200], coupling_terms=[[1],[1]])
cavity.set_reservoir_pump_rates(np.array([500]))

from PyPBEC.Solver import ODE, MonteCarlo
solver_ode = ODE(cavity_obj=cavity, T=10, n_points=100)
solver_ode.set_initial_photons(np.array([0,0]))
solver_ode.set_initial_excited_molecules(np.array([0]))
solved_cavity_ode = solver_ode.solve()

solver_mc = MonteCarlo(cavity_obj=cavity, T=10, n_points=200)
solver_mc.set_initial_photons(solved_cavity_ode.photons[-1])
solver_mc.set_initial_excited_molecules(solved_cavity_ode.emols[-1])
solved_cavity_mc = solver_mc.solve()

# Save data to a dictionary (or a numpy file)

data = {'time_ODE':solved_cavity_ode.t,'pops_ODE':solved_cavity_ode.photons,'time_MC':solved_cavity_mc.t,'pops_MC':solved_cavity_mc.photons}
# np.save("data_stochastic.npy",data,allow_pickle=True)

# Plot the data

from matplotlib import *
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Computer Modern Sans serif'],'size':20})

def stochastic_plot(save_fig = False,data=data):
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    labels = ['Mode 0','Mode 1']
    markers = ['x',None]
    for i in range(np.shape(data['pops_ODE'])[1]):
        ax1.plot(data['time_ODE'],np.transpose(data['pops_ODE'])[i],label=labels[i],marker=markers[i],markevery=3)
        ax2.plot(data['time_MC'],np.transpose(data['pops_MC'])[i],marker=markers[i],markevery=3)
    ax1.legend(loc='best')
    ax1.set_ylabel('Population',labelpad=10)
    ax1.set_xlabel(r'Time (in units of $\kappa$)',labelpad=10)
    ax2.set_xlabel(r'Time (in units of $\kappa$)',labelpad=10)
    tight_layout(pad = 2.0)
    if save_fig:
        savefig('stochastic_pop.pdf',bbox_inches='tight',dpi=300)
    else:
        plt.show()

stochastic_plot()

