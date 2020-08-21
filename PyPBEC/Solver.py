"""
Written by: Joao Rodrigues
June 2020



"""

import numpy as np
import copy
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath('..'))




class Solver():

	"""
		Solver() is an abstraction class that must be inherited by particular solvers.

	"""

	def __init__(self, cavity_obj, **kwargs):
		"""
			Parameters:
		
				cavity_obj (Cavity object):		The cavity object.
				T (float):						Total integration time


		"""

		self.T = None
		self.DYNAMICS = False
		for name, value in kwargs.items():
			setattr(self, name, value)
		from PyPBEC.Cavity import Cavity
		if not isinstance(cavity_obj, Cavity):
			raise Exception("Solver expects a Cavity() object")
		self.cavity_obj = copy.deepcopy(cavity_obj)
		if (not self.T == None) and self.T <= 0:
			raise Exception("Total integration time must be positive")
		self.check_solver_specific_parameters()



	def check_solver_specific_parameters(self):
		pass



	def set_initial_photons(self, initial_photons):
		"""
			Parameters:

				initial_photons (numpy array):	Array with the initial number of photons in each cavity mode.
												Expected shape is (M,)

		"""
		initial_photons = np.array(initial_photons, dtype=float)
		if not (len(initial_photons.shape)==1 and initial_photons.shape[0] == self.cavity_obj.M):
			raise Exception("Shape of initial photon population not consistent with {0} photonic modes".format(self.cavity_obj.M))
		self.initial_photons = initial_photons
		self.DYNAMICS = True


	def set_initial_excited_molecules(self, initial_excited_molecules):
		"""
			Parameters:

				initial_excited_molecules (numpy array):	Array with the initial number of excited molecules in spatial bin. 
													Expected shape is (J,)

		"""
		initial_excited_molecules = np.array(initial_excited_molecules, dtype=float)
		if not (len(initial_excited_molecules.shape)==1 and initial_excited_molecules.shape[0] == self.cavity_obj.J):
			raise Exception("Shape of initial excited molecular population not consistent with {0} spatial bins".format(self.cavity_obj.J))
		if any([initial_excited_molecules[i]>self.cavity_obj.mols[i] for i in range(0, self.cavity_obj.J)]):
			raise Exception("Some bins are initialized with a fraction of excited molecules above one")
		self.initial_excited_molecules = initial_excited_molecules
		self.DYNAMICS = True


	def solve(self):
		if self.DYNAMICS == True:
			self.cavity_obj.photons[0] = 1.0*self.initial_photons
			self.cavity_obj.emols[0] = 1.0*self.initial_excited_molecules
			self.cavity_obj.gmols[0] = self.cavity_obj.mols - self.initial_excited_molecules
		self.call_solver()
		return copy.deepcopy(self.cavity_obj)


	def call_solver(self):
		raise Exception("Solver not implemented")
		



class MonteCarlo(Solver):

	"""
		Solves the incoherent cavity dynamics by performing a stochastic Monte Carlo realization.

		Parameters:
			n_point (int):		Number of points to output the solution. These points will not be exact, but instead the algorithm will
								...search the nearby time instant in the stochastic evolution

	"""


	def check_solver_specific_parameters(self):
		if not hasattr(self, "n_points"):
			raise Exception("Monte Carlo Solver needs a n_points variable")
		if not type(self.n_points)==int:
			raise Exception("n_points must be an integer")


	def call_solver(self):

		while self.cavity_obj.t[-1] < self.T:

			##### total rates array
			f = self.cavity_obj.emols[-1] / (self.cavity_obj.emols[-1]+self.cavity_obj.gmols[-1])
			rates = []
			# cavity loss
			[rates.append(self.cavity_obj.rates_kappa[i]*self.cavity_obj.photons[-1][i]) for i in range(0, self.cavity_obj.M)]
			# molecular emission
			[rates.append(self.cavity_obj.rates_E[i]*(self.cavity_obj.photons[-1][i]+1)*np.sum(self.cavity_obj.g[i,:]*f)) for i in range(0, self.cavity_obj.M)]
			# molecular absorption
			[rates.append(self.cavity_obj.rates_A[i]*self.cavity_obj.photons[-1][i]*np.sum(self.cavity_obj.g[i,:]*(1-f))) for i in range(0, self.cavity_obj.M)]
			# molecular decay into non-cavity modes
			rates.append(np.sum(self.cavity_obj.rates_Gamma_down*f))
			# molecular pump 
			rates.append(np.sum(self.cavity_obj.rates_Gamma_up*(1-f)))

			##### Random dynamics
			# event time
			event_time = np.random.exponential(1.0/np.sum(rates))
			# event
			probabilities = np.array(rates, dtype=float) / np.sum(rates)
			event_type = np.random.choice(a=len(rates), p=probabilities)

			##### Prepares the population arrays
			self.cavity_obj.photons.append(copy.deepcopy(self.cavity_obj.photons[-1]))
			self.cavity_obj.gmols.append(copy.deepcopy(self.cavity_obj.gmols[-1]))
			self.cavity_obj.emols.append(copy.deepcopy(self.cavity_obj.emols[-1]))
			self.cavity_obj.t.append(self.cavity_obj.t[-1]+event_time)
			
			##### Updates the populations
			# Cavity loss event
			if event_type <= self.cavity_obj.M-1:
				cavity_mode_index = event_type
				self.cavity_obj.photons[-1][cavity_mode_index] -= 1
			# molecular emission event
			elif event_type <= 2*self.cavity_obj.M-1:
				cavity_mode_index = event_type - self.cavity_obj.M
				aux_p = self.cavity_obj.emols[-1]*self.cavity_obj.g[cavity_mode_index,:]
				aux_p = aux_p/np.sum(aux_p)
				molecular_mode_index = np.random.choice(a=self.cavity_obj.J, p=aux_p)
				self.cavity_obj.photons[-1][cavity_mode_index] += 1
				self.cavity_obj.emols[-1][molecular_mode_index] -= 1
				self.cavity_obj.gmols[-1][molecular_mode_index] += 1
			# molecular absorption event
			elif event_type <= 3*self.cavity_obj.M-1:
				cavity_mode_index = event_type - 2*self.cavity_obj.M
				aux_p = self.cavity_obj.gmols[-1]*self.cavity_obj.g[cavity_mode_index,:]
				aux_p = aux_p/np.sum(aux_p)
				molecular_mode_index = np.random.choice(a=self.cavity_obj.J, p=aux_p)
				self.cavity_obj.photons[-1][cavity_mode_index] -= 1
				self.cavity_obj.emols[-1][molecular_mode_index] += 1
				self.cavity_obj.gmols[-1][molecular_mode_index] -= 1
			# molecular decay into non-cavity modes
			elif event_type == 3*self.cavity_obj.M:
				aux_p = self.cavity_obj.rates_Gamma_down*self.cavity_obj.emols[-1] / np.sum(self.cavity_obj.rates_Gamma_down*self.cavity_obj.emols[-1])
				molecular_mode_index = np.random.choice(a=self.cavity_obj.J, p=aux_p)
				self.cavity_obj.emols[-1][molecular_mode_index] -= 1
				self.cavity_obj.gmols[-1][molecular_mode_index] += 1
			# molecular pump
			elif event_type == 3*self.cavity_obj.M+1:
				aux_p = self.cavity_obj.rates_Gamma_up*self.cavity_obj.gmols[-1] / np.sum(self.cavity_obj.rates_Gamma_up*self.cavity_obj.gmols[-1])
				molecular_mode_index = np.random.choice(a=self.cavity_obj.J, p=aux_p)
				self.cavity_obj.emols[-1][molecular_mode_index] += 1	
				self.cavity_obj.gmols[-1][molecular_mode_index] -= 1			
			else:
				raise Exception("Coding error")

		# Selects the dynamics at uniformly distributed time instants
		t_points = np.linspace(0, self.T, self.n_points)
		time = np.array(self.cavity_obj.t)
		indices = [np.argmin(np.abs(t_points[i]-time)) for i in range(0, len(t_points))]

		# Saves the solution
		self.cavity_obj.load_dynamics(
			t=time[indices], 
			photons=np.array(self.cavity_obj.photons)[indices,:],
			emols=np.array(self.cavity_obj.emols)[indices,:],
			gmols=np.array(self.cavity_obj.gmols)[indices,:])



	
class ODE(Solver):

	"""
		Solves the incoherent cavity dynamics by integrating the mean-field rate equations.

		Parameters:
			n_point (int):		Number of points to calculate the mean-field solution.

	"""


	def check_solver_specific_parameters(self):
		if not hasattr(self, "n_points"):
			raise Exception("ODE Solver needs a n_points variable")
		if not type(self.n_points)==int:
			raise Exception("n_points must be an integer")


	def call_solver(self):
		# Defines the system of equations
		def dydt(t, y):
			photons = y[0:self.cavity_obj.M]
			excited_molecules = y[self.cavity_obj.M:]
			derivatives = list()
			# photonic part
			f = excited_molecules/self.cavity_obj.mols
			[derivatives.append(
				-self.cavity_obj.rates_kappa[i]*y[i]
				+self.cavity_obj.rates_E[i]*(y[i]+1)*np.sum(self.cavity_obj.g[i,:]*f)
				-self.cavity_obj.rates_A[i]*y[i]*np.sum(self.cavity_obj.g[i,:]*(1-f)))
				for i in range(0, self.cavity_obj.M)]
			# molecules part
			[derivatives.append(
				-(self.cavity_obj.rates_Gamma_down[i]+np.sum(self.cavity_obj.g[:,i]*self.cavity_obj.rates_E*(photons+1)))*y[i+self.cavity_obj.M]
				+(self.cavity_obj.rates_Gamma_up[i]+np.sum(self.cavity_obj.g[:,i]*self.cavity_obj.rates_A*photons))*(self.cavity_obj.mols[i]-y[i+self.cavity_obj.M]))
				for i in range(0, self.cavity_obj.J)]
			return np.array(derivatives, dtype=float)

		# Defines the initial conditions
		y0 = list()
		[y0.append(self.initial_photons[i]) for i in range(0, len(self.initial_photons))]
		[y0.append(self.initial_excited_molecules[i]) for i in range(0, len(self.initial_excited_molecules))]

		# Solves the initial value problem
		t_eval = np.linspace(0, self.T, self.n_points)
		sol = solve_ivp(dydt, t_span=(0, self.T), y0=y0, t_eval=t_eval, vectorized=False)

		# Saves the solution
		aux_gmols = np.array([self.cavity_obj.mols-sol.y[self.cavity_obj.M:,i] for i in range(0, self.n_points)])
		self.cavity_obj.load_dynamics(
			t=sol.t, 
			photons=np.transpose(sol.y[0:self.cavity_obj.M,:]),
			emols=np.transpose(sol.y[self.cavity_obj.M:,:]),
			gmols=aux_gmols)




class SteadyState(Solver):

	"""
		Solves the steady-state of the mean-field rate equations.

		Parameters:
			ANNEALING (bool):	If True, slowing increases the pump up to the user defined value, computing steady-state solutions
								at all cases and using them as initial guesses for the next pump value. Default = True
	
	"""


	def check_solver_specific_parameters(self):

		if not hasattr(self, "ANNEALING"):
			self.ANNEALING = True



	def call_solver(self):
		# Defines the system of equations
		def dydt(y):
			photons = y[0:self.cavity_obj.M]
			excited_molecules = y[self.cavity_obj.M:]
			derivatives = list()
			# photonic part
			f = excited_molecules/self.cavity_obj.mols
			[derivatives.append(
				-self.cavity_obj.rates_kappa[i]*y[i]
				+self.cavity_obj.rates_E[i]*(y[i]+1)*np.sum(self.cavity_obj.g[i,:]*f)
				-self.cavity_obj.rates_A[i]*y[i]*np.sum(self.cavity_obj.g[i,:]*(1-f)))
				for i in range(0, self.cavity_obj.M)]
			# molecules part
			[derivatives.append(
				-(self.cavity_obj.rates_Gamma_down[i]+np.sum(self.cavity_obj.g[:,i]*self.cavity_obj.rates_E*(photons+1)))*y[i+self.cavity_obj.M]
				+(self.cavity_obj.rates_Gamma_up[i]+np.sum(self.cavity_obj.g[:,i]*self.cavity_obj.rates_A*photons))*(self.cavity_obj.mols[i]-y[i+self.cavity_obj.M]))
				for i in range(0, self.cavity_obj.J)]
			return np.array(derivatives, dtype=float)

		# Defines the starting point
		y0 = list()
		[y0.append(self.initial_photons[i]) for i in range(0, len(self.initial_photons))]
		[y0.append(self.initial_excited_molecules[i]) for i in range(0, len(self.initial_excited_molecules))]

		# Gradually increases the pump value and uses each iteration as the initial guess for the next
		if self.ANNEALING:
			n_annealing_steps = 20
		else:
			n_annealing_steps = 1
		aux = copy.deepcopy(self.cavity_obj.rates_Gamma_up)
		values = np.flip([1.0 / (2.0**i) for i in range(0, n_annealing_steps)])
		for value in values:
			self.cavity_obj.rates_Gamma_up = value*aux
			sol = root(dydt, x0=y0)
			y0 = copy.deepcopy(sol.x)

		# Saves the solution
		self.cavity_obj.load_dynamics(
			t=np.array([-np.inf]), 
			photons=np.array([sol.x[0:self.cavity_obj.M]]),
			emols=np.array([sol.x[self.cavity_obj.M:]]),
			gmols=self.cavity_obj.mols-np.array([sol.x[self.cavity_obj.M:]]))		



		