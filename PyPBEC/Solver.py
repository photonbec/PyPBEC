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
		
				cavity_obj (Cavity object):		The cavity object. The cavity object is passed as a reference. This means that the
												cavity object inside and outside the solver object is actually the same. This makes
												it easy to change the pump power, for instance, without having to instantiate a 
												new Solver object
				T (float):						Total integration time


		"""

		self.T = None
		self.DYNAMICS = False
		for name, value in kwargs.items():
			setattr(self, name, value)
		from PyPBEC.Cavity import Cavity
		if not isinstance(cavity_obj, Cavity):
			raise Exception("Solver expects a Cavity() object")
		self.cavity_obj = cavity_obj
		if (not self.T == None) and self.T <= 0:
			raise Exception("Total integration time must be positive")
		self.check_solver_specific_parameters()

		# Resets the cavity populations, in the cavity object
		self.cavity_obj.reset_cavity_populations()	



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

		##### Checks if populations are integers:
		labels = ["excited molecular", "ground-state molecular", "photon"]

		pops = [self.cavity_obj.emols[-1], self.cavity_obj.gmols[-1], self.cavity_obj.photons[-1]]
		for label, pop in zip(labels, pops):
			i = 0
			for j in range(0, len(pop)):
				i += 1
				if not int(pop[j])==pop[j]:
					if i==1:
						print("Warning: Monte-Carlo solver found non-integer values in the "+label+" population")
					pop[j] = int(pop[j])



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

		# Defines the initial conditions
		y0 = list()
		[y0.append(self.initial_photons[i]) for i in range(0, len(self.initial_photons))]
		[y0.append(self.initial_excited_molecules[i]) for i in range(0, len(self.initial_excited_molecules))]

		# Solves the initial value problem
		t_eval = np.linspace(0, self.T, self.n_points)
		sol = solve_ivp(self.dydt, t_span=(0, self.T), y0=y0, t_eval=t_eval, vectorized=False)

		# Saves the solution
		aux_gmols = np.array([self.cavity_obj.mols-sol.y[self.cavity_obj.M:,i] for i in range(0, self.n_points)])
		self.cavity_obj.load_dynamics(
			t=sol.t, 
			photons=np.transpose(sol.y[0:self.cavity_obj.M,:]),
			emols=np.transpose(sol.y[self.cavity_obj.M:,:]),
			gmols=aux_gmols)


	def dydt(self, t, y): # rate equations

		# Cavity parameters
		Abs = self.cavity_obj.rates_A
		Emi = self.cavity_obj.rates_E
		Mol = self.cavity_obj.mols
		ka = self.cavity_obj.rates_kappa
		gDown = self.cavity_obj.rates_Gamma_down
		g = self.cavity_obj.g
		pump = self.cavity_obj.rates_Gamma_up		

		photons = y[0:self.cavity_obj.M]
		excited_molecules = y[self.cavity_obj.M:]
		f = excited_molecules/self.cavity_obj.mols

		dn_dt = -ka*photons + Emi*(photons+1)*(g@f) - Abs*photons*(g@(-f+1))
		df_dt = -(gDown+(g.T@(Emi*(photons+1))))*excited_molecules + (pump+(g.T@(Abs*photons)))*(Mol-excited_molecules)
		y_dot = np.array(list(dn_dt)+list(df_dt))

		return y_dot


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









class RotatedBasisODE(Solver):

	"""
		Solves the cavity dynamics by performing an hierarchical dimensionality reduction on the molecular reservoir basis.
		This reduces the overall number of equations at different levels of accuracy in relation to the original set of
		differential equations. The smaller number of equations allows a faster computation.
		More details check: 
		Walker et al, Physical Review A 100, 053828 (2019), link: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.053828

		Parameters:
			order (int):		Order of the rotated basis approximation
			VERBOSE (bool):		If True, plots information about rotated basis. (default=True)

	"""
		

	def check_solver_specific_parameters(self):

		if not hasattr(self, "order"):
			raise Exception("Please set the order of the rotated basis hierarchical approximation")
		if not hasattr(self, "VERBOSE"):
			self.VERBOSE = True	


	def call_solver(self):

		if not hasattr(self, "mode_num"):
			self.mode_num, self.basis_num, self.RinvWinv, self.RW, self.RinvWinvGdA, self.RLRT = self.setup_rotated_basis()

		# Defines the initial conditions
		y0 = np.zeros(self.mode_num+self.basis_num)
		print("***")
		print("Warning: Initial conditions are set to zero in the rotated basis solver.")
		print("A future release will implement the ability to defined non-zero initial conditions")
		print("***")

		# Solves the initial value problem
		t_eval = np.linspace(0, self.T, self.n_points)
		sol = solve_ivp(self.dydt, t_span=(0, self.T), y0=y0, t_eval=t_eval, vectorized=False)

		# Saves the solution
		self.cavity_obj.load_dynamics(
			t=sol.t, 
			photons=np.transpose(sol.y[0:self.cavity_obj.M,:]),
			emols=[None]*self.cavity_obj.J,
			gmols=[None]*self.cavity_obj.J)



	def setup_rotated_basis(self):

		g = self.cavity_obj.g
		order = self.order
		Abs = self.cavity_obj.rates_A
		Emi = self.cavity_obj.rates_E
		M = self.cavity_obj.M
		J = self.cavity_obj.J

		# singular value decomposition
		U, D, Vd = np.linalg.svd(g)
		DMat = np.zeros(np.shape(g), dtype=float)
		sz = np.min(np.shape(DMat))
		DMat[:sz, :sz] = np.diag(D)
		

		Dinv = np.zeros(np.shape(g)[::-1], dtype=float)
		sz = np.min(np.shape(Dinv))
		Dinv[:sz, :sz] = np.diag(1./D)

		Bp = np.linalg.inv(U.dot(DMat)[:,:M])
		B = np.identity(J)
		B[:M,:M] = copy.deepcopy(Bp)
		
		def HConj(mat):
			return np.conj(mat).T
		
		W = HConj(Vd).dot(B)
		Winv = np.linalg.inv(W)
		
		LMats = []
		for i in range(M):
			L = Winv.dot(np.diag(g[i])@W)
			LMats.append(L)
		
		fZero = np.zeros([M,J])
		fZero[:M,:M] = np.diag([1]*M)
		
		def calcRinvLoop():
			prevO = list(fZero)
			allVs = copy.deepcopy(prevO)
			
			independenceValLg = 1e-5     
			independenceValSm = 1e-15

			orderSzA = [len(prevO)]
			orderBndA = [len(prevO)]
			for orderI in range(1, order):
				tmp = np.array(LMats).dot(np.transpose(prevO))
				vs = np.reshape(np.transpose(tmp, (0,2,1)), (M*len(prevO),J))

				nextO = []
				for v in vs:
					i = 0
					while(True):
						mOver = 0
						v /= np.sqrt(v.dot(v))
						for u in allVs:
							ov = u.dot(v)
							if ov>mOver:
								mOver = ov
							v -= u*(ov)
						rsz = v.dot(v)
						if rsz < independenceValLg:
							break
						if rsz>1+1e-5:
							raise Exception("q")
						i += 1
						if mOver<independenceValSm:
							v /= np.sqrt(v.dot(v))
							allVs.append(v)
							nextO.append(v)
							break
				prevO = nextO

				if len(nextO) == 0:
					break

				orderSzA.append(len(nextO))
				orderBndA.append(len(allVs))

			return [np.array(allVs),orderSzA,orderBndA]
		
		Rinv, orderSzA, orderBndA = calcRinvLoop()
		R = HConj(Rinv)
		RinvWinv = Rinv.dot(Winv)
		RW = W.dot(R)
		RinvWinv1 = Rinv.dot(np.sum(Winv,1))
		RinvWinvGdA = Rinv.dot(Winv).dot(HConj(g))*Abs   # RinvWinvGdE = Rinv.dot(Winv).dot(HConj(g))*Emi
		RLR = []
		for L in np.array(LMats):
		    RLR.append(Rinv.dot(L).dot(np.transpose(Rinv)) )
		RLRT = np.transpose(RLR,[1,2,0])
		basis_num = len(Rinv)
		mode_num = M

		if self.VERBOSE:
			print("-> Rotated basis ode solver:")
			print("    -> Total number of photonic modes                  = ", str(mode_num))
			print("    -> Total number of (real space) molecular modes    = ", str(J))
			print("    -> Total number of (rotated space) molecular modes = ", str(basis_num))

		return mode_num, basis_num, RinvWinv, RW, RinvWinvGdA, RLRT


	def dydt(self, t, y): # rate equations
		
		# Cavity parameters
		Abs = self.cavity_obj.rates_A
		Emi = self.cavity_obj.rates_E
		Mol = self.cavity_obj.mols
		ka = self.cavity_obj.rates_kappa
		gDown = self.cavity_obj.rates_Gamma_down
		g = self.cavity_obj.g
		pump = self.cavity_obj.rates_Gamma_up

		mode_num, RinvWinv, RW, RinvWinvGdA, RLRT = self.mode_num, self.RinvWinv, self.RW, self.RinvWinvGdA, self.RLRT

		n = y[:mode_num]
		fh = y[mode_num:]

		#dn_dt = -ka*n + n*(Emi+Abs)*fh[:mode_num] + Emi*fh[:mode_num] - n*Abs*np.sum(g*Mol, 1)
		#dm_dt = RinvWinv.dot(pump*Mol) - (RinvWinv*(pump+gDown)).dot(RW).dot(fh) + RinvWinv.dot(Mol)*RinvWinvGdA.dot(n) - RLRT.dot(((Abs + Emi)*n + Emi)).dot(fh)

		dn_dt = -ka*n + n*(Emi+Abs)*fh[:mode_num] + Emi*fh[:mode_num] - n*Abs*np.sum(g, 1)
		dm_dt = RinvWinv.dot(pump) - (RinvWinv*(pump+gDown)).dot(RW).dot(fh) + RinvWinvGdA.dot(n) - RLRT.dot(((Abs+Emi)*n + Emi)).dot(fh)

		y_dot = np.array(list(dn_dt)+list(dm_dt))

		return y_dot



