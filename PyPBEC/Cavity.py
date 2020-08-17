"""
Written by: Joao Rodrigues
June 2020



"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib
import copy

class Cavity():
	
	"""
		A Cavity() object defines all the physical properties of the cavity, including details of both the photonic and molecular modes

	"""

	def __init__(self, M, J):
		"""
			Parameters:

				M (int):	Number of photonic modes
				J (int):	Number of molecular modes. This is simply the number of spatial bins

		"""
		self.M = M
		self.J = J
		self.t = [0.0]
		self.photons = [np.zeros(self.M)]
		self.gmols = [np.zeros(self.J)]
		self.emols = [np.zeros(self.J)]


	def set_cavity_loss_rates(self, rates):
		"""
			Parameters:

				rates (numpy array):	Array of cavity loss rates. Expected shape is (M,)

		"""
		rates = np.array(rates, dtype=float)
		if not (len(rates.shape)==1 and rates.shape[0] == self.M):
			raise Exception("Shape of cavity loss rates not consistent with {0} photonic modes".format(self.M))
		if any([rates[i]<0 for i in range(0, len(rates))]):
			raise Exception("All rates must be non-negative")
		self.rates_kappa = rates


	def set_cavity_emission_rates(self, rates):
		"""
			Parameters:

				rates (numpy array):	Array of cavity emission rates. Expected shape is (M,)

		"""
		rates = np.array(rates, dtype=float)
		if not (len(rates.shape)==1 and rates.shape[0] == self.M):
			raise Exception("Shape of cavity emission rates not consistent with {0} photonic modes".format(self.M))
		if any([rates[i]<0 for i in range(0, len(rates))]):
			raise Exception("All rates must be non-negative")		
		self.rates_E = rates


	def set_cavity_absorption_rates(self, rates):
		"""
			Parameters:

				rates (numpy array):	Array of cavity absorption rates. Expected shape is (M,)

		"""
		rates = np.array(rates, dtype=float)
		if not (len(rates.shape)==1 and rates.shape[0] == self.M):
			raise Exception("Shape of cavity absorption rates not consistent with {0} photonic modes".format(self.M))
		if any([rates[i]<0 for i in range(0, len(rates))]):
			raise Exception("All rates must be non-negative")		
		self.rates_A = rates


	def set_reservoir_decay_rates(self, rates):
		"""
			Parameters:

				rates (numpy array):	Array of reversoir decay rates. These are the rates of emission into non-cavity modes.
									Expected shape is (J,)

		"""
		rates = np.array(rates, dtype=float)
		if not (len(rates.shape)==1 and rates.shape[0] == self.J):
			raise Exception("Shape of molecular reservoir decay rates not consistent with {0} molecular modes".format(self.J))
		if any([rates[i]<0 for i in range(0, len(rates))]):
			raise Exception("All rates must be non-negative")		
		self.rates_Gamma_down = rates


	def set_reservoir_pump_rates(self, rates):
		"""
			Parameters:

				rates (numpy array):	Array of reversoir pump rates. Expected shape is (J,)

		"""
		rates = np.array(rates, dtype=float)
		if not (len(rates.shape)==1 and rates.shape[0] == self.J):
			raise Exception("Shape of molecular reservoir pump rates not consistent with {0} molecular modes".format(self.J))
		if any([rates[i]<0 for i in range(0, len(rates))]):
			raise Exception("All rates must be non-negative")		
		self.rates_Gamma_up = rates


	def set_reservoir_population(self, population):
		"""
			Parameters:

				population (numpy array):	Array of reversoir population, i.e. the (total) number of molecules in each bin. 
											Expected shape is (J,)

		"""

		if any([not (type(number)==int or type(number)==np.int64) for number in population]):
			raise Exception("Molecular population must all be integers")
		population = np.array(population, dtype=float)
		if not (len(population.shape)==1 and population.shape[0] == self.J):
			raise Exception("Shape of molecular reservoir population not consistent with {0} molecular modes".format(self.J))
		if any([population[i]<0 for i in range(0, len(population))]):
			raise Exception("All molecular populations must be non-negative")		
		self.mols = population
		self.gmols[0] = 1.0*self.mols



	def set_coupling_terms(self, coupling_terms):
		"""
			Parameters:

				coupling_terms (numpy array): 	Matrix with coefficients determining the coupling between the photonic and
												molecular modes. Expected shape is (M, J)

		"""
		coupling_terms = np.array(coupling_terms, dtype=float)
		if not (len(coupling_terms.shape)==2 and coupling_terms.shape[0]==self.M and coupling_terms.shape[1]==self.J):
			raise Exception("Shape of coupling terms not consistent. Expected shape is [{0}, {1}].".format(self.M, self.J))
		if any([not (np.sum(coupling_terms[i,:])-1)<1e-10 for i in range(0, coupling_terms.shape[0])]):
			raise Exception("Coupling term not correctly normalized across the molecular reservoir")
		if any([coupling_terms[i,j]<0 for i in range(0, coupling_terms.shape[0]) for j in range(0, coupling_terms.shape[1])]):
			raise Exception("All coupling terms must be non-negative")
		self.g = coupling_terms



	def load_dynamics(self, t, photons, gmols, emols):
		"""
			Parameters:

				t (numpy array):		Array with time values. Shape is (n, )
				photons (numpy array):	Array with photon populations. Shape is (n, M)
				gmols (numpy array):	Array with ground state molecular populations. Shape is (n, J)
				emols (numpy array):	Array with excited state molecular populations. Shape is (n, J)

		"""

		if not len(set([t.shape[0], photons.shape[0], gmols.shape[0], emols.shape[0]])) == 1:
			raise Exception("Cavity populations do not have consistent shapes")
		if not photons.shape[1] == self.M:
			raise Exception("Photon population not consistent with {0} photonic modes".format(self.M))
		if not gmols.shape[1] == self.J:
			raise Exception("Ground state molecular population not consistent with {0} molecular modes".format(self.J))
		if not emols.shape[1] == self.J:
			raise Exception("Excited state molecular population not consistent with {0} molecular modes".format(self.J))

		self.t = copy.deepcopy(t)
		self.photons = copy.deepcopy(photons)
		self.gmols = copy.deepcopy(gmols)
		self.emols = copy.deepcopy(emols)



class Modes():

	"""
		A Modes() object defines the geometry of the cavity mirrors and calculates the respective mode structure.

	"""

	def __init__(self, grid_size, grid_delta, L0, lambda0, n, n_modes):
		"""

			Parameters:
				grid_size (float):			Total grid size, in meters. The numerical grid is always square.
				grid_delta (float):			Grid resolution, in meters. It will be the same in x and y.
				L0 (float):					Cavity lenght, in meters. The mirror shape is defined as perturbations on top of ...
											... overall cavity length.
				lambda0 (float):			Cavity cutoff, in meters.
				n (float):					Refractive index of the intracavity medium.
				n_modes (int):				Number of modes to calculate.

		"""

		if grid_size<=0:
			raise Exception("Grid size must be positive")
		if grid_delta<=0 or grid_delta>grid_size:
			raise Exception("Invalid grid resolution")
		if L0<=0:
			raise Exception("Cavity length must be positive")
		if lambda0<=0:
			raise Exception("Cavity cutoff must be positive")
		if n<=0:
			raise Exception("Refractive index must be positive")
		if n_modes<=0 or not type(n_modes)==int:
			raise Exception("Invalid number of modes")

		self.grid_size = 1e6*float(grid_size)
		self.grid_delta = 1e6*float(grid_delta)
		self.L0 = 1e6*float(L0)
		self.lambda0 = 1e6*float(lambda0)
		self.n = float(n)
		self.n_modes = int(n_modes)
		self.pump = None

		# Creates the grid
		x = np.arange(-self.grid_size/2, self.grid_size/2, self.grid_delta)
		y = np.arange(-self.grid_size/2, self.grid_size/2, self.grid_delta)
		self.X, self.Y = np.meshgrid(x, y)
		


	##### The following methods define a collection of typical cavity geometries


	def set_geometry(self, geometry):
		
		"""
			Sets a user-defined mirror shape. The mirror shape array size must be compatible with the grid size and grid resolution defined.
			The mirror shape is defined as the spatially dependent perturbation of the cavity length (on top of L0). The units are meters.
			The geometry is set such that the longest cavity point corresponds to geometry=0, such that geometry(x,y) >= 0

			Parameters:

				mirror_shape (numpy array):

		"""
		if not type(geometry) == np.ndarray:
			raise Exception("geometry is expected to be a numpy array")
		if not geometry.shape == self.X.shape:
			raise Exception("Size of geometry is not compatible with grid. Expected size is ({0},{1})".format(self.X.shape[0], self.X.shape[1]))
		self.geometry = np.array(1e6*copy.deepcopy(geometry), dtype=float)



	def set_geometry_spherical(self, RoC, depth):
		
		"""

			Parameters:
				RoC (float):		Radius of curvature, in meters.
				depth (float):		Feature depth, in meters.

		"""

		RoC = 1e6*RoC
		depth = 1e6*depth
		aux = (np.sqrt(RoC**2 - (self.X**2 + (self.Y)**2)) - RoC)
		self.geometry = (np.heaviside(aux + depth, 1)*aux) - (np.heaviside(-aux - depth, 1)*depth)
		self.geometry = - self.geometry



	def set_geometry_box(self, width, height, depth):

		"""

			Parameters:
				width (float):		Box width, in meters.
				height (float):		Box height, in meters
				depth (float):		Feature depth, in meters.

		"""

		width = 1e6*width
		height = 1e6*height
		depth = 1e6*depth
		self.geometry = -depth + depth*(1-np.heaviside(self.X-width/2, 1)) * (1-np.heaviside(-self.X-width/2, 1)) * \
					(1-np.heaviside(self.Y-height/2, 1)) * (1-np.heaviside(-self.Y-height/2, 1))
		self.geometry = - self.geometry


	def  set_geometry_circular_box(self, radius, depth):

		"""

			Parameters:
				radius (float):		Spherical box radius, in meters.
				depth (float):		Feature depth, in meters.

		"""

		radius = 1e6*radius
		depth = 1e6*depth
		self.geometry = -depth*np.heaviside(np.sqrt(self.X**2+self.Y**2)-radius, 1)
		self.geometry = - self.geometry


	######################################################################



	def compute_cavity_modes(self):
		
		"""

			Parameters:


			Return:
				lambdas (numpy array):		Wavelength of the cavity eigenmodes, in meters
				modes (numpy array):		Cavity eigenmodes (squared amplitude of the wavefunctions).

		"""

		# Renormalizes and reshapes the cavity geometry
		deltaL_normalized = -self.geometry/self.L0
		deltaL_reshaped = np.reshape(deltaL_normalized, [deltaL_normalized.shape[0]*deltaL_normalized.shape[1]])
		deltaL_sparse = sp.diags(diagonals=deltaL_reshaped)

		# Calculates the sparse matrix representation of the 2D Laplacian
		N = deltaL_normalized.shape[0]
		diag = np.ones([N*N])
		mat = sp.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
		I = sp.eye(N)
		laplacian = sp.kron(I,mat,format='csr')+sp.kron(mat,I)

		# Computes eigenmodes and eigenvalues
		cte1 = (self.lambda0)**2 / (8*(np.pi**2)*(self.n**2))
		cte1 = (1.0 / (self.grid_size / deltaL_normalized.shape[0])**2) * cte1
		hamiltonian_operator = cte1*laplacian + deltaL_sparse
		eigenvalues, eigenvectors = linalg.eigs(A=hamiltonian_operator, which='SM', k=self.n_modes)

		# Orders by energy
		sorted_ind = np.flip(np.argsort(eigenvalues))
		eigenvalues = eigenvalues[sorted_ind]
		eigenvectors = eigenvectors[:,sorted_ind]

		# Converts eigenvalues to wavelenght units
		self.lambdas = np.real(self.n*(self.lambda0 / (self.n-eigenvalues)))
		

		# Reshapes the eigenmodes and calculates their square amplitude
		self.modes = np.abs(np.reshape(eigenvectors, [deltaL_normalized.shape[0], deltaL_normalized.shape[1], self.n_modes]))**2

		return 1e-6*self.lambdas, copy.deepcopy(self.modes)



	def plot_cavity(self):

		matplotlib.rcParams.update({'font.size': 8})
		fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(14*0.8, 6*0.8))
		axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

		aux = axes[0].pcolor(self.X, self.Y, self.geometry)
		axes[0].set_xlabel(r'x (microns)')
		axes[0].set_ylabel(r'y (microns)')
		axes[0].set_title("Cavity Geometry")

		axes[1].plot(1e3*self.lambdas)
		axes[1].set_xlabel(r'Mode order')
		axes[1].set_ylabel(r'Mode wavelenght (nm)')
		axes[1].set_title("Cavity modes")

		if type(self.pump)==np.ndarray:
			aux = axes[2].pcolor(self.X, self.Y, self.pump)
			axes[2].set_xlabel(r'x (microns)')
			axes[2].set_ylabel(r'y (microns)')
			axes[2].set_title("Cavity pump")
			first_mode_ax_ind = 3
		else:
			first_mode_ax_ind = 2		

		for i in range(first_mode_ax_ind, 8):
			aux_ind = i-first_mode_ax_ind
			if self.n_modes > aux_ind:
				axes[i].pcolor(np.squeeze(self.modes[:,:,aux_ind]))
				axes[i].set_xlabel(r'x (microns)')
				axes[i].set_ylabel(r'y (microns)')
				axes[i].set_title(r'Mode {0} (Squared amplitude)'.format(aux_ind))

		plt.tight_layout()
		plt.show()



	def get_coupling_matrix(self):

		g = np.reshape(self.modes, [self.modes.shape[0]*self.modes.shape[1], self.n_modes])
		g = np.transpose(g)
		g = np.array([g[i,:]/np.sum(np.squeeze(g[i,:])) for i in range(0, self.n_modes)], dtype=float)

		return copy.deepcopy(g)


	def get_cavity_grid(self):

		return copy.deepcopy(1e-6*self.X), copy.deepcopy(1e-6*self.Y)


	def load_pump(self, pump):
		"""
			Parameters:

				pump (numpy array):		2D array with the pump shape. Dimensions must match the cavity grid shape

		"""

		pump = np.array(pump, dtype=float)
		if not (pump.shape[0]==self.X.shape[0] and pump.shape[1]==self.X.shape[1]):
			raise Exception("Expected pump shape is", self.X.shape) 

		self.pump = copy.deepcopy(pump)