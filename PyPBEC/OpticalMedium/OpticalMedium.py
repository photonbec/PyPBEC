import numpy as np
from scipy import constants as sc
from scipy.interpolate import interp1d
from pathlib import Path
import sys
import os
import csv

class OpticalMedium():

	available_media = list()
	available_media.append("Rhodamine6G")

	def __init__(self, optical_medium):

		"""
			Initiazies an optical medium object.

			Parameters:

				optical_medium (str):		Optical medium

		"""

		if not type(optical_medium) == str:
			raise Exception("optical_medium is expected to be a string")

		if not optical_medium in self.available_media:
			raise Exception(optical_medium+" is an unknown optical medium")

		if optical_medium == "Rhodamine6G":
			self.medium = Rhodamine6G()


	def get_rates(self, lambdas, **kwargs):

		"""
			Calculates the rates of absorption and emission, for a specific optical medium.

			Parameters:

				lambdas (list, or other iterable): Wavelength points where the rates are to be calculated. Wavelength is in meters
				other medium specific arguments
				
		"""

		return self.medium.get_rates(lambdas=lambdas, **kwargs)




class Rhodamine6G(OpticalMedium):

	def __init__(self):
		pass


	def get_rates(self, lambdas, dye_concentration, n):

		"""
			Rates for Rhodamine 6G

			Parameters:

				lambdas (list, or other iterable):  	Wavelength points where the rates are to be calculated. Wavelength is in meters
				dye_concentration (float):				In mM (milimolar) 1 mM = 1 mol / m^3			
				n (float): 								index of refraction

		"""

		datafile = Path("data") / "absorption_cross_sections_R6G_in_EthyleneGlycol.csv"
		datafile = Path(os.path.dirname(os.path.abspath(__file__))) / datafile
		all_extinction_values = []
		for exti_data_index in range(1,5):
			exti_file = datafile.open()
			reader = csv.reader(exti_file)
			dump = [thing for thing in reader]
			exti_file.close()
			min_index = 100
			extinction_wavelengths = np.array([float(x[0]) for x in dump[min_index:]])
			extinction_values = np.array([float(x[exti_data_index]) for x in dump[min_index:]])
			all_extinction_values.append(list(extinction_values))
			concn = dump[0][exti_data_index].split("concn=")[-1]
		
		# Combine multiple concentrations into one dataset
		break_indices = [169,181,191] # points at which various concentrations SNR become better than others
		combined_extinction_values = all_extinction_values[0][:break_indices[0]]
		combined_extinction_values +=all_extinction_values[1][break_indices[0]:break_indices[1]]
		combined_extinction_values +=all_extinction_values[2][break_indices[1]:break_indices[2]]
		combined_extinction_values +=all_extinction_values[3][break_indices[2]:]
		
		normalised_absorption = interp1d(extinction_wavelengths*1e-9,np.array(combined_extinction_values)/np.max(combined_extinction_values))

		lamZPL = 545e-9
		temperature = 300
		peak_Xsectn = 2.45e-20
		n_mol_per_vol= dye_concentration*sc.Avogadro
		absorption_Xsctns = [normalised_absorption(a_l)*peak_Xsectn for a_l in lambdas]
		absorption_rates = np.array(absorption_Xsctns)*n_mol_per_vol*sc.c/n
		ks_factors = np.array([np.exp(sc.h*sc.c*(1/lamZPL - 1/lam)/(sc.Boltzmann*temperature)) for lam in lambdas])
		emission_rates = absorption_rates*ks_factors
		
		return absorption_rates, emission_rates