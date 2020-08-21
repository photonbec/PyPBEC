import numpy as np
from scipy import constants as sc
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.special import erf as Erf
import pandas as pd
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

		# absorption data
		min_wavelength = 480
		max_wavelength = 650
		absorption_spectrum_datafile = Path("data") / 'absorption_cross_sections_R6G_in_EthyleneGlycol_corrected.csv'
		absorption_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / absorption_spectrum_datafile
		raw_data2 = pd.read_csv(absorption_spectrum_datafile)
		initial_index = raw_data2.iloc[(raw_data2['wavelength (nm)']-min_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data2.iloc[(raw_data2['wavelength (nm)']-max_wavelength).abs().argsort()].index[0]
		raw_data2 = raw_data2.iloc[:final_index].reset_index(drop=True)
		absorption_data = raw_data2
		absorption_data_normalized = absorption_data['absorption cross-section (m^2)'].values / np.max(absorption_data['absorption cross-section (m^2)'].values)
		absorption_spectrum = np.squeeze(np.array([[absorption_data['wavelength (nm)'].values], [absorption_data_normalized]], dtype=float))
		interpolated_absorption_spectrum = interp1d(absorption_spectrum[0,:], absorption_spectrum[1,:], kind='cubic')
		
		# emission data
		fluorescence_spectrum_datafile =  Path("data") / 'fluorescence_spectrum_R6G_in_EthyleneGlycol_corrected.csv'
		fluorescence_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / fluorescence_spectrum_datafile
		raw_data = pd.read_csv(fluorescence_spectrum_datafile)
		initial_index = raw_data.iloc[(raw_data['wavelength (nm)']-min_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data.iloc[(raw_data['wavelength (nm)']-max_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[:final_index].reset_index(drop=True)
		fluorescence_data = raw_data
		fluorescence_data_normalized = fluorescence_data['fluorescence (arb. units)'].values / np.max(fluorescence_data['fluorescence (arb. units)'].values)
		emission_spectrum = np.squeeze(np.array([[fluorescence_data['wavelength (nm)'].values], [fluorescence_data_normalized]], dtype=float))
		interpolated_emission_spectrum = interp1d(emission_spectrum[0,:], emission_spectrum[1,:], kind='cubic')

		# Uses both datasets
		if np.min(1e9*np.array(lambdas)) < 480 or np.max(1e9*np.array(lambdas)) > 650:
			raise Exception('*** Restrict wavelength to the range between 480 and 650 nm ***')

		temperature = 300
		lamZPL = 545e-9
		n_mol_per_vol= dye_concentration*sc.Avogadro
		peak_Xsectn = 2.45e-20*n_mol_per_vol*sc.c/n
		wpzl = 2*np.pi*sc.c/lamZPL/1e12

		def freq(wl):
			return 2*np.pi*sc.c/wl/1e12
		def single_exp_func(det):
			f_p = 2*np.pi*sc.c/(wpzl+det)*1e-3
			f_m = 2*np.pi*sc.c/(wpzl-det)*1e-3
			return (0.5*interpolated_absorption_spectrum(f_p)) + (0.5*interpolated_emission_spectrum(f_m))
		def Err(det):
			return Erf(det*1e12)
		def single_adjust_func(det):
			return ((1+Err(det))/2.0*single_exp_func(det)) + ((1-Err(det))/2.0*single_exp_func(-1.0*det)*np.exp(sc.h/(2*np.pi*sc.k*temperature)*det*1e12)) 
			
		emission_rates = np.array([single_adjust_func(-1.0*freq(a_l)+wpzl) for a_l in lambdas])*peak_Xsectn
		absorption_rates = np.array([single_adjust_func(freq(a_l)-wpzl) for a_l in lambdas])*peak_Xsectn

		return absorption_rates, emission_rates