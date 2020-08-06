import numpy as np
from scipy import constants as sc
import csv
from pylab import *
from scipy.interpolate import interp1d
from pathlib import Path
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d



class CavityRates():

	def __init__(self, rawDataPath='R6G_data', model='exp_data'):

		self.model = model
		self.temperature = 300
		self.lamZPL = 545e-9
		self.solvent_n = 1.43
		self.rawDataPath = rawDataPath
		#Get normalised absorption as a function of wavelength in nm
		if self.model == 'exp_data':
			self.get_normalised_cross_section_from_exp_data()
		elif self.model == 'magic_numbers':
			self.get_normalised_cross_section_from_magic_numbers()
		elif self.model == 'rates_0D':
			self.get_normalized_emission_spectrum()
		else:
			raise Exception("Unknown rate model")
	
	
	#This uses experimental data from our group. We later rescale the maximum to 2.45e-20m^2. Something to do with not trusting the monochromator
	def get_normalised_cross_section_from_exp_data(self):
		exti_file_name = Path(self.rawDataPath) / "absorption_cross_sections_R6G_in_EthyleneGlycol.csv"
		exti_file_name = Path(os.path.dirname(os.path.abspath(__file__))) / exti_file_name
		#exti_file_name = self.rawDataPath+"absorption_cross_sections_R6G_in_EthyleneGlycol.csv"
		#NOTE: extinctions are in fact cross-sections, and there are several inferred at different concentrations, i.e. valid in different wavelength regions.

		all_extinction_values = []
		for exti_data_index in range(1,5):
			exti_file = exti_file_name.open()
			reader = csv.reader(exti_file)
			dump = [thing for thing in reader]
			exti_file.close()
			min_index = 100
			extinction_wavelengths = array([float(x[0]) for x in dump[min_index:]])
			extinction_values = array([float(x[exti_data_index]) for x in dump[min_index:]])
			all_extinction_values.append(list(extinction_values))
			concn = dump[0][exti_data_index].split("concn=")[-1]
		
		
		#--------------------------
		#COMBINE MULTIPLE CONCENTRATIONS INTO ONE DATASET
		#----------------------------
		break_indices = [169,181,191] #points at which various concentrations SNR become better than others
		combined_extinction_values = all_extinction_values[0][:break_indices[0]]
		combined_extinction_values +=all_extinction_values[1][break_indices[0]:break_indices[1]]
		combined_extinction_values +=all_extinction_values[2][break_indices[1]:break_indices[2]]
		combined_extinction_values +=all_extinction_values[3][break_indices[2]:]
		
		self.normalised_absorption = interp1d(extinction_wavelengths*1e-9,array(combined_extinction_values)/max(combined_extinction_values))

		
	#Previous simulations have used magic numbers to fit the functional form of the absorption (and equivalently emission) cross sections. It comes from Henry. 
	# ...I don't know where he got it from. Could be worth asking.
	def get_normalised_cross_section_from_magic_numbers(self):
		def magic_normalised_absorption(wavelength_m, maxAbsorption = 1.):  ## in nm
			wavelength = wavelength_m*1e9
			if (  (array([wavelength])>620).any()  or  (array([wavelength])<440).any()  ):
				raise Exception("wavelength outside data range")
			def gausF(x,b2,c2,s2):
				return b2*exp(-((x-c2)/s2)**2)
		
			def biLin(x,m1,dm1,c1,s1):
				return (m1+dm1/2.)*x + sqrt(s1**2 + (x-c1)**2)*dm1/2. - (m1+dm1/2.)*c1-s1*dm1/2.
		    
			def hevi(x,b,c,s):
				return b/(1 + exp((x-c)/s)  )
		        
			c0, m1,dm1,c1,s1, ba,ca,sa, b2,c2,s2, b3,c3,s3, b4,c4,s4, b5,c5,s5, b6,c6,s6, b7,c7,s7 = [  0.049808076209124454,
		         2.43455165e-02,  -8.33075469e-02,
		         5.36791110e+02,   6.10755335e+00,   3.48108875e-01,
		         5.01668090e+02,   5.01237925e+00,  -1.87545255e-01,
		         5.86464915e+02,   1.44929782e+01,   1.95216417e-01,
		         5.54695067e+02,   1.50626766e+01,  -1.11029752e-01,
		         5.36081866e+02,   7.97020549e+00,   1.37292408e-02,
		         5.18791586e+02,   5.49142664e+00,  -5.19777967e-02,
		         4.73843772e+02,   9.46064300e+00,  -4.79382020e-02,
		         4.46133098e+02,   7.10639919e+00]
		    
			x = array(wavelength)
		    
			log10Ab = c0 + biLin(x,m1,dm1,c1,s1) + hevi(x,ba,ca,sa) + gausF(x,b2,c2,s2)  + gausF(x,b3,c3,s3)+ \
			gausF(x,b4,c4,s4) + gausF(x,b5,c5,s5) + gausF(x,b6,c6,s6) + gausF(x,b7,c7,s7)
		    
			return maxAbsorption*10**log10Ab
		self.normalised_absorption = magic_normalised_absorption


	def get_normalized_emission_spectrum(self):

		fluorescence_spectrum_datafile =  Path(self.rawDataPath) / 'fluorescence_spectrum_R6G_in_EthyleneGlycol_corrected.csv'
		fluorescence_spectrum_datafile = Path(os.path.dirname(os.path.abspath(__file__))) / fluorescence_spectrum_datafile
		min_wavelength = 490
		max_wavelength = 650
		raw_data = pd.read_csv(fluorescence_spectrum_datafile)
		initial_index = raw_data.iloc[(raw_data['wavelength (nm)']-min_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[initial_index:].reset_index(drop=True)
		final_index = raw_data.iloc[(raw_data['wavelength (nm)']-max_wavelength).abs().argsort()].index[0]
		raw_data = raw_data.iloc[:final_index].reset_index(drop=True)
		fluorescence_data = raw_data
		fluorescence_data_normalized = fluorescence_data['fluorescence (arb. units)'].values / np.max(fluorescence_data['fluorescence (arb. units)'].values)
		self.emission_spectrum = np.squeeze(np.array([[fluorescence_data['wavelength (nm)'].values], [fluorescence_data_normalized]], dtype=float))
		self.interpolated_emission_spectrum = interp1d(self.emission_spectrum[0,:], self.emission_spectrum[1,:], kind='cubic')


		
	###################################### Default model
	def compute_rates(self, lambdas, rho=2.3, n=1.43, peak_Xsectn=2.45e-20, temperature=300, Gamma_down=None, Gamma_0=None, degeneracy=None):
		'''
			Returns absorption and emission rates:

			Parameters:
				rho:      (float) dye density, in units of mM=1mol/m^3
				n:        (float) index of refraction
				lambdas:  (list of floats) wavelengths where to compute the aborption and emission rates (in meters)
				peak_Xsectn: (float) peak absorption cross-section
				temperature: (float) temperature

					For the 0D model, extra parameters are necessary:
				Gamma_down:   (float) Emission rate into non cavity modes (in s^-1)
				Gamma_0:      (float) Total emission rate, around 1 / 4ns for R6G (in s^-1)
				degeneracy    (np array) Degeneracy structure of the cavity. Order is assumed to correspond to the same order as "lambdas"

			Return:
				absorption_rates: (list of floats) of the same length as lambdas (in s^-1)
				emission_rates:   (list of floats) of the same length as lambdas (in s^-1)
		'''

		if self.model == 'exp_data' or self.model == 'magic_numbers':
			self.rho = rho
			self.solvent_n = n
			self.temperature = temperature
			self.peak_Xsectn = peak_Xsectn
			n_mol_per_vol= self.rho*sc.Avogadro
			absorption_Xsctns = [self.normalised_absorption(a_l)*self.peak_Xsectn for a_l in lambdas]
			absorption_rates = array(absorption_Xsctns)*n_mol_per_vol*sc.c/self.solvent_n
			ks_factors = array([exp(sc.h*sc.c*(1/self.lamZPL - 1/lam)/(sc.Boltzmann*self.temperature)) for lam in lambdas])
			emission_rates = absorption_rates*ks_factors
			return absorption_rates, emission_rates

		elif self.model == 'rates_0D':
			if Gamma_down is None or Gamma_0 is None:
				raise Exception('*** 0D rate model requieres the definition of Gamma_down and Gamma_0 ***')
			if degeneracy is None:
				raise Exception('*** 0D rate model requieres the definition of cavity degeneracy structure ***')
			if np.min(1e9*lambdas) < 490 or np.max(1e9*lambdas) > 650:
				raise Exception('*** Restrict wavelength to the range between 450 and 650 nm ***')

			# interpolates the emission spectrum at the query points
			unnormalized_emission_rates = self.interpolated_emission_spectrum(1.0*1e9*lambdas)
			# Computes the emission rates
			emission_rates = (Gamma_0-Gamma_down) * unnormalized_emission_rates / np.sum(degeneracy*unnormalized_emission_rates)
			# Computes the absorption rates
			kennard_stepanov_factor = np.array([np.exp((sc.h*sc.c*(1.0/lamb - 1.0/self.lamZPL)/(sc.Boltzmann*temperature))) for lamb in lambdas])
			absorption_rates = emission_rates * kennard_stepanov_factor

			return absorption_rates, emission_rates
           



#---------------


