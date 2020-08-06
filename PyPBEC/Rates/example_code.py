# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:36:48 2019

@author: btw15
"""

from pylab import *
from CavityRates import CavityRates
import matplotlib.pyplot as plt

wavelengths = arange(440,620,1)*1e-9

cavr_exp = CavityRates(model='exp_data')
cavr_magic = CavityRates(model='magic_numbers')
cavr_exp_adj = CavityRates(model='exp_data_adj')

#Compare normalised cross sections from the 2 models
plt.figure(1)
plt.clf()
plt.semilogy(wavelengths*1e9,cavr_exp.normalised_absorption(wavelengths),label='experimental data')
plt.semilogy(wavelengths*1e9,cavr_magic.normalised_absorption(wavelengths),label='magic numbers')
plt.xlabel('Wavelength/nm')
plt.ylabel('Normalised Absorption')
plt.legend(loc='best')
plt.title('Compare normalised cross sections from the 2 models')
plt.show()

cutoff = 590e-9
spacing = 1e-9
n_modes = 5
lams = [cutoff-i*spacing for i in range(n_modes)]
ab_rates, em_rates = cavr_exp.compute_rates(lams)
print('Ab rates: ', ab_rates)
print('Em rates: ', em_rates)


# Prints emission and absporption rates
ab_rates, em_rates = cavr_exp.compute_rates(wavelengths)
plt.figure(2),clf()
plt.semilogy(wavelengths*1e9,em_rates, label='Emission rates')
plt.semilogy(wavelengths*1e9,ab_rates, label='Absorption rates')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Rates')
plt.legend(loc='best')
plt.show()
