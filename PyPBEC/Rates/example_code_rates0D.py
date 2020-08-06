import matplotlib.pyplot as plt
from CavityRates import CavityRates
import numpy as np

##### Parameters
# query wavelengths (in meters)
wavelengths = np.arange(570,600,10)*1e-9
# total spontaneous emission lifetime (in s)
tau_zero = 4.0 * 1e-9
# total emission into noncavity modes lifetime (in s)
tau_down = 40.0 * 1e-9
# mode degeneracy
mode_degeneracy = np.flip(np.arange(0, len(wavelengths), 1) + 1)


##### Cavity rates, in 1/s
cavityrates = CavityRates(model='rates_0D')
absorption_rates, emission_rates = cavityrates.compute_rates(
	lambdas=wavelengths,
	temperature=300,
	Gamma_0=1.0/tau_zero,
	Gamma_down=1.0/tau_down,
	degeneracy=mode_degeneracy)


print('Ab rates: ', absorption_rates)
print('Em rates: ', emission_rates)


##### Plots the rates
wavelengths = np.arange(520,610,0.1)*1e-9
mode_degeneracy = np.flip(np.arange(0, len(wavelengths), 1) + 1)
cavityrates = CavityRates(model='rates_0D')
absorption_rates, emission_rates = cavityrates.compute_rates(
	lambdas=wavelengths,
	temperature=300,
	Gamma_0=1.0/tau_zero,
	Gamma_down=1.0/tau_down,
	degeneracy=mode_degeneracy)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.plot(wavelengths*1e9, absorption_rates, label='Absorption')
plt.plot(wavelengths*1e9, emission_rates, label='Emission')
plt.legend()
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'Rates ps$^{-1}$')
plt.show()