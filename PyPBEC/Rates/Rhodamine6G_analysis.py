#exit()

#ipython --pylab
#execfile("Rhodamine6G_analysis.py")
import csv
from scipy.signal import argrelmax
from numpy import argmax
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy import constants

import matplotlib
matplotlib.rcParams["font.size"]=20

#figure(1),clf()
#figure(2),clf()

compound_name = "Rhodamine 6G"

mypath = "/Users/btw15/Documents/Academic/PhD/Calculations/Jitter/20180504/"
emis_file_name = mypath+"fluorescence_spectrum_R6G_in_EthyleneGlycol.csv"
exti_file_name = mypath+"absorption_cross_sections_R6G_in_EthyleneGlycol.csv"
#NOTE: extinctions are in fact cross-sections, and there are several inferred at different concentrations, i.e. valid in different wavelength regions.

emis_file = open(emis_file_name)
reader = csv.reader(emis_file)
dump = [thing for thing in reader]
emis_file.close()
min_index = 250
emission_wavelengths = array([float(x[0]) for x in dump[min_index:]]) 
emission_values = array([float(x[1]) for x in dump[min_index:]])#ignore v. short wavelengths

#CROP OUT UGLY FLUO DATA
#Data is cut out because of spurious peaks which come from the room lighting.
kill_indices_pairs=[(380,420),(631,647),(670,688),(962,1014)] #NOTE: this list gives indices AFTER earlier pairs are removed!
for kill_indices in kill_indices_pairs:
    emission_wavelengths2 = [em for i,em in enumerate(emission_wavelengths) if not(kill_indices[0]<i<kill_indices[1])]
    emission_values2 = [em for i,em in enumerate(emission_values) if not(kill_indices[0]<i<kill_indices[1])]
    emission_wavelengths=emission_wavelengths2
    emission_values = emission_values2
'''
figure(1)
plot(emission_wavelengths ,emission_values,label="Emission")
xlabel("Wavelength (nm)")
ylabel("Fluorescence value (arb. units)")

grid(1)
xlim(450,680)
title("Fluorescence for "+compound_name)
subplots_adjust(bottom=0.12,left=0.18,right=0.98,top=0.92)
savefig("R6G_fluorescence.png")
'''

all_extinction_values = []
for exti_data_index in range(1,5):
    exti_file = open(exti_file_name)
    reader = csv.reader(exti_file)
    dump = [thing for thing in reader]
    exti_file.close()
    min_index = 100
    extinction_wavelengths = array([float(x[0]) for x in dump[min_index:]])
    extinction_values = array([float(x[exti_data_index]) for x in dump[min_index:]])
    all_extinction_values.append(list(extinction_values))
    concn = dump[0][exti_data_index].split("concn=")[-1]
'''
    figure(2)
    semilogy(extinction_wavelengths ,extinction_values,label=concn+"")
    ylabel(r"Absorption cross section (m $^{-2}$ )")
    grid(1)
    xlim(450,680)
    legend()

figure(2)
title("Absorption for various concentrations")
xlabel("Wavelength (nm)")
grid(0)
suptitle("")
subplots_adjust(bottom=0.12,left=0.16,right=0.98,top=0.92)
'''

#--------------------------
#COMBINE MULTIPLE CONCENTRATIONS INTO ONE DATASET
#----------------------------
break_indices = [169,181,191] #points at which various concentrations SNR become better than others
combined_extinction_values = all_extinction_values[0][:break_indices[0]]
combined_extinction_values +=all_extinction_values[1][break_indices[0]:break_indices[1]]
combined_extinction_values +=all_extinction_values[2][break_indices[1]:break_indices[2]]
combined_extinction_values +=all_extinction_values[3][break_indices[2]:]
'''
figure(3),clf()
semilogy(extinction_wavelengths ,combined_extinction_values)
ylabel(r"Absorption cross section (m $^{-2}$ )")
grid(1)
xlim(450,630)
ylim(1e-25,)
legend()
title("Absorption for "+compound_name)
xlabel("Wavelength (nm)")
subplots_adjust(bottom=0.12,left=0.16,right=0.98,top=0.92)
savefig("R6G_absorption.png")
'''
experimental_absorption = interp1d(extinction_wavelengths,combined_extinction_values)
experimental_emission = interp1d(emission_wavelengths,array(emission_values)*max(combined_extinction_values)/max(emission_values))
#execfile("Rhodamine6G_analysis.py")
#EoF