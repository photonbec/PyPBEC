"""
Microlaser example
--------
This short script is intended to be a minimum working example 
of the pypbec package operating. It solves for the steady-state
population of a single-mode microlaser as a function of pump
rate, using Rhodamine 6G as the gain medium.
"""

import PyPBEC
from PyPBEC.Rates.CavityRates import CavityRates