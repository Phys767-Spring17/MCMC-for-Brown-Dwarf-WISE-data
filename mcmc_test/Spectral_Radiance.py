from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import PyAstronomy
import subprocess
from astropy.io import ascii
import os

np.seterr(divide='ignore', invalid='ignore')
np.seterr(over='ignore', invalid='ignore')

results = ascii.read(os.path.dirname(os.path.realpath(__file__))+"\\results.dat")

h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

def my_own_Planck(x,T):
    #x is the wavelength
    #returns spectral radiance per sr
    return((a/x**5)*(1/((np.exp(b/(T*x))-1))))

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

X = np.linspace(0,50,1000,endpoint=True)

# Temperatures of other Brown Dwarfs used for comparison
MCMC_Temp = results[0][0]
T1, T1_name = 227, "WISE 1506+7027"
T2, T2_name = 450, "WISE 0410+1502"
T3, T3_name = 350, "WISE 1541−2250"

# temperature and Brown Dwarf decriptions
T1_description = T1_name + " (" + str(T1)+"K" + ")"
T2_description = T1_name + " (" + str(T2)+"K" + ")"
T3_description = T1_name + " (" + str(T3)+"K" + ")"
MCMC_description = "WISE 0855-0714" + " (" + str(MCMC_Temp)+"K" + ")"

# Ys used for plotting
Y1 = (planck(X*10**-6,T1))
Y2 = (planck(X*10**-6,T2))
Y3 = (planck(X*10**-6,T3))
Y4 = (planck(X*10**-6,MCMC_Temp))

# Plotting the planck function for each brown dwarf
plot1, = pl.plot(X,Y1, color="red", label=T1_description )
plot2, = pl.plot(X,Y2, color="green", label=T2_description )
plot3, = pl.plot(X,Y3, color="blue", label=T3_description )
plot4, = pl.plot(X,Y4, color="black", label=MCMC_description )
pl.legend([plot1,plot2,plot3,plot4],[T1_description, T2_description, T3_description, MCMC_description])
pl.xlabel("Wavelength (um)")
pl.ylabel("Spectral Radiance")

pl.show()
