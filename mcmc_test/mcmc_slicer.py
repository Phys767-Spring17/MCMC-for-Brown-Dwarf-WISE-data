import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con
import emcee
import corner
import scipy.optimize as op
import random
import os
import sampyl
import pymc

# Get the data using the astropy ascii
data = ascii.read(os.path.dirname(os.path.realpath(__file__))+"\\SED.dat", data_start=6)

x = data[0][:]      # Wavelength column
y = data[1][:]     # log10(flux)
yerr = data[2][:]  # Error on log10(flux)

#Constants
h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

Teffmin = 10.0 #effective temperature minimum
Teffmax = 1000.0 #effective temperature maximum
logfacmin = -100.0 #log factor minimum
logfacmax = 0.0 #log factor maximum

#theatshape is 2 X 2 array
thetashape=np.array([[Teffmin,Teffmax],[logfacmin,logfacmax]])

#Conversion Matrix required for the MHSampler to run
cov = np.cov(x,y)

#Model for the log of the Spectral Radiance
def model(x, T,logfactor):

    #takes in the wavelength array approximate Temp and the log factor and returns and array of logflux
    wav = x * 1.0e-6
    flux = np.empty([len(wav)])
    logflux = np.empty([len(wav)])

    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    for i in range(len(wav)):
        a = 2.0*h*c**2
        b = h*c/(wav[i]*k*T)
        flux[i] = a/ ( (wav[i]**5) * (np.exp(b) - 1.0) )
        logflux[i] = logfactor + np.log10(flux[i])
    return logflux

#log of likelyhood function
def log_like(x,logf,errlogf,theta):
    residuals = logf - model(x,theta[0],theta[1])
    loglike=0.0
    for i in range(len(x)):
        loglike = loglike - np.log(errlogf[i]) - 0.5*(residuals[i]/errlogf[i])**2
    loglike = loglike - 0.5*len(x)*np.log(2.0*np.pi)
    return loglike

#log of the priors
def log_prior(theta,thetashape):

    logpriors=np.empty([len(theta)])
    #logprior=0.0

    # Prior for theta[0]: Teff~logU[Teffmin,Teffmax]
    Teff = theta[0]
    Teffmin = thetashape[0][0]
    Teffmax = thetashape[0][1]
    if Teffmin < Teff < Teffmax:
        logpriors[0] = ( 1.0/(np.log(Teffmax) - np.log(Teffmin)) )/Teff
    else:
        logpriors[0] = -1.0e99 # -infinity

    # Prior for theta[1]: logfac~U[logfacmin,logfacmax]
    logfac = theta[1]
    logfacmin = thetashape[1][0]
    logfacmax = thetashape[1][1]
    if logfacmin < logfac < logfacmax:
        logpriors[1] = 1.0/(logfacmax - logfacmin)
    else:
        logpriors[1] = -1.0e99 # -infinity

    #logprior = np.sum(logpriors)

    return np.sum(logpriors)

# Initialize the MCMC from a random point drawn from the prior
Teffinitial = np.exp( np.random.uniform(np.log(thetashape[0][0]),np.log(thetashape[0][1])) )
logfacinitial=np.random.uniform(thetashape[1][0],thetashape[1][1])
samples=np.array([[Teffinitial,logfacinitial]])

# Calculate the associated modified loglike
loglikechain=np.empty([1])
loglikechain[0]=log_prior(samples[0],thetashape) + log_like(x,y,yerr,samples[0])

#log of the probability function
def lnprob(theta, x, y, yerr):
    lp = log_prior(theta,thetashape)

    loglikechain=np.empty([1])
    loglikechain[0]=log_prior(samples[0],thetashape) + log_like(x,y,yerr,samples[0])
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_like(x,y,yerr,theta) #loglikechain[0]

"""




#Set the number of dimensions the MCMC algorithm will look for and the number of walkers
ndim, nwalkers = 2, 100
#identify the initial position for MCMC algorithm
pos = [[Teffinitial,logfacinitial] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#Set up the sampler function
sampler = emcee.MHSampler(cov, dim = ndim, lnprobfn = lnprob, args=(x, y, yerr))


# Clear and run the production chain.
number_of_samples = 100000
print("Running MCMC...")
sampler.run_mcmc(pos[0], number_of_samples, rstate0=np.random.get_state())
print("Done.")

# Give a generic burning point and choose the sample points used and identify the final result
burnin = 500
samples = sampler.chain[burnin:,:].reshape((-1, 2))

# Compute the quantiles.
samples[:] #= np.exp(samples[:])
T_mcmc, logfac_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""
    #MCMC result:
    #T = {0[0]} +{0[1]} -{0[2]}
    #Log Factor = {1[0]} +{1[1]} -{1[2]}
""".format(T_mcmc, logfac_mcmc))

# Calculate the bruning point and use it to calculate the final result once again
loglikeburn=np.median(samples[:,1])
j=-1
while True:
    j=j+1
    if samples[:,1][j] > loglikeburn:
        break
burnj=j
print( 'Burn point = ',burnj)

samples = sampler.chain[burnj:,:].reshape((-1, 2))

# Compute the quantiles.
samples[:] #= np.exp(samples[:])
T_mcmc, logfac_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""
    #MCMC result with calculated burn point:
    #T = {0[0]} +{0[1]} -{0[2]}
    #Log Factor = {1[0]} +{1[1]} -{1[2]}
""".format(T_mcmc, logfac_mcmc))
"""
