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

# Get the data using the astropy ascii
data = ascii.read("SED.dat", data_start=6)

x = data[0][:]      # Wavelength column
y = data[1][:]     # log10(flux)
yerr = data[2][:]  # Error on log10(flux)

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

Teffmin = 10.0 #effective temperature minimum
Teffmax = 1000.0 #effective temperature maximum
logfacmin = -100.0 #log factor minimum
logfacmax = 0.0 #log factor maximum
#theatshape is 2 X 2 array
thetashape=np.array([[Teffmin,Teffmax],[logfacmin,logfacmax]])

cov = np.cov(x,y)

class MCMCSetup(object):

    def __init__(self):
        pass

    def model(self,x, T,logfactor):

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

    def log_like(self,x,logf,errlogf,theta):
        residuals = logf - self.model(x,theta[0],theta[1])
        loglike=0.0
        for i in range(len(x)):
            loglike = loglike - np.log(errlogf[i]) - 0.5*(residuals[i]/errlogf[i])**2
        loglike = loglike - 0.5*len(x)*np.log(2.0*np.pi)
        return loglike

    def log_prior(self,theta,thetashape):

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

    def lnprob(self,theta, x, y, yerr):
        lp = self.log_prior(theta,thetashape)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_like(x,y,yerr,theta) #loglikechain[0]

MCMC = MCMCSetup()

# Initialize the MCMC from a random point drawn from the prior
Teffinitial = np.exp( np.random.uniform(np.log(thetashape[0][0]),np.log(thetashape[0][1])) )
logfacinitial=np.random.uniform(thetashape[1][0],thetashape[1][1])
thetachain=np.array([[Teffinitial,logfacinitial]])

ndim, nwalkers = 2, 100
pos = [[Teffinitial,logfacinitial] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.MHSampler(cov, dim = ndim, lnprobfn = MCMC.lnprob , args=(x, y, yerr))

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos[0], 5000,rstate0=np.random.get_state())
print("Done.")

# Make the triangle plot.
burnin = 500
samples = sampler.chain[burnin:,:]#.reshape((-1, 2))

# Compute the quantiles.
samples[:] #= np.exp(samples[:])
T_mcmc, logfac_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))

print("""MCMC result:
    T = {0[0]} +{0[1]} -{0[2]}
    Log Factor = {1[0]} +{1[1]} -{1[2]}
""".format(T_mcmc, logfac_mcmc))


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

print("""MCMC result with calculated burn point:
    T = {0[0]} +{0[1]} -{0[2]}
    Log Factor = {1[0]} +{1[1]} -{1[2]}
""".format(T_mcmc, logfac_mcmc))

# ------------------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\emcee_model_graphs"
os.makedirs(dir_path, exist_ok=True)

# Ploting MCMC
plotting_wavelength = np.arange(x[0], 25.0, 0.01)

# Plot the initial guess results.
plt.errorbar(x, y, yerr=yerr, fmt=".r")
plt.plot(plotting_wavelength, MCMC.model(plotting_wavelength,samples[0][0],samples[0][1]), "--k", lw=1)
plt.savefig(dir_path+"\Initial Guess Result.png")
plt.close()

# Plot the MCMC results.
plt.errorbar(x, y, yerr=yerr,fmt='.r',ms="6")
plt.plot(plotting_wavelength, MCMC.model(plotting_wavelength,T_mcmc[0],logfac_mcmc[0]), "--k", lw=1)
plt.savefig(dir_path+"\MCMC results.png")
plt.close()

# Plot
jlist=np.arange(len(samples))
plt.scatter(samples[:,0], samples[:,1], c=jlist, cmap='coolwarm')
plt.xlabel('Temperature [K]')
plt.ylabel('log10(factor)')
plt.savefig(dir_path+"\\1B Temp vs logfactor.png")
plt.close()

np.max(samples[:,1])


plt.plot(samples[:,1])
plt.xlabel('Chain number')
plt.ylabel('loglike')
plt.savefig(dir_path+"\\2B Chain number vs loglike.png")
#plt.show()
plt.close()

jlist=np.arange(len(samples))
plt.scatter(samples[burnj:,0], samples[burnj:,1], c=jlist[burnj:], cmap='coolwarm',alpha=0.5)
plt.xlabel('Temperature [K]')
plt.ylabel('log10(factor)')
plt.savefig(dir_path+"\\3B Temperatur vs log10(factor) B.png")
#plt.show()
plt.close()

#print( 'Temperature [K] = ',np.round(np.median(samples[burnj:,0]),1),'-',np.round(np.median(samples[burnj:,0])-np.percentile(samples[burnj:,0],15.9),1),'+',np.round(np.percentile(samples[burnj:,0],84.1)-np.median(samples[burnj:,0]),1))
#print( 'log10(factor) = ',np.round(np.median(samples[burnj:,1]),3),'-',np.round(np.median(samples[burnj:,1])-np.percentile(samples[burnj:,1],15.9),3),'+',np.round(np.percentile(samples[burnj:,1],84.1)-np.median(samples[burnj:,1]),3))

ascii.write(samples[burnj:,:], "chains.dat")


plt.plot(samples[burnj:,0])
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('Temperature [K]')
plt.savefig(dir_path+"\\4B Check mixing, Temperature A.png")
#plt.show()
plt.close()


plt.plot(samples[burnj:,1])
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('log10(factor)')
plt.savefig(dir_path+"\\5B Check mixing, log10(factor) A.png")
#plt.show()
plt.close()


temp=np.empty([len(samples)-burnj])
temp[0]=samples[burnj,0]
for i in range(burnj+1,len(samples)):
    temp[i-burnj]=np.mean(samples[burnj:i,0])
plt.plot(temp)
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('Temperature [K]')
plt.savefig(dir_path+"\\6B Check mixing, Temperature B.png")
#plt.show()
plt.close()

temp=np.empty([len(samples)-burnj])
temp[0]=samples[burnj,1]
for i in range(burnj+1,len(samples)):
    temp[i-burnj]=np.mean(samples[burnj:i,1])
plt.plot(temp)
plt.title('Check mixing')
plt.xlabel('Chain number')
plt.ylabel('log10(factor)')
plt.savefig(dir_path+"\\7B Check mixing, log10(factor) B.png")
#plt.show()
plt.close()

T_mcmc_array = np.array(T_mcmc)
ascii.write(T_mcmc_array, "results.dat", names= ("Temperature", "+", "-"))
