from mcmc_test.mcmc_clone_emceee import model
from mccm_test.mcmc_clone_emceee import acceptfrac

import pytest
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import scipy.constants as con
import os
import emcee
import corner
import scipy.optimize as op
import random

from inspect import signature

#determine if the function recieves correct number of arguments
def test_model_format():
    sig = signature(model)
    params = sig.parameters
    print(len(params))
    assert len(params) == 3

#Determines if the MCMC results are within the acceptable range
def test_mcmc_clone_emceee_output():
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\mcmc_test"
    results = ascii.read(path+"\\results.dat")
    assert   225 <= results[0][0] <= 260 and 225 <= results[0][0]+results[0][1] <= 260 and 225 <= results[0][0]+results[0][2] <= 260

#Test to determine if mcmc_clone_emceee outputs the right amount of graphs into the proper directory
def test_mcmc_clone_emceee_output_number():
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\mcmc_test\\emcee_model_graphs"
    assert len(os.listdir(path)) == 9
