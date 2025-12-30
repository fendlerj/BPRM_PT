# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:00:20 2023

@author: FENDLER-JUL
"""

import os
os.chdir('path_code') #path where the files basics, parameters_AM, parameters_EM, parameters_DM and MCMC are.
import numpy as np
import basics as basics
import parameters_AM as pAM
import parameters_EM as pEM
import parameters_DM as pDM
import MCMC as MCMC
import random
import scipy.stats as stats
import time

#path where the data file is
filename = 'path_data/data.csv' 

#Set seed
seed = 123

#EHR or Cox
disease_model = 'EHR'

#name of the chain - 
chain = 'chain1'
#path where the results should be saved
path_results = os.path.join('path_res')

#Load the data frame
frame = basics.load_data(filename)

#Get the numpy arrays of each covariate
Z_R = frame['Rad'].values 
Z_G = frame['Gam'].values 
Z_P = frame['Pous'].values 
Z_A = frame['Age'].values
#Get the numpy arrays of the outcome
y = frame['Y'].values
trunc_y = np.zeros(len(y))
event = frame['delta'].values

#Get the index of non-exposed individuals
index_non_expo = frame[(frame['Rad']== 0) & (frame['Gam']== 0) & (frame['Pous']== 0)].index.values
i_expo = np.ones(len(y), dtype=bool)
i_expo[index_non_expo] = False


#Total number of iterations of the MCMC algorithm
nb_iter= 30000
#Number of iterations during the burnin period
burnin = 10000
#Number of iterations during the burnin phase
nb_phases = 50


#Maximum number of cluster - if too small a warning will be printed
Cmax = 50


#Initialisation of alpha
alpha_start = pAM.Alpha(alpha = 0.5, prior_shape = 2, prior_intensity = 1)

#Initialisation of C
C_start = pAM.C(np.random.randint(15, size=len(y)))
C_start.C_curr[i_expo == False] = -1

#Initialisation of V and phi
V_start = pAM.V(np.random.beta(1, alpha_start.alpha_curr, size=Cmax))
phi_start = pAM.Phi(np.zeros(Cmax))
phi_start.update_parameter(V_start.V_curr)

#Initialisation of U - latent variable of the slice sampler
U_start = pAM.U(np.random.rand(len(y))*phi_start.phi_curr[C_start.C_curr])

#Initialisation of the parameters mu and tau
mu_R_start = pEM.Mu(np.random.rand(Cmax), 0.10, 1/2.25**2)
tau_R_start = pEM.Tau(np.random.rand(Cmax), 1,1)
mu_G_start = pEM.Mu(np.random.rand(Cmax), -2.3, 1/8.08**2)
tau_G_start = pEM.Tau(np.random.rand(Cmax), 1, 1)
mu_P_start = pEM.Mu(np.random.rand(Cmax), 1.01, 1/11.79**2)
tau_P_start = pEM.Tau(np.random.rand(Cmax), 1, 1)
mu_A_start = pEM.Mu(np.random.rand(Cmax), 0, 1/10**2)
tau_A_start = pEM.Tau(np.random.rand(Cmax), 1, 1)

#Initialisation of Beta
##Uncomment this line as well of the relevant lines in the file parameters_DM for a normal prior distribution on beta
#beta_start = pDM.Beta(np.exp(np.random.normal(0,1,Cmax)), np.random.rand(Cmax), 0, 100)
##Uncomment this line as well of the relevant lines in the file parameters_DM for a gamma prior distribution on beta
#beta_start = pDM.Beta_Gibbs_EHR(np.exp(np.random.normal(0,10,Cmax)), 0.1,0.1)
##Beta follow a Beta PERT prior distribution
beta_start = pDM.Beta(stats.truncnorm.rvs(a = (-1 - 0)/10, b = (10 - 0)/10,loc = 0, scale =10, size = Cmax), np.random.rand(Cmax), -1, 0, 40)

#Initialisation of eta
eta_start = pDM.Eta(5, 0.8, 0.001, 0.001)

#Initialisation of xi
xi_start = pDM.Xi(10, 1, 1)

#Initialisation of the seed 
np.random.seed(seed)
random.seed(seed)

#Initialisation of the MCMC sampler
MCMC = MCMC.MCMC_PT(chain, path_results, disease_model, alpha_start, C_start, V_start, phi_start,
                 U_start, mu_R_start, tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                 mu_A_start, tau_A_start, beta_start, eta_start, xi_start,
                 Z_R, Z_G, Z_P, Z_A, y, trunc_y, event, i_expo, seed)


#Start and time the MCMC algorithm
t1 = time.time()
MCMC.run_adaptive_MCMC(nb_phases,nb_iter, burnin)
t2 = time.time()
print(t2- t1)

