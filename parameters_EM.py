# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:33:00 2023

@author: FENDLER-JUL
"""

from parameters import Parameter_Gibbs
import numpy as np
import scipy.stats as stats


class Mu(Parameter_Gibbs):
    """Mu is the location of the log-normale distributions modelling the continuous covariates.
       Its prior distribution is a normal distribution N(prior_mean, prior_accuracy = 1/var)"""
    def __init__(self, mu, prior_mean, prior_accuracy):
        self.mu_curr = mu
        self.prior_mean = prior_mean
        self.prior_accuracy = prior_accuracy
        
    def update_parameter(self, tau, C, Z , API, T):
        mean = (np.array([tau[k]*np.dot((C == k), np.log(Z)) for k in API])/T+ self.prior_accuracy*self.prior_mean)/(np.array([np.sum(C==k)*tau[k] for k in API])/T+ self.prior_accuracy) 
        cov = np.diag(1/(np.array([np.sum(C==k)*tau[k] for k in API])/T + self.prior_accuracy))
        self.mu_curr[API] = stats.multivariate_normal.rvs(mean, cov)
        
class Tau(Parameter_Gibbs):
    """ Tau is the precision = 1/scale^2 of the log-normale distributions modelling the continuous covariates.
        Its prior distribution is a Gamma distribution G(prior_shape, prior_intensity = 1/scale)"""
    def __init__(self, tau, prior_shape, prior_intensity):
        self.tau_curr = tau
        self.prior_shape = prior_shape
        self.prior_intensity = prior_intensity
        
        
    def update_parameter(self, mu, C, Z, API, T):
        for c in API:
            shape = np.sum(C==c)/(2*T) + self.prior_shape
            intensity = self.prior_intensity + (1/(2*T))*np.dot((C == c), (np.log(Z) - mu[c])**2)
            self.tau_curr[c] = stats.gamma.rvs(a= shape, scale= 1/intensity)
  
