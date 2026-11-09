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
        API = np.asarray(API)
        obs = Z > 0
        K = len(self.mu_curr)
        n_k = np.bincount(C[obs], minlength=K)[API]
        s_k = np.bincount(C[obs], weights=np.log(Z[obs]), minlength=K)[API]
        prec = n_k*tau[API]/T + self.prior_accuracy
        mean = (tau[API]*s_k/T + self.prior_accuracy*self.prior_mean)/prec
        self.mu_curr[API] = mean + np.random.standard_normal(len(API))/np.sqrt(prec)

        
        
class Tau(Parameter_Gibbs):
    """ Tau is the precision = 1/scale^2 of the log-normale distributions modelling the continuous covariates.
        Its prior distribution is a Gamma distribution G(prior_shape, prior_intensity = 1/scale)"""
    def __init__(self, tau, prior_shape, prior_intensity):
        self.tau_curr = tau
        self.prior_shape = prior_shape
        self.prior_intensity = prior_intensity
        
        
    def update_parameter(self, mu, C, Z, API, T):
        API = np.asarray(API)
        obs = Z > 0
        K = len(self.tau_curr)
        Co = C[obs]
        n_k = np.bincount(Co, minlength=K)[API]
        ss_k = np.bincount(Co, weights=(np.log(Z[obs]) - mu[Co])**2, minlength=K)[API]
        shape = n_k/(2*T) + self.prior_shape
        rate = self.prior_intensity + ss_k/(2*T)
        self.tau_curr[API] = np.random.gamma(shape, 1/rate)

    
  
