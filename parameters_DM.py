# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:33:02 2023

@author: FENDLER-JUL
"""
from parameters import Parameter_MH, Parameter_Gibbs
import basics as basics
import numpy as np
import scipy.stats as stats


#There are different options for updating Beta with different prior distributions.
#Uncomment the desired one. The file chain MCMC might need to be updated as well.

# class Beta(Parameter_MH):
#     """ Beta is the vector of risk parameters in each clusters.
#         Its prior distribution is a normal distribution N(prior_mean, prior_sd) """
#
#     def __init__(self, beta, proposal_sds, prior_mean, prior_sd):
#         self.beta_curr = beta
#         self.prior_mean = prior_mean
#         self.prior_sd = prior_sd
#         self.proposal_sds = proposal_sds
#         self.size = len(self.beta_curr)
    
#     def update_parameter(self, disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, API, i_expo):
#         accept = np.zeros(len(self.beta_curr))
#         for index in API:
#             accept[index] =  self.update_component(disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, index, i_expo)
#         return accept[API]
    
#     def update_component(self, disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, index, i_expo):
#         beta_curr_index = self.beta_curr[index] 
#         beta_cand = np.array(self.beta_curr, copy = True)
#         beta_cand_index =  stats.lognorm.rvs(s = self.proposal_sds[index], scale = np.exp(-self.proposal_sds[index]**2/2 + np.log(beta_curr_index + 1))) -0.999
#         beta_cand[index] = beta_cand_index
#         logratio = np.dot(event[i_expo], np.log(basics.predictor(disease_model, beta_cand, C[i_expo])) - np.log(basics.predictor(disease_model, self.beta_curr, C[i_expo])))
#         logratio += (prior_param_xi[0] + nevent)*(np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, self.beta_curr, eta, C, y, trunc_y, i_expo)) 
#                                                   - np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, beta_cand, eta, C, y, trunc_y, i_expo)))
#         logratio -= 1/(2*self.prior_sd**2)*((beta_cand_index - self.prior_mean)**2 
#                           - (beta_curr_index - self.prior_mean)**2 )
#         logratio += np.log(beta_cand_index + 1) - np.log(beta_curr_index + 1)
#         logratio += 2*self.proposal_sds[index]*(np.log(beta_curr_index + 1) - np.log(beta_cand_index + 1))
#         accept = (np.log(np.random.uniform(0,1))<logratio)
#         if beta_cand_index == -1:
#             accept = False
#         if accept :
#             self.beta_curr[index] = beta_cand_index
#         return accept
    
# class Beta_Gibbs_EHR(Parameter_Gibbs):
#     """ Beta is the vector of risk parameters in each clusters.
#         Its prior distribution is a gamma distribution G(prior_shape, prior_intentisity = 1/prior_scale)"""
#
#     def __init__(self, beta,  prior_shape, prior_intensity):
#         self.beta_curr = beta
#         self.prior_shape = prior_shape
#         self.prior_intensity = prior_intensity
#         self.size = len(self.beta_curr)
        
#     def update_parameter(self, eta, C, y, trunc_y, event, xi, API, T):
#         ibh = 1/T*10**(-25)*xi/eta * basics.integrated_baseline_hazard(eta, trunc_y, y)
#         for index in API:
#             self.update_component(C, ibh, event, T, index)
            
#     def update_component(self, C, ibh, event, T, index):
#         shape = np.dot(C == index, event)/T + self.prior_shape
#         intensity  = np.dot(C == index, ibh) + self.prior_intensity
#         new_beta = stats.gamma.rvs(a= shape, scale= 1/intensity) - 1
#         self.beta_curr[index] = new_beta

class Beta(Parameter_MH):
    """ Beta is the vector of risk parameters in each clusters.
        Its prior distribution is a beta PERT distribution with parameters (prior_min, prior_mode, prior_max)"""
    def __init__(self, beta, proposal_sds, prior_min, prior_mode, prior_max):
        self.beta_curr = beta
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.proposal_sds = proposal_sds
        self.prior_mode = prior_mode
        self.size = len(self.beta_curr)
        self.param_alpha = (4*self.prior_mode + self.prior_max - 5*self.prior_min)/(self.prior_max - self.prior_min)
        self.param_beta = (5*self.prior_max - self.prior_min - 4*self.prior_mode)/(self.prior_max - self.prior_min)
    
    def update_parameter(self, disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, API, i_expo, T):
        accept = np.zeros(len(self.beta_curr))
        for index in API:
            accept[index] =  self.update_component(disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, index, i_expo, T)
        return accept[API]
        
    def update_component(self, disease_model, eta, C, y, trunc_y, event, nevent, prior_param_xi, index, i_expo, T):
        beta_curr_index = self.beta_curr[index] 
        beta_cand = np.array(self.beta_curr, copy = True)
        proposal_sds_index = self.proposal_sds[index]
        a = (self.prior_min - beta_curr_index)/proposal_sds_index
        b = (self.prior_max - beta_curr_index)/proposal_sds_index
        beta_cand_index =  stats.truncnorm.rvs(a = a, b= b, loc = beta_curr_index, scale = self.proposal_sds[index]) 
        if beta_cand_index == -1.:
            beta_cand_index = -0.99999
        beta_cand[index] = beta_cand_index
        logratio = 1/T*np.dot(event[i_expo], np.log(basics.predictor(disease_model, beta_cand, C[i_expo])) - np.log(basics.predictor(disease_model, self.beta_curr, C[i_expo])))
        logratio += 1/T*(prior_param_xi[0] + nevent)*(np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, self.beta_curr, eta, C, y, trunc_y, i_expo)) 
                                                  - np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, beta_cand, eta, C, y, trunc_y, i_expo)))
        logratio += ((self.param_alpha - 1)*(np.log(beta_cand_index - self.prior_min) - np.log(beta_curr_index - self.prior_min))
                     + (self.param_beta - 1)*(np.log(self.prior_max - beta_cand_index) - np.log(self.prior_max - beta_curr_index)))
        logratio += np.log(stats.norm.cdf((self.prior_max - beta_curr_index)/proposal_sds_index) - stats.norm.cdf((self.prior_min - beta_curr_index)/proposal_sds_index))
        logratio -= np.log(stats.norm.cdf((self.prior_max - beta_cand_index)/proposal_sds_index) - stats.norm.cdf((self.prior_min - beta_cand_index)/proposal_sds_index))
        accept = (np.log(np.random.uniform(0,1))<logratio)
        if accept :
            self.beta_curr[index] = beta_cand_index
        return accept
    
class Eta(Parameter_MH):
    """ Eta is the shape parameter of the baseline hazard.
    Its prior distribution is a gamma distribution G(prior_shape, prior_intentisity = 1/prior_scale)"""

    def __init__(self, eta, proposal_sds, prior_shape, prior_intensity):
        self.eta_curr = eta
        self.etab_curr = eta - 1
        self.prior_shape = prior_shape
        self.prior_intensity = prior_intensity
        self.proposal_sds = proposal_sds
            
    def update_parameter(self, disease_model, beta, prior_param_xi, C, y, 
                         trunc_y, event, nevent, i_expo, T):
        etab_cand = self.etab_curr * np.exp(stats.norm.rvs(0, self.proposal_sds))
        eta_cand = etab_cand + 1
        logratio = self.prior_shape*(np.log(etab_cand) - np.log(self.etab_curr))
        logratio -= self.prior_intensity*(etab_cand - self.etab_curr)
        logratio += 1/T*np.dot(event,np.log(y))*(eta_cand - self.eta_curr)
        logratio += 1/T*((prior_param_xi[0] + nevent)*(np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, beta, self.eta_curr, C, y, trunc_y, i_expo))
                                                  - np.log(prior_param_xi[1] + 10**(-25)* basics.S_beta_eta_C(disease_model, beta, eta_cand, C, y, trunc_y, i_expo))))
        accept = (np.log(np.random.uniform(0,1))<logratio)
        
        self.eta_curr += (eta_cand-self.eta_curr)*accept
        self.etab_curr += (etab_cand-self.etab_curr)*accept
        return(accept)
    
class Xi(Parameter_Gibbs):
    """ Xi the scale parameter of the baseline hazard.
        Its prior distribution is a gamma distribution G(prior_shape, prior_intentisity = 1/prior_scale).
    """

    def __init__(self, xi,  prior_shape, prior_intensity):

        self.xi_curr = xi
        self.prior_shape = prior_shape
        self.prior_intensity = prior_intensity
        self.prior_param = np.array([self.prior_shape, self.prior_intensity])
            
    def update_parameter(self, disease_model, beta, eta, C, y, trunc_y, nevent, i_expo, T):
        shape = self.prior_shape + nevent/T
        intensity = self.prior_intensity+ 10**(-25)*basics.S_beta_eta_C(disease_model, beta, eta, C, y, trunc_y, i_expo)/T
        new_xi = stats.gamma.rvs(a= shape, scale= 1/intensity)
        self.xi_curr = new_xi

        