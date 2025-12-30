# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:07:22 2023

@author: FENDLER-JUL
"""

import numpy as np
import functions_mcmc as f_MCMC
from jax import random
from copy import deepcopy

class chain_MCMC():
    
    def __init__(self, disease_model, alpha_start, C_start, V_start, phi_start,
                 U_start, mu_R_start, tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                 mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                 Z_G, Z_P, Z_A, y, trunc_y, event,
                 i_expo, seed, T):
        self.disease_model = disease_model
        self.alpha = deepcopy(alpha_start)
        self.beta = deepcopy(beta_start)
        self.C = deepcopy(C_start)
        self.V = deepcopy(V_start)
        self.U = deepcopy(U_start)
        self.alpha = deepcopy(alpha_start)
        self.phi = deepcopy(phi_start)
        self.mu_R = deepcopy(mu_R_start)
        self.tau_R = deepcopy(tau_R_start)
        self.mu_G = deepcopy(mu_G_start)
        self.tau_G = deepcopy(tau_G_start)
        self.mu_P = deepcopy(mu_P_start)
        self.tau_P = deepcopy(tau_P_start)
        self.mu_A = deepcopy(mu_A_start)
        self.tau_A = deepcopy(tau_A_start)
        self.eta = deepcopy(eta_start)
        self.xi = deepcopy(xi_start)
        self.Z_R = Z_R
        self.Z_G = Z_G
        self.Z_P = Z_P
        self.Z_A = Z_A
        self.y = y
        self.trunc_y = trunc_y
        self.event = event
        self.nevent = np.sum(event)
        self.acceptance_eta = np.zeros(3)
        self.acceptance_beta = np.zeros((1, len(self.beta.beta_curr)))* np.nan
        self.set_A = np.array([0])
        self.set_P = np.array([0])
        self.CA_star = f_MCMC.compute_CA_star(self.C.C_curr)
        self.i_expo = i_expo
        self.key = random.PRNGKey(seed)
        self.T = T
    
        
    def iter_1(self, iter_i):
        """
        Iter once the chain of the MCMC parameters
        """
        #compute CA_star and set_A
        self.CA_star = f_MCMC.compute_CA_star(self.C.C_curr)
        self.set_A = np.array([k for k in range(0,self.CA_star +1)])
        
        #update V[set_A]
        self.V.update_parameter_A(self.set_A, self.C.C_curr, self.alpha.alpha_curr)
        
        #update phi
        self.phi.update_parameter(self.V.V_curr)
       
        #update mu[set_A] et tau[set_A]
        self.mu_R.update_parameter(self.tau_R.tau_curr, self.C.C_curr[self.i_expo], self.Z_R[self.i_expo] , self.set_A, self.T)
        self.tau_R.update_parameter(self.mu_R.mu_curr, self.C.C_curr[self.i_expo], self.Z_R[self.i_expo], self.set_A, self.T)
        self.mu_G.update_parameter(self.tau_G.tau_curr, self.C.C_curr[self.i_expo], self.Z_G[self.i_expo] , self.set_A, self.T)
        self.tau_G.update_parameter(self.mu_G.mu_curr, self.C.C_curr[self.i_expo], self.Z_G[self.i_expo], self.set_A, self.T)
        self.mu_P.update_parameter(self.tau_P.tau_curr, self.C.C_curr[self.i_expo], self.Z_P[self.i_expo] , self.set_A, self.T)
        self.tau_P.update_parameter(self.mu_P.mu_curr, self.C.C_curr[self.i_expo], self.Z_P[self.i_expo], self.set_A, self.T)
        self.mu_A.update_parameter(self.tau_A.tau_curr, self.C.C_curr[self.i_expo], self.Z_A[self.i_expo] , self.set_A, self.T)
        self.tau_A.update_parameter(self.mu_A.mu_curr, self.C.C_curr[self.i_expo], self.Z_A[self.i_expo], self.set_A, self.T)
        
        #update beta_A
        #Metropolis Hasting
        self.acceptance_beta[:,self.set_A] = self.beta.update_parameter(self.disease_model, self.eta.eta_curr, self.C.C_curr, self.y, self.trunc_y, self.event, 
                                   self.nevent, self.xi.prior_param, self.set_A, self.i_expo, self.T)
        #Gibbs
        #self.beta.update_parameter(self.eta.eta_curr, self.C.C_curr, self.y, self.trunc_y, self.event, self.xi.xi_curr, self.set_A, self.T)
        
        if self.CA_star > 1:
            # label switching move 1
            label_j, label_l, accept_switch = f_MCMC.label_switching_m1(self.C.C_curr, self.phi.phi_curr, self.CA_star)
            if accept_switch : 
                self.C.C_curr[label_j], self.C.C_curr[label_l] = self.C.C_curr[label_l], self.C.C_curr[label_j]
            #label_switching move 2
            label_j, accept_switch = f_MCMC.label_switching_m2(self.C.C_curr, self.V.V_curr, self.CA_star)
            if accept_switch : 
                self.C.C_curr[label_j], self.C.C_curr[label_j +1] = self.C.C_curr[label_j +1], self.C.C_curr[label_j]
            # label switching move 3
            c, accept_switch = f_MCMC.label_switching_m3(
                self.C.C_curr, self.phi.phi_curr, self.V.V_curr, self.alpha.alpha_curr, self.CA_star)
            if accept_switch:
                self.C.C_curr[c], self.C.C_curr[c +1] = self.C.C_curr[c +1], self.C.C_curr[c]
        
        # update U
        self.U.update_parameter(self.phi.phi_curr, self.C.C_curr[self.i_expo], self.i_expo)
        
        #compute U_star et CA_star
        U_star = f_MCMC.compute_U_star(self.U.U_curr)
        self.CA_star = f_MCMC.compute_CA_star(self.C.C_curr[self.i_expo])
        
        #update alpha
        self.alpha.update_parameter(self.V.V_curr, self.set_A)
        
        #update V[set_P] et CP_star
        CP_star = self.V.update_parameter_P(self.CA_star, U_star, self.alpha.alpha_curr, self.phi.phi_curr)
        
        #update phi
        self.phi.update_parameter(self.V.V_curr)
        self.set_P = np.array([k for k in range(self.CA_star + 1, min(CP_star,len(self.beta.beta_curr)))])
        self.U.update_parameter(self.phi.phi_curr, self.C.C_curr[self.i_expo], self.i_expo)
        
        if len(self.set_P) >0: # If the potential set is nor empty
            #update phi
            self.phi.update_parameter(self.V.V_curr)
            
            #update mu[set_P] et tau[set_P]
            self.mu_R.update_parameter(self.tau_R.tau_curr, self.C.C_curr[self.i_expo], self.Z_R[self.i_expo] , self.set_P, self.T)
            self.tau_R.update_parameter(self.mu_R.mu_curr, self.C.C_curr[self.i_expo], self.Z_R[self.i_expo], self.set_P, self.T)
            self.mu_G.update_parameter(self.tau_G.tau_curr, self.C.C_curr[self.i_expo], self.Z_G[self.i_expo] , self.set_P, self.T)
            self.tau_G.update_parameter(self.mu_G.mu_curr, self.C.C_curr[self.i_expo], self.Z_G[self.i_expo], self.set_P, self.T)
            self.mu_P.update_parameter(self.tau_P.tau_curr, self.C.C_curr[self.i_expo], self.Z_P[self.i_expo] , self.set_P, self.T)
            self.tau_P.update_parameter(self.mu_P.mu_curr, self.C.C_curr[self.i_expo], self.Z_P[self.i_expo], self.set_P, self.T)
            self.mu_A.update_parameter(self.tau_A.tau_curr, self.C.C_curr[self.i_expo], self.Z_A[self.i_expo] , self.set_P, self.T)
            self.tau_A.update_parameter(self.mu_A.mu_curr, self.C.C_curr[self.i_expo], self.Z_A[self.i_expo], self.set_P, self.T)
           
            #update beta[set_P]
            self.acceptance_beta[:,self.set_P] = self.beta.update_parameter(self.disease_model, self.eta.eta_curr, self.C.C_curr, self.y, self.trunc_y, self.event, 
                                       self.nevent, self.xi.prior_param, self.set_P, self.i_expo, self.T)
            #Gibbs
            #self.beta.update_parameter(self.eta.eta_curr, self.C.C_curr, self.y, self.trunc_y, self.event, self.xi.xi_curr, self.set_P, self.T)
        
        #update eta and xi
        self.acceptance_eta = self.eta.update_parameter(self.disease_model, self.beta.beta_curr, self.xi.prior_param, self.C.C_curr, 
                                  self.y, self.trunc_y, self.event, self.nevent,  self.i_expo, self.T)
        self.xi.update_parameter(self.disease_model, self.beta.beta_curr, self.eta.eta_curr, self.C.C_curr, 
                                  self.y, self.trunc_y, self.nevent, self.i_expo, self.T)
        # update C
        self.C.update_parameter(self.disease_model, self.beta.beta_curr, self.xi.xi_curr,
                                self.xi.prior_param[0], self.xi.prior_param[1], self.eta.eta_curr, self.mu_R.mu_curr, 
                                self.mu_G.mu_curr, self.mu_P.mu_curr, self.mu_A.mu_curr, 
                                self.tau_R.tau_curr, self.tau_G.tau_curr, self.tau_P.tau_curr, 
                                self.tau_A.tau_curr,
                                self.phi.phi_curr, self.Z_R, self.Z_G, self.Z_P, self.Z_A, 
                                self.y, self.trunc_y, self.event, self.i_expo, self.nevent, self.U.U_curr,
                                 self.phi.phi_curr, self.T, self.key, CP_star)
        