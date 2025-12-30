# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:59:08 2023

@author: FENDLER-JUL
"""
import numpy as np
from parameters import Parameter_Gibbs
import scipy.stats as stats
from jax import numpy as jnp
from jax import jit, random, vmap
import basics

class Alpha(Parameter_Gibbs):
    """Alpha is the concentration parameter of the Dirichlet process
       Its prior distribution is a Gamma G(shape, intensity)"""
    def __init__(self, alpha, prior_shape, prior_intensity):
        self.alpha_curr = alpha
        self.prior_shape = prior_shape
        self.prior_intensity = prior_intensity
        
    def update_parameter(self, V, A):
        shape = len(A) + self.prior_shape
        V[V == 1] = 0.999999999 #avoid numeric errors
        rate = self.prior_intensity - np.sum(np.log(1 - V[A]))
        self.alpha_curr = stats.gamma.rvs(a = shape, scale = 1/rate, size = 1)[0]
    

class V(Parameter_Gibbs):
    """
    V is the vector of parameters involved in the sitck breaking construction of the mixture weigths.
    It prior distribution is a Gamma(1, alpha)
    """
    def __init__(self, V) :
        self.V_curr = V
        
    def update_parameter_A(self, A, C, alpha):
        #update the alaments of V in the active set
        for c in A :
            a = 1 + np.sum(C ==c)
            b = alpha + np.sum(C > c)
            self.V_curr[c] = stats.beta.rvs(a,b)
    
    def update_parameter_P(self, CA_star, U_star, alpha, Phi):
        #update the element of V in the potential set
        c = CA_star
        while np.sum(Phi[:c]) <= (1 - U_star) and c < len(self.V_curr):
            self.V_curr[c] = stats.beta.rvs(1, alpha)
            c+=1
        return(c)
            
            
class U(Parameter_Gibbs):
    """
    U is the vector of parameters from the independent slice sampler.
    Its prior distribution is a uniform distribution U(0,phi)
    """
    def __init__(self, U):
        self.U_curr = U
        self.size = len(U)
        
    def update_parameter(self, Phi, C, i_expo):
        self.U_curr[i_expo] = np.random.random(len(C))*Phi[C]
        
class Phi():
    """
    Phi is the vector of mixture weights. 
    Knowing V, its values are determinist.
    """
    def __init__(self, phi):
        self.phi_curr = phi
        
    def update_parameter(self, V):
        self.phi_curr[0] = V[0]
        for c in range(1,len(self.phi_curr)):
            self.phi_curr[c] = V[c]*np.prod((1 - V[:c]))
            
        
class C(Parameter_Gibbs):
    def __init__(self, C):
        self.C_curr = C
        self.size = len(C)
        
    def change_value(self, new_C):
        self.C_curr = new_C
    
    def update_parameter(self, disease_model, beta, xi, prior_param_xi_0, prior_param_xi_1, eta, mu_R, mu_G,
                         mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, 
                         phi, Z_R, Z_G, Z_P, Z_A,
                         y, trunc_y, event, i_expo, nevent, U, phi_curr, T, key, C_star):
        integrated_baseline_hazard = ibh_jit(xi, eta, y, trunc_y)
        
        #Compute risk ratio
        if disease_model == 'Cox':
            risk_ratio = np.exp(beta)
        else : risk_ratio = 1 + beta
        
        risk_ratio[risk_ratio == 0] = 0.000000001 #To avoid numeric errors
        
        #Compute the baseline hasard
        S_eta_beta_C = basics.S_beta_eta_C(disease_model, beta, eta, self.C_curr, y, trunc_y, i_expo)
        
        #For each individual, find the clusters for which the probability of belonging to the cluster 
        # of the individual is null
        template_logproba = np.array((U[:, None] > phi_curr[None, :]), dtype = float)
        template_logproba[template_logproba == 1.0] = -np.inf

        # For each exposed indidividual, draw the cluster membership
        res_vec = draw_C_vec( mu_R, mu_G, mu_P, mu_A,
                             tau_R, tau_G, tau_P, tau_A, 
                            phi, risk_ratio, xi, prior_param_xi_0, prior_param_xi_1, S_eta_beta_C,
                            integrated_baseline_hazard[i_expo], 
                            Z_R[i_expo], Z_G[i_expo], Z_P[i_expo],
                            Z_A[i_expo], event[i_expo],  nevent, template_logproba[i_expo,], T,
                            random.split(key, num = int(jnp.sum(i_expo))), C_star) 
        
        self.C_curr[i_expo] = res_vec 
    
    
                  
def log_proba_cont_indiv(mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, Z_R_i, Z_G_i, Z_P_i, Z_A_i, T):
    #For a given individual i, compute [Z_R_i, Z_G_i, Z_P_i, Z_A_i | mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A]
    logproba = 0.
    logproba += jnp.where(Z_R_i >0, -tau_R[:]*((jnp.log(Z_R_i) - mu_R[:])**2)/(2*T) + jnp.log(tau_R[:])/(2*T), 0)
    logproba += jnp.where(Z_G_i >0, -tau_G[:]*((jnp.log(Z_G_i) - mu_G[:])**2)/(2*T) + jnp.log(tau_G[:])/(2*T), 0)
    logproba += jnp.where(Z_P_i >0, -tau_P[:]*((jnp.log(Z_P_i) - mu_P[:])**2)/(2*T) + jnp.log(tau_P[:])/(2*T), 0)
    logproba += -tau_A[:]*((jnp.log(Z_A_i) - mu_A[:])**2)/(2*T) + jnp.log(tau_A[:])/(2*T)
    return logproba

log_proba_cont_indiv_jit = jit(log_proba_cont_indiv)

def log_proba_indiv( mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, 
                    phi, risk_ratio, xi, prior_param_xi_0, prior_param_xi_1, S_eta_beta_C, ibh_i,
                    Z_R_i, Z_G_i, Z_P_i, Z_A_i, event_i, nevent, T):
    #For a given individual i, compute the non-normalised log probability of belonging to each cluster
    logproba = event_i*jnp.log(risk_ratio)/T - ibh_i*risk_ratio/T
    logproba +=  10**(-25)*xi*S_eta_beta_C/T - (prior_param_xi_0 + nevent/T)*jnp.log(prior_param_xi_1 + 10**(-25)/T*S_eta_beta_C)
    logproba += jnp.log(phi) 
    logproba += log_proba_cont_indiv_jit(mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, Z_R_i, Z_G_i, Z_P_i, Z_A_i, T)
    return logproba

log_proba_indiv_jit = log_proba_indiv

def draw_C( mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, 
           phi, risk_ratio, xi, prior_param_xi_0, prior_param_xi_1, S_eta_beta_C, integrated_baseline_hazard_i,
           Z_R_i, Z_G_i,Z_P_i, Z_A_i, event_i, nevent, template_logproba_i, T, key_cat, C_star):
    #Compute the vector of non-normalised log probability of belonging to each cluster
    logproba = log_proba_indiv_jit( mu_R, mu_G, mu_P, mu_A, tau_R, tau_G, tau_P, tau_A, 
                        phi, risk_ratio, xi, prior_param_xi_0, prior_param_xi_1, S_eta_beta_C,
                        integrated_baseline_hazard_i, 
                        Z_R_i, Z_G_i, Z_P_i, Z_A_i, event_i, nevent, T)
    
    #For the cluster for which the probability of belonging is null, set the log-proba to -inf
    logproba = logproba + template_logproba_i

    logproba = logproba - jnp.max(logproba) #To avoid numarical errors
    
    #Draw the cluster membership
    res = random.categorical(key_cat, logits = logproba, axis=-1, shape=None)
    return res

draw_C_vec = vmap(draw_C, in_axes = (None, None, None, None, None, 
                                     None, None, None, None, None, None, None, None, None, 0, 0, 0, 
                                     0, 0, 0, None, 0, None, 0, None)) #Parallelise the drawing of the cluster membership over all the individuals



def ibh(xi, eta, y, trunc_y):
    #Compute the integrated baseline hazard
    return ((10**(-25))*xi/eta)*(y**eta -trunc_y**eta)

ibh_jit = jit(ibh)

    

        
