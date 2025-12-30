# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:33:37 2023

@author: FENDLER-JUL
"""
import numpy as np
import basics

def label_switching_m1(C, phi, Cmax):
    """
    label switching move 1
    """
    j = np.random.randint(0, Cmax)
    l = np.random.randint(0, Cmax)
    if l!= j :
        p_switch = (phi[j]/phi[l])**(np.sum(C == l) - np.sum(C == j))
        p_switch = min(1, p_switch)
        accept = np.random.uniform() < p_switch
    else :
        accept = False
    return(j,l, accept)


def label_switching_m2(C, V, Cmax):
    """
    label switching move 2
    """
    j = np.random.randint(0, Cmax - 1)
    p_switch = ((1 - V[j+1])**(np.sum(C == j)))/((1 -V[j])**(np.sum(C == j + 1)))
    p_switch = min(1, p_switch)
    accept = np.random.uniform() < p_switch
    return(j, accept)

def label_switching_m3(C, phi, V, alpha,  Cmax):
    """
    label switching move 3
    """
    c = np.random.randint(0, Cmax-1)
    C_cand = np.array(C, copy = True)
    C_cand[C == c], C_cand[C == c+1] = c+1,c
    phi_cand = np.array(phi, copy = True)
    phi_plus = phi[c] + phi[c+1]
    E_phi0 = (1 + alpha + np.sum(C == c + 1) + np.sum(C > c + 1))/(alpha + np.sum(C == c + 1) + np.sum(C > c + 1))
    E_phi1 = 1/E_phi0
    phi_prime = phi[c+1]*E_phi0 + phi[c]*E_phi1
    phi_cand[c] = phi[c+1]*phi_plus/phi_prime*E_phi0
    phi_cand[c+1] = phi[c]*phi_plus/phi_prime*E_phi1
    V_cand = np.array(V, copy = True)
    V_cand[c] = phi_cand[c]/(np.prod(1 - V[:c]))
    V_cand[c+1] = phi_cand[c+1]/((1-V_cand[c])*np.prod(1 - V[:c]))
    R1 = ((1 + alpha + np.sum(C == (c+1)) + np.sum([np.sum(C  == k) for k in range(c+2,Cmax)]))
          /(alpha + + np.sum(C == (c+1)) + np.sum([np.sum(C  == k) for k in range(c+2, Cmax)])))
    R2 = ((1 + alpha + np.sum(C == (c)) + np.sum([np.sum(C  == k) for k in range(c+2,Cmax)]))
          /(alpha + + np.sum(C == (c)) + np.sum([np.sum(C  == k) for k in range(c+2, Cmax)])))
    R = (phi_plus/(phi[c+1]*R1 + phi[c]*R2))**(np.sum(C == c) + np.sum(C==c+1))*R1**(np.sum(C == c+1))*R2**(np.sum(C ==c))
    accept = np.random.uniform() < R
    phi += accept*(phi_cand - phi)
    V += accept*(V_cand - V)
    return(c, accept)


def compute_U_star(U):
    return np.min(U)

def compute_CA_star(C):
    return np.max(C)

def compute_CP_star(phi, U_star):
    c = 0
    while (np.sum(phi[:c+1]) <= 1 - U_star):
        c += 1
    return c

def adaptive_proposals(acceptance_rates, proposal_sds, target):
    """
    This functions update the standard deviation of the proposal distribution
    to target either 0.44 for a single parameter, or 0.2 for a vector
    """
    proposal_sds *= (1+np.sign(acceptance_rates-target)*0.20*(np.abs(acceptance_rates-target)>0.02))
    return(proposal_sds)

def save_chain(sample, path_results, chain, param_name):
    """
    Save the chain of the parameter parameter_name in a .npy file.
    """
    f = open(path_results + "/v1"  + chain + param_name + ".npy", 'wb')
    np.save(f, sample)
    f.close()
    
   
def log_posterior_density(disease_model, beta, eta, xi,
                      mu_R, tau_R, mu_G, tau_G, mu_P, tau_P, mu_A, tau_A, C, 
                      phi, V, alpha, y, trunc_y, event, nevent, i_expo, Z_R, Z_G, Z_P, 
                      Z_A, prior_min_beta, prior_mode_beta,
                      prior_max_beta, prior_shape_eta, prior_intensity_eta,
                      prior_shape_xi, prior_intensity_xi,
                      prior_mean_mu_R, prior_accuracy_mu_R, prior_shape_tau_R, 
                      prior_intensity_tau_R, prior_mean_mu_G, prior_accuracy_mu_G, prior_shape_tau_G, 
                      prior_intensity_tau_G, prior_mean_mu_P, prior_accuracy_mu_P, prior_shape_tau_P, 
                      prior_intensity_tau_P, prior_mean_mu_A, prior_accuracy_mu_A, prior_shape_tau_A, 
                      prior_intensity_tau_A, prior_shape_alpha, prior_intensity_alpha):
    """
    Compute the non-normalised logposterior density of the model 
    """
    #likelihood
    prob = np.dot(event,(eta - 1)*np.log(y) + np.log(basics.predictor(disease_model, beta +0.00000000001, C)))
    prob -= (prior_shape_xi + nevent)*np.log(prior_intensity_xi + 10**(-25) * basics.S_beta_eta_C(disease_model, beta, eta, C, y, trunc_y, i_expo))
    prob += prior_intensity_xi * xi -(prior_shape_xi - 1)*np.log(xi)
    
    #prior
    q1 = 1 + 4*(prior_mode_beta - prior_min_beta)/(prior_max_beta - prior_min_beta)
    q2 = 1 + 4*(prior_max_beta - prior_mode_beta)/(prior_max_beta - prior_min_beta)
    prob += np.sum((q1 - 1)*np.log(beta[:np.max(C)] - prior_min_beta) + (q2 - 1)*np.log(prior_max_beta - beta[:np.max(C)]))
    prob += (prior_shape_eta - 1)*np.log(eta + 1) - prior_intensity_eta*eta
    prob += (prior_shape_xi - 1)*np.log(xi) - prior_intensity_xi*xi
    prob += np.sum(np.log(tau_R[C[Z_R >0]])/2 - np.log(Z_R[Z_R >0]) - tau_R[C[Z_R >0]]/2*(mu_R[C[Z_R >0]]-np.log(Z_R[Z_R >0]))**2)
    prob += np.sum(- prior_accuracy_mu_R/2*(mu_R[:np.max(C)]- prior_mean_mu_R)**2 + (prior_shape_tau_R - 1)*np.log(tau_R[:np.max(C)]) - prior_intensity_tau_R*tau_R[:np.max(C)])
    prob += np.sum(np.log(tau_G[C[Z_G >0]])/2 - np.log(Z_G[Z_G >0]) - tau_G[C[Z_G >0]]/2*(mu_G[C[Z_G >0]]-np.log(Z_G[Z_G >0]))**2)
    prob += np.sum(- prior_accuracy_mu_G/2*(mu_G[:np.max(C)]- prior_mean_mu_G)**2 + (prior_shape_tau_G - 1)*np.log(tau_G[:np.max(C)]) - prior_intensity_tau_G*tau_G[:np.max(C)])
    prob += np.sum(np.log(tau_P[C[Z_P >0]])/2 - np.log(Z_P[Z_P >0]) - tau_P[C[Z_P >0]]/2*(mu_P[C[Z_P >0]]-np.log(Z_P[Z_P >0]))**2)
    prob += np.sum(- prior_accuracy_mu_P/2*(mu_P[:np.max(C)]- prior_mean_mu_P)**2 + (prior_shape_tau_P - 1)*np.log(tau_P[:np.max(C)]) - prior_intensity_tau_P*tau_P[:np.max(C)])
    prob += np.sum(np.log(tau_A[C[i_expo]])/2 - np.log(Z_A[i_expo]) - tau_A[C[i_expo]]/2*(mu_A[C[i_expo]]-np.log(Z_A[i_expo]))**2)
    prob += np.sum(- prior_accuracy_mu_A/2*(mu_A[:np.max(C)]- prior_mean_mu_A)**2 + (prior_shape_tau_A - 1)*np.log(tau_A[:np.max(C)]) - prior_intensity_tau_A*tau_A[:np.max(C)])
    prob += np.sum(np.log(phi[C] + 0.0000000001))
    prob += (alpha -1)*np.sum(np.log(1 - V[:np.max(C)]))
    prob += (prior_shape_alpha - 1)*np.log(alpha) - prior_intensity_alpha*alpha
    
    return(prob)
           
        
def accept_swap_PT(log_prob_i, log_prob_j, T_i, T_j):
    """
    Accept the swap of two temperature.
    """
    log_prob_swap = (1/T_i - 1/T_j)*(log_prob_j - log_prob_i)
    accept = (np.log(np.random.uniform(0,1))<log_prob_swap)
    return accept