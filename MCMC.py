# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:40:45 2023

@author: FENDLER-JUL
"""

import chain_MCMC
import numpy as np
import functions_mcmc as f_MCMC
from random import sample

class MCMC_PT():
    
    def __init__(self, chain, path_results, disease_model, 
               alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
               tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
               mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
               Z_G, Z_P, Z_A, y, trunc_y, event,
               i_expo, seed = 0, vect_T = [1,2,5,10,20]):
        
        # initialise the chain at temperature number 1
        self.chain1 = chain_MCMC.chain_MCMC(disease_model, 
                   alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
                   tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                   mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                   Z_G, Z_P, Z_A,  y, trunc_y, event,
                   i_expo, seed, vect_T[0])
        
        # initialise the chain at temperature number 2
        self.chain2 = chain_MCMC.chain_MCMC(disease_model, 
                   alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
                   tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                   mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                   Z_G, Z_P, Z_A, y, trunc_y, event,
                   i_expo, seed, vect_T[1])
        
        # initialise the chain at temperature number 3
        self.chain3 = chain_MCMC.chain_MCMC(disease_model, 
                   alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
                   tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                   mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                   Z_G, Z_P, Z_A, y, trunc_y, event,
                   i_expo, seed, vect_T[2])
        
        # initialise the chain at temperature number 4
        self.chain4 = chain_MCMC.chain_MCMC(disease_model, 
                   alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
                   tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                   mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                   Z_G, Z_P, Z_A, y, trunc_y, event,
                   i_expo, seed, vect_T[3])
        
        # initialise the chain at temperature number 5
        self.chain5 = chain_MCMC.chain_MCMC(disease_model, 
                   alpha_start, C_start, V_start, phi_start,U_start, mu_R_start, 
                   tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                   mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R, 
                   Z_G, Z_P, Z_A, y, trunc_y, event,
                   i_expo, seed, vect_T[4])
        
        self.nb_T = len(vect_T)
        self.path_results = path_results
        self.chain = chain
        self.dict_chain = {0 : self.chain1, 1 : self.chain2, 2 : self.chain3,
                           3 : self.chain4, 4 : self.chain5}
        
        
        
        
    def run_MCMC(self, niter):
        """
        Run niter iteration of the MCMC algorithm
        """

        #Initialise the vector of acceptance and the chain for each parameter
        self.acceptance_eta_chain1 = np.zeros(niter)
        self.acceptance_eta_chain2 = np.zeros(niter)
        self.acceptance_eta_chain3 = np.zeros(niter)
        self.acceptance_eta_chain4 = np.zeros(niter)
        self.acceptance_eta_chain5 = np.zeros(niter)
        self.acceptance_beta_chain1 = np.zeros((niter, len(self.chain1.beta.beta_curr)))* np.nan
        self.acceptance_beta_chain2 = np.zeros((niter, len(self.chain1.beta.beta_curr)))* np.nan
        self.acceptance_beta_chain3 = np.zeros((niter, len(self.chain1.beta.beta_curr)))* np.nan
        self.acceptance_beta_chain4 = np.zeros((niter, len(self.chain1.beta.beta_curr)))* np.nan
        self.acceptance_beta_chain5 = np.zeros((niter, len(self.chain1.beta.beta_curr)))* np.nan
        self.acceptance_swap = []
        chain_beta = np.zeros((niter, len(self.chain1.beta.beta_curr)))
        chain_eta = np.zeros(niter)
        chain_xi = np.zeros(niter)
        chain_alpha = np.zeros(niter)
        chain_mu_R = np.zeros((niter, len(self.chain1.mu_R.mu_curr)))
        chain_tau_R = np.zeros((niter, len(self.chain1.tau_R.tau_curr)))
        chain_mu_G = np.zeros((niter, len(self.chain1.mu_G.mu_curr)))
        chain_tau_G = np.zeros((niter, len(self.chain1.tau_G.tau_curr)))
        chain_mu_P = np.zeros((niter, len(self.chain1.mu_P.mu_curr)))
        chain_tau_P = np.zeros((niter, len(self.chain1.tau_P.tau_curr)))
        chain_mu_A = np.zeros((niter, len(self.chain1.mu_A.mu_curr)))
        chain_tau_A = np.zeros((niter, len(self.chain1.tau_A.tau_curr)))
        chain_C = np.zeros((niter, len(self.chain1.C.C_curr)))
       #Run niter iterations for the 5 chains
        for i in range(niter):
            self.chain1.iter_1(i)
            self.chain2.iter_1(i)
            self.chain3.iter_1(i)
            self.chain4.iter_1(i)
            self.chain5.iter_1(i)
            self.acceptance_eta_chain1[i] = self.chain1.acceptance_eta
            self.acceptance_eta_chain2[i] = self.chain2.acceptance_eta
            self.acceptance_eta_chain3[i] = self.chain3.acceptance_eta
            self.acceptance_eta_chain4[i] = self.chain4.acceptance_eta
            self.acceptance_eta_chain5[i] = self.chain5.acceptance_eta
            self.acceptance_beta_chain1[i,:] = self.chain1.acceptance_beta
            self.acceptance_beta_chain2[i,:] = self.chain2.acceptance_beta
            self.acceptance_beta_chain3[i,:] = self.chain3.acceptance_beta
            self.acceptance_beta_chain4[i,:] = self.chain4.acceptance_beta
            self.acceptance_beta_chain5[i,:] = self.chain5.acceptance_beta
        
            #swap two chains every 100 iterations
            if i%100 == 0 :
                i_swap = sample([0,1,2,3], 1)[0]
                j_swap = i_swap + 1 
                chain_i = self.dict_chain[i_swap]
                chain_j = self.dict_chain[j_swap]
                accept_swap = False
                log_prob_j = f_MCMC.log_posterior_density(chain_j.disease_model, chain_j.beta.beta_curr,
                                                  chain_j.eta.eta_curr, chain_j.xi.xi_curr,  
                                                  chain_j.mu_R.mu_curr, 
                                                  chain_j.tau_R.tau_curr, chain_j.mu_G.mu_curr, 
                                                  chain_j.tau_G.tau_curr, chain_j.mu_P.mu_curr, 
                                                  chain_j.tau_P.tau_curr, chain_j.mu_A.mu_curr, 
                                                  chain_j.tau_A.tau_curr, chain_j.C.C_curr, 
                                                  chain_j.phi.phi_curr, chain_j.V.V_curr,
                                                  chain_j.alpha.alpha_curr, chain_j.y, 
                                                  chain_j.trunc_y, chain_j.event, chain_j.nevent, chain_j.i_expo, 
                                                  chain_j.Z_R, chain_j.Z_G, chain_j.Z_P, 
                                                  chain_j.Z_A, 
                                                  chain_j.beta.prior_min, chain_j.beta.prior_mode, chain_j.beta.prior_max, 
                                                  chain_j.eta.prior_shape, chain_j.eta.prior_intensity,
                                                  chain_j.xi.prior_shape, chain_j.xi.prior_intensity,
                                                  chain_j.mu_R.prior_mean, 
                                                  chain_j.mu_R.prior_accuracy, chain_j.tau_R.prior_shape,
                                                  chain_j.tau_R.prior_intensity, chain_j.mu_G.prior_mean,
                                                  chain_j.mu_G.prior_accuracy, chain_j.tau_G.prior_shape, 
                                                  chain_j.tau_G.prior_intensity, chain_j.mu_P.prior_mean, 
                                                  chain_j.mu_P.prior_accuracy, chain_j.tau_P.prior_shape, 
                                                  chain_j.tau_P.prior_intensity, chain_j.mu_A.prior_mean,
                                                  chain_j.mu_A.prior_accuracy, chain_j.tau_A.prior_shape,
                                                  chain_j.tau_A.prior_intensity, chain_j.alpha.prior_shape, 
                                                  chain_j.alpha.prior_intensity)
                
                log_prob_i = f_MCMC.log_posterior_density(chain_i.disease_model, chain_i.beta.beta_curr,
                                                  chain_i.eta.eta_curr, chain_i.xi.xi_curr,  
                                                  chain_i.mu_R.mu_curr, 
                                                  chain_i.tau_R.tau_curr, chain_i.mu_G.mu_curr, 
                                                  chain_i.tau_G.tau_curr, chain_i.mu_P.mu_curr, 
                                                  chain_i.tau_P.tau_curr, chain_i.mu_A.mu_curr, 
                                                  chain_i.tau_A.tau_curr, chain_i.C.C_curr, 
                                                  chain_i.phi.phi_curr, chain_i.V.V_curr,
                                                  chain_i.alpha.alpha_curr, chain_i.y, 
                                                  chain_i.trunc_y, chain_i.event, chain_i.nevent, chain_i.i_expo, 
                                                  chain_i.Z_R, chain_i.Z_G, chain_i.Z_P, 
                                                  chain_i.Z_A,
                                                  chain_i.beta.prior_min, chain_i.beta.prior_mode, chain_i.beta.prior_max, 
                                                  chain_i.eta.prior_shape, chain_i.eta.prior_intensity,
                                                  chain_i.xi.prior_shape, chain_i.xi.prior_intensity,
                                                  chain_i.mu_R.prior_mean, 
                                                  chain_i.mu_R.prior_accuracy, chain_i.tau_R.prior_shape,
                                                  chain_i.tau_R.prior_intensity, chain_i.mu_G.prior_mean,
                                                  chain_i.mu_G.prior_accuracy, chain_i.tau_G.prior_shape, 
                                                  chain_i.tau_G.prior_intensity, chain_i.mu_P.prior_mean, 
                                                  chain_i.mu_P.prior_accuracy, chain_i.tau_P.prior_shape, 
                                                  chain_i.tau_P.prior_intensity, chain_i.mu_A.prior_mean,
                                                  chain_i.mu_A.prior_accuracy, chain_i.tau_A.prior_shape,
                                                  chain_i.tau_A.prior_intensity, chain_i.alpha.prior_shape, 
                                                  chain_i.alpha.prior_intensity)
                
                if f_MCMC.accept_swap_PT(log_prob_i, log_prob_j, chain_i.T, chain_j.T) :
                    #If accept swap, swap all the parameters of the two chains
                    chain_i.beta.beta_curr, chain_j.beta.beta_curr = chain_j.beta.beta_curr, chain_i.beta.beta_curr
                    chain_i.eta.eta_curr, chain_j.eta.eta_curr = chain_j.eta.eta_curr, chain_i.eta.eta_curr
                    chain_i.xi.xi_curr, chain_j.xi.xi_curr = chain_j.xi.xi_curr, chain_i.xi.xi_curr
                    chain_i.mu_R.mu_curr, chain_j.mu_R.mu_curr = chain_j.mu_R.mu_curr, chain_i.mu_R.mu_curr
                    chain_i.mu_G.mu_curr, chain_j.mu_G.mu_curr = chain_j.mu_G.mu_curr, chain_i.mu_G.mu_curr
                    chain_i.mu_P.mu_curr, chain_j.mu_P.mu_curr = chain_j.mu_P.mu_curr, chain_i.mu_P.mu_curr
                    chain_i.mu_A.mu_curr, chain_j.mu_A.mu_curr = chain_j.mu_A.mu_curr, chain_i.mu_A.mu_curr
                    chain_i.tau_R.tau_curr, chain_j.tau_R.tau_curr = chain_j.tau_R.tau_curr, chain_i.tau_R.tau_curr
                    chain_i.tau_G.tau_curr, chain_j.tau_G.tau_curr = chain_j.tau_G.tau_curr, chain_i.tau_G.tau_curr
                    chain_i.tau_P.tau_curr, chain_j.tau_P.tau_curr = chain_j.tau_P.tau_curr, chain_i.tau_P.tau_curr
                    chain_i.tau_A.tau_curr, chain_j.tau_A.tau_curr = chain_j.tau_A.tau_curr, chain_i.tau_A.tau_curr
                    chain_i.C.C_curr, chain_j.C.C_curr = chain_j.C.C_curr, chain_i.C.C_curr 
                    chain_i.U.U_curr, chain_j.U.U_curr = chain_j.U.U_curr, chain_i.U.U_curr 
                    chain_i.phi.phi_curr, chain_j.phi.phi_curr = chain_j.phi.phi_curr, chain_i.phi.phi_curr 
                    chain_i.V.V_curr, chain_j.V.V_curr = chain_j.V.V_curr, chain_i.V.V_curr 
                    chain_i.alpha.alpha_curr, chain_j.alpha.alpha_curr = chain_j.alpha.alpha_curr, chain_i.alpha.alpha_curr 
                    accept_swap = True
                self.acceptance_swap.append(accept_swap)

            #save chains
            chain_beta[i,:] = self.chain1.beta.beta_curr
            chain_eta[i] = self.chain1.eta.eta_curr
            chain_xi[i] = self.chain1.xi.xi_curr
            chain_alpha[i] = self.chain1.alpha.alpha_curr
            chain_mu_R[i,:] = self.chain1.mu_R.mu_curr
            chain_tau_R[i,:] = self.chain1.tau_R.tau_curr
            chain_mu_G[i,:] = self.chain1.mu_G.mu_curr
            chain_tau_G[i,:] = self.chain1.tau_G.tau_curr
            chain_mu_P[i,:] = self.chain1.mu_P.mu_curr
            chain_tau_P[i,:] = self.chain1.tau_P.tau_curr
            chain_mu_A[i,:] = self.chain1.mu_A.mu_curr
            chain_tau_A[i,:] = self.chain1.tau_A.tau_curr 
            chain_C[i,:] = self.chain1.C.C_curr
            
            if i > 0 and i% 500 == 0: #Intermediate save every 500 iterations
                print(i)
                # print('beta')
                # print(np.mean(chain_beta[i-500:i,], axis = 0))
                # print('eta')
                # print(np.mean(self.acceptance_eta_chain1[i-500:i]))
                # print(np.mean(chain_eta[i-500:i]))
                # print(np.mean(chain_xi[i-500:i]))
                # print(np.mean(chain_alpha[i-500:i]))
                # print('acceptance rate swap')
                # print(np.mean(self.acceptance_swap))

                #save_beta
                f_MCMC.save_chain(chain_beta[:i,] , self.path_results, self.chain, 'beta')
                #save_eta
                f_MCMC.save_chain(chain_eta[:i] , self.path_results, self.chain, 'eta')
                #save xi
                f_MCMC.save_chain(chain_xi [:i], self.path_results, self.chain, 'xi')
                #save_alpha
                f_MCMC.save_chain(chain_alpha [:i], self.path_results, self.chain, 'alpha')
                #save_mu
                f_MCMC.save_chain(chain_mu_R[:i,] , self.path_results, self.chain, 'mu_R')
                f_MCMC.save_chain(chain_mu_G[:i,] , self.path_results, self.chain, 'mu_G')
                f_MCMC.save_chain(chain_mu_P[:i,] , self.path_results, self.chain, 'mu_P')
                f_MCMC.save_chain(chain_mu_A[:i,] , self.path_results, self.chain, 'mu_A')
                #save_tau
                f_MCMC.save_chain(chain_tau_R[:i,] , self.path_results, self.chain, 'tau_R')
                f_MCMC.save_chain(chain_tau_G[:i,] , self.path_results, self.chain, 'tau_G')
                f_MCMC.save_chain(chain_tau_P[:i,] , self.path_results, self.chain, 'tau_P')
                f_MCMC.save_chain(chain_tau_A[:i,] , self.path_results, self.chain, 'tau_A')
                
                #save C
                f_MCMC.save_chain(chain_C[:i,] , self.path_results, self.chain, 'C')
        
        #save_beta
        f_MCMC.save_chain(chain_beta , self.path_results, self.chain, 'beta')
        
        #save_eta
        f_MCMC.save_chain(chain_eta , self.path_results, self.chain, 'eta')
        
        #save xi
        f_MCMC.save_chain(chain_xi , self.path_results, self.chain, 'xi')
        
        #save_alpha
        f_MCMC.save_chain(chain_alpha , self.path_results, self.chain, 'alpha')
        
        #save_mu
        f_MCMC.save_chain(chain_mu_R , self.path_results, self.chain, 'mu_R')
        f_MCMC.save_chain(chain_mu_G , self.path_results, self.chain, 'mu_G')
        f_MCMC.save_chain(chain_mu_P , self.path_results, self.chain, 'mu_P')
        f_MCMC.save_chain(chain_mu_A , self.path_results, self.chain, 'mu_A')
        
        #save_tau
        f_MCMC.save_chain(chain_tau_R , self.path_results, self.chain, 'tau_R')
        f_MCMC.save_chain(chain_tau_G , self.path_results, self.chain, 'tau_G')
        f_MCMC.save_chain(chain_tau_P , self.path_results, self.chain, 'tau_P')
        f_MCMC.save_chain(chain_tau_A , self.path_results, self.chain, 'tau_A')
        
        #save C
        f_MCMC.save_chain(chain_C , self.path_results, self.chain, 'C')
        
        # print(np.mean(chain_beta, axis = 0))
        # print(np.mean(chain_eta))
        # print(np.mean(chain_xi))
        # print(np.mean(chain_alpha))
        # print('acceptance rate swap')
        # print(np.mean(self.acceptance_swap))
                
    def run_adaptive_MCMC(self, nb_phases, niter, burnin):
        """
        Run the MCMC with an adaptive phase at the beginning.
        """
        i = 0
        while (i < nb_phases) :
            print("Adaptive phase {} :".format(str(i)))
            self.run_MCMC(100) #Run 100 iteration of MCMC

            #Compute the acceptance rates and update the proposal standard deviations
            acceptance_rates_eta_c1 = np.mean(self.acceptance_eta_chain1)
            acceptance_rates_eta_c2 = np.mean(self.acceptance_eta_chain2)
            acceptance_rates_eta_c3 = np.mean(self.acceptance_eta_chain3)
            acceptance_rates_eta_c4 = np.mean(self.acceptance_eta_chain4)
            acceptance_rates_eta_c5 = np.mean(self.acceptance_eta_chain5)
            self.chain1.eta.proposal_sds = f_MCMC.adaptive_proposals(acceptance_rates_eta_c1, self.chain1.eta.proposal_sds, 0.44)
            self.chain2.eta.proposal_sds = f_MCMC.adaptive_proposals(acceptance_rates_eta_c2, self.chain2.eta.proposal_sds, 0.44)
            self.chain3.eta.proposal_sds = f_MCMC.adaptive_proposals(acceptance_rates_eta_c3, self.chain3.eta.proposal_sds, 0.44)
            self.chain4.eta.proposal_sds = f_MCMC.adaptive_proposals(acceptance_rates_eta_c4, self.chain4.eta.proposal_sds, 0.44)
            self.chain5.eta.proposal_sds = f_MCMC.adaptive_proposals(acceptance_rates_eta_c5, self.chain5.eta.proposal_sds, 0.44)
            acceptance_rates_beta_c1 = np.nanmean(self.acceptance_beta_chain1, axis = 0)
            acceptance_rates_beta_c1[np.isnan(acceptance_rates_beta_c1)] = 0.44
            acceptance_rates_beta_c1[np.nansum(self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c1[np.nansum(1-self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c2 = np.nanmean(self.acceptance_beta_chain1, axis = 0)
            acceptance_rates_beta_c2[np.isnan(acceptance_rates_beta_c1)] = 0.44
            acceptance_rates_beta_c2[np.nansum(self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c2[np.nansum(1-self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c3 = np.nanmean(self.acceptance_beta_chain1, axis = 0)
            acceptance_rates_beta_c3[np.isnan(acceptance_rates_beta_c1)] = 0.44
            acceptance_rates_beta_c3[np.nansum(self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c3[np.nansum(1-self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c4 = np.nanmean(self.acceptance_beta_chain1, axis = 0)
            acceptance_rates_beta_c4[np.isnan(acceptance_rates_beta_c1)] = 0.44
            acceptance_rates_beta_c4[np.nansum(self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c4[np.nansum(1-self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c5 = np.nanmean(self.acceptance_beta_chain1, axis = 0)
            acceptance_rates_beta_c5[np.isnan(acceptance_rates_beta_c1)] = 0.44
            acceptance_rates_beta_c5[np.nansum(self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            acceptance_rates_beta_c5[np.nansum(1-self.chain1.acceptance_beta, axis = 0) == 1] = 0.44
            for j in range(len(self.chain1.beta.beta_curr)) :
                   self.chain1.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(acceptance_rates_beta_c1[j], self.chain1.beta.proposal_sds[j], 0.44)  
                   self.chain2.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(acceptance_rates_beta_c2[j], self.chain2.beta.proposal_sds[j], 0.44)
                   self.chain3.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(acceptance_rates_beta_c3[j], self.chain3.beta.proposal_sds[j], 0.44)
                   self.chain4.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(acceptance_rates_beta_c4[j], self.chain4.beta.proposal_sds[j], 0.44)
                   self.chain5.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(acceptance_rates_beta_c5[j], self.chain5.beta.proposal_sds[j], 0.44)
            
            print(self.chain1.CA_star)
            print('beta')
            print(self.chain1.beta.beta_curr)
            print(acceptance_rates_beta_c1)
            print('eta')
            print(acceptance_rates_eta_c1)
            print(self.chain1.xi.xi_curr)
            i+= 1
        
        self.run_MCMC(niter)