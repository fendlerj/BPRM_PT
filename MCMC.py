# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:40:45 2023

@author: FENDLER-JUL

MCMC with parallel temering
"""

import chain_MCMC
import numpy as np
import functions_mcmc as f_MCMC
from random import randrange

# Parameters exchanged between two chains when a swap is accepted
# (object attribute, value attribute)
SWAP_ATTRIBUTES = [('beta', 'beta_curr'), ('eta', 'eta_curr'), ('xi', 'xi_curr'),
                   ('mu_R', 'mu_curr'), ('mu_G', 'mu_curr'), ('mu_P', 'mu_curr'), ('mu_A', 'mu_curr'),
                   ('tau_R', 'tau_curr'), ('tau_G', 'tau_curr'), ('tau_P', 'tau_curr'), ('tau_A', 'tau_curr'),
                   ('C', 'C_curr'), ('U', 'U_curr'), ('phi', 'phi_curr'), ('V', 'V_curr'),
                   ('alpha', 'alpha_curr')]

# Parameters of the cold chain saved to disk: name -> (object attribute, value attribute)
SAVED_PARAMETERS = {'beta': ('beta', 'beta_curr'), 'eta': ('eta', 'eta_curr'),
                    'xi': ('xi', 'xi_curr'), 'alpha': ('alpha', 'alpha_curr'),
                    'mu_R': ('mu_R', 'mu_curr'), 'mu_G': ('mu_G', 'mu_curr'),
                    'mu_P': ('mu_P', 'mu_curr'), 'mu_A': ('mu_A', 'mu_curr'),
                    'tau_R': ('tau_R', 'tau_curr'), 'tau_G': ('tau_G', 'tau_curr'),
                    'tau_P': ('tau_P', 'tau_curr'), 'tau_A': ('tau_A', 'tau_curr'),
                    'C': ('C', 'C_curr')}


class MCMC_PT():

    def __init__(self, chain, path_results, disease_model,
                 alpha_start, C_start, V_start, phi_start, U_start, mu_R_start,
                 tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                 mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R,
                 Z_G, Z_P, Z_A, y, trunc_y, event,
                 i_expo, seed=0, vect_T=[1.0, 1.10, 1.21, 1.33, 1.46, 1.61, 1.77], swap_every=10):

        self.vect_T = list(vect_T)
        self.nb_T = len(self.vect_T)
        self.path_results = path_results
        self.chain = chain
        self.swap_every = swap_every

        # One chain per temperature
        self.chains = [chain_MCMC.chain_MCMC(disease_model,
                                             alpha_start, C_start, V_start, phi_start, U_start, mu_R_start,
                                             tau_R_start, mu_G_start, tau_G_start, mu_P_start, tau_P_start,
                                             mu_A_start, tau_A_start, beta_start, eta_start, xi_start, Z_R,
                                             Z_G, Z_P, Z_A, y, trunc_y, event,
                                             i_expo, seed + k, T)
                       for k, T in enumerate(self.vect_T)]

        self.dict_chain = {k: ch for k, ch in enumerate(self.chains)}
        self.chain1 = self.chains[0]

        # Swap attempts / acceptances 
        self.swap_attempts = np.zeros(max(self.nb_T - 1, 0))
        self.swap_accepts = np.zeros(max(self.nb_T - 1, 0))

    def _log_likelihood(self, ch):
        """
        Untempered log-likelihood of the current state of chain ch
        """
        return f_MCMC.log_likelihood(ch.disease_model, ch.beta.beta_curr,
                                     ch.eta.eta_curr, ch.xi.xi_curr,
                                     ch.mu_R.mu_curr, ch.tau_R.tau_curr,
                                     ch.mu_G.mu_curr, ch.tau_G.tau_curr,
                                     ch.mu_P.mu_curr, ch.tau_P.tau_curr,
                                     ch.mu_A.mu_curr, ch.tau_A.tau_curr,
                                     ch.C.C_curr, ch.y, ch.trunc_y, ch.event, ch.nevent, ch.i_expo,
                                     ch.Z_R, ch.Z_G, ch.Z_P, ch.Z_A,
                                     ch.xi.prior_shape, ch.xi.prior_intensity)

    def _swap_states(self, chain_i, chain_j):
        """
        Exchange the states of two chains (the temperatures stay with the chains)
        """
        for obj_name, val_name in SWAP_ATTRIBUTES:
            obj_i, obj_j = getattr(chain_i, obj_name), getattr(chain_j, obj_name)
            val_i, val_j = getattr(obj_i, val_name), getattr(obj_j, val_name)
            setattr(obj_i, val_name, val_j)
            setattr(obj_j, val_name, val_i)

    def attempt_swap(self):
        """
        Propose to swap the states of a random pair of neighbouring temperatures
        """
        if self.nb_T < 2:
            return False
        k = randrange(self.nb_T - 1)
        chain_i, chain_j = self.chains[k], self.chains[k + 1]
        log_prob_i = self._log_likelihood(chain_i)
        log_prob_j = self._log_likelihood(chain_j)
        accept = f_MCMC.accept_swap_PT(log_prob_i, log_prob_j, chain_i.T, chain_j.T)
        self.swap_attempts[k] += 1
        if accept:
            self._swap_states(chain_i, chain_j)
            self.swap_accepts[k] += 1
        return accept

    def swap_acceptance_rates(self):
        """
        Acceptance rate of the swaps for each neighbouring pair (nan if never attempted)
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.swap_accepts / self.swap_attempts

    def _save(self, samples, last=None):
        for name, arr in samples.items():
            f_MCMC.save_chain(arr if last is None else arr[:last], self.path_results, self.chain, name)

    def run_MCMC(self, niter):
        """
        Run niter iterations of the MCMC algorithm for all the chains
        """
        n_beta = len(self.chain1.beta.beta_curr)

        # Acceptance vectors
        self.acceptance_eta = np.zeros((self.nb_T, niter))
        self.acceptance_beta = np.full((self.nb_T, niter, n_beta), np.nan)
        self.acceptance_swap = []

        # Samples of the cold chain
        samples = {}
        for name, (obj_name, val_name) in SAVED_PARAMETERS.items():
            value = np.asarray(getattr(getattr(self.chain1, obj_name), val_name))
            samples[name] = np.zeros((niter,) + value.shape)

        for i in range(niter):
            for k, ch in enumerate(self.chains):
                ch.iter_1(i)
                self.acceptance_eta[k, i] = ch.acceptance_eta
                self.acceptance_beta[k, i, :] = ch.acceptance_beta

            # Swap two neighbouring chains every swap_every iterations
            if self.nb_T > 1 and i % self.swap_every == 0:
                self.acceptance_swap.append(self.attempt_swap())

            # Store the cold chain
            for name, (obj_name, val_name) in SAVED_PARAMETERS.items():
                samples[name][i] = getattr(getattr(self.chain1, obj_name), val_name)

            if i > 0 and i % 500 == 0:  # Intermediate save every 500 iterations
                print(i)
                self._save(samples, last=i)

        self._save(samples)
        if self.nb_T > 1:
            print('swap acceptance rate per pair of temperatures')
            print(dict(zip([(self.vect_T[k], self.vect_T[k + 1]) for k in range(self.nb_T - 1)],
                     np.round(self.swap_acceptance_rates(), 3).tolist())))

    def run_adaptive_MCMC(self, nb_phases, niter, burnin):
        """
        Run the MCMC with an adaptive phase at the beginning.
        """
        for phase in range(nb_phases):
            print("Adaptive phase {} :".format(str(phase)))
            self.run_MCMC(100)  # Run 100 iterations of MCMC

            # Compute the acceptance rates and update the proposal standard deviations
            for k, ch in enumerate(self.chains):
                rate_eta = np.mean(self.acceptance_eta[k])
                ch.eta.proposal_sds = f_MCMC.adaptive_proposals(rate_eta, ch.eta.proposal_sds, 0.44)

                with np.errstate(invalid='ignore'):
                    rates_beta = np.nanmean(self.acceptance_beta[k], axis=0)
                rates_beta[np.isnan(rates_beta)] = 0.44
                rates_beta[np.nansum(ch.acceptance_beta, axis=0) == 1] = 0.44
                rates_beta[np.nansum(1 - ch.acceptance_beta, axis=0) == 1] = 0.44
                for j in range(len(ch.beta.beta_curr)):
                    ch.beta.proposal_sds[j] = f_MCMC.adaptive_proposals(rates_beta[j], ch.beta.proposal_sds[j], 0.44)

                if k == 0:
                    rates_beta_c1, rate_eta_c1 = rates_beta, rate_eta

            print("n clusters")
            print(len(np.unique(self.chain1.C.C_curr[self.chain1.i_expo])))
            print("active clusters")
            print(np.unique(self.chain1.C.C_curr[self.chain1.i_expo]))
            print('beta')
            print(self.chain1.beta.beta_curr)
            print(rates_beta_c1)
            print('eta')
            print(rate_eta_c1)
            print(self.chain1.xi.xi_curr)
            if self.nb_T > 1:
                print('swap acceptance rate per pair of temperatures')
                print(dict(zip([(self.vect_T[k], self.vect_T[k + 1]) for k in range(self.nb_T - 1)],
                               np.round(self.swap_acceptance_rates(), 3).tolist())))

        self.run_MCMC(niter)

