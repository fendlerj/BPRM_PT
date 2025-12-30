# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:34:19 2023

@author: FENDLER-JUL
"""

import numpy as np
import os
os.chdir('C:/Users/Julie/Documents/Publication/article_BPRM/BPRM_PT_gamma_data_simu/')#path where the files basics, parameters_AM, parameters_EM, parameters_DM and MCMC are.
import function_postproc as f_postproc

#Path to save the results
path_results = os.path.join('C:/Users/Julie/Documents/Publication/article_BPRM//test_res_code_python')
chain = 'chain1'
chain_C = np.load(path_results + "/v1"  + chain + 'C'+ ".npy")

#compute mean similarity matrix
n_burnin = 10000
nb_iter = chain_C.shape[0]
S_mean = np.zeros((len(chain_C[0]),len(chain_C[0])))
for iter_C in range(n_burnin +1 ,nb_iter) :
    if iter_C % 500 == 0:
        print(iter_C)
    S_mean += f_postproc.compute_similarity_matrix(chain_C[iter_C])/(nb_iter - n_burnin)

#Save the mean similarity matrix    
np.savetxt(path_results +"/S_mean_" + chain +".csv", S_mean, delimiter=",")
