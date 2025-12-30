# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:33:08 2023

@author: FENDLER-JUL
"""
import numpy as np
import pandas as pd

def load_data(name):
    """ The function load_data takes the name of a csv file and loads it as a
        pandas data frame in Python
        """
    data = pd.read_csv(name, sep=',', quotechar='"', skiprows=0)
    return(data)

def predictor(disease_model, beta, C):
    """
    Return a vector of the value of the risk knowing the cluster membership for each individual
    """
    if disease_model == 'EHR':
        return (1.0 + beta[C])
    elif disease_model == 'Cox':
        return (np.exp(beta[C]))
    
    
def integrated_baseline_hazard(eta, t0, t1):
   """
       Return the integrated (between t0 and t1) baseline hazard
   """
   return((t1)**eta-(t0)**eta)

    
def S_beta_eta_C(disease_model, beta, eta, C, y, trunc_y, i_expo):
    S = 1/eta * (np.dot(predictor(disease_model, beta, C[i_expo]), 
                        integrated_baseline_hazard(eta, trunc_y[i_expo], y[i_expo])) + 
                        np.sum(integrated_baseline_hazard(eta, trunc_y[i_expo == False], y[i_expo== False])))
    return S
