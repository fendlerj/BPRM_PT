# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:33:04 2023

@author: FENDLER-JUL
"""


#Class templates for the parameters with Gibbs sampler (Paraneter_Gibbs) and 
#Metropolis-Hasting sampler (Parameter_MH)

class Parameter_Gibbs():
    
    def __init__(self, parameter):
        self.parameter = parameter
    
    def update_parameter(self):
        raise NotImplementedError()
        

class Parameter_MH(Parameter_Gibbs):
    
    def __init__(self, parameter, adaptive_sd):
        self.parameter = parameter
        self.adaptive_sd = adaptive_sd
    
    def update_adaptive_sd(self, new_adaptive_sd):
        self.adaptive_sd = new_adaptive_sd
    
    def update_parameter(self):
        raise NotImplementedError()
        

