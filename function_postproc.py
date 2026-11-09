# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:33:37 2023

@author: FENDLER-JUL
"""
import numpy as np

def compute_similarity_matrix(C):
    """
    This function compute the similarity matrix
    """
    dim = len(C)
    mat = np.zeros((dim,dim))
    for indiv1 in range(dim):
        mat[indiv1, :] = np.array(C[indiv1] == C, dtype = int)
    return(mat)
    