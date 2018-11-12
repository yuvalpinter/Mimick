from __future__ import division
import numpy as np
from numpy import random

def props(total, golds, weighting):
    return weighting(golds)
    #p = golds/total
    #return weighting(p)

# PROPORTION SCHEMES

def sum_w(p):
    # macro
    return 1.0
    
def linear_w(p):
    # micro
    return p
    
def sq_w(p):
    return p * p
    
def rand_w(p):
    return random.random()
    
def sqrt_w(p):
    return np.sqrt(p)
    
def log_w(p):
    return np.log2(p)
    #return np.log2(np.e)*np.log1p(p)
    
def inv_w(p):
    return 1.-(2./(1.+p))
    
#inverse_w(x):
#   1.5-(1./(x+1.))
#counter_w(x):
#   2.0 - x
#cnlog_w(x):
#   1.0 - np.log2(x)
