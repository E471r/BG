import numpy as np

import tensorflow as tf

import pickle

##

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

def KL_discrete_(p, q):
    ''' p, q : histograms : arrays of same shape, both sum to 1.
    '''
    return tf.reduce_sum( p * tf.math.log((p+1e-6) / (q+1e-6)) )

def S_discrete_(p):
    ''' p : histogram : array sums to 1.
    '''  
    return - tf.reduce_sum( p * tf.math.log(p+1e-6) )
