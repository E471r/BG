import numpy as np

import tensorflow as tf

import pickle

## Saving/Loading:

def save_npy_(x,name):
    with open(name, "wb") as f: np.save(f, x) ; print('saved',name)
    
def load_npy_(name):
    with open(name, "rb") as f: x = np.load(f) ; return x

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

##

def MCMC_r_(potential, # object
            x_init : np.ndarray, # (1,dim)
            n_steps : int = 1000000,
            kT : float = 1.0,
            SD : float = 0.08, # standard deviation of step sizes.
            ):

    dim = x_init.shape[1]
    
    x = x_init
    u = potential.evaluate_potential(x)

    acc = 0
    
    xTraj=np.zeros([n_steps,dim])

    for step in range(n_steps):
        
        xTrial = x + SD * np.random.randn(1,dim)
        uTrial = potential.evaluate_potential(xTrial)

        deltaU = uTrial - u

        if deltaU < 0:

            x = xTrial
            u = uTrial
            acc = acc + 1

        elif np.random.rand() < np.exp(-deltaU/kT):
            
            x = xTrial
            u = uTrial
            acc = acc + 1
            
        else:
            pass
        
        xTraj[step] = x

    print('acceptance ratio:', acc/n_steps) # to tune maxDr
    
    return xTraj

##

def KL_discrete_(p, q):
    ''' p, q : histograms : arrays of same shape, both sum to 1.
    '''
    return tf.reduce_sum( p * tf.math.log((p+1e-6) / (q+1e-6)) )

def S_discrete_(p):
    ''' p : histogram : array sums to 1.
    '''  
    return - tf.reduce_sum( p * tf.math.log(p+1e-6) )

##

## Rigid body aligment:

def least_squares_rotation_matrix_(x, z, ws):
    # x,z ~ (n,3)
    u,s,v = np.linalg.svd( x.T.dot(ws).dot(z) )
    R = u.dot(v)
    return R

def get_tripod_(x):
    # x ~ (n,3)
    v01 = x[0]-x[1]
    v02 = x[0]-x[2]
    u012 = np.cross(v01,v02)
    return np.stack([v01,v02,u012],axis=0)

def rigid_allign_(X : np.ndarray,
                  z : np.ndarray,
                  subset_inds : list = None,
                  centre_on_subset0 : bool = False,
                  d3_subset_inds_planar : bool = False,
                  return_tensor : bool = True,
                  verbose : bool = False):

    ''' Rigid body alignment (just one iteration) of cartesian coordinates: 
    
    Inputs:
        X : (m,n,3) shaped array to be aligned to fixed reference (z)
            m = number of molecules, or frames containing one molecule.
            n = number of atoms in the molecule.
            3 = three cartesian coordinates.
        z : (n,3) shaped array : structure alignment template.

    Parameters:
        subset_inds : list (default is None : all atoms used) of indices 
                      for atoms to use in alignment (e.g., [1,2,3,..]).
        centre_on_subset0 : if True (default is False) and subset_inds is not None, 
                            the first atom in subset_inds list will be used instead 
                            of centre of mass, as the centre for the rotation fit.
        d3_subset_inds_planar : if all subset_inds are atoms which may sometimes appear on 
                                a 2D plane (e.g., all are part of a ring) set this to True.
        verbose : bool.

    Output:
        Y : (m,n,3) shaped array of aligned conformers, where
             all conformers X[i] least squares superposed to fit on z.
    '''

    X = np.array(X) ; N,n,d = X.shape # (N,n,d) 
    z = np.array(z)                   # (n,d)

    if subset_inds is None: subset_inds = np.arange(n).tolist()
    else: subset_inds = np.array(subset_inds).flatten().tolist()
 
    X_ = X[:,subset_inds,:] ; z_ = z[subset_inds,:]
    
    if centre_on_subset0: mu_z_ =  np.array(z_[0])[np.newaxis,:]
    else: mu_z_ = z_.mean(0, keepdims=True)
    z_ -= mu_z_
    
    ##
    ws_ = np.eye(z_.shape[0]) 
    ##

    Y = np.zeros([N,n,d])
    for i in range(N):
        x_ = np.array(X_[i]) # (n_s, 3)
        
        if centre_on_subset0:  mu_x_ = np.array(x_[0])[np.newaxis,:]
        else: mu_x_ = x_.mean(0, keepdims=True)

        x_ -= mu_x_

        if d3_subset_inds_planar:
            R = least_squares_rotation_matrix_(get_tripod_(x_[[0,1,2]]),
                                               get_tripod_(z_[[0,1,2]]),
                                               np.eye(3))
        else:
            R = least_squares_rotation_matrix_(x_, z_, ws_)

        err_before = np.linalg.norm(x_-z_)
        y_ = x_.dot(R)
        err_after = np.linalg.norm(y_-z_)

        if err_after < err_before:
            if verbose:
                stars = (10*(err_before-err_after)/err_before).astype(int)
                print('% err drop:  [','*'*stars,'.'*(10-stars),']')
            else: pass
            Xi_ali = (X[i] - mu_x_).dot(R) + mu_z_
            Y[i,:] = Xi_ali

        else:
            if verbose: print('no change at frame:', i) # rotation skipped.
            else: pass
            Y[i,:] = X[i] - mu_x_ + mu_z_

    if return_tensor: return tf.constant(Y, dtype=tf.float32)
    else: return Y
