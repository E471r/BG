import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Pipeline import DPA
from kde import *

##

# Semi-unsupervised way to find parameters for MARGINAL_GMM_PRIOR from data.

# TODO: further test/improve user experience.

##
"""
## plotting what is happening is important, both after clustering and after fitting:

# 15*5=75 dimensions here fit well into plt.subplots(15,5, figsize=(12,16))

n_columns = 15
n_rows = 5

fig, ax = plt.subplots(n_columns, n_rows, figsize=(12,16))
fig.tight_layout()

a = 0 ; b = 0
for i in range(obj_fmp.dim):

    # available after initialising find_the_marginal_parameters:
    ax[a,b].plot(obj_fmp.grid, obj_fmp.histograms[:,i])
    ax[a,b].scatter(obj_fmp.x[:,i], obj_fmp.marginal_probabilities[:,i],s=1, color='green')

    # available after find_centroids_():
    ax[a,b].scatter(obj_fmp.centroids[i], obj_fmp.probabilities_of_centroids[i][:len(obj_fmp.centroids[i])],s=40, color='red')
    
    # available after train_():
    ax[a,b].plot(obj_fmp.grid, obj_fmp.histograms_fitted[:,i], color='m')
    ax[a,b].set_title(i)
    
    a+=1
    if a == n_columns:
        a = 0
        b += 1
    else : pass

"""
##

pi = np.pi

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

def KL_discrete_(p, q):
    ''' p, q : histograms : arrays of same shape, both sum to 1.
    '''
    return tf.reduce_sum( p * tf.math.log((p+1e-6) / (q+1e-6)) )

def clamp_range_(x, range = [-1.0, 1.0]):
    Min, Max = range
    return tf.clip_by_value(x, Min, Max)

class find_the_marginal_parameters:
    def __init__(self,
                 X, # all data (N,dim) after ic_map.forward(r) (i.e., all elements in model_range [-1,1])
                 periodic_mask : list, # 1d list
                 n_subsample : int = 3001,
                 Z : float = 1.5, # number of modes found depends on this, but clip_mode_counts_ is safer to run than changing Z.
                 ):

        x = np.array(X)
        N, dim = x.shape
        inds_subsample = np.random.choice(N, n_subsample, replace=False)
        periodic_mask = np.array(periodic_mask).reshape(dim,)
        x = x[inds_subsample]

        histograms = []
        marginal_probabilities = []
        
        self.n_bins = 40
        self.c = 2.0/self.n_bins

        for i in range(dim):
            obj_kde = KDE(x[:,i], x_range=[-1.0,1.0], n_bins=self.n_bins, param=300, periodic=[periodic_mask[i]])
            ax = obj_kde.axes[0] # same for all
            p_xi = obj_kde.normalised_px ; marginal_probabilities.append(p_xi)
            # ^ only for init_heights.
            histograms.append(obj_kde.normalised_histogram)
            # histograms.append(np.histogram(X[:,i], bins=self.n_bins, range=[-1.0,1.0], density=True)[0])
            # ^ target shapes

        self.histograms = np.stack(histograms, axis=1) # !
        self.marginal_probabilities = np.stack(marginal_probabilities, axis=1)

        self.grid = ax
        self.dim = dim
        self.periodic_mask = periodic_mask
        self.Z = Z ; self.reset_Z_(Z)
        self.x = x

        self.inds_subsample = inds_subsample # not used later
        
        # self.dropper_ = lambda rate : 1-(np.random.rand(self.n_bins,1)>1-rate).astype(int)
        self.reset_results_()

    def reset_results_(self, reset_indexed=None):
        if reset_indexed is None:
            self.fitted_locs = dict(zip(np.arange(self.dim), [None]*self.dim)) # list
            self.fitted_scales = dict(zip(np.arange(self.dim), [None]*self.dim)) # list
            self.fitted_heights = dict(zip(np.arange(self.dim), [None]*self.dim)) # list
        else:
            for i in reset_indexed:
                self.fitted_locs[i] = None
                self.fitted_scales[i] = None
                self.fitted_heights[i] = None

    def reset_Z_(self, new_Z):
        if new_Z != self.Z:
            print('reseting Z from',self.Z,'to',self.new_Z)
            self.Z = new_Z
        else: pass
        self.EST = DPA.DensityPeakAdvanced(Z=self.Z)

    def find_centroids_(self):
        centroids = []
        inds_centoids = []

        for i in range(self.dim):
            xi = self.x[:,i:i+1]

            print('running independent DPA on',i+1,'/',self.dim,'dimension.')
            if self.periodic_mask[i] == 1:
                est = self.EST.fit( np.concatenate([np.cos(xi*pi),np.sin(xi*pi)],axis=1) )
            else:
                est = self.EST.fit(xi)

            centroids.append(xi[est.centers_].flatten())
            inds_centoids.append(est.centers_)

        self.centroids = centroids
        self.inds_centoids = inds_centoids

        self.probabilities_of_centroids = [self.marginal_probabilities[inds_centoids[i],i] for i in range(self.dim)]
        self.init_heights = [x/x.sum() for x in self.probabilities_of_centroids]
        # ^ both raggen
        self.mode_counts = np.array([len(x) for x in self.init_heights])

        n_3modes = np.where(self.mode_counts==3,1,0).sum()
        n_over_3modes = np.where(self.mode_counts>3,1,0).sum()

        print('number of dimensions where there were 3 modes identified:', n_3modes)
        print('number of dimensions where there were >3 modes identified:', n_over_3modes)
        if n_over_3modes > 0:
            print('can remove redundant modes by running clip_mode_counts_(max_mode_count)')
        else: pass

    def clip_mode_counts_(self, max_mode_count : int):
        cs, hs, ps = [], [], []
        for i in range(self.dim):
            if self.mode_counts[i] > max_mode_count:
                c = self.centroids[i][:max_mode_count]
                h = self.init_heights[i][:max_mode_count]
                p = self.probabilities_of_centroids[i][:max_mode_count]
                self.mode_counts[i] = max_mode_count
            else:
                c = self.centroids[i]
                h = self.init_heights[i]
                p = self.probabilities_of_centroids[i]
            cs.append(c)
            hs.append(h)
            ps.append(p)
            
        self.centroids = cs
        self.init_heights = hs
        self.probabilities_of_centroids = ps

    def train_(self,
               n_itter : int = 200,
               learning_rates : list or float = [0.001]*3, 
               init_width : float = 0.1,
               fit_indexed : list = None):
        
        if type(learning_rates) in [float,int]:
            learning_rates = [learning_rates]*3
        else: pass
        
        if fit_indexed is None: dims = np.arange(self.dim)
        else: dims = np.array(fit_indexed)

        self.errs_history = [] ; self.histograms_fitted = []
        for j in dims:
            n_modes_j = self.mode_counts[j]
            grid_j = tf.constant(np.stack([self.grid]*n_modes_j, axis=1), tf.float32)
            histogram_j_target = tf.constant(self.histograms[:,j:j+1], dtype=tf.float32)

            # loc ~ (Kj)
            if self.fitted_locs[j] is not None: l = tf.constant(self.fitted_locs[j], dtype=tf.float32)
            else:                            l = tf.constant(self.centroids[j], dtype=tf.float32)

            # scale ~ (Kj)
            if self.fitted_scales[j] is not None: s = tf.constant(self.fitted_scales[j], dtype=tf.float32)
            else:                              s = tf.constant(np.ones(n_modes_j,)*init_width, dtype=tf.float32)

            # height ~ (1,Kj)
            if self.fitted_heights[j] is not None: h = tf.constant(self.fitted_heights[j], dtype=tf.float32)
            else: h = tf.constant(self.init_heights[j][np.newaxis,:], dtype=tf.float32)

            if self.periodic_mask[j] == 1:
                f_j = lambda loc, scale, grid : tfp.distributions.VonMises(loc=loc*pi, concentration=1.0/scale**2).prob(grid*pi)*pi
                type_j = 'periodic'
            else:
                f_j = lambda loc, scale, grid : tfp.distributions.TruncatedNormal(loc=loc, scale=scale/pi,low=-1.0,high=1.0).prob(grid)
                type_j = 'non-periodic'
                
            errs = []
            print('running independent SGD on',type_j,'dimension',j)#,'from',dims)
            for i in range(n_itter):
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(l) # 
                    tape.watch(s) ##
                    tape.watch(h) ##
                    
                    l = clamp_range_(l)
                    s = tf.abs(s)
                    h /= tf.reduce_sum(h)
                    
                    histogram_j_hat = sum_(f_j(l,s,grid_j) * h)
                    err = 0.0
                    err += KL_discrete_(histogram_j_hat*self.c, histogram_j_target*self.c) 
                    err += KL_discrete_(histogram_j_target*self.c, histogram_j_hat*self.c)
                    
                l -= learning_rates[0]*tape.gradient(err, l) ; l = clamp_range_(l)
                s -= learning_rates[1]*tape.gradient(err, s) ; s = tf.abs(s)
                h -= learning_rates[2]*tape.gradient(err, h) ; h /= tf.reduce_sum(h)

                errs.append(err.numpy())

            self.errs_history.append(errs)
            self.histograms_fitted.append(histogram_j_hat.numpy().flatten())

            self.fitted_locs[j] = l.numpy()
            self.fitted_scales[j] = s.numpy()
            self.fitted_heights[j] = h.numpy().flatten()
            
        self.errs_history = np.array(self.errs_history).T
        self.histograms_fitted = np.array(self.histograms_fitted).T

    def retrain_indexed_(self,
                         inds : list,
                         n_itter : int = 200,
                         learning_rates : list or float = [0.001]*3,
                         init_width : float = 0.1,
                        ):
        self.reset_results_(reset_indexed=inds)
        self.train_(n_itter = n_itter,
                    learning_rates = learning_rates, 
                    init_width = init_width,
                    fit_indexed= inds)
    
    @property
    def results(self):
        marginal_centres = [self.fitted_locs[i].tolist() for i in range(self.dim)]
        marginal_widths = [self.fitted_scales[i].tolist() for i in range(self.dim)]
        marginal_heights = [self.fitted_heights[i].tolist() for i in range(self.dim)]
        periodic_mask = self.periodic_mask.tolist()
        print('returning all inputs respectively ready for MARGINAL_GMM_PRIOR')
        return marginal_centres, marginal_widths, marginal_heights, periodic_mask 

