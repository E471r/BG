import numpy as np

import tensorflow as tf

from spline_layer import SPLINE_LAYER

from shift_layer import SHIFT_LAYER_STATIC, SHIFT_LAYER_PARAMETRIC

from prior import MARGINAL_GMM_PRIOR # requires tensorflow_probability to be installed.

from utils_nf import save_pickle_, load_pickle_

##

from mlp import MLP

from spline import rational_quadratic_

##

def get_permutation_inds_(dim : int, n_sets : int, inds = None):
    permutation_inds = []
    for i in range(n_sets):
        if inds is None : permute_indices = np.random.permutation(np.arange(dim))
        else : permute_indices = np.array(inds[i])
        unpermute_indices = np.array([int(np.where(permute_indices == i)[0]) for i in range(dim)])
        permutation_inds.append( [permute_indices, unpermute_indices] )
    return permutation_inds

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)
    
pi = np.pi
def cos_sin_1_(x, nk=1):
    return tf.concat( [tf.cos(x*pi),tf.sin(x*pi)], axis=1 )


def cos_sin_2_(x, nk=1):
    return tf.concat( [tf.cos(x*pi),tf.sin(x*pi),
                       tf.cos(x*pi*2.0),tf.sin(x*pi*2.0)], axis=1 )

def cos_sin_3_(x, nk=1):
    return tf.concat( [tf.cos(x*pi),tf.sin(x*pi),
                       tf.cos(x*pi*2.0),tf.sin(x*pi*2.0),
                       tf.cos(x*pi*3.0),tf.sin(x*pi*3.0)], axis=1 )

def cos_sin_(x, nk=1):
    return tf.concat( [tf.cos(x*pi),tf.sin(x*pi),
                       tf.cos(x*pi*2.0),tf.sin(x*pi*2.0),
                       tf.cos(x*pi*3.0),tf.sin(x*pi*3.0),
                       tf.cos(x*pi*4.0),tf.sin(x*pi*4.0)], axis=1 )


def selective_cos_sin_(x, mask_x):
    x_non_periodic = tf.gather(x, tf.where(mask_x != 1)[:,1], axis=1)
    x_periodic = cos_sin_( tf.gather(x, tf.where(mask_x == 1)[:,1], axis=1) )
    return tf.concat([x_non_periodic, x_periodic], axis=1)

##

class MODEL_4(tf.keras.models.Model):
    def __init__(self,
                 dim_flow: int = 22,
                 periodic_mask = None,
                 prior : list = None, # list of lists of lists, or None.
                 n_layers : int = 6,
                 n_bins : int = 20,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 use_joined_MLPs : bool = False,
                 permutation_inds = None,
                 verbose : bool = True,
                 ):
        super(MODEL_4, self).__init__()
        
        self.dim_flow = dim_flow
        if periodic_mask is None: self.periodic_mask = tf.constant(np.ones([self.dim_flow]).reshape(1,self.dim_flow,1), dtype=tf.int32)
        else:  self.periodic_mask = tf.constant(np.array(periodic_mask).reshape(1,self.dim_flow,1), dtype=tf.int32)

        if prior is None :
            self.log_prior_uniform = - self.dim_flow*np.log(2.0)
            self.evaluate_log_prior_ = lambda z : self.log_prior_uniform
            self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        else :
            marginal_centres, marginal_widths, marginal_heights = prior
            for x in [len(marginal_centres), len(marginal_widths), len(marginal_heights)]:
                if x != self.dim_flow: print('!! : prior : incorrect dimensions provided')
                else: pass
            self.obj_prior = MARGINAL_GMM_PRIOR(marginal_centres, marginal_widths, marginal_heights, self.periodic_mask)
            self.evaluate_log_prior_ = lambda z : self.obj_prior.evaluate_log_prior_(z) # (m,1)
            self.sample_prior_ = lambda batch_size : self.obj_prior.sample_prior_(batch_size) # (m,1)

        self.prior = prior
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.dims_hidden = dims_hidden
        self.hidden_activation = hidden_activation
        self.use_joined_MLPs = use_joined_MLPs
        if permutation_inds is None: self.permutation_inds = get_permutation_inds_(self.dim_flow, self.n_layers)
        else: self.permutation_inds = permutation_inds
        self.verbose = verbose
        #self.eps = eps
        self.eps_bin = 2.0/(self.n_bins*5.0)
        self.eps_slope = 1e-2

        self.dimA = self.dim_flow//2
        self.dimB = self.dim_flow - self.dimA
        
        self.layer_grid = np.arange(n_layers)
        self.pi = np.pi

        self.SA_book = { i: SPLINE_LAYER(dim = self.dimA,
                                         n_bins = n_bins,
                                         dims_hidden = dims_hidden,
                                         joined_MLP = use_joined_MLPs,
                                         hidden_activation = hidden_activation,
                                         eps_bin = self.eps_bin, eps_slope=  self.eps_slope) for i in self.layer_grid }

        self.SB_book = { i: SPLINE_LAYER(dim = self.dimB,
                                         n_bins = n_bins,
                                         dims_hidden = dims_hidden,
                                         joined_MLP = use_joined_MLPs,
                                         hidden_activation = hidden_activation,
                                         eps_bin = self.eps_bin, eps_slope=  self.eps_slope) for i in self.layer_grid }

        self.SHIFT_STATIC = SHIFT_LAYER_STATIC(flow_range = [-1.0, 1.0])
        self.layer_grid_shift = [0,1] * self.n_layers
        self.layer_grid_shift = np.array(self.layer_grid_shift[:self.n_layers])

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)

        if verbose: self.print_model_size()
        else: pass

        self.store_initial_parameters_()

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

    def save_model(self, path_and_name : str):
        init_args = [self.dim_flow, self.periodic_mask, self.prior, self.n_layers, self.n_bins, self.dims_hidden, self.hidden_activation, self.use_joined_MLPs,  self.permutation_inds, self.verbose]
        save_pickle_([init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_4, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    ##
    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(tf.Variable(self.trainable_variables[i]))

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(list_params[i])
        self.store_initial_parameters_()
    ##

    def split_(self, AB, i):
        AB = tf.gather(AB, self.permutation_inds[i][0], axis=1)
        return AB[:,:self.dimA], AB[:,self.dimA:]
    
    def join_(self, A, B, i):
        return tf.gather(tf.concat([A, B], axis=1), self.permutation_inds[i][1], axis=1)       

    def selective_shift_(self, x, mask_x, forward : bool):

        inds_periodic = tf.where(mask_x[0]==1)[:,0]
        inds_other = tf.where(mask_x[0]!=1)[:,0]
        x_periodic = tf.gather(x, inds_periodic, axis=1)
        x_other = tf.gather(x, inds_other, axis=1)

        if forward: x_periodic = self.SHIFT_STATIC.forward(x_periodic)
        else:       x_periodic = self.SHIFT_STATIC.inverse(x_periodic)

        return tf.gather(tf.concat([x_periodic, x_other], axis=1),
                         tf.stack([tf.where(tf.concat([inds_periodic, inds_other], axis=0) == i)[0,0] for i in range(mask_x.shape[1])]),
                         axis=1)

    def forward(self, AB, drop_rate : float = 0.0): # Fxz
        mask = self.periodic_mask
        ladJxz = 0.0
        for i in self.layer_grid:
            
            A, B = self.split_(AB, i)
            mask_A, mask_B = self.split_(mask, i)
            
            cond_B = selective_cos_sin_(B, mask_B[:,:,0])

            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=True)
            A, ladJ = self.SA_book[i].forward(A, periodic_mask=mask_A, cond = cond_B, drop_rate=drop_rate) ; ladJxz  += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=False)

            cond_A = selective_cos_sin_(A, mask_A[:,:,0])

            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=True)
            B, ladJ = self.SB_book[i].forward(B, periodic_mask=mask_B, cond = cond_A, drop_rate=drop_rate) ; ladJxz  += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=False)

            AB = self.join_(A, B, i)
            mask = self.join_(mask_A, mask_B , i)
            
        return AB, ladJxz

    def inverse(self, AB, drop_rate : float = 0.0): # Fzx
        mask = self.periodic_mask
        ladJzx = 0.0
        for i in np.flip(self.layer_grid):
            
            A, B = self.split_(AB, i)
            mask_A, mask_B = self.split_(mask, i)
            
            cond_A = selective_cos_sin_(A, mask_A[:,:,0])

            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=True)
            B, ladJ = self.SB_book[i].inverse(B, periodic_mask=mask_B, cond = cond_A, drop_rate=drop_rate) ; ladJzx += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=False)

            cond_B = selective_cos_sin_(B, mask_B[:,:,0])

            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=True)
            A, ladJ = self.SA_book[i].inverse(A, periodic_mask=mask_A, cond = cond_B, drop_rate=drop_rate) ; ladJzx += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=False)

            AB = self.join_(A, B, i)
            mask = self.join_(mask_A, mask_B , i)

        return AB, ladJzx
    
    def inverse_special(self, AB): # Fzx
        mask = self.periodic_mask
        
        list_AB = [] ; list_ladJzx = []
        for i in np.flip(self.layer_grid):

            ladJzx = 0.0
            
            A, B = self.split_(AB, i)
            mask_A, mask_B = self.split_(mask, i)
            
            cond_A = selective_cos_sin_(A, mask_A[:,:,0])

            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=True)
            B, ladJ = self.SB_book[i].inverse(B, periodic_mask=mask_B, cond = cond_A) ; ladJzx += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: B = self.selective_shift_(B, mask_B[:,:,0], forward=False)

            cond_B = selective_cos_sin_(B, mask_B[:,:,0])

            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=True)
            A, ladJ = self.SA_book[i].inverse(A, periodic_mask=mask_A, cond = cond_B) ; ladJzx += sum_(ladJ)
            if self.layer_grid_shift[i] == 1: A = self.selective_shift_(A, mask_A[:,:,0], forward=False)

            AB = self.join_(A, B, i)
            mask = self.join_(mask_A, mask_B , i)
            list_AB.append(AB) ; list_ladJzx.append(ladJzx)
            
        return list_AB, list_ladJzx

    def forward_np(self, AB):
        a, b = self.forward( tf.constant(AB, dtype=tf.float32) )
        return a.numpy(), b.numpy()
        
    def inverse_np(self, AB):
        a, b = self.inverse( tf.constant(AB, dtype=tf.float32) ) 
        return a.numpy(), b.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))
    
##

class MODEL_P(tf.keras.models.Model):
    def __init__(self,
                 dim_flow : int,
                 prior : list = None, # gmm prior does not work here.
                 n_layers : int = 6,
                 n_bins : int = 20,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 use_joined_MLPs : bool = False,
                 permutation_inds = None):
        super(MODEL_P, self).__init__()

        self.dim_flow = dim_flow

        ##
        self.periodic_mask = tf.constant(np.ones([self.dim_flow]).reshape(1,self.dim_flow,1), dtype=tf.int32)
        
        if prior is None :
            self.log_prior_uniform = - self.dim_flow*np.log(2.0)
            self.evaluate_log_prior_ = lambda z : self.log_prior_uniform
            self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        else :
            marginal_centres, marginal_widths, marginal_heights = prior
            for x in [len(marginal_centres), len(marginal_widths), len(marginal_heights)]:
                if x != self.dim_flow: print('!! : prior : incorrect dimensions provided')
                else: pass
            self.obj_prior = MARGINAL_GMM_PRIOR(marginal_centres, marginal_widths, marginal_heights, self.periodic_mask)
            self.evaluate_log_prior_ = lambda z : self.obj_prior.evaluate_log_prior_(z) # (m,1)
            self.sample_prior_ = lambda batch_size : self.obj_prior.sample_prior_(batch_size) # (m,1)
            
        ##

        self.prior = prior
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.dims_hidden = dims_hidden
        self.hidden_activation = hidden_activation
        self.use_joined_MLPs = use_joined_MLPs
        if permutation_inds is None: self.permutation_inds = get_permutation_inds_(self.dim_flow, self.n_layers)
        else: self.permutation_inds = permutation_inds

        self.eps_bin = 2.0/(self.n_bins*5.0)
        self.eps_slope = 1e-2

        self.dimA = self.dim_flow//2
        self.dimB = self.dim_flow - self.dimA

        self.layer_grid = np.arange(n_layers)
        self.layer_grid_flipped = np.flip(self.layer_grid)

        self.SA_book = {i: SPLINE_LAYER(dim = self.dimA,
                                        n_bins = n_bins,
                                        dims_hidden = dims_hidden,
                                        joined_MLP = use_joined_MLPs,
                                        hidden_activation = hidden_activation,
                                        eps_bin = self.eps_bin, eps_slope=  self.eps_slope) for i in self.layer_grid }
        self.SB_book = {i: SPLINE_LAYER(dim = self.dimB,
                                        n_bins = n_bins,
                                        dims_hidden = dims_hidden,
                                        joined_MLP = use_joined_MLPs,
                                        hidden_activation = hidden_activation,
                                        eps_bin = self.eps_bin, eps_slope=  self.eps_slope) for i in self.layer_grid }

        self.sA_book = {i: SHIFT_LAYER_PARAMETRIC(dim = self.dimA, dims_hidden = dims_hidden, hidden_activation = hidden_activation,) for i in self.layer_grid }
        self.sB_book = {i: SHIFT_LAYER_PARAMETRIC(dim = self.dimB, dims_hidden = dims_hidden, hidden_activation = hidden_activation,) for i in self.layer_grid }

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_variables)
        self.print_model_size()

        self.store_initial_parameters_()
        
    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

    def save_model(self, path_and_name : str):
        init_args = [self.dim_flow, self.prior, self.n_layers, self.n_bins, self.dims_hidden, self.hidden_activation, self.use_joined_MLPs, self.permutation_inds]
        save_pickle_([init_args, self.trainable_variables], path_and_name)
    
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_P, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    ##
    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(tf.Variable(self.trainable_variables[i]))

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(list_params[i])
        self.store_initial_parameters_()
    ##

    def split_(self, AB, i):
        AB = tf.gather(AB, self.permutation_inds[i][0], axis=1)
        return AB[:,:self.dimA], AB[:,self.dimA:]
    
    def join_(self, A, B, i):
        return tf.gather(tf.concat([A, B], axis=1), self.permutation_inds[i][1], axis=1)       

    def forward(self, AB, drop_rate : float = 0.0):
        ladJs = 0.0

        for i in self.layer_grid:
            A, B = self.split_(AB, i)
            ## A
            csB = cos_sin_(B)
            A = self.sA_book[i].forward(A, cond = csB)
            A, ladJ = self.SA_book[i].forward(A, cond = csB, drop_rate = drop_rate) ; ladJs += sum_(ladJ)
            A = self.sA_book[i].inverse(A, cond = csB)
            ## #
            ## B
            csA = cos_sin_(A)
            B = self.sB_book[i].forward(B, cond = csA)
            B, ladJ = self.SB_book[i].forward(B, cond = csA, drop_rate = drop_rate) ; ladJs += sum_(ladJ)
            B = self.sB_book[i].inverse(B, cond = csA)
            ## #
            AB = self.join_(A, B, i)

        return AB, ladJs

    def inverse(self, AB, drop_rate : float = 0.0):
        ladJs = 0 

        for i in self.layer_grid_flipped:
            A, B = self.split_(AB, i)
            ## B
            csA = cos_sin_(A)
            B = self.sB_book[i].forward(B, cond = csA)
            B, ladJ = self.SB_book[i].inverse(B, cond = csA, drop_rate = drop_rate) ; ladJs += sum_(ladJ)
            B = self.sB_book[i].inverse(B, cond = csA) 
            ## #
            ## A
            csB = cos_sin_(B)
            A = self.sA_book[i].forward(A, cond = csB)
            A, ladJ = self.SA_book[i].inverse(A, cond = csB, drop_rate = drop_rate) ; ladJs += sum_(ladJ)
            A = self.sA_book[i].inverse(A, cond = csB)
            ## #
            AB = self.join_(A, B, i)

        return AB, ladJs

    def forward_np(self, AB):
        a, b = self.forward( tf.constant(AB, dtype=tf.float32) )
        return a.numpy(), b.numpy()
        
    def inverse_np(self, AB):
        a, b = self.inverse( tf.constant(AB, dtype=tf.float32) ) 
        return a.numpy(), b.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))
    
##

class MODEL_5(tf.keras.models.Model):
    def __init__(self,
                 dim_flow: int = 22,
                 periodic_mask = None,
                 prior : list = None, # list of lists of lists, or None.
                 n_bins : int = 20,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 verbose : bool = True,
                 ):
        super(MODEL_5, self).__init__()
        
        self.dim_flow = dim_flow
        if periodic_mask is None: self.periodic_mask = tf.constant(np.ones([self.dim_flow]).reshape(1,self.dim_flow,1), dtype=tf.int32)
        else:  self.periodic_mask = tf.constant(np.array(periodic_mask).reshape(1,self.dim_flow,1), dtype=tf.int32)

        if prior is None :
            self.log_prior_uniform = - self.dim_flow*np.log(2.0)
            self.evaluate_log_prior_ = lambda z : self.log_prior_uniform
            self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        else :
            import tensorflow_probability as tfp
            marginal_centres, marginal_widths, marginal_heights = prior
            for x in [len(marginal_centres), len(marginal_widths), len(marginal_heights)]:
                if x != self.dim_flow: print('!! : prior : incorrect dimensions provided')
                else: pass
            self.obj_prior = MARGINAL_GMM_PRIOR(marginal_centres, marginal_widths, marginal_heights, self.periodic_mask)
            self.evaluate_log_prior_ = lambda z : self.obj_prior.evaluate_log_prior_(z) # (m,1)
            self.sample_prior_ = lambda batch_size : self.obj_prior.sample_prior_(batch_size) # (m,1)

        self.prior = prior
        self.n_bins = n_bins
        self.dims_hidden = dims_hidden
        self.hidden_activation = hidden_activation
        self.verbose = verbose
        #self.eps = eps
        self.eps_bin = 2.0/(self.n_bins*5.0)
        self.eps_slope = 1e-2
        self.left, self.right = [-1.0, 1.0]

        self.layer_grid_forward = np.arange(self.dim_flow)
        self.layer_grid_inverse = np.flip(self.layer_grid_forward)

        #self.complements = [list(set(np.arange(self.dim_flow).tolist()) - set([i])) for i in range(self.dim_flow)]
        self.cond_trues = tf.constant((1-np.eye(self.dim_flow)*True).astype(bool)[:,np.newaxis,:], dtype=bool)
        self.flow_trues = tf.constant((np.eye(self.dim_flow)*True).astype(bool)[:,np.newaxis,:], dtype=bool)

        self.whs_0_ =  MLP(dims_outputs = [self.dim_flow * self.n_bins, self.dim_flow * self.n_bins, self.dim_flow * (self.n_bins+1)],
                            outputs_activations = None,
                            dims_hidden = dims_hidden,
                            hidden_activation = hidden_activation,
                           )

        self.whs_1_ =  MLP(dims_outputs = [self.dim_flow * self.n_bins, self.dim_flow * self.n_bins, self.dim_flow * (self.n_bins+1)],
                            outputs_activations = None,
                            dims_hidden = dims_hidden,
                            hidden_activation = hidden_activation,
                           )

        self.SHIFT_STATIC = SHIFT_LAYER_STATIC(flow_range = [-1.0, 1.0])

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)

        if verbose: self.print_model_size()
        else: pass

        self.store_initial_parameters_()

    ##
    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(tf.Variable(self.trainable_variables[i]))

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(list_params[i])
        self.store_initial_parameters_()
    ##

    def WHS_0_(self, cond, i, drop_rate = 0.0):
        w, h, s = self.whs_0_(cond, drop_rate = drop_rate) # (m,d,n_bins), (m,d,n_bins), (m,d,n_bins+1) 
        m = w.shape[0]
        w = tf.gather(tf.reshape(w, [m, self.dim_flow, self.n_bins]), [i], axis=1)
        h = tf.gather(tf.reshape(h, [m, self.dim_flow, self.n_bins]), [i], axis=1)
        s = tf.gather(tf.reshape(s, [m, self.dim_flow, self.n_bins + 1]), [i], axis=1)
        return w, h, s # (m,1,n_bins), (m,1,n_bins), (m,1,n_bins+1)

    def WHS_1_(self, cond, i, drop_rate = 0.0):
        w, h, s = self.whs_1_(cond, drop_rate = drop_rate)
        m = w.shape[0]
        w = tf.gather(tf.reshape(w, [m, self.dim_flow, self.n_bins]), [i], axis=1)
        h = tf.gather(tf.reshape(h, [m, self.dim_flow, self.n_bins]), [i], axis=1)
        s = tf.gather(tf.reshape(s, [m, self.dim_flow, self.n_bins + 1]), [i], axis=1)
        return w, h, s # (m,1,n_bins), (m,1,n_bins), (m,1,n_bins+1)

    def S_0_(self, x, cond, i, inverse=False, drop_rate = 0.0):
        w, h, s = self.WHS_0_(cond, i, drop_rate = drop_rate)
        y, ladJ = rational_quadratic_(x=x, w=w, h=h, d=s,
                                            periodic=True, periodic_mask = self.periodic_mask[:,i:i+1,:],
                                            inverse=inverse, 
                                            left=self.left, right=self.right,
                                            bottom=self.left, top=self.right,
                                            eps_bin=self.eps_bin, eps_slope=self.eps_slope)
        return y, ladJ # (m,1), (m,1)

    def S_1_(self, x, cond, i, inverse=False, drop_rate = 0.0):
        w, h, s = self.WHS_1_(cond, i, drop_rate = drop_rate)
        y, ladJ = rational_quadratic_(x=x, w=w, h=h, d=s,
                                            periodic=True, periodic_mask = self.periodic_mask[:,i:i+1,:],
                                            inverse=inverse, 
                                            left=self.left, right=self.right,
                                            bottom=self.left, top=self.right,
                                            eps_bin=self.eps_bin, eps_slope=self.eps_slope)
        return y, ladJ # (m,1), (m,1)

    def split_(self, AB, i):
        x = tf.gather(AB, [i], axis=1) # (m,1)
        cond = tf.where(self.cond_trues[i],AB,0.0) # (1,d,1) # zeros on column i
        cond = selective_cos_sin_(cond, self.periodic_mask[:,:,0])
        return x, cond # (m,1), (m,d)

    def join_(self, x_i, AB, i):
        AB = tf.where(self.flow_trues[i],x_i,AB)
        return AB

    def forward(self, AB, drop_rate : float = 0.0): # Fxz
        ladJxz = 0.0

        for i in self.layer_grid_forward:
            x_i, cond_i = self.split_(AB, i) # (m,1), (m,d)
            x_i, ladJ = self.S_0_(x_i, cond_i, i, drop_rate = drop_rate) # (m,1), (m,1)
            AB = self.join_(x_i, AB, i)
            ladJxz += ladJ

        AB = self.selective_shift_(AB, self.periodic_mask[:,:,0], forward=True)
        for i in self.layer_grid_forward:
            x_i, cond_i = self.split_(AB, i) # (m,1), (m,d)
            x_i, ladJ = self.S_1_(x_i, cond_i, i, drop_rate = drop_rate) # (m,1), (m,1)
            AB = self.join_(x_i, AB, i)
            ladJxz += ladJ
        AB = self.selective_shift_(AB, self.periodic_mask[:,:,0], forward=False)

        return AB, ladJxz

    def inverse(self, AB, drop_rate : float = 0.0): # Fxz
        ladJzx = 0.0

        AB = self.selective_shift_(AB, self.periodic_mask[:,:,0], forward=True)
        for i in self.layer_grid_inverse:
            x_i, cond_i = self.split_(AB, i) # (m,1), (m,d)
            x_i, ladJ = self.S_1_(x_i, cond_i, i, inverse=True, drop_rate = drop_rate) # (m,1), (m,1)
            AB = self.join_(x_i, AB, i)
            ladJzx += ladJ
        AB = self.selective_shift_(AB, self.periodic_mask[:,:,0], forward=False)

        for i in self.layer_grid_inverse:
            x_i, cond_i = self.split_(AB, i) # (m,1), (m,d)
            x_i, ladJ = self.S_0_(x_i, cond_i, i, inverse=True, drop_rate = drop_rate) # (m,1), (m,1)
            AB = self.join_(x_i, AB, i)
            ladJzx += ladJ

        return AB, ladJzx

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

    def save_model(self, path_and_name : str):
        init_args = [self.dim_flow,
                     self.periodic_mask,
                     self.prior,
                     self.n_bins, 
                     self.dims_hidden, 
                     self.hidden_activation, 
                     self.verbose]
        save_pickle_([init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_5, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    def selective_shift_(self, x, mask_x, forward : bool):

        inds_periodic = tf.where(mask_x[0]==1)[:,0]
        inds_other = tf.where(mask_x[0]!=1)[:,0]
        x_periodic = tf.gather(x, inds_periodic, axis=1)
        x_other = tf.gather(x, inds_other, axis=1)

        if forward: x_periodic = self.SHIFT_STATIC.forward(x_periodic)
        else:       x_periodic = self.SHIFT_STATIC.inverse(x_periodic)

        return tf.gather(tf.concat([x_periodic, x_other], axis=1),
                         tf.stack([tf.where(tf.concat([inds_periodic, inds_other], axis=0) == i)[0,0] for i in range(mask_x.shape[1])]),
                         axis=1)

    def forward_np(self, AB):
        a, b = self.forward( tf.constant(AB, dtype=tf.float32) )
        return a.numpy(), b.numpy()
        
    def inverse_np(self, AB):
        a, b = self.inverse( tf.constant(AB, dtype=tf.float32) ) 
        return a.numpy(), b.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))

