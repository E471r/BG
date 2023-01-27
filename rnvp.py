
import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp # only needed if wish to set specific normal as prior.

from mlp import MLP

from utils_nf import save_pickle_, load_pickle_

##

class RNVP_LAYER(tf.keras.layers.Layer):
    def __init__(self,
                 dim : int,
                 dims_hidden = [100],
                 joined_MLP = False,
                 hidden_activation = tf.nn.relu,
                 outputs_activations = ['linear', 'linear'],
                 ):
        super().__init__()

        if joined_MLP:
            self.ST = MLP(dims_outputs = [dim, dim],
                          outputs_activations = outputs_activations,
                          dims_hidden = dims_hidden,
                          hidden_activation = hidden_activation)
        else:
            self.S = MLP(dims_outputs = [dim],
                         outputs_activations = [outputs_activations[0]],
                         dims_hidden = dims_hidden,
                         hidden_activation = hidden_activation)
            
            self.T = MLP(dims_outputs = [dim],
                         outputs_activations = [outputs_activations[1]],
                         dims_hidden = dims_hidden,
                         hidden_activation = hidden_activation)
            
            self.ST = lambda cond, drop_rate  : self.S(cond, drop_rate=drop_rate) + self.T(cond, drop_rate=drop_rate)

    def forward(self, x, cond, drop_rate : float = 0.0):
        s, t = self.ST(cond, drop_rate = drop_rate)
        return tf.exp(s)*x + t, s             # (m,dim), (m,dim)

    def inverse(self, x, cond, drop_rate : float = 0.0):
        s, t = self.ST(cond, drop_rate = drop_rate)
        return tf.exp(-s)*(x - t), -s         # (m,dim), (m,dim)

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

##

def get_permutation_inds_(dim : int, n_sets : int, inds = None):
    permutation_inds = []
    for i in range(n_sets):
        if inds is None : permute_indices = np.random.permutation(np.arange(dim))
        else : permute_indices = np.array(inds[i])
        unpermute_indices = np.array([int(np.where(permute_indices == i)[0]) for i in range(dim)])
        permutation_inds.append( [permute_indices, unpermute_indices] )
    return permutation_inds

class MODEL_RNVP(tf.keras.models.Model):
    def __init__(self,
                 dim_flow : int,
                 prior :  list = None, # list of 1 or 2 arrays, both shaped (dim_flow,)
                 n_layers : int = 4,
                 dims_hidden : list = [60],
                 hidden_activation = tf.nn.relu,
                 outputs_activations = ['linear', 'linear'], # tf.nn.tanh, linear
                 use_joined_MLPs : bool = False,
                 permutation_inds : list = None,
                 verbose : bool = True,
                ):
        super(MODEL_RNVP, self).__init__()
        
        self.dim_flow = dim_flow

        if prior is None : # -> unit gaussian.
            self.log_prior_denominator =  - 0.5 * self.dim_flow * np.log(2.0*np.pi)
            self.evaluate_log_prior_ = lambda z : - 0.5 * tf.reduce_sum(z**2, axis=1, keepdims=True) + self.log_prior_denominator
            self.sample_prior_ = lambda batch_size : tf.random.normal(shape=[batch_size, self.dim_flow])
        else:
            if len(prior) == 1: loc = np.array(prior[0]) ; scale = np.ones([self.dim_flow,])
            else: loc = np.array(prior[0]) ; scale = np.array(prior[1])
            if len(loc) != self.dim_flow or len(scale) != self.dim_flow:
                print('!! : prior : incorrect dimensions provided')
            else: pass
            self.obj_normal = tfp.distributions.Normal(loc=loc, scale=scale)
            self.evaluate_log_prior_ = lambda z : sum_(self.obj_normal.log_prob(z))
            self.sample_prior_ = lambda batch_size : self.obj_normal.sample(batch_size)

        self.prior = prior
        self.n_layers = n_layers
        self.dims_hidden = dims_hidden
        self.hidden_activation = hidden_activation
        self.outputs_activations = outputs_activations
        self.use_joined_MLPs = use_joined_MLPs
        if permutation_inds is None: self.permutation_inds = get_permutation_inds_(dim_flow, n_layers)
        else: self.permutation_inds = permutation_inds
        self.verbose = verbose

        self.dimA = dim_flow //2
        self.dimB = dim_flow - self.dimA

        self.layer_grid = np.arange(n_layers)
        self.layer_grid_flipped = np.flip(self.layer_grid)

        self.RA_book = { i : RNVP_LAYER(dim = self.dimA,
                                        dims_hidden = dims_hidden,
                                        joined_MLP = use_joined_MLPs,
                                        hidden_activation =  hidden_activation,
                                        outputs_activations = outputs_activations,
                                        ) for i in self.layer_grid }

        self.RB_book = { i : RNVP_LAYER(dim = self.dimB,
                                        dims_hidden = dims_hidden,
                                        joined_MLP = use_joined_MLPs,
                                        hidden_activation =  hidden_activation,
                                        outputs_activations = outputs_activations,
                                        ) for i in self.layer_grid }

        
        _ = self.forward( tf.zeros([1, dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)
        self.uniform_prior = False

        if verbose: self.print_model_size()
        else: pass

        self.store_initial_parameters_()

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        self.shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]

    def save_model(self, path_and_name : str):
        init_args = [self.dim_flow, self.prior, self.n_layers, self.dims_hidden, self.hidden_activation, self.outputs_activations, self.use_joined_MLPs, self.permutation_inds, self.verbose]
        save_pickle_([init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_RNVP, init_args)
        [loaded_model.trainable_variables[i].assign(ws[i]) for i in range(len(ws))]
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
        ladJxz = 0
        for i in self.layer_grid:

            A, B = self.split_(AB, i)
            A, ladJ = self.RA_book[i].forward(A, cond = B, drop_rate=drop_rate) ; ladJxz += sum_(ladJ)
            B, ladJ = self.RB_book[i].forward(B, cond = A, drop_rate=drop_rate) ; ladJxz += sum_(ladJ)
            AB = self.join_(A, B, i)
            
        return AB, ladJxz

    def inverse(self, AB, drop_rate : float = 0.0):
        ladJzx = 0
        for i in self.layer_grid_flipped:

            A, B = self.split_(AB, i)   
            B, ladJ = self.RB_book[i].inverse(B, cond = A, drop_rate=drop_rate) ; ladJzx += sum_(ladJ)
            A, ladJ = self.RA_book[i].inverse(A, cond = B, drop_rate=drop_rate) ; ladJzx += sum_(ladJ)
            AB = self.join_(A, B, i)

        return AB, ladJzx

    ##

    def forward_special(self, AB, drop_rate : float = 0.0):
        list_ladJxz = []
        for i in self.layer_grid:

            ladJxz = 0.0
            A, B = self.split_(AB, i)
            A, ladJ = self.RA_book[i].forward(A, cond = B, drop_rate=drop_rate) ; ladJxz += sum_(ladJ)
            B, ladJ = self.RB_book[i].forward(B, cond = A, drop_rate=drop_rate) ; ladJxz += sum_(ladJ)
            AB = self.join_(A, B, i)
            list_ladJxz.append(ladJxz)

        return AB, list_ladJxz

    def inverse_special(self, AB):
        list_AB = [] ; list_ladJzx = []
        for i in self.layer_grid_flipped:

            ladJzx = 0.0
            A, B = self.split_(AB, i)   
            B, ladJ = self.RB_book[i].inverse(B, cond = A) ; ladJzx += sum_(ladJ)
            A, ladJ = self.RA_book[i].inverse(A, cond = B) ; ladJzx += sum_(ladJ)
            AB = self.join_(A, B, i)
            list_AB.append(AB) ; list_ladJzx.append(ladJzx)

        return list_AB, list_ladJzx

    ##

    def forward_np(self, AB):
        a, b = self.forward( tf.constant(AB, dtype=tf.float32) )
        return a.numpy(), b.numpy()

    def inverse_np(self, AB):
        a, b = self.inverse( tf.constant(AB, dtype=tf.float32) ) 
        return a.numpy(), b.numpy()

    def sample_np(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))

## Not used:

class LAYER_MCMC(tf.keras.layers.Layer):
    def __init__(self,
                 flow_layer_index : int, # [0,1,2,...] starting from the Boltzmann side.
                 n_flow_layers : int,
                 target_energy_model,
                 prior_energy_model,
                ):
        super().__init__()

        Lambda = flow_layer_index/n_flow_layers # in each layer place this layer first, and then one more after p(z) is created (i.e., at flow_layer_index=n_flow_layers).

        self.energy_ratio = [1-Lambda, Lambda]
        self.target_energy_model = target_energy_model
        self.prior_energy_model = prior_energy_model

    def energy_(self, x):
        ur_x = self.energy_ratio[0]*self.target_energy_model(x)
        uz_x = self.energy_ratio[1]*self.prior_energy_model(x)
        return ur_x + uz_x

    def run_it(self, x, n_steps : int, step_size : float):
        '''
        At each layer (unique Lambda value), the forward and inverse function is the same.
        '''
        m, dim = x.shape
        E0 = self.energy_(x) # any reshapes should be inside those functions.
        E = E0

        for i in range(n_steps):

            dx = step_size * tf.random.normal([m, dim])
            x_prop = x + dx
            E_prop = self.energy_(x_prop)

            acc = tf.cast(tf.random.uniform([m, 1]) < tf.exp(-(E_prop - E)), tf.float32)
            x = (1.0 - acc)*x + acc*x_prop
            E = (1.0 - acc)*E + acc*E_prop

        return x, E - E0

