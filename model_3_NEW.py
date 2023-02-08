import numpy as np

import tensorflow as tf

from spline_layer_new import SPLINE_LAYER_NEW

from utils_nf import save_pickle_, load_pickle_

##

def get_list_cond_masks_unsupervised_(dim_flow):
    """ barcode
    """
    list_cond_masks = []
    for i in range(dim_flow):
        a = 2**i
        x = np.array((([0]*a + [1]*a)*D)[:dim_flow])
        if 1 in x: pass
        else: break
        list_cond_masks.append(x)
    return list_cond_masks # plt.matshow(list_cond_masks) ; ref: i-flow paper.

class MODEL_3_NEW(tf.keras.models.Model):
    def __init__(self,
                 periodic_mask, #~(dim_flow,)
                 list_cond_masks = None, # non-random variable permutations
                 # ^ length of this list determines number of layers.
                 # each element of the list is a mask s.t. len(mask) = len(periodic_mask)

                 n_bins_periodic = 10,           # in each layer (if relevant)
                 number_of_splines_periodic = 1, # in each layer (if relevant)
                 n_bins_other = 10,              # in each layer (if relevant)

                 n_hidden = 1, # dim of hidden layers set same as output dim
                 hidden_activation = tf.nn.silu,

                 # flow_range = [-1.0,1.0],      # not adjustable here yet!

                 min_bin_width = 0.1,
                 # trainable_slopes = False,     # not used here yet.
                 # min_knot_slope = 0.1,         # not used here yet.

                 nk_for_periodic_MLP_encoding = 1,

                 verbose : bool = True,
                 ):
        super(MODEL_3_NEW, self).__init__()

        self.init_args = [periodic_mask,
                         list_cond_masks,
                         n_bins_periodic,
                         number_of_splines_periodic,
                         n_bins_other,
                         n_hidden,
                         hidden_activation,
                         min_bin_width,
                         nk_for_periodic_MLP_encoding,
                         verbose,
                         ]

        periodic_mask = np.array(periodic_mask).flatten()
        self.dim_flow = len(periodic_mask)
        if list_cond_masks is not None: list_cond_masks = [np.array(x).flatten() for x in list_cond_masks]
        else: list_cond_masks = get_list_cond_masks_unsupervised_(self.dim_flow)
        
        self.periodic_mask = periodic_mask
        self.list_cond_masks = list_cond_masks
        self.n_bins_periodic = n_bins_periodic
        self.number_of_splines_periodic = number_of_splines_periodic
        self.n_bins_other = n_bins_other
        self.n_hidden = n_hidden
        self.hidden_activation = hidden_activation
        self.flow_range = [-1.0,1.0] #
        self.min_bin_width = min_bin_width
        self.trainable_slopes = False
        self.min_knot_slope = 0.1    #
        self.nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding 
        self.verbose = verbose

        self.n_layers = len(list_cond_masks)
        self.inds_layers_forward = np.arange(self.n_layers)
        self.inds_layers_inverse = np.flip(self.inds_layers_forward)

        self.LAYERS = [SPLINE_LAYER_NEW( periodic_mask = periodic_mask, 
                                         cond_mask = x, 
                                         n_bins_periodic = n_bins_periodic,
                                         number_of_splines_periodic = number_of_splines_periodic,
                                         n_bins_other = n_bins_other,
                                         n_hidden = n_hidden,
                                         hidden_activation = hidden_activation,
                                         flow_range = self.flow_range,
                                         min_bin_width = min_bin_width,
                                         nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding)

                      for x in list_cond_masks
                      ]

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)

        #################

        ## generic should be ingerited (TODO):
        self.log_prior_uniform = - self.dim_flow*np.log(2.0)
        self.evaluate_log_prior_ = lambda z : self.log_prior_uniform
        self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        ##

        if verbose: self.print_model_size()
        else: pass

        self.store_initial_parameters_()

    def forward(self, x, drop_rate : float = 0.0):
        ladJ_forward = 0.0
        for i in self.inds_layers_forward:
            x, ladJ = self.LAYERS[i].forward(x) ; ladJ_forward += ladJ
        return x, ladJ_forward 

    def inverse(self, x, drop_rate : float = 0.0):
        ladJ_inverse = 0.0
        for i in self.inds_layers_inverse:
            x, ladJ = self.LAYERS[i].inverse(x) ; ladJ_inverse += ladJ
        return x, ladJ_inverse

    ###################################################################################

    ## generic functions: 
    # TODO: generic functions for saving loading etc should be inherited.

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

    def save_model(self, path_and_name : str):
        save_pickle_([self.init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_3_NEW, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(tf.Variable(self.trainable_variables[i]))

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(list_params[i])
        self.store_initial_parameters_()


    def forward_np(self, x):
        x, ladJ = self.forward( tf.constant(x, dtype=tf.float32) )
        return x.numpy(), ladJ.numpy()
        
    def inverse_np(self, x):
        x, ladJ  = self.inverse( tf.constant(x, dtype=tf.float32) ) 
        return x.numpy(), ladJ.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))
