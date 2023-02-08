import numpy as np

import tensorflow as tf

from spline_layer_new import SPLINE_LAYER_NEW

from utils_nf import save_pickle_, load_pickle_

##

class MODEL_3_NEW(tf.keras.models.Model):
    def __init__(self,
                 periodic_mask,
                 list_cond_masks, # non-random variable permutations [TODO: automate internally]
                 # ^ length of this list determines number of layers.
                 # each element of the list is a mask s.t. len(mask) = len(periodic_mask)

                 n_bins_periodic = 10,           # in each layer (if relevant)
                 number_of_splines_periodic = 1, # in each layer (if relevant)
                 n_bins_other = 10,              # in each layer (if relevant)

                 n_hidden = 1, # dim of hidden layers set same as output dim
                 hidden_activation = tf.nn.silu,

                 # flow_range = [-1.0,1.0],      # not adjustable here yet!

                 min_bin_width = 0.1,
                 # min_knot_slope = 0.1,         # not used here yet.

                 nk_for_periodic_MLP_encoding = 1,

                 verbose : bool = True,
                 ):
        super(MODEL_3_NEW, self).__init__()

        " TODO "

        # just stack layers based on list_cond_masks
        # this part is much less cumbersome now, all is handled inside each layer.
        





