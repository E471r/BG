import numpy as np

import tensorflow as tf

from mlp import MLP

'''
class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 dims_outputs : list,
                 outputs_activations : list = None,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 **kwargs):
        super().__init__(**kwargs)

        """ MLP : class : Multilayer Perceptron.

        Inputs:
            dims_outputs : list of ints.
            outputs_activations : list of functions, or None (default: unconstrained linear outputs).
            dims_hidden : list of ints. [The length of this list determines how many hidden layers.]
            hidden_activation : non-linear function. [After each hidden later this non-linearity is applied.]

        """

        n_hidden_layers = len(dims_hidden)
        n_output_layers = len(dims_outputs)

        if outputs_activations is None: outputs_activations = ['linear']*n_output_layers
        else: pass

        self.hidden_layers = [tf.keras.layers.Dense(dims_hidden[i], activation = hidden_activation) for i in range(n_hidden_layers)]
        self.output_layers = [tf.keras.layers.Dense(dims_outputs[j], activation = outputs_activations[j]) for j  in range(n_output_layers)]

    def call(self, x, drop_rate = 0.0):
        """
        Inputs:
            x : (m,d) shaped tensor. 
                d, and the (d,dims_hidden[0]) shaped weights of the first hidden layer, become defined after the first call. 
            drop_rate : float in range [0,1]. 
                Default is 0.0, and always set to zero (from outside) when evaluating. 
                During training around 0.1 is approximate heuristic.
        Output:
            ys : list of tensors with shapes (m,dims_outputs[i]) for every output layer i.
        """
        for layer in self.hidden_layers:
            x = layer(x)
        if drop_rate > 0.0: x = tf.keras.layers.Dropout(rate = drop_rate)(x, training=True) # drop_rate set to zero (from outside) every time when evaluating.
        else: pass
        ys = [layer(x) for layer in self.output_layers]
        return ys
'''

import tensorflow_probability as tfp
RQS_class = tfp.bijectors.RationalQuadraticSpline

##

def bin_positons_(MLP_output, # (m,dim*n_bins)
                  dim,
                  n_bins,
                  domain_width,
                  min_bin_width,
                 ):
    MLP_output = tf.reshape(MLP_output, [-1, dim, n_bins])
    c = domain_width - n_bins*min_bin_width
    bin_positons = tf.nn.softmax(MLP_output, axis=-1) * c + min_bin_width
    return bin_positons # (m, dim, n_bins)

def knot_slopes_(MLP_output,
                dim,
                n_bins,
                min_knot_slope,
               ):
    MLP_output = tf.reshape(MLP_output, [-1, dim, n_bins - 1])
    knot_slopes = tf.nn.softplus(MLP_output) + min_knot_slope
    return knot_slopes # (m, dim, n_bins)

def rqs_(x,
         w,
         h, 
         s = None, 
         forward = True,
         xy_range = [0.0, 1.0],
         min_bin_width = 0.1,
         min_knot_slope = 0.1,
        ):
    m, dim = x.shape
    n_bins = w.shape[1] // dim
    
    xy_min, xy_max = xy_range
    domain_width = xy_max - xy_min
    
    bin_positons_x = bin_positons_(w,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_bin_width) # (m, dim, n_bins)
    bin_positons_y = bin_positons_(h,
                                   dim = dim,
                                   n_bins = n_bins,
                                   domain_width = domain_width,
                                   min_bin_width = min_knot_slope) # (m, dim, n_bins)
    
    if s is None: knot_slopes = tf.ones([m,dim,(n_bins-1)])
    else: knot_slopes = knot_slopes_(s, dim = dim, n_bins = n_bins, min_knot_slope = min_knot_slope)
    
    RQS_obj = RQS_class(bin_widths = bin_positons_x,
                        bin_heights = bin_positons_y,
                        knot_slopes = knot_slopes,
                        range_min = xy_min,
                       )
    if forward:
        return RQS_obj.forward(x), RQS_obj.forward_log_det_jacobian(x)
    else:
        return RQS_obj.inverse(x), RQS_obj.inverse_log_det_jacobian(x)
    

def shift_(x, shifts, forward=True, xy_range = [0.0,1.0]):
    A, B = xy_range
    if forward: return tf.math.floormod(x+shifts - A, B-A) + A
    else: return       tf.math.floormod(x-shifts - A, B-A) + A
    
def rqs_with_periodic_shift_(x,
                             list_w,     # (m,dim*n_bins) * n_transforms
                             list_h,     # (m,dim*n_bins) * n_transforms
                             list_shifts, # (m,dim)
                             list_s = None,
                             forward = True,
                             xy_range = [0.0, 1.0],
                             min_bin_width = 0.05, # 0.1 for xy_range of [-1,1]
                             min_knot_slope = 0.05, #not use here, reduce # of parameters.
                            ):
    n_transforms = len(list_h)
    if list_s is None: list_s = [None]*n_transforms
    ladJsum = 0.0
    
    if forward: inds_list = [i for i in range(n_transforms)]
    else:       inds_list = [n_transforms-1-i for i in range(n_transforms)]
    
    for i in inds_list:
        x = shift_(x, list_shifts[i], forward=True, xy_range = xy_range)
        x, ladJ = rqs_(x,
                       w = list_w[i],
                       h = list_h[i],
                       s = list_s[i],
                       forward = forward,
                       xy_range = xy_range,
                       min_bin_width = min_bin_width,
                       min_knot_slope = min_knot_slope,
                      ) ; ladJsum += ladJ
        x = shift_(x, list_shifts[i], forward=False, xy_range = xy_range)
    return x, ladJsum

##

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)
pi = 3.1415926535897932384626433832795028841971693993751058209 # 3.141592653589793
def cos_sin_1_(x):
    x*=pi
    return tf.concat( [tf.cos(x),tf.sin(x)], axis=1 )
def cos_sin_2_(x):
    x*=pi
    return tf.concat( [tf.cos(x),tf.sin(x),
                       tf.cos(2.0*x),tf.sin(2.0*x)], axis=1 )
def cos_sin_3_(x):
    x*=pi
    return tf.concat( [tf.cos(x),tf.sin(x*pi),
                       tf.cos(2.0*x),tf.sin(2.0*x),
                       tf.cos(3.0*x),tf.sin(3.0*x)], axis=1 )
def cos_sin_4_(x):
    x*=pi
    return tf.concat( [tf.cos(x),tf.sin(x),
                       tf.cos(2.0*x),tf.sin(2.0*x),
                       tf.cos(3.0*x),tf.sin(3.0*x),
                       tf.cos(4.0*x),tf.sin(4.0*x)], axis=1 )
def cos_sin_5_(x):
    x*=pi
    return tf.concat( [tf.cos(x),tf.sin(x),
                       tf.cos(2.0*x),tf.sin(2.0*x),
                       tf.cos(3.0*x),tf.sin(3.0*x),
                       tf.cos(4.0*x),tf.sin(4.0*x),
                       tf.cos(5.0*x),tf.sin(5.0*x)], axis=1 )

list_cos_sin_ = [cos_sin_1_,cos_sin_2_,cos_sin_3_,cos_sin_4_,cos_sin_5_]
##

def broadcasting_app_axis1_(x, n):
    # x ~ (m,n*d)
    m = x.shape[1] // n
    inds_axis_1 = [tf.range(m)+i*m for i in range(n)]
    y = tf.stack([tf.gather(x,inds_axis_1[i],axis=1) for i in range(n)])
    return y # ~ (n,m,d)

class SPLINE_LAYER_NEW(tf.keras.layers.Layer):
    def __init__(self,
                 periodic_mask, #~(dim_flow,) ; 1 if periodic, other if non-periodic
                 cond_mask, #~(dim_flow,) ; 1 if variable is being transformed, 0 if variable is being used to condition.

                 n_bins_periodic = 10,
                 number_of_splines_periodic = 1,
                 n_bins_other = 10,
                 
                 n_hidden = 1, # dim of hidden layers set same as output dim
                 hidden_activation = tf.nn.silu,
                 # joined_MLPs =  True,
                 flow_range = [-1.0,1.0], # not adjustable here yet!
                 min_bin_width = 0.1,
                
                 #trainable_slopes = False,
                 min_knot_slope = 0.1, #  not used here yet.
                 
                 nk_for_periodic_MLP_encoding = 1,
                ):

        super().__init__()
        periodic_mask = np.array(periodic_mask).flatten()
        cond_mask = np.array(cond_mask).flatten()
        self.periodic_mask = periodic_mask
        self.cond_mask = cond_mask

        self.n_bins_P = n_bins_periodic
        self.n_splines_P = number_of_splines_periodic 
        self.n_bins_O = n_bins_other

        self.n_hidden = n_hidden 
        self.hidden_activation = hidden_activation 
        self.joined_MLPs = True #joined_MLPs
        self.flow_range = flow_range
        self.min_bin_width = min_bin_width

        self.trainable_slopes = False #trainable_slopes
        self.min_knot_slope = min_knot_slope

        self.nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding
        ##

        self.n_variables = len(periodic_mask)
        if len(cond_mask) != self.n_variables: print('!! SPLINE_LAYER_NEW : lengths of both masks should be equal')
        else: pass

        self.inds_A_P = np.where((cond_mask==1)&(periodic_mask==1))[0]  # inds periodic in 1st part
        self.inds_A_O = np.where((cond_mask==1)&(periodic_mask!=1))[0]  # inds other in 1st part
        self.inds_A_cP = np.where((cond_mask==0)&(periodic_mask==1))[0] # inds not flowing periodic in 1st part
        self.inds_A_cO = np.where((cond_mask==0)&(periodic_mask!=1))[0] # inds not flowing other in 1st part

        cat_inds_A = np.concatenate([self.inds_A_P, self.inds_A_O, self.inds_A_cP, self.inds_A_cO])
        self.inds_unpermute_A = np.array([int(np.where(cat_inds_A == i)[0]) for i in range(len(cat_inds_A))])

        self.inds_B_P = np.where((cond_mask==0)&(periodic_mask==1))[0]  # inds periodic in 2nd part
        self.inds_B_O = np.where((cond_mask==0)&(periodic_mask!=1))[0]  # inds other in 2nd part
        self.inds_B_cP = np.where((cond_mask==1)&(periodic_mask==1))[0] # inds not flowing periodic in 2nd part
        self.inds_B_cO = np.where((cond_mask==1)&(periodic_mask!=1))[0] # inds not flowing other in 2st part

        cat_inds_B = np.concatenate([self.inds_B_P, self.inds_B_O, self.inds_B_cP, self.inds_B_cO])
        self.inds_unpermute_B = np.array([int(np.where(cat_inds_B == i)[0]) for i in range(len(cat_inds_B))])

        self.n_A_P = len(self.inds_A_P)
        self.n_A_O = len(self.inds_A_O)
        self.n_A_cP = len(self.inds_A_cP)
        self.n_A_cO = len(self.inds_A_cO)

        self.n_B_P = len(self.inds_B_P)
        self.n_B_O = len(self.inds_B_O)
        self.n_B_cP = len(self.inds_B_cP)
        self.n_B_cO = len(self.inds_B_cO)
        ##

        self.output_dims_MLP_A =  []
        if self.n_A_P > 0:
            self.output_dims_MLP_A.append( self.n_splines_P * (self.n_A_P*self.n_bins_P*2 + 0 + self.n_A_P) ) 
            # ^ if self.trainable_slopes 0:= self.n_A_P*(self.n_bins_P-1)
            if self.n_A_O > 0:
                self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + 0 )
                # ^ if self.trainable_slopes 0:= self.n_A_O*(self.n_bins_O-1)
                self.A_ = self.A_PO_
            else: 
                self.A_ = self.A_P_
        else:
            self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + 0 )
            # ^ if self.trainable_slopes 0:= self.n_A_O*(self.n_bins_O-1)
            self.A_ = self.A_O_
            
        self.MLP_A = MLP(dims_outputs = self.output_dims_MLP_A,
                         outputs_activations = None,
                         dims_hidden = [sum(self.output_dims_MLP_A)]*n_hidden, # check.
                         hidden_activation = hidden_activation)

        self.output_dims_MLP_B =  []
        if self.n_B_P > 0:
            self.output_dims_MLP_B.append( self.n_splines_P * (self.n_B_P*self.n_bins_P*2 + 0 + self.n_B_P) ) 
            # ^ if self.trainable_slopes 0:= self.n_B_P*(self.n_bins_P-1)
            if self.n_B_O > 0:
                self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + 0 )
                # ^ if self.trainable_slopes 0:= self.n_B_O*(self.n_bins_O-1)
                self.B_ = self.B_PO_
            else: 
                self.B_ = self.B_P_
        else:
            self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + 0 )
            # ^ if self.trainable_slopes 0:= self.n_B_O*(self.n_bins_O-1)
            self.B_ = self.B_O_
            
        self.MLP_B = MLP(dims_outputs = self.output_dims_MLP_B,
                         outputs_activations = None,
                         dims_hidden = [sum(self.output_dims_MLP_B)]*n_hidden, # check.
                         hidden_activation = hidden_activation)
        ##
        self.cos_sin_ = list_cos_sin_[nk_for_periodic_MLP_encoding-1]

    def A_PO_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xAP = tf.gather(x, self.inds_A_P, axis=1)
        xAO = tf.gather(x, self.inds_A_O, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAP, pAO = self.MLP_A(xAc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        yAP,ladJ = rqs_with_periodic_shift_(xAP,                          # (m,dim)
                                            list_w = pAP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pAP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pAP[:,:,2*n:],  # (m,dim)
                                            list_s = None,
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:], 
                        s = None, 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order(join):
        # yAP, yAO, xAcP, xAcO ; where the former two were transformed conditioned on the latter two.

        cat_y = tf.concat([yAP, yAO, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_PO_(self, x, forward = True):
        """ roles of flowing vs. conditinoing swapped
        """
        # split all types of variables and run transformations:
        
        xBP = tf.gather(x, self.inds_B_P, axis=1)
        xBO = tf.gather(x, self.inds_B_O, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBP, pBO = self.MLP_B(xBc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        yBP,ladJ = rqs_with_periodic_shift_(xBP,                          # (m,dim)
                                            list_w = pBP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pBP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pBP[:,:,2*n:],  # (m,dim)
                                            list_s = None,
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:], 
                        s = None, 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order(join):
        # yBP, yBO, xBcP, xBcO ; where the former two were transformed conditioned on the latter two, wich were tranformed in A.

        cat_y = tf.concat([yBP, yBO, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def A_P_(self, x, forward = True):
        
        # split all types of variables and run transformations:
        
        xAP = tf.gather(x, self.inds_A_P, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAP = self.MLP_A(xAc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        yAP,ladJ = rqs_with_periodic_shift_(xAP,                          # (m,dim)
                                            list_w = pAP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pAP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pAP[:,:,2*n:],  # (m,dim)
                                            list_s = None,
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        # put everything back in the right order(join):
        # yAP, xAcP, xAcO ; where the former was transformed conditioned on the latter two.

        cat_y = tf.concat([yAP, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_P_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xBP = tf.gather(x, self.inds_B_P, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBP = self.MLP_B(xBc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        yBP,ladJ = rqs_with_periodic_shift_(xBP,                          # (m,dim)
                                            list_w = pBP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                            list_h = pBP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                            list_shifts = pBP[:,:,2*n:],  # (m,dim)
                                            list_s = None,
                                            forward = forward,
                                            xy_range = self.flow_range,
                                            min_bin_width = self.min_bin_width,
                                            min_knot_slope = self.min_knot_slope,
                                            ) ; ladJ_sum += sum_(ladJ)

        # put everything back in the right order(join):
        # yBP, xBcP, xBcO ; where the former was transformed conditioned on the latter two.

        cat_y = tf.concat([yBP, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def A_O_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xAO = tf.gather(x, self.inds_A_O, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([self.cos_sin_(xAcP), xAcO], axis=1)
        pAO = self.MLP_A(xAc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:], 
                        s = None, 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order(join):
        # yAO, xAcP, xAcO ; where the former was transformed conditioned on the latter two.

        cat_y = tf.concat([yAO, xAcP, xAcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_A, axis=1)

        return y, ladJ_sum

    def B_O_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xBO = tf.gather(x, self.inds_B_O, axis=1)

        xBcP = tf.gather(x, self.inds_B_cP, axis=1)
        xBcO = tf.gather(x, self.inds_B_cO, axis=1)
        xBc = tf.concat([self.cos_sin_(xBcP), xBcO], axis=1)
        pBO = self.MLP_B(xBc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:], 
                        s = None, 
                        forward = forward,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        ) ; ladJ_sum += sum_(ladJ)
        
        # put everything back in the right order(join):
        # yBO, xBcP, xBcO ; where the former was transformed conditioned on the latter two.

        cat_y = tf.concat([yBO, xBcP, xBcO], axis=1)
        y = tf.gather(cat_y, self.inds_unpermute_B, axis=1)

        return y, ladJ_sum

    def forward(self,x):
        ladJ_forward = 0.0
        x, ladJ = self.A_(x, forward=True) ; ladJ_forward += ladJ
        x, ladJ = self.B_(x, forward=True) ; ladJ_forward += ladJ
        return x, ladJ_forward

    def inverse(self,x):
        ladJ_inverse = 0.0 
        x, ladJ = self.B_(x, forward=False) ; ladJ_inverse += ladJ
        x, ladJ = self.A_(x, forward=False) ; ladJ_inverse += ladJ
        return x, ladJ_inverse

