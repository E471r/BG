import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class_RQS = tfp.bijectors.RationalQuadraticSpline

##

import pickle

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

##

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 dims_outputs : list,
                 outputs_activations : list = None,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 **kwargs):
        super().__init__(**kwargs)

        self.dims_outputs = dims_outputs
        self.outputs_activations = outputs_activations
        self.dims_hidden = dims_hidden
        self.hidden_activation = hidden_activation

        n_hidden_layers = len(dims_hidden)
        n_output_layers = len(dims_outputs)

        if outputs_activations is None: outputs_activations = ['linear']*n_output_layers
        else: pass

        self._hidden_layers = [tf.keras.layers.Dense(dims_hidden[i], activation = hidden_activation) for i in range(n_hidden_layers)]
        self._output_layers = [tf.keras.layers.Dense(dims_outputs[j], activation = outputs_activations[j]) for j  in range(n_output_layers)]

    def call(self, x, drop_rate = 0.0):
        for _layer in self._hidden_layers:
            x = _layer(x)
        # if drop_rate > 0.0: x = tf.keras.layers.Dropout(rate = drop_rate)(x, training=True) # drop_rate set to zero (from outside) every time when evaluating.
        # else: pass
        return [layer(x) for layer in self._output_layers]

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

def knot_slopes_(MLP_output, # (m, dim*(n_bins-1))
                 dim,
                 n_bins,
                 min_knot_slope,
                 ):
    MLP_output = tf.reshape(MLP_output, [-1, dim, n_bins - 1])
    knot_slopes = tf.nn.softplus(MLP_output) + min_knot_slope
    return knot_slopes # (m, dim, n_bins-1)

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

def rqs_(x,         # (m,dim)
         w,         # (m,dim*n_bins)
         h,         # (m,dim*n_bins) 
         s = None,  # (m,dim*(n_bins-1))
         forward = True,
         xy_range = [-1.0, 1.0],
         min_bin_width = 0.025, # 10 bins
         min_knot_slope = 0.1,
         whs_already_reshaped = False,
         ):
    m, dim = x.shape

    if not whs_already_reshaped: n_bins = w.shape[1] // dim
    else: n_bins = w.shape[2]

    xy_min, xy_max = xy_range ; domain_width = xy_max - xy_min
    #
    if s is None or s.shape[-1]==0: s = tf.ones([m,dim*(n_bins-1)]) # -> knot_slopes = 1.4132617 > 1   
    else: pass
    #
    bin_positons_x = bin_positons_(w, dim=dim, n_bins=n_bins, domain_width=domain_width, min_bin_width=min_bin_width) 
    # (m, dim, n_bins)
    bin_positons_y = bin_positons_(h, dim=dim, n_bins=n_bins, domain_width=domain_width, min_bin_width=min_bin_width) 
    # (m, dim, n_bins)
    knot_slopes = knot_slopes_(s, dim = dim, n_bins = n_bins, min_knot_slope = min_knot_slope)
    # (m, dim, n_bins-1)
    #
    obj_RQS = class_RQS(bin_widths=bin_positons_x, bin_heights=bin_positons_y, knot_slopes=knot_slopes, range_min=xy_min)
    #
    if forward: return obj_RQS.forward(x), sum_(obj_RQS.forward_log_det_jacobian(x)) # (m,dim), (m,1)
    else:       return obj_RQS.inverse(x), sum_(obj_RQS.inverse_log_det_jacobian(x)) # (m,dim), (m,1)

def shift_(x,      # (m,dim)
           shifts, # (m,dim) OR (,)
           forward = True,
           xy_range = [0.0,1.0],
           ):
    A, B = xy_range
    if forward: return tf.math.floormod(x+shifts - A, B-A) + A
    else: return       tf.math.floormod(x-shifts - A, B-A) + A

def rqs_with_periodic_shift_(x,
                             list_w,             # (m,dim*n_bins) * n_transforms
                             list_h,             # (m,dim*n_bins) * n_transforms
                             list_shifts = None, # (m,dim)
                             list_s = None,      # (m,dim*(n_bins-1)) * n_transforms
                             forward = True,
                             xy_range = [-1.0, 1.0],
                             min_bin_width = 0.025,
                             min_knot_slope = 0.1,
                            ):
    n_transforms = len(list_h)
    #
    if list_s is None or list_s[0].shape[-1] == 0: list_s = [None]*n_transforms
    else: pass
    #
    xy_min, xy_max = xy_range ; domain_width = xy_max - xy_min
    if list_shifts is None: list_shifts = [[0.0, 0.5*domain_width]*n_transforms][0][:n_transforms]
    else: pass 
    #
    if forward: inds_list = [i for i in range(n_transforms)]
    else:       inds_list = [n_transforms-1-i for i in range(n_transforms)]
    #
    ladJ = 0.0
    for i in inds_list:
        x = shift_(x, list_shifts[i], forward=True,  xy_range=xy_range)
        x, ladj = rqs_(x,w=list_w[i],h=list_h[i],s=list_s[i],forward=forward,xy_range=xy_range,min_bin_width=min_bin_width,min_knot_slope=min_knot_slope)
        x = shift_(x, list_shifts[i], forward=False, xy_range=xy_range)
        ladJ += ladj
    return x, ladJ # (m,dim), (m,1)
##

pi = 3.1415926535897932384626433832795028841971693993751058209 # 3.141592653589793
def cos_sin_(x, nk=1):
    x*=pi
    output = []
    for k in range(1,nk+1):
        output.append(tf.cos(k*x))
        output.append(tf.sin(k*x))
    return tf.concat(output, axis=-1)

def broadcasting_app_axis1_(x, n):
    # x ~ (m,n*d)
    m = x.shape[1] // n
    inds_axis_1 = [tf.range(m)+i*m for i in range(n)]
    y = tf.stack([tf.gather(x,inds_axis_1[i], axis=1) for i in range(n)])
    return y # ~ (n,m,d)

class SPLINE_LAYER(tf.keras.layers.Layer):
    def __init__(self,
                 periodic_mask, #~(dim_flow,) ; 1 if periodic, other if non-periodic
                 cond_mask, #~(dim_flow,) ; 1 if variable is being transformed, 0 if variable is being used to condition.

                 n_bins_periodic = 10,
                 number_of_splines_periodic = 2,
                 trainable_shifts = False,
                 n_bins_other = 10,
                 
                 n_hidden = 1, # dim of hidden layers set same as output dim
                 hidden_activation = tf.nn.relu,
                 min_bin_width = 0.001,
                
                 trainable_slopes = False,
                 min_knot_slope = 0.001,
                 
                 dims_hidden = None,

                 nk_for_periodic_MLP_encoding = 1,
                ):
        super().__init__()
        periodic_mask = np.array(periodic_mask).flatten()
        cond_mask = np.array(cond_mask).flatten()
        self.periodic_mask = periodic_mask
        self.cond_mask = cond_mask

        self.n_bins_P = n_bins_periodic
        self.n_splines_P = number_of_splines_periodic
        self.trainable_shifts = trainable_shifts
        self.n_bins_O = n_bins_other

        self.n_hidden = n_hidden 
        self.hidden_activation = hidden_activation 
        self.joined_MLPs = True # TODO
        self.flow_range = [-1.0,1.0]
        self.min_bin_width = min_bin_width

        self.trainable_slopes = trainable_slopes
        self.min_knot_slope = min_knot_slope

        self.dims_hidden = dims_hidden

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

        if self.trainable_slopes:
            dim_slopes_A_P = self.n_A_P*(self.n_bins_P-1) ; dim_slopes_B_P = self.n_B_P*(self.n_bins_P-1)
            dim_slopes_A_O = self.n_A_O*(self.n_bins_O-1) ; dim_slopes_B_O = self.n_B_O*(self.n_bins_O-1)
        else:
            dim_slopes_A_P = dim_slopes_B_P = 0
            dim_slopes_A_O = dim_slopes_B_O = 0 

        if self.trainable_shifts:
            dim_shifts_A = self.n_A_P
            dim_shifts_B = self.n_B_P
        else:
            dim_shifts_A = 0
            dim_shifts_B = 0


        self.output_dims_MLP_A =  []
        if self.n_A_P > 0:

            self.output_dims_MLP_A.append( self.n_splines_P * (self.n_A_P*self.n_bins_P*2 + dim_slopes_A_P + dim_shifts_A) ) 
            if self.n_A_O > 0:
                self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + dim_slopes_A_O )
                self.A_ = self.A_PO_
            else: 
                self.A_ = self.A_P_
        else:
            self.output_dims_MLP_A.append( self.n_A_O*self.n_bins_O*2 + dim_slopes_A_O )
            self.A_ = self.A_O_
            
        if dims_hidden is None: dims_hidden_A = [sum(self.output_dims_MLP_A)]*n_hidden
        else: dims_hidden_A = dims_hidden
        if self.joined_MLPs:
            self.MLP_A = MLP(dims_outputs = self.output_dims_MLP_A,
                            outputs_activations = None,
                            dims_hidden = dims_hidden_A,
                            hidden_activation = hidden_activation)
        else:
            self.MLP_A = 'f_cat(MLP,MLP,...) implement'

        self.output_dims_MLP_B =  []
        if self.n_B_P > 0:
            self.output_dims_MLP_B.append( self.n_splines_P * (self.n_B_P*self.n_bins_P*2 + dim_slopes_B_P + dim_shifts_B) ) 
            if self.n_B_O > 0:
                self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + dim_slopes_B_O )
                self.B_ = self.B_PO_
            else: 
                self.B_ = self.B_P_
        else:
            self.output_dims_MLP_B.append( self.n_B_O*self.n_bins_O*2 + dim_slopes_B_O )
            self.B_ = self.B_O_
            
        if dims_hidden is None: dims_hidden_B = [sum(self.output_dims_MLP_B)]*n_hidden
        else: dims_hidden_B = dims_hidden
        if self.joined_MLPs:
            self.MLP_B = MLP(dims_outputs = self.output_dims_MLP_B,
                            outputs_activations = None,
                            dims_hidden = dims_hidden_B,
                            hidden_activation = hidden_activation)
        else:
            self.MLP_B = 'f_cat(MLP,MLP,...) implement'
        ##

    def A_PO_(self, x, forward = True):

        # split all types of variables and run transformations:
        
        xAP = tf.gather(x, self.inds_A_P, axis=1)
        xAO = tf.gather(x, self.inds_A_O, axis=1)

        xAcP = tf.gather(x, self.inds_A_cP, axis=1)
        xAcO = tf.gather(x, self.inds_A_cO, axis=1)
        xAc = tf.concat([cos_sin_(xAcP, nk=self.nk_for_periodic_MLP_encoding), xAcO], axis=1)
        pAP, pAO = self.MLP_A(xAc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        if self.trainable_shifts:
            yAP,ladJ = rqs_with_periodic_shift_(xAP,                                        # (m,dim)
                                                list_w = pAP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                                list_h = pAP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                                list_shifts = pAP[:,:,2*n:2*n+self.n_A_P],  # (m,dim)
                                                list_s = pAP[:,:,2*n+self.n_A_P:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)
        else:
            yAP,ladJ = rqs_with_periodic_shift_(xAP,                                        # (m,dim)
                                                list_w = pAP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                                list_h = pAP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                                list_shifts = None,  # (m,dim)
                                                list_s = pAP[:,:,n:2*n:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)


        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:2*m], 
                        s = pAO[:,2*m:], 
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
        xBc = tf.concat([cos_sin_(xBcP, nk=self.nk_for_periodic_MLP_encoding), xBcO], axis=1)
        pBP, pBO = self.MLP_B(xBc)

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        if self.trainable_shifts:
            yBP,ladJ = rqs_with_periodic_shift_(xBP,                                        # (m,dim)
                                                list_w = pBP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                                list_h = pBP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                                list_shifts = pBP[:,:,2*n:2*n+self.n_B_P],  # (m,dim)
                                                list_s = pBP[:,:,2*n+self.n_B_P:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)
        else:
            yBP,ladJ = rqs_with_periodic_shift_(xBP,                                        # (m,dim)
                                                list_w = pBP[:,:,:n],                       # (m,dim*n_bins) * n_transforms
                                                list_h = pBP[:,:,n:2*n],                    # (m,dim*n_bins) * n_transforms
                                                list_shifts = None,  # (m,dim)
                                                list_s = pBP[:,:,2*n:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:2*m], 
                        s = pBO[:,2*m:], 
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
        xAc = tf.concat([cos_sin_(xAcP, nk=self.nk_for_periodic_MLP_encoding), xAcO], axis=1)
        pAP = self.MLP_A(xAc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pAP = broadcasting_app_axis1_(pAP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_A_P*self.n_bins_P
        if self.trainable_shifts:
            yAP,ladJ = rqs_with_periodic_shift_(xAP,                          # (m,dim)
                                                list_w = pAP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                                list_h = pAP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                                list_shifts = pAP[:,:,2*n:2*n+self.n_A_P],  # (m,dim)
                                                list_s = pAP[:,:,2*n+self.n_A_P:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)
        else:
            yAP,ladJ = rqs_with_periodic_shift_(xAP,                          # (m,dim)
                                                list_w = pAP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                                list_h = pAP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                                list_shifts = None,  # (m,dim)
                                                list_s = pAP[:,:,2*n:],
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
        xBc = tf.concat([cos_sin_(xBcP, nk=self.nk_for_periodic_MLP_encoding), xBcO], axis=1)
        pBP = self.MLP_B(xBc)[0] # raw params.

        # [m, n_splines_P*(n_A_P*n_bins_P*2 + 0 + n_B_P)] -> [n_splines_P, m, (n_A_P*n_bins_P*2 + n_B_P)]
        pBP = broadcasting_app_axis1_(pBP, self.n_splines_P)

        ladJ_sum = 0.0

        n = self.n_B_P*self.n_bins_P
        if self.trainable_shifts:
            yBP,ladJ = rqs_with_periodic_shift_(xBP,                          # (m,dim)
                                                list_w = pBP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                                list_h = pBP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                                list_shifts = pBP[:,:,2*n:2*n+self.n_B_P],  # (m,dim)
                                                list_s = pBP[:,:,2*n+self.n_B_P:],
                                                forward = forward,
                                                xy_range = self.flow_range,
                                                min_bin_width = self.min_bin_width,
                                                min_knot_slope = self.min_knot_slope,
                                                ) ; ladJ_sum += sum_(ladJ)
        else: 
            yBP,ladJ = rqs_with_periodic_shift_(xBP,                          # (m,dim)
                                                list_w = pBP[:,:,:n],         # (m,dim*n_bins) * n_transforms
                                                list_h = pBP[:,:,n:2*n],      # (m,dim*n_bins) * n_transforms
                                                list_shifts = None,  # (m,dim)
                                                list_s = pBP[:,:,2*n:],
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
        xAc = tf.concat([cos_sin_(xAcP, nk=self.nk_for_periodic_MLP_encoding), xAcO], axis=1)
        pAO = self.MLP_A(xAc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_A_O*self.n_bins_O
        yAO,ladJ = rqs_(xAO,
                        w = pAO[:,:m],
                        h = pAO[:,m:2*m], 
                        s = pAO[:,2*m:],
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
        xBc = tf.concat([cos_sin_(xBcP, nk=self.nk_for_periodic_MLP_encoding), xBcO], axis=1)
        pBO = self.MLP_B(xBc)[0] # raw params.

        ladJ_sum = 0.0

        m = self.n_B_O*self.n_bins_O
        yBO,ladJ = rqs_(xBO,
                        w = pBO[:,:m],
                        h = pBO[:,m:2*m], 
                        s = pBO[:,2*m:], 
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

##

def get_list_cond_masks_unsupervised_(dim_flow):
    """ barcode
        plt.matshow(list_cond_masks) # ref: i-flow paper.
    """
    list_cond_masks = []
    for i in range(dim_flow):
        a = 2**i
        x = np.array((([0]*a + [1]*a)*dim_flow)[:dim_flow])
        if 1 in x: pass
        else: break
        list_cond_masks.append(x)
    return list_cond_masks

##

class GAUSSIAN_truncated_PRIOR_neg1_1_range:
    def __init__(self,
                 dim_flow,
                 scale = 0.3, # 0.35 ok, but 0.4 starts to approach samples from tails.
                ):
        self.dim_flow = dim_flow
        self.scale_default = scale
        self.obj_pdf_centre = tfp.distributions.TruncatedNormal(loc=[0.0]*self.dim_flow,
                                                        scale=[scale]*self.dim_flow,
                                                        low = -1.0, high = 1.0)
        # dont need VonMise for periodic only because obj_pdf_centre is in the centre of the flow range!
        # the purpose is to avoid samples near edges.

    def evaluate_log_prior_(self, x):
        marginal_log_prob = self.obj_pdf_centre.log_prob(x)
        return sum_(marginal_log_prob) # (m,1)

    def sample_prior_(self, batch_size, scale='default'):
        if type(scale) == str:
            z_samples = self.obj_pdf_centre.sample(batch_size)
        else: # float or int
            obj_pdf_centre = tfp.distributions.TruncatedNormal(loc=[0.0]*self.dim_flow,
                                                            scale=[scale]*self.dim_flow,
                                                            low = -1.0, high = 1.0)
            # sort this out if sampling at non trained (narrower ; lower T) prior: 
            # take into account analytic entropy difference.
            # tf.reduce_sum(obj_pdf_centre.entropy()) = -ln(Z_z) # (,)
            # or keep to same perturbation for all models.
            z_samples = obj_pdf_centre.sample(batch_size)
        return z_samples # (m,dim_flow)

##
class MODEL_C(tf.keras.models.Model):
    def __init__(self,
                 periodic_mask, #~(dim_flow,)
                 list_cond_masks = None, # non-random variable permutations
                 # ^ length of this list determines number of layers.
                 # each element of the list is a mask s.t. len(mask) = len(periodic_mask)

                 n_bins_periodic = 10,
                 number_of_splines_periodic = 2, # 1 or 2
                 trainable_shifts = False,
                 n_bins_other = 10,

                 n_hidden = 1, # dim of hidden layers set same as output dim [TODO: choices]
                 hidden_activation = tf.nn.relu,

                 min_bin_width = 0.001,
                 trainable_slopes = True, # True
                 min_knot_slope = 0.001,

                 dims_hidden = None,

                 nk_for_periodic_MLP_encoding = 1,

                 verbose = True,

                 prior = 'gauss', # gauss
                 ):
        super(MODEL_C, self).__init__()

        self.init_args = [periodic_mask,
                         list_cond_masks,
                         n_bins_periodic,
                         number_of_splines_periodic,
                         trainable_shifts,
                         n_bins_other,
                         n_hidden,
                         hidden_activation,
                         min_bin_width,
                         trainable_slopes,
                         min_knot_slope,
                         dims_hidden,
                         nk_for_periodic_MLP_encoding,
                         verbose,
                         prior,
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
        self.trainable_slopes = trainable_slopes
        self.min_knot_slope = min_knot_slope
        self.dims_hidden = dims_hidden
        self.nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding 
        self.verbose = verbose
        self.prior = prior

        self.n_layers = len(list_cond_masks)
        self.inds_layers_forward = np.arange(self.n_layers)
        self.inds_layers_inverse = np.flip(self.inds_layers_forward)

        self.LAYERS = [SPLINE_LAYER( periodic_mask = periodic_mask, 
                                     cond_mask = x, 
                                     n_bins_periodic = n_bins_periodic,
                                     number_of_splines_periodic = number_of_splines_periodic,
                                     trainable_shifts = trainable_shifts,
                                     n_bins_other = n_bins_other,
                                     n_hidden = n_hidden,
                                     hidden_activation = hidden_activation,
                                     min_bin_width = min_bin_width,
                                     trainable_slopes = trainable_slopes,
                                     min_knot_slope =  min_knot_slope,
                                     dims_hidden = dims_hidden,
                                     nk_for_periodic_MLP_encoding = nk_for_periodic_MLP_encoding)

                    for x in list_cond_masks
                    ]

        #################
        ##
        # prior:
        if self.prior == 'flat':
            print('self.prior: flat')
            self.log_prior_uniform = - float(self.dim_flow*np.log(2.0))
            self.evaluate_log_prior_ = lambda x : self.log_prior_uniform
            self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        elif self.prior == 'gauss':
            print('self.prior: gauss')
            self.obj_prior = GAUSSIAN_truncated_PRIOR_neg1_1_range(dim_flow = self.dim_flow,
                                                                   scale = 0.3, # can be tuned manually here.
                                                                   )
            self.evaluate_log_prior_ = self.obj_prior.evaluate_log_prior_ # inputs: x
            self.sample_prior_ = self.obj_prior.sample_prior_             # inputs: m, scale='default' or float/int
        #################

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)

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
        loaded_model = (lambda f, args : f(*args))(MODEL_C, init_args)
        for i in range(len(ws)):
            loaded_model.trainable_variables[i].assign(ws[i])
        return loaded_model

    def store_initial_parameters_(self):
        self.initial_parameters = []
        for i in range(self.n_trainable_tensors):
            self.initial_parameters.append(self.trainable_variables[i].numpy())

    def replace_paremeters(self, list_params):
        for i in range(self.n_trainable_tensors):
            self.trainable_variables[i].assign(tf.Variable(list_params[i], dtype=tf.float32))
        self.store_initial_parameters_()


    def forward_np(self, x):
        x, ladJ = self.forward( tf.constant(x, dtype=tf.float32) )
        return x.numpy(), ladJ.numpy()
        
    def inverse_np(self, x):
        x, ladJ  = self.inverse( tf.constant(x, dtype=tf.float32) ) 
        return x.numpy(), ladJ.numpy()

    def sample(self, n_samples):
        return self.inverse_np(self.sample_prior_(n_samples))

    def check_invertible(self, x):
        
        y, ladJxy = self.forward(x)
        xhat, ladJyx  = self.inverse(y)

        plt.plot(x[:,:], color='black', linewidth=2)
        plt.plot(xhat[:,:], color='red', linewidth=1)
        plt.show()
        
        #print(ladJxy)
        #print(-ladJyx)
        plt.plot(ladJxy, color='black', linewidth=2)
        plt.plot(-ladJyx, color='red', linewidth=1)
        plt.show()
