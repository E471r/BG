import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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

tfd = tfp.distributions
tfb = tfp.bijectors

class MODEL_M(tf.keras.models.Model):
    """ simple MAF with splines
        for cartesian only (can't encode periodic because different in/out dim is not yet implemented inside tfb.AutoregressiveNetwork)
    """
    # MODEL_M is better than coupled flow.
    # loss = - tf.reduce_mean(ladJ) - tf.reduce_mean(model_m.evaluate_log_prior_(y)) ; where ;  y, ladJ = model_m.forward(xbatch)
    def __init__(self,
                 dim_flow,
                 orders = [None,None],

                 n_bins = 10,
                 min_bin_width = 0.02,
                 min_knot_slope = 0.01,
                 
                 dims_hidden = [100],
                 hidden_activation = tf.nn.relu,

                 prior = 'flat', # or float (standard deviaion on [-1,1] interval.
                 ):
        super(MODEL_M, self).__init__()

        self.init_args = [dim_flow,
                          orders,
                          n_bins,
                          min_bin_width,
                          min_knot_slope,
                          dims_hidden,
                          hidden_activation,
                          prior,
                         ]
        self.dim_flow = dim_flow
        self.n_passes = len(orders)
        self.inds_layers_forward = np.arange(self.n_passes)
        self.inds_layers_inverse = np.flip(self.inds_layers_forward)
        
        self.inds_permute = []
        for i in range(self.n_passes):
            if orders[i] is None:
                if i == 0:   self.inds_permute.append(np.arange(self.dim_flow))
                elif i == 1: self.inds_permute.append(np.flip(np.arange(self.dim_flow)))
                else:        self.inds_permute.append(np.arange(self.dim_flow)[np.random.choice(self.dim_flow,self.dim_flow,replace=False)])
            else: self.inds_permute.append(orders[i])
        self.inds_unpermute = [np.array([int(np.where(self.inds_permute[j] == i)[0]) for i in range(self.dim_flow)]) for j in  self.inds_layers_forward]

        # (self.inds_permute[i] -> ... -> self.inds_unpermute[i]) for i in either self.inds_layers_forward OR self.inds_layers_inverse

        self.n_bins = n_bins
        self.min_bin_width = min_bin_width ; self.flow_range = [-1.0,1.0]
        self.min_knot_slope = min_knot_slope 
        self.dims_hidden  = dims_hidden
        self.hidden_activation = hidden_activation
        self.prior = prior

        ##

        self.MLPs_W = [tfb.AutoregressiveNetwork(params=self.n_bins,
                                                 event_shape=[self.dim_flow],
                                                 hidden_units=dims_hidden,
                                                 activation=hidden_activation) for i in range(self.n_passes)]
        self.MLPs_H = [tfb.AutoregressiveNetwork(params=self.n_bins,
                                                 event_shape=[self.dim_flow],
                                                 hidden_units=dims_hidden,
                                                 activation=hidden_activation) for i in range(self.n_passes)]
        self.MLPs_S = [tfb.AutoregressiveNetwork(params=self.n_bins-1,
                                                 event_shape=[self.dim_flow],
                                                 hidden_units=dims_hidden,
                                                 activation=hidden_activation) for i in range(self.n_passes)]
        
        ##

        # prior:
        if self.prior == 'flat':
            print('self.prior: flat')
            self.log_prior_uniform = - self.dim_flow*np.log(2.0)
            self.evaluate_log_prior_ = lambda x : self.log_prior_uniform
            self.sample_prior_ = lambda batch_size : tf.random.uniform(shape=[batch_size, self.dim_flow], minval=-1.0,  maxval=1.0)
        elif self.prior == 'gauss':
            print('self.prior: gauss')
            self.obj_prior = GAUSSIAN_truncated_PRIOR_neg1_1_range(dim_flow = self.dim_flow,
                                                                   scale = 0.3, # can be tuned manually here.
                                                                   )
            self.evaluate_log_prior_ = self.obj_prior.evaluate_log_prior_ # inputs: x
            self.sample_prior_ = self.obj_prior.sample_prior_             # inputs: m, scale='default' or float/int
        ##

        _ = self.forward( tf.zeros([1, self.dim_flow]) )
        self.n_trainable_tensors = len(self.trainable_weights)
        self.print_model_size()

    def transform_forward_(self, x, i):
        w = self.MLPs_W[i](x) # (m,dim_flow,n_bins) shape already, that is fine.
        h = self.MLPs_H[i](x) # (m,dim_flow,n_bins)
        s = self.MLPs_S[i](x) # (m,dim_flow,n_bins-1)

        xi, ladj = rqs_(x = x,  # (m,dim_flow)
                        w = w,  # (m,dim_flow,n_bins)
                        h = h,  # (m,dim_flow,n_bins)
                        s = s,  # (m,dim_flow,n_bins-1)
                        forward = True,
                        xy_range = self.flow_range,
                        min_bin_width = self.min_bin_width,
                        min_knot_slope = self.min_knot_slope,
                        whs_already_reshaped = True)
        return xi, ladj # (m,dim_flow), (m,1)

    def transform_inverse_(self, x, i):
        ladj = 0.0
        output = []

        for j in range(self.dim_flow):
            cond = tf.concat(output+[x[:,j:]],axis=1) # (m,dim)
            w = self.MLPs_W[i](cond)[:,j:j+1,:] # (m,1,n_bins)
            h = self.MLPs_H[i](cond)[:,j:j+1,:] # (m,1,n_bins)
            s = self.MLPs_S[i](cond)[:,j:j+1,:] # (m,1,n_bins-1)
            xj, _ladj = rqs_(x = x[:,j:j+1],  # (m,1)
                             w = w,           # (m,1,n_bins)
                             h = h,           # (m,1,n_bins)
                             s = s,           # (m,1,n_bins-1)
                             forward = False,
                             xy_range = self.flow_range,
                             min_bin_width = self.min_bin_width,
                             min_knot_slope = self.min_knot_slope,
                             whs_already_reshaped = True)
            ladj += _ladj
            output.append(xj)
        x = tf.concat(output,axis=1)
        return x, ladj

    def forward(self, x):
        ladJ = 0.0
        for i in self.inds_layers_forward:
            x = tf.gather(x, self.inds_permute[i], axis=1)
            x, ladj = self.transform_forward_(x, i=i) ; ladJ += ladj
            x = tf.gather(x, self.inds_unpermute[i], axis=1)
        return x, ladJ # (m,dim), (m,1)

    def inverse(self, x):
        ladJ = 0.0
        for i in self.inds_layers_inverse:
            x = tf.gather(x, self.inds_permute[i], axis=1)
            x, ladj = self.transform_inverse_(x, i=i) ; ladJ += ladj
            x = tf.gather(x, self.inds_unpermute[i], axis=1)
        return x, ladJ # (m,dim), (m,1)

    ##

    def save_model(self, path_and_name : str):
        save_pickle_([self.init_args, self.trainable_variables], path_and_name)
        
    @staticmethod
    def load_model(path_and_name : str):
        init_args, ws = load_pickle_(path_and_name)
        loaded_model = (lambda f, args : f(*args))(MODEL_M, init_args)
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

    def print_model_size(self):
        ws = self.trainable_weights
        n_trainable_variables = sum([np.product(ws[i].shape) if 0 not in ws[i].shape else np.sum(ws[i].shape) for i in range(len(ws))])
        print('There are',n_trainable_variables,'trainable parameters in this model, among', len(ws),'trainable_variables.' )
        shapes = [tuple(x.shape) for x in ws]
        shapes_str = ['W: '+str(shapes[i*2])+' b: '+str(shapes[2*i+1])+' ' for i in range(len(shapes)//2)]
        self.shapes_trainable_variables = [''.join([(' ' * (8 - len(y))) + y for y in [x.split(' ')  for x in shapes_str][i]]) for i in range(len(shapes)//2)]
        print('[NB: To see dimensionalities of the trainable variables print(list(self.shapes_trainable_variables)).] ')

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

'''
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

pi = 3.1415926535897932384626433832795028841971693993751058209 # 3.141592653589793
def cos_sin_(x, nk=1):
    x*=pi
    output = []
    for k in range(1,nk+1):
        output.append(tf.cos(k*x))
        output.append(tf.sin(k*x))
    return tf.concat(output, axis=-1)
'''
