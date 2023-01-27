import tensorflow as tf

from mlp import MLP

from spline import rational_quadratic_

##

class SPLINE_LAYER(tf.keras.layers.Layer):
    def __init__(self,
                 dim : int,             # number of marginal variables being transformed.
                 n_bins : int = 32,     # number of knots in spline.
                 dims_hidden = [100],   # list of ints : number nodes in hidden layers.
                 joined_MLP = False,    # whether to have MLPs seperate for each type of spline parameter.
                 hidden_activation = tf.nn.silu,
                 flow_range = [-1.0,1.0],
                 eps_bin = 1e-2, # or 2.0/(n_bins*5.0)
                 eps_slope = 1e-2,
                 ):
        super().__init__()
        
        self.dim = dim
        self.n_bins = n_bins
        self.left, self.right = flow_range
        self.eps_bin = eps_bin 
        self.eps_slope = eps_slope
        #self.flow_range_safe = [self.left+eps_clamp, self.right-eps_clamp]

        if joined_MLP:
            self.whs_ = MLP(dims_outputs = [dim * n_bins, dim * n_bins, dim * (n_bins+1)],
                            outputs_activations = None,
                            dims_hidden = dims_hidden,
                            hidden_activation = hidden_activation,
                           )
            self.params_net_ = self.joined_MLP_
        else:
            self.w_ = MLP(dims_outputs = [dim * n_bins],
                          outputs_activations = None,
                          dims_hidden = dims_hidden,
                          hidden_activation = hidden_activation,
                         )
            self.h_ = MLP(dims_outputs = [dim * n_bins],
                          outputs_activations = None,
                          dims_hidden = dims_hidden,
                          hidden_activation = hidden_activation,
                         )
            self.s_ = MLP(dims_outputs = [dim * (n_bins+1)],
                          outputs_activations = None,
                          dims_hidden = dims_hidden,
                          hidden_activation = hidden_activation,
                         )
            self.params_net_ = self.seperate_MLP_


    def seperate_MLP_(self, cond, drop_rate = 0.0):
        w = self.w_(cond, drop_rate = drop_rate)[0]
        h = self.h_(cond, drop_rate = drop_rate)[0]
        s = self.s_(cond, drop_rate = drop_rate)[0]

        m = w.shape[0]

        w = tf.reshape(w, [m, self.dim, self.n_bins])
        h = tf.reshape(h, [m, self.dim, self.n_bins])
        s = tf.reshape(s, [m, self.dim, self.n_bins + 1])

        return w, h, s
    
    def joined_MLP_(self, cond, drop_rate = 0.0):
        w, h, s = self.whs_(cond, drop_rate = drop_rate)

        m = w.shape[0]

        w = tf.reshape(w, [m, self.dim, self.n_bins])
        h = tf.reshape(h, [m, self.dim, self.n_bins])
        s = tf.reshape(s, [m, self.dim, self.n_bins + 1])

        return w, h, s
    
    def forward(self, x, cond, periodic_mask=None, drop_rate = 0.0):
        w, h, s = self.params_net_(cond, drop_rate = drop_rate)

        #x = clamp_range_(x, self.flow_range_safe)
        y, ladJ = rational_quadratic_(x=x, w=w, h=h, d=s,
                                      periodic=True, periodic_mask = periodic_mask,
                                      inverse=False, 
                                      left=self.left, right=self.right,
                                      bottom=self.left, top=self.right,
                                      eps_bin=self.eps_bin, eps_slope=self.eps_slope)
        return y, ladJ # (m,dim), (m,dim)

    def inverse(self, y, cond, periodic_mask=None, drop_rate = 0.0):
        w, h, s = self.params_net_(cond, drop_rate = drop_rate)

        #y = clamp_range_(y, self.flow_range_safe)
        x, ladJ = rational_quadratic_(x=y, w=w, h=h, d=s,
                                      periodic=True, periodic_mask = periodic_mask,
                                      inverse=True,
                                      left=self.left, right=self.right,
                                      bottom=self.left, top=self.right,
                                      eps_bin=self.eps_bin, eps_slope=self.eps_slope)
        return x, ladJ # (m,dim), (m,dim)

