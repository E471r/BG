import tensorflow as tf
import warnings

##

def searchsorted_(sorted_sequence, values):
    '''
    ans = torch.sum(values[..., None] >= sorted_sequence, dim=-1) - 1
    '''
    bool2int = tf.cast( values[..., None] >= sorted_sequence, dtype=tf.int32 )
    ans = tf.reduce_sum( bool2int, axis=-1 ) - 1
    return ans

def select_bins_(x, idx):
    # idx ~ (batch_dims, input_dim, 1)
    # x ~ (context_batch_dims, input_dim, count_bins)
    '''
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    '''
    idx = tf.clip_by_value( idx, 0, x.shape[-1]-1 )
    if len(idx.shape) >= len(x.shape):
        '''
        x = x.reshape((1,) * (len(idx.shape) - len(x.shape)) + x.shape)
        x = x.expand(idx.shape[:-2] + (-1,) * 2)
        idx = x.gather(-1, idx).squeeze(-1)
        '''
        x = tf.reshape( x, shape = ((1,) * (len(idx.shape) - len(x.shape)) + x.shape) )
        # skipping expand()
        idx = tf.gather_nd( x, idx, batch_dims = len(x.shape)-1 )
    else: pass
    return idx

def normalize_knot_slopes_(unnormalized_knot_slopes, min_knot_slope: float):
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    offset = tf.math.log(tf.exp(1. - min_knot_slope) - 1.)
    return tf.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope

def rational_quadratic_(
    x, # (m,dim)
    w, # (m,dim,n_bins)
    h, # (m,dim,n_bins)
    d, # (m,dim,n_bins+1)

    periodic = False,
    periodic_mask = None, # (1,dim,1) tensor of 1 where periodic.
    
    inverse=False,
    
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    
    eps_bin = 1e-2,
    eps_slope = 1e-2
    ):
    
    min_bin_w = min_bin_h = eps_bin
    min_d = eps_slope
        
    n_bins = w.shape[-1]
    
    if min_bin_w * n_bins > 1.0: warnings.warn("Minimal bin width too large for the number of bins")
    if min_bin_h * n_bins > 1.0: warnings.warn("Minimal bin height too large for the number of bins")
    
    # w :
    w = tf.nn.softmax(w, axis=-1)
    w = min_bin_w + (1 - min_bin_w * n_bins) * w
    c_w = tf.math.cumsum(w, axis=-1)
    c_w = tf.pad( c_w, tf.constant([[0, 0],[0, 0],[1, 0]]) )
    c_w = (right - left) * c_w + left
    w = c_w[..., 1:] - c_w[..., :-1]
    
    # h :
    h = tf.nn.softmax(h, axis=-1)
    h = min_bin_h + (1 - min_bin_h * n_bins) * h
    c_h = tf.math.cumsum(h, axis=-1)
    c_h = tf.pad( c_h, tf.constant([[0, 0],[0, 0],[1, 0]]) )
    c_h = (top - bottom) * c_h + bottom
    h = c_h[..., 1:] - c_h[..., :-1]
    
    # d :
    d = normalize_knot_slopes_(d, min_d) # d = min_d + tf.math.softplus(d)
    
    ####

    if periodic and periodic_mask is None: # True, None --> all
        d = tf.concat([d[..., :-1], d[..., :1]], axis=-1) # all
    else: 
        if periodic_mask is None: pass # False, None --> none
        else: d = tf.where(periodic_mask==1, tf.concat([d[..., :-1], d[..., :1]], axis=-1), d) # some

    __min = tf.reduce_min(x)
    __max = tf.reduce_max(x)
    if __min < left or __max > right:
        # warnings.warn("spline input outside domain") # , found ["+str(__min)+","+str(__max)+"]
        ## UserWarning: spline input outside domain, found [-0.99994534,1.0000002]
        L = right - left
        if periodic and periodic_mask is None: # True, None --> all
            x = tf.where(x < left, x + L, x)
            x = tf.where(x > right, x - L, x)
            
        else:
            if periodic_mask is None: pass # none
            else: # some
                x = tf.where(periodic_mask[:,:,0]==1, tf.where(x < left, x + L, x), x) # # (1,dim,1) - > (1,dim)
                x = tf.where(periodic_mask[:,:,0]==1, tf.where(x > right, x - L, x), x)

                x = tf.where(periodic_mask[:,:,0]==0, tf.where(x < left, left + 1e-6, x), x) # # (1,dim,1) - > (1,dim)
                x = tf.where(periodic_mask[:,:,0]==0, tf.where(x > right, right - 1e-6, x), x)
    else: pass

    ####

    if inverse:
        #bin_idx = tf.searchsorted(c_h, x)[..., None] - 1
        #bin_idx = tf.where(bin_idx<0, 0, bin_idx)
        bin_idx = tf.expand_dims(searchsorted_(c_h, x), axis=-1)
        bin_idx = tf.where(bin_idx==n_bins,n_bins-1,bin_idx) # ..
    else:
        #bin_idx = tf.searchsorted(c_w, x)[..., None] - 1
        #bin_idx = tf.where(bin_idx<0, 0, bin_idx)
        bin_idx = tf.expand_dims(searchsorted_(c_w, x), axis=-1)
        bin_idx = tf.where(bin_idx==n_bins,n_bins-1,bin_idx) # ..
    
    #in_c_w = tf.gather_nd( c_w, bin_idx, batch_dims = len(c_w.shape)-1 )
    in_c_w = select_bins_(c_w, bin_idx)
    
    #in_w = tf.gather_nd( w, bin_idx, batch_dims = len(w.shape)-1 )
    in_w = select_bins_(w, bin_idx)
    
    #in_c_h = tf.gather_nd( c_h, bin_idx, batch_dims = len(c_h.shape)-1 )
    in_c_h = select_bins_(c_h, bin_idx)
    
    #in_h = tf.gather_nd( h, bin_idx, batch_dims = len(h.shape)-1 )
    in_h = select_bins_(h, bin_idx)
    
    #in_delta = tf.gather_nd( h / w, bin_idx, batch_dims = len(h.shape)-1 )
    in_delta = select_bins_(h / w, bin_idx)
    
    #in_d = tf.gather_nd( d, bin_idx, batch_dims = len(d.shape)-1 )
    in_d = select_bins_(d[...,:-1], bin_idx)
    
    #in_d_1 = tf.gather_nd( d[..., 1:], bin_idx, batch_dims = len(d.shape)-1 )
    in_d_1 = select_bins_(d[..., 1:], bin_idx)
    
    if inverse:
        
        _C = in_d + in_d_1 - 2. * in_delta
        
        a = (x - in_c_h) * _C + in_h * (in_delta - in_d)
        b = in_h * in_d - (x - in_c_h) * _C
        c = - in_delta * (x - in_c_h)
        
        discriminant = b**2 - 4. * a * c
        
        # assert (discriminant >= 0.).all()
        discriminant = tf.where(discriminant<0.0, 0.0, discriminant)
        
        root = - 2. * c / (b + tf.sqrt(discriminant))
        
        y = root * in_w + in_c_w
        
        theta_one_minus_theta = root * (1. - root)
        
        denominator = in_delta + theta_one_minus_theta * _C
        
        derivative_numerator = (in_delta**2) * (
            in_d_1 * root**2
            + 2. * in_delta * theta_one_minus_theta
            + in_d * (1. - root)**2  
        )
        
        ladJ_forward = tf.math.log(derivative_numerator) - 2. * tf.math.log(denominator)
        
        ladJ = - ladJ_forward
    
    else:

        theta = (x - in_c_w) / in_w
        theta_one_minus_theta = theta * (1. - theta)
        
        numerator = in_h * (
            in_delta * theta**2 + in_d * theta_one_minus_theta
        )
        denominator = in_delta + (
            (in_d + in_d_1 - 2. * in_delta) * theta_one_minus_theta
        )
        y = in_c_h + numerator / denominator
        
        derivative_numerator = (in_delta**2) * (
            in_d_1 * theta**2
            + 2. * in_delta * theta_one_minus_theta
            + in_d * (1 - theta)**2
        )

        ladJ_forward = tf.math.log(derivative_numerator) - 2. * tf.math.log(denominator)
        
        ladJ = ladJ_forward

    return tf.clip_by_value(y, left+1e-6, right-1e-6), ladJ #, [w,h,d]
