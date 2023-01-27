import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

##

pi = np.pi

def get_gather_inds_(ragged):
    gather_inds = []
    a = 0
    for i in range(len(ragged)):
        inds = []
        for k in range(len(ragged[i])):
            inds.append(a)
            a+=1
        gather_inds.append(inds)
    return gather_inds

def sum_(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

class MARGINAL_GMM_PRIOR:
    def __init__(self,
                 marginal_centres, # list of lists # of floats
                 marginal_widths,  # list of lists # of floats
                 marginal_heights, # list of lists # of floats
                 periodic_mask,    # list          # of ones and zeros
                ):

        n_vars = len(marginal_centres)
        periodic_mask = np.array(periodic_mask).reshape(n_vars,)
        n_centres_var = np.array([len(x) for x in marginal_centres])

        check1 = np.array([len(x) for x in marginal_widths])
        check2 = np.array([len(x) for x in marginal_heights])
        if np.allclose(n_centres_var, check1) and np.allclose(n_centres_var, check2): pass
        else: print('!! : MARGINAL_GMM_PRIOR : initialisation arrays do not match in shape.')

        inds_periodic = np.where(periodic_mask==1)[0] ; self.n_periodic = len(inds_periodic)
        inds_other = np.where(periodic_mask!=1)[0] ; self.n_other = len(inds_other)
        
        if self.n_periodic == 0: pass
        else:
            marginal_centres_periodic = [marginal_centres[i] for i in inds_periodic]
            self.inds_gather_periodic = get_gather_inds_(marginal_centres_periodic) # s!
            loc_periodic = np.concatenate(marginal_centres_periodic)*pi # s
            scale_periodic = 1.0/np.concatenate([marginal_widths[i] for i in inds_periodic])**2 # s
            normalised_heights_periodic = [np.array(marginal_heights[i])/np.array(marginal_heights[i]).sum() for i in inds_periodic]
            normalised_heights_periodic = [tf.constant(x.reshape(1,len(x)), dtype=tf.float32) for x in normalised_heights_periodic] # s!
            
            loc_periodic = tf.constant(loc_periodic, dtype=tf.float32)
            scale_periodic = tf.constant(scale_periodic, dtype=tf.float32)
            self.obj_pdf_P = tfp.distributions.VonMises(loc=loc_periodic, concentration=scale_periodic)
            
            self.mode_counts_periodic = [len(x) for x in marginal_centres_periodic]
            
            self.normalised_heights_periodic_flat_tf = tf.concat(normalised_heights_periodic,axis=1)
            self.normalised_heights_periodic = [x.numpy().flatten() for x in normalised_heights_periodic]
            
        if self.n_other == 0: pass
        else:
            marginal_centres_other = [marginal_centres[i] for i in inds_other]
            self.inds_gather_other = get_gather_inds_(marginal_centres_other) # s!
            loc_other = np.concatenate(marginal_centres_other)*pi # s
            scale_other = np.abs(np.concatenate([marginal_widths[i] for i in inds_other])) # s
            normalised_heights_other = [np.array(marginal_heights[i])/np.array(marginal_heights[i]).sum() for i in inds_other]
            normalised_heights_other = [tf.constant(x.reshape(1,len(x)), dtype=tf.float32) for x in normalised_heights_other] # s!
            
            loc_other = tf.constant(loc_other, dtype=tf.float32)
            scale_other = tf.constant(scale_other, dtype=tf.float32)        
            self.obj_pdf_O = tfp.distributions.TruncatedNormal(loc=loc_other, scale=scale_other, low = -pi, high = pi)
            
            self.mode_counts_other = [len(x) for x in marginal_centres_other]
            
            self.normalised_heights_other_flat_tf = tf.concat(normalised_heights_other,axis=1)
            self.normalised_heights_other = [x.numpy().flatten() for x in normalised_heights_other]
            
        self.n_vars = n_vars
        
        self.inds_periodic = inds_periodic
        self.inds_other = inds_other
        
        self.log_rescaling_factor = n_vars*np.log(pi)

    def split_(self, z):
        zp = tf.gather(z, self.inds_periodic, axis=1)
        zo = tf.gather(z, self.inds_other, axis=1)
        return zp, zo

    def join_(self, zp, zo):
        gather_inds = tf.concat([self.inds_periodic, self.inds_other], axis=0) 
        permutation_inds = tf.stack([tf.where(gather_inds == i)[0,0] for i in range(self.n_vars)])
        z = tf.gather(tf.concat([zp,zo], axis=1), permutation_inds, axis=1)
        return z

    def log_pz_periodic_(self, zp):
        
        zp_branches = tf.concat([tf.concat([zp[:,i:i+1]]*self.mode_counts_periodic[i], axis=1) for i in range(self.n_periodic)], axis=1)
        p_zp = self.obj_pdf_P.prob(zp_branches) * self.normalised_heights_periodic_flat_tf
        
        log_pz_periodic = 0.0
        for i in range(self.n_periodic):
            log_pz_periodic += tf.math.log(sum_(tf.gather(p_zp, self.inds_gather_periodic[i], axis=1)))
    
        return log_pz_periodic
    
    def log_pz_other_(self, zo):

        zo_branches = tf.concat([tf.concat([zo[:,i:i+1]]*self.mode_counts_other[i], axis=1) for i in range(self.n_other)], axis=1)
        p_zo = self.obj_pdf_O.prob(zo_branches) * self.normalised_heights_other_flat_tf
        
        log_pz_other = 0.0
        for i in range(self.n_other):
            log_pz_other += tf.math.log(sum_(tf.gather(p_zo, self.inds_gather_other[i], axis=1)))

        return log_pz_other
    
    def evaluate_log_prior_(self, z):
        
        zp,zo = self.split_(z*pi) # self.log_rescaling_factor added to correct for this.
         
        log_pz = 0.0
        
        if self.n_periodic == 0: pass
        else: log_pz += self.log_pz_periodic_(zp)
            
        if self.n_other == 0: pass
        else: log_pz += self.log_pz_other_(zo)
            
        return log_pz + self.log_rescaling_factor
    
    def sample_pz_periodic_(self, m):
        samples_modes = self.obj_pdf_P.sample(m)
        samples = []
        for i in range(self.n_periodic):
            samples_modes_dim_i = tf.gather(samples_modes,self.inds_gather_periodic[i],axis=1) # (m,K)
            Ns_modes_dim_i = np.random.multinomial(m, self.normalised_heights_periodic[i]) # (K)
            
            samples_dim_i = []
            for j in range(self.mode_counts_periodic[i]):
                samples_dim_i.append(samples_modes_dim_i[:Ns_modes_dim_i[j], j:j+1]) # (Nj,)
                
            samples.append(tf.concat(samples_dim_i,axis=0)) # (m,1)
            
        return tf.concat(samples,axis=1)
    
    def sample_pz_other_(self, m):
        samples_modes = self.obj_pdf_O.sample(m)
        samples = []
        for i in range(self.n_other):
            samples_modes_dim_i = tf.gather(samples_modes,self.inds_gather_other[i],axis=1) # (m,K)
            Ns_modes_dim_i = np.random.multinomial(m, self.normalised_heights_other[i]) # (K)
            
            samples_dim_i = []
            for j in range(self.mode_counts_other[i]):
                samples_dim_i.append( samples_modes_dim_i[:Ns_modes_dim_i[j], j:j+1] ) # (Nj,)
                
            samples.append(tf.concat(samples_dim_i,axis=0))
            
        return tf.concat(samples,axis=1)  
    
    def sample_prior_(self, batch_size):
        if self.n_periodic == 0:
            zp = tf.constant(np.array([]).reshape(batch_size,0), dtype=tf.float32)
        else:
            zp = self.sample_pz_periodic_(batch_size)
        
        if self.n_other == 0:
            zo = tf.constant(np.array([]).reshape(batch_size,0), dtype=tf.float32)
        else:
            zo = self.sample_pz_other_(batch_size)
            
        return self.join_(zp,zo)/pi # z
    
    @property
    def number_of_modes(self):
        if self.n_periodic == 0: p = 1
        else: p = np.product(self.mode_counts_periodic)
        if self.n_other == 0: o = 1
        else: o = np.product(self.mode_counts_other)
        return p*o
        
