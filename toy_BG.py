import numpy as np
import tensorflow as tf 

## Ground truth functions:

"""
class D2_double_well: 
    def __init__(self,
                 params_well_0 = [50.0,1.3,1,-2.5,0],
                 params_well_1 = [50.0,1.3,200,2.5,0],
                 params_wells = [4,0.0005],
                ):
        self.theta_0 = params_well_0
        self.theta_1 = params_well_1
        self.theta_01 = params_wells
    
    def evaluate_potential(self,
                           positions : np.ndarray,
                           return_grad : bool = False,
                          ):
        
        forces = np.zeros_like(positions)
        
        a,s,l,x,y = self.theta_0
        W0 = a*np.exp(-0.5*(positions[:,0]-x)**2/(s**2) -0.5*(positions[:,1]-y)**2/(l**2))
        dW0dx = -(positions[:,0]-x)*W0/s**2
        dW0dy = -(positions[:,1]-y)*W0/l**2
        
        a,s,l,x,y = self.theta_1
        W1 = a*np.exp(-0.5*(positions[:,0]-x)**2/(s**2) -0.5*(positions[:,1]-y)**2/(l**2))
        dW1dx = -(positions[:,0]-x)*W1/s**2
        dW1dy = -(positions[:,1]-y)*W1/l**2        
        
        a,b = self.theta_01
        
        energies = -(W0+W1) + a*(positions[:,0])**2 + b*(positions[:,1])**2
        energies = energies[:,np.newaxis]
        
        forces[:,0] = -dW0dx-dW1dx + 2*a*positions[:,0]
        forces[:,1] = -dW0dy-dW1dy + 2*b*positions[:,1]
        
        if return_grad:
            return energies, forces
        else:
            return energies, 0.0
"""

class GAUS:
    def __init__(self,
                mu,
                C,
                height,
                build_tf : bool = False):
        self.d = mu.shape[0]
        self.mu = mu.reshape(1,self.d) # (1,d)
        self.C = C
        self.height = float(height)
        
        u,s,uT = np.linalg.svd(C)
        self.det_C = np.product(s)
        
        self.C_inv = u.dot((1/s)*np.eye(self.d)).dot(uT)
        self.C_sqrt = u.dot(np.sqrt(s)*np.eye(self.d)).dot(uT)
        self.normalisation_constant = ((np.sqrt(2*np.pi)**self.d)*np.sqrt(self.det_C)) # (1,)
        
        if build_tf:
            self.mu_tf = tf.constant(self.mu, dtype=tf.float32)
            self.C_inv_tf = tf.constant(self.C_inv, dtype=tf.float32)
            
        self.set_norm_constant_(height = height)
            
    def set_norm_constant_(self, height : float = None):
        self.height = float(height)
        self.c = self.height / self.normalisation_constant
        
    def gaussian_np_(self, X : np.ndarray, return_grad : bool = False):
        
        dx = (X - self.mu) # (N,d) - (1,d) = (N,d)
        distances = np.einsum('oi,ij,oj->o', dx, self.C_inv, dx) # (N,)
        p_X = self.c * np.exp(-0.5*distances)[:,np.newaxis] # (N,1)
        
        if return_grad:
            dp_dX = - p_X * np.einsum('oi,ij->oj', dx, self.C_inv) # (N,1)*(N,d) -> N,d
            return p_X, dp_dX
        else: return p_X

    def gaussian_tf_(self, X : np.ndarray, return_grad : bool = False):

        dx = (X - self.mu_tf) # (N,d) - (1,d) = (N,d)
        distances = tf.einsum('oi,ij,oj->o', dx, self.C_inv_tf, dx) # (N,)
        p_X = self.c * tf.expand_dims( tf.exp(-0.5*distances), axis=1 ) # (N,1)
        
        if return_grad:
            dp_dX = - p_X * tf.einsum('oi,ij->oj', dx, self.C_inv_tf) # (N,1)*(N,d) -> N,d
            return p_X, dp_dX
        else: return p_X
        
    def gaussian(self, X : np.ndarray, tf : bool = False, return_grad : bool = False):
        if tf: return self.gaussian_tf_(X, return_grad=return_grad)
        else: return self.gaussian_np_(X, return_grad=return_grad)

    def sample(self, n_samples):
        return self.mu + np.random.randn(n_samples,self.d).dot(self.C_sqrt)
    
class TOY_POTENTIAL:
    def __init__(self,
                 mus : list or np.ndarray,
                 Cs : list or np.ndarray or float,
                 heights : list or np.ndarray, # = depths of energy wells.
                 compatible_tf : bool = False,
                ):
        # mus ~ [(d,), (d,), ...]
        # Cs ~ [(d,d), (d,d), ...]
        # heights ~ [h0, h1, ...]
        self.n_wells = len(mus)
        
        if type(Cs) == float:
            self.dim = max(list(mus[0].shape))
            Cs = np.array([np.eye(self.dim)*Cs]*len(mus))
        else: pass        
        
        self.heights = np.array(heights).reshape(self.n_wells,)
        self.sum_heights = self.heights.sum()
        self.normalised_heights = self.heights / self.sum_heights
    
        self.gaussians = {i: GAUS(mus[i], Cs[i], self.normalised_heights[i], build_tf=compatible_tf) for i in range(self.n_wells) }
        self.Zs = np.array([self.gaussians[i].normalisation_constant for i in range(self.n_wells)])
        self.volumes = self.Zs * self.heights
        
        self.normalised_volumes = self.Zs * self.volumes / self.volumes.sum()
        self.gaussians = {i: GAUS(mus[i], Cs[i], self.normalised_volumes[i], build_tf=compatible_tf) for i in range(self.n_wells) }
        
        self.potentials = {i: GAUS(mus[i], Cs[i], self.volumes[i], build_tf=compatible_tf) for i in range(self.n_wells) }
        
        self.pi = self.volumes / self.volumes.sum()
        
    def evaluate_potential(self,
                           positions : np.ndarray,
                           tf : bool = False,
                           return_grad : bool = False,
                          ):
        if return_grad:
            energies, forces = 0.0, 0.0
            for i in range(self.n_wells):
                u, f = self.potentials[i].gaussian(X=positions, tf=tf, return_grad=True)
                energies -= u
                forces -= f
            return energies, forces
        else:
            energies = 0.0
            for i in range(self.n_wells):
                u = self.potentials[i].gaussian(X=positions, tf=tf, return_grad=False)
                energies -= u
            return energies       

##
def d1_gaus_p_tf_(x, z, var):
    cx = tf.cos(x) ; cz = tf.cos(z)
    sx = tf.sin(x) ; sz = tf.sin(z)
    distance = (cx-cz)**2 + (sx-sz)**2
    arg = (1./var) * distance / 8.
    return tf.exp(-arg)
    
def d2_well_tf_(x, mu, var, h = 1):
    px = d1_gaus_p_tf_(x[:,0], mu[0], var[0])           
    py = d1_gaus_p_tf_(x[:,1], mu[1], var[1])
    return h * px * py
    
def d2_n_well_tf_(x, list_mu, list_var, list_h):
    n_wells = len(list_h)
    e = 0.0
    for i in range(n_wells):
        e += d2_well_tf_(x, list_mu[i], list_var[i], h = list_h[i])
    return tf.expand_dims(e,axis=1)

class TOY_POTENTIAL_P:
    def __init__(self,
                 list_mu,
                 list_var,
                 list_h
                ):
        self.list_mu = tf.constant(list_mu,dtype=tf.float32)
        self.list_var = tf.constant(list_var,dtype=tf.float32)
        self.list_h = tf.constant(list_h,dtype=tf.float32)
        
    def evaluate_potential(self, x, return_grad=False):
        x = tf.constant(x, dtype=tf.float32)
        if return_grad:
            with tf.GradientTape() as tape:
                tape.watch(x)
                energies = d2_n_well_tf_(x,
                                         list_mu = self.list_mu,
                                         list_var = self.list_var,
                                         list_h = self.list_h)
                Energies = energies[:,0]

            forces = tape.gradient(Energies, x)
            return energies, forces
        else:
            return d2_n_well_tf_(x,
                                 list_mu = self.list_mu,
                                 list_var = self.list_var,
                                 list_h = self.list_h)
##

class MODEL_DIST:
    def __init__(self,
                 mus : list or np.ndarray,
                 Cs : list or np.ndarray or float,  # Temperature (T)
                 heights : list or np.ndarray,      # np.exp(-energies/T))
                 daxes : list or np.ndarray = None, # [dx**2,dy**2] ; dx, dy are bin widths.
                 compatible_tf : bool = False,
                ):
        # mus ~ [(d,), (d,), ...]
        # Cs ~ [(d,d), (d,d), ...]
        # heights ~ [h0, h0, ...]
        self.dim = max(mus[0].shape)
        self.c = 1.0 / np.array(heights).sum()
        
        if daxes is None: daxes = np.ones((self.dim))
        else: daxes = np.array(daxes).reshape(self.dim,)
        
        if type(Cs) == float:
            dim = max(list(mus[0].shape))
            Cs = np.array([daxes*np.eye(dim)*Cs]*len(mus))
        else: pass
        self.n_wells = len(mus)
        self.gaussians = {i: GAUS(mus[i], Cs[i], heights[i], build_tf=compatible_tf) for i in range(self.n_wells) }
        self.pi = np.array([heights[i]/np.sum(heights) for i in range(self.n_wells)]).reshape(self.n_wells,)


    def evaluate_dist_np(self, positions, return_grad : bool = False): # very slow.
        p_x, dp_dx = 0, 0
        if return_grad:
            for i in range(self.n_wells):
                px, dpdx = self.gaussians[i].gaussian(X=positions, tf=False, return_grad=True)
                p_x += px
                dp_dx += dpdx
            return self.c*p_x, self.c*dp_dx
        else:
            for i in range(self.n_wells):
                p_x += self.gaussians[i].gaussian(X=positions, tf=False, return_grad=False)
            return self.c*p_x

    def sample(self, N_samples, shuffle=False, which_wells=None):
        if which_wells is None:
            which_wells = np.arange(self.n_wells)
        else: pass
        # N = sample size
        pi = (N_samples*self.pi[which_wells]).astype(int)
        samples = []
        j = 0
        for i in which_wells:
            pi_i = pi[j]
            if pi_i > 0:
                samples.append(self.gaussians[i].sample(pi_i))
            else: pass
            j+=1
        
        samples = np.concatenate(samples, axis=0)
        if shuffle:
            inds_shuffle = np.random.choice(samples.shape[0], size=samples.shape[0], replace=False)
            return samples[inds_shuffle]
        else: return samples
    
##

class XR_MAP_toy:
    
    def __init__(self,
                 raw_data : np.ndarray):
        
        self.dim = raw_data.shape[1]
        self.data_ranges = [[raw_data[:,i].min()-1e-3, raw_data[:,i].max()+1e-3] for i in range(self.dim)]
        
        self.model_range = [-1.0, 1.0]

    def fit_to_range_(self, x, physical_range : list, forward : bool = True):
        
        # shape: x ~ (m,n) ; m = batch_size , n = number of alike variables. 
        
        # forward : physical_range -> model_range
        # else    : model_range -> physical_range
        
        x_min, x_max = physical_range
        min_model, max_model = self.model_range
        
        J = (max_model - min_model)/(x_max - x_min)
        
        if forward:
            return J*(x - x_min) + min_model , tf.cast(tf.math.log(J)*x.shape[1], tf.float32)
        else:
            return (x - min_model)/J + x_min , tf.cast(-tf.math.log(J)*x.shape[1], tf.float32)

    def forward(self, R, scale_ranges : bool = True):

        if len(R.shape) < 2: R = R.reshape(len(R), self.dim)
        else: pass
        
        ladJrx = 0
        
        X = []
        for i in range(self.dim):
            x, ladJ = self.fit_to_range_(R[:,i:i+1], physical_range=self.data_ranges[i], forward=True)
            X.append(x)
            ladJrx += ladJ
        
        X = tf.concat(X, axis=1)
        return X, ladJrx

    def inverse(self, X, flatten : bool = False):

        if len(X.shape) < 2: X = X.reshape(len(X), self.dim)
        else: pass

        ladJxr = 0
        
        R = []
        for i in range(self.dim):
            r, ladJ = self.fit_to_range_(X[:,i:i+1], physical_range=self.data_ranges[i], forward=False)
            R.append(r)
            ladJxr += ladJ
        
        R = tf.concat(R, axis=1)
        return R, ladJxr

class XR_MAP_none:
    
    def __init__(self):
        ''

    def forward(self, R, scale_ranges : bool = True):

        return R, 0.0

    def inverse(self, X, flatten : bool = False):

        return X, 0.0



