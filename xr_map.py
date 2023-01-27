from rdkit import Chem

import numpy as np

import tensorflow as tf

# v2

##

pi = np.pi

##

def depth_first_search_(graph, node):
    # Reference:  https://www.educative.io/edpresso/how-to-implement-depth-first-search-in-python
    nodes = set() ; visited = set()
    def dfs(visited, graph, node):
        if node not in visited:
            nodes.add(node)
            visited.add(node)
            for neighbour in graph[node]:
                dfs(visited, graph, neighbour)
    dfs(visited, graph,node)
    return nodes 

def check_neighbours_ranks_(me, am):
    ranks = []
    for neighbour in range(len(am)):
        if am[me,neighbour] > 0:
            am_cut = np.array(am)
            am_cut[me,neighbour] = 0 ; am_cut[neighbour,me] = 0
            graph = {i: np.nonzero(row)[0].tolist() for i,row in enumerate(am_cut)}
            ranks.append(len(depth_first_search_(graph,neighbour)))
        else: ranks.append(0)
    return np.array(ranks)

def get_neighbour_lists_(am):
    DCBA = []
    for D in range(am.shape[0]):
        am_cut = np.array(am)
        C = np.argmax(am_cut[D]*check_neighbours_ranks_(D,am)) ; am_cut[C,D] = 0 ; am_cut[D,C] = 0
        B = np.argmax(am_cut[C]*check_neighbours_ranks_(C,am)) ; am_cut[B,C] = 0 ; am_cut[C,B] = 0
        A = np.argmax(am_cut[B]*check_neighbours_ranks_(B,am)) ; am_cut[A,B] = 0 ; am_cut[B,A] = 0
        DCBA.append([D,C,B,A])
    return np.array(DCBA) # (n,4)

##

def get_distance_tf_(R, inds_2_atoms):
    # R ~ (# frames, # atoms, 3)
    # inds_3_atoms ~ (3,)

    clip_low_at = 1e-6
    clip_high_at = 1e+6

    # m = R.shape[0]

    A,B = inds_2_atoms
    rA = R[:,A,:] # (m,3)
    rB = R[:,B,:] # (m,3)

    vBA = rA - rB # (m,3)

    d = tf.norm(vBA, axis=1, keepdims=True) # (m,1)
    d = tf.clip_by_value(d, clip_low_at, clip_high_at) # (m,1)

    dd_drA = vBA / d # (m,3)

    return d, dd_drA # (m,1), (m,3)

def get_angle_tf_(R, inds_3_atoms, return_all_derivatives=False):
    # R ~ (# frames, # atoms, 3)
    # inds_3_atoms ~ (3,)

    clip_low_at = 1e-6
    clip_high_at = 1e+6

    # m = R.shape[0]

    A,B,C = inds_3_atoms
    rA = R[:,A,:] # (m,3)
    rB = R[:,B,:] # (m,3)
    rC = R[:,C,:] # (m,3)

    vBA = rA - rB # (m,3)
    vBC = rC - rB # (m,3)

    NORM_vBA = tf.norm(vBA, axis=-1, keepdims=True) # (m,1)
    NORM_vBA = tf.clip_by_value(NORM_vBA, clip_low_at, clip_high_at) # (m,1)
    NORM_vBC = tf.norm(vBC, axis=-1, keepdims=True) # (m,1)
    NORM_vBC = tf.clip_by_value(NORM_vBC, clip_low_at, clip_high_at) # (m,1)

    uBA = vBA / NORM_vBA # (m,3)
    uBC = vBC / NORM_vBC # (m,3)

    dot = tf.reduce_sum(uBA*uBC, axis=1, keepdims=True) # (m,1)
    dot = tf.clip_by_value(dot, -1.0, 1.0) # (m,1)
    
    theta = tf.acos(dot) # (m,1)
    theta = tf.clip_by_value(theta, clip_low_at, 3.14159265358979323846264338327950288-clip_low_at) # (m,1)

    one_minus_x_sq = 1.0 - dot**2 # (m,1)
    one_minus_x_sq = tf.clip_by_value(one_minus_x_sq, clip_low_at, 1.0)
    dacos_dx = - 1.0 / tf.sqrt(one_minus_x_sq) # (m,1)
    dtheta_drA = dacos_dx * (uBC - uBA * tf.reduce_sum(uBA*uBC, axis=1, keepdims=True)) / NORM_vBA

    return theta, dtheta_drA # (m,1), (m,3)

def get_torsion_tf_(R, inds_4_atoms):
    # R ~ (# frames, # atoms, 3)
    # inds_4_atoms ~ (4,)
    
    clip_low_at = 1e-6
    clip_high_at = 1e+6
    
    m = R.shape[0]
    
    A,B,C,D = inds_4_atoms
    rA = R[:,A,:] # (m,3)
    rB = R[:,B,:] # (m,3)
    rC = R[:,C,:] # (m,3)
    rD = R[:,D,:] # (m,3)
    
    vBA = rA - rB # (m,3)
    vBC = rC - rB # (m,3)
    vCD = rD - rC # (m,3)
    
    NORM_vBC = tf.norm(vBC, axis=-1, keepdims=True) # (m,1)
    NORM_vBC = tf.clip_by_value(NORM_vBC, clip_low_at, clip_high_at) # (m,1)
    uBC = vBC / NORM_vBC # (m,3)
    
    w = vCD - tf.reduce_sum(vCD*uBC, axis=-1, keepdims=True)*uBC # (m,3)
    v = vBA - tf.reduce_sum(vBA*uBC, axis=-1, keepdims=True)*uBC # (m,3)
    
    uBC1 = uBC[:,0] # (m,)
    uBC2 = uBC[:,1] # (m,)
    uBC3 = uBC[:,2] # (m,)
    
    zero = tf.zeros([m,]) # (m,)
    S = tf.stack([tf.stack([ zero, uBC3,-uBC2],axis=-1),
                  tf.stack([-uBC3, zero, uBC1],axis=-1),
                  tf.stack([ uBC2,-uBC1, zero],axis=-1)],axis=-1) # (m,3,3)
    
    y = tf.einsum('ij,ijk,ik->i',w,S,v) # (m,)
    x = tf.einsum('ij,ij->i',w,v) # (m,)
    y = tf.expand_dims(y,axis=-1) # (m,1)
    x = tf.expand_dims(x,axis=-1) # (m,1)
    
    phi = tf.math.atan2(y,x) # (m,1) the torsional angle.
    
    denominator = x**2 + y**2 # (m,1)
    denominator = tf.clip_by_value(denominator, clip_low_at, clip_high_at) # (m,1)
    S_transpose_w = tf.einsum('ijk,ij->ik',S,w) # (m,3)
    ##
    # numerator = x*S_transpose_w - y*w # (m,3)
    safe = tf.eye(3)[None, :, :] - uBC[..., None] * uBC[..., None, :] # (m,3,3)  # in case second derivative.
    numerator  = x*tf.einsum('ij,ijk->ik', S_transpose_w, safe) - y*tf.einsum('ij,ijk->ik', w, safe) # (m,3)
    ##
    dphi_drA = numerator/denominator # (m,3) derivative of phi w.r.t. position vector rA.
    
    return phi, dphi_drA # (m,1), (m,3)

def NeRF_tf_(d, theta, phi, rB, rC, rD):

    # Reference: DOI 10.1002/jcc.20237
    
    # xA = [d,theta,phi]
    
    # d : (m,)
    # theta : (m,)
    # phi : (m,)
    
    # rB : (m,3)
    # rD : (m,3)
    # rD : (m,3)
    
    #clip_low_at = 1e-6 # not needed here, faster without.
    #clip_high_at = 1e+6 # not needed here, faster without.
    
    m = d.shape[0]

    vCB = rB-rC # (m,3)
    vDC = rC-rD # (m,3)

    NORM_vCB = tf.norm(vCB ,axis=1, keepdims=True) # (m,1)
    #NORM_vCB = tf.clip_by_value(NORM_vCB, clip_low_at, clip_high_at) # (m,1)
    NORM_vDC = tf.norm(vDC, axis=1, keepdims=True) # (m,1)
    #NORM_vDC = tf.clip_by_value(NORM_vDC, clip_low_at, clip_high_at) # (m,1)
    
    uCB = vCB / NORM_vCB # (m,3)
    uDC = vDC / NORM_vDC # (m,3)
    
    nv = tf.linalg.cross(uDC,uCB) ; nv /= tf.norm(nv, axis=1, keepdims=True) # (m,3)
    
    M = tf.stack([ uCB, tf.linalg.cross(nv,uCB), nv ], axis=2) # (m,3,3)

    the = pi - theta # (m,)
    c_t = tf.cos(the) ; s_t = tf.sin(the) # (m,)
    c_p = tf.cos(phi) ; s_p = tf.sin(phi) # (m,)

    v = tf.stack([ d*c_t, d*s_t*c_p, d*s_t*s_p ], axis=1) # (m,3) # spherical vector.

    rA = rB + tf.einsum('oij,oj->oi', M, v) # (m,3)
    
    zero = tf.zeros([m]) # (m,)
    partials = tf.stack([tf.stack([  c_t,      d*s_t,      zero      ], axis=-1), # (m,3)
                         tf.stack([  s_t*c_p, -d*c_t*c_p, -d*s_t*s_p ], axis=-1), # (m,3)
                         tf.stack([  s_t*s_p, -d*c_t*s_p,  d*s_t*c_p ], axis=-1), # (m,3)
                        ], axis=-1) # (m,3,3) # stacks as transpose of what it looks like.
    
    jacobian_drA_dxA = tf.einsum('oij,okj->oik', partials, M) # (m,3,3) # partials.dot(M.T)

    return rA, jacobian_drA_dxA # (m,3), (m,3,3)

##

def ladJ_from_list_of_jacobians_tf_(jacobians):
    # m = number of frames
    # jacobians : list ~ [(m,3,3),(m,3,3),...] of length n_IC

    Js = tf.stack(jacobians, axis=1) # (m, n_IC, 3, 3)

    detJs = tf.reduce_sum( tf.linalg.cross(Js[...,0], Js[...,1]) * Js[...,2] , axis=2) # (m, n_IC)

    log_abs_detJs = tf.math.log( tf.abs( detJs ) ) # (m, n_IC)

    ladJ = tf.reduce_sum(log_abs_detJs, axis=1, keepdims=True) # (m,1)

    return ladJ # (m,1)

def R_to_IC_tf_(R, all_or_some_rows_from_ABCD):
    # R ~ (m, n, 3)
    # all_or_some_rows_from_ABCD ~ (n_IC, 4) # list or array

    IC0 = [] # bonds
    IC1 = [] # angles
    IC2 = [] # torsions
    list_of_jacobians = []
    
    for inds_4_atoms in all_or_some_rows_from_ABCD:
        b, db_drA = get_distance_tf_( R, inds_4_atoms[:2] ) # (m,1), (m,3)
        a, da_drA = get_angle_tf_(    R, inds_4_atoms[:3] ) # (m,1), (m,3)
        t, dt_drA = get_torsion_tf_(  R, inds_4_atoms[:4] ) # (m,1), (m,3)

        IC0.append(b)
        IC1.append(a)
        IC2.append(t)

        jacobian = tf.stack([db_drA, da_drA, dt_drA], axis=-1) # (m,3,3)
        list_of_jacobians.append(jacobian)

    IC = tf.concat([IC0,IC1,IC2],axis=-1) # (nIC,m,3)
    IC = tf.einsum('kij->ikj', IC)  # (m,nIC,3)

    ladJ = ladJ_from_list_of_jacobians_tf_(list_of_jacobians)

    return IC, ladJ # (m,nIC,3), (m,1)
    
def IC_to_R_from_origin_tf_(X,
                            ABCD,
                            reconstriction_sequence,
                            index_of_starting_distance,
                            ):

    # X ~ (m,n,3) = (# frame, # atoms, 3 IC variable i.e., [b,a,t])
    # ABCD ~ (n,4)
    # reconstriction_sequence : order in which ABCD is used build molecule
    # index_of_starting_distance ~ int 

    m = X.shape[0]
    distances = X[:,:,0] # (m,n)
    angles = X[:,:,1]    # (m,n)
    torsions = X[:,:,2]  # (m,n)

    R = [] ; list_of_jacobians = []
    
    A, B, C, D = ABCD[reconstriction_sequence[0]]

    dCB = distances[:, index_of_starting_distance:index_of_starting_distance + 1] # (m,1)
    
    rB = tf.zeros( [m, 3] )                         # (m,3)
    rC = tf.concat( [dCB, tf.zeros((m,2))], axis=1) # (m,3) # origin
    rD = rC + tf.constant( [[0.,10.,0.]] )          # (m,3)
    
    R.append(rB) ; R.append(rC) ; permutation = [B,C]
    
    rA, JA = NeRF_tf_(distances[:,A], # (m,)
                      angles[:,A],    # (m,)
                      torsions[:,A],  # (m,)
                      rB,             # (m,3)
                      rC,             # (m,3)
                      rD,             # (m,3)
                      )               # outputs ~ (m,3), (m,3,3)

    R.append(rA) ; permutation.append(A) ; list_of_jacobians.append(JA)
    
    for A in reconstriction_sequence[1:]:
        A, B, C, D = ABCD[A]

        rB = tf.gather(R, tf.where(tf.equal(permutation, B))[0][0]) # (m,3)
        rC = tf.gather(R, tf.where(tf.equal(permutation, C))[0][0]) # (m,3)
        rD = tf.gather(R, tf.where(tf.equal(permutation, D))[0][0]) # (m,3)
        
        rA, JA = NeRF_tf_(distances[:,A], # (m,)
                          angles[:,A],     # (m,)
                          torsions[:,A],   # (m,)
                          rB,              # (m,3)
                          rC,              # (m,3)
                          rD,              # (m,3)
                          )                # outputs ~ (m,3), (m,3,3)

        R.append(rA) ; permutation.append(A) ; list_of_jacobians.append(JA)

    R = tf.stack(R)                                   # (n, m, 3)
    R = tf.gather(R, tf.argsort(permutation), axis=0) # (n, m, 3)
    R = tf.einsum('ijk->jik', R)                      # (m, n, 3)

    ladJ = ladJ_from_list_of_jacobians_tf_(list_of_jacobians) # (m,1)

    return R, ladJ # (m,n,3), (m,1)

def IC_to_R_from_Rseeds_tf_(X,
                            R_seeds,
                            ABCD,
                            inds_IC,
                            inds_XYZ,
                            reconstriction_sequence,
                            ):   
    # n = n_IC + n_XYZ 
    # X ~ (m, n_IC, 3)
    # R_seeds ~ (m, n_XYZ, 3)
    # ABCD ~ (n,4)
    # inds_IC ~ (n_IC,)
    # inds_XYZ ~ (n_XYZ,)
    # reconstriction_sequence ~ (n_IC,)

    n_XYZ = R_seeds.shape[1]

    distances = X[:,:,0] # (m,n_IC)
    angles = X[:,:,1]    # (m,n_IC)
    torsions = X[:,:,2]  # (m,n_IC)

    R = [R_seeds[:,i,:] for i in range(n_XYZ)] ; permutation = list(inds_XYZ)

    list_of_jacobians = []

    for A in reconstriction_sequence:
        A, B, C, D = ABCD[A]

        _A = tf.where(tf.equal(inds_IC, A))[0][0]

        rB = tf.gather(R, tf.where(tf.equal(permutation, B))[0][0]) # (m,3)
        rC = tf.gather(R, tf.where(tf.equal(permutation, C))[0][0]) # (m,3)
        rD = tf.gather(R, tf.where(tf.equal(permutation, D))[0][0])   # (m,3) 

        rA, JA = NeRF_tf_(distances[:,_A], angles[:,_A], torsions[:,_A], rB, rC, rD)
        R.append(rA) ; permutation.append(A) ; list_of_jacobians.append(JA)

    R = tf.stack(R)                                   # (n, m, 3)
    R = tf.gather(R, tf.argsort(permutation), axis=0) # (n, m, 3)
    R = tf.einsum('ijk->jik', R)                      # (m, n, 3)

    ladJ = ladJ_from_list_of_jacobians_tf_(list_of_jacobians) # (m,1)

    return R, ladJ # (m,n,3), (m,1)

##

def fit_to_range_tf_(x,
                     current_range : list, # ~ [2,]
                     target_range : list,  # ~ [2,]
                     ):
    
    # x ~ (m,n) ; m = batch_size , n = number of alike variables. 

    n = x.shape[1]

    x_min, x_max = current_range

    y_min, y_max = target_range
    
    J = (y_max - y_min)/(x_max - x_min) # (,) scalar
    
    y = J*(x - x_min) + y_min

    ladJ = tf.cast( n * tf.math.log(J), tf.float32) # (,) scalar

    return y, ladJ # (m,n), (,)

def try_get_conditioned_reconstriction_sequence_(inds_placed,  # list
                                                 ABCD,
                                                 verbose : bool = False):

    # inds_placed : [A,B,C] from ABCD[A], in case of full IC trasformation.
    #               inds_XYZ, in case of tartial (mixed) IC trasformation.
    # ABCD ~ (n,4) ; n = # atoms 

    n = ABCD.shape[0]

    placed_set = set(inds_placed)
    not_placed_set = list( set(np.arange(n)) - placed_set )

    reconstriction_sequence = []
    while len(placed_set) != n:
        placed_something = False
        for A in not_placed_set:
            if A not in placed_set:
                BCD = ABCD[A][1:] 
                if set(BCD).issubset(placed_set):
                    reconstriction_sequence.append(A)
                    placed_set |= set([A])
                    placed_something = True
        if placed_something is False:
            break
    if placed_something:
        if verbose: print('These indeces work. All atoms can be reached!')
        else: pass
        return reconstriction_sequence
    else: 
        if verbose: print('!! Invalid indeces proved (some atoms can not be reached).')
        else: pass    
        return None

###

import networkx as nx

def make_snakes(am, head):
    G = nx.from_numpy_matrix(am)
    L = 4
    result = []
    for paths in (nx.all_simple_paths(G, head, target, L) for target in G.nodes()):
        result+=paths    
        
    return [x for x in result if len(x)==4]


def get_ABCD_v3_(am):
    am = np.array(am)
    n_atoms = len(am)
    all_snakes = []

    for i in range(n_atoms):
        ith_snakes = make_snakes(am, i)
        all_snakes += (ith_snakes)

    str_bodies = [str(x[1:3]) for x in all_snakes]
    unique_bodies = list(set(str_bodies))
    
    inds = np.arange(len(am))
    che = []

    a = 0
    while len(che) < n_atoms:
        try:
            for i in range(len(unique_bodies)):
                for j in range(len(str_bodies)):
                    if unique_bodies[i] in str_bodies[j] and all_snakes[j][0] == inds[a]:
                        a+=1
                        che.append(all_snakes[j])
                    else: pass
        except: break
            
    return che

###

class XR_MAP:

    def automatic_search_for_stA_and_stBC_(self):
        yes = False
        for A in range(self.n):
            B, C =  self.ABCD[A,[1,2]]
            for i in range(self.n):
                if set([B,C]).issubset(self.ABCD[i,:2]):
                    yes = True
                    break
                else: pass
            if yes:
                reconstriction_sequence = try_get_conditioned_reconstriction_sequence_(self.ABCD[A][:3], self.ABCD)
                if reconstriction_sequence is not None: break
                else: pass
            else: pass

        if reconstriction_sequence is not None:
            self.reconstriction_sequence = [A] + reconstriction_sequence 
            self.stBC = i
            self.stA = A  # only for information, not used later.
        else: print('\n !!! : There was a problem.')

    @property
    def what_is_ABCD(self):
        print('The z-matrix (indices) can be viewed under self.ABCD')
        print('The numbers in self.ABCD are illustrated in the cartoon self.mol')
        print('self.mol can also be viewed in 3D using .. ')

##

    def __init__(self,
                PDB, # of a single molecule in vacuum.
                ):
        """
        1.0    <-- excecution order
        """
        self.model_range = [-1.0, 1.0]

        # m : brach_size
        # n : # atoms

        # n_IC : # atoms represetned using internal coodiantes (IC)
        # inds_IC : indices of atoms which are epresetned using IC

        # n_XYZ : # atoms represetned using Cartesian coordiantes (XYZ)
        # inds_XYZ : indices of atoms which are epresetned using XYZ
        
        # n = n_IC + n_XYZ

        # n_H : number of hydrogen atoms
        # inds_H : indices of atoms which are hydrogens

        # ABCD : (n,4) arrays of indices
            # first 2 inds in each row are for bond
            # first 3 inds in each row are for angle
            # all 4 inds in each row are for torsion
        # mol : rdkit object 
        
        self.mol = Chem.MolFromPDBFile(PDB, removeHs = False) # obj
        for i, a in enumerate(self.mol.GetAtoms()): a.SetAtomMapNum(i)

        # hydrogen atoms:
        self.masses = np.array([x.GetMass() for x in self.mol.GetAtoms()]) # (n,)
        self.inds_H = np.where(self.masses<1.009)[0] # (n_H,)
        self.n_H = self.inds_H.shape[0] # (,)

        self.AdjacencyMatrix = Chem.rdmolops.GetAdjacencyMatrix( self.mol ) # (n,n)
        self.n = self.AdjacencyMatrix.shape[0] # (,)

        self.ABCD = get_neighbour_lists_(self.AdjacencyMatrix) # (n,4)
        
        #self.ABCD = np.array(get_ABCD_v3_(self.AdjacencyMatrix))

        self.automatic_search_for_stA_and_stBC_() # defines: stA, stBC, reconstriction_sequence 

        self.n_IC = int(self.n)
        self.inds_IC = np.arange(self.n)
        self.n_XYZ = 0
        self.inds_XYZ = None

        self._set_IC_ranges_general() # defines: IC_ranges ~ (n,3,2)
        self.XYZ_ranges = None
        self._set_IC_mask()

        self.H_bonds_are_constant = False

    def _set_IC_ranges_general(self,
                               range_bond = [0.0, 4.0],
                               range_angle = [0.0, pi],
                              ):
        """
        1.5    <-- excecution order
            optional if to set different general ranges (mindful of A or nm of data in range_bond)
        """
        self.IC_ranges = np.array([[range_bond, range_angle, [-pi, pi]]]*self.n) # (n,3,2)
        self.IC_ranges_are_from_data = False

    def _show_this_object_some_data(self,
                                    R_full_dataset, # (N,n,3) 
                                    make_IC_ranges_come_from_data = True,
                                    pad : float = 1e-5,
                                    ):
        """
        2.0    <-- excecution order
            needed if set_which_atoms_to_be_kept_cartisian(**inds**) is used
                if that is used input here needs to be already alligned to the **inds** atoms.
            needed if set_IC_mask( make_H_bonds_constant = True) is ran
        """
        R = np.array(R_full_dataset)

        self.COMs = R.mean(0) # (n,3)
        self.XYZ_ranges = np.stack([R.min(0),R.max(0)], axis=-1) # (n,3,2)

        IC = R_to_IC_tf_(tf.constant(R, dtype=tf.float32), self.ABCD)[0].numpy()

        IC_ranges = np.stack([IC.min(0),IC.max(0)], axis=-1) # (n,3,2)
        
        IC_ranges_before_padding = IC_ranges[:,:,1] - IC_ranges[:,:,0]
        self.XYZ_ranges_before_padding = self.XYZ_ranges[:,:,1] - self.XYZ_ranges[:,:,0]

        # making these ranges safe:
        if pad is not None:
            eps = 2*pad
            for i in range(self.n):
                for j in range(2):
                    Min, Max  = IC_ranges[i,j] ; Range = Max - Min
                    if Range < eps:
                        IC_ranges[i,j,0] -= pad
                        IC_ranges[i,j,1] += pad
                    else: pass

            for i in range(self.n):
                for j in range(3):
                    Min, Max  = self.XYZ_ranges[i,j] ; Range = Max - Min
                    if Range < eps:
                        self.XYZ_ranges[i,j,0] -= pad
                        self.XYZ_ranges[i,j,1] += pad
                    else: pass
        else: pass
        
        IC_ranges = np.concatenate([IC_ranges[:,:2,:], self.IC_ranges[:,-1:,:]], axis=1)
        IC_ranges_after_padding = IC_ranges[:,:,1] - IC_ranges[:,:,0]
        self.XYZ_ranges_after_padding = self.XYZ_ranges[:,:,1] - self.XYZ_ranges[:,:,0]   

        # self.H_bonds = IC[:,self.inds_H,0].mean(0) # (n_H,)
        self.H_bonds = self.scale_X_tf_(IC, type=0, forward = True)[0].numpy()[:,self.inds_H,0].mean(0) # (n_H,)

        if make_IC_ranges_come_from_data:
            self.IC_ranges = IC_ranges
            self.H_bonds = self.scale_X_tf_(IC, type=0, forward = True)[0].numpy()[:,self.inds_H,0].mean(0) # (n_H,)
            self.IC_ranges_are_from_data = True
            self.IC_ranges_before_padding = IC_ranges_before_padding
            self.IC_ranges_after_padding = IC_ranges_after_padding
        else: pass

        # summary of what is take from the data:
        # self.XYZ_ranges
        # self.IC_ranges
        # self.H_bonds # meaning distances along the bond which connects H to other atom.
        ''

    def _set_which_atoms_to_be_kept_cartisian(self, inds_keep_these_atoms_cartesian : list):
        """
        2.5    <-- excecution order
            optional : requires _show_this_object_some_data to be ran.
        """
        inds_keep_these_atoms_cartesian = sorted(inds_keep_these_atoms_cartesian)

        reconstriction_sequence = try_get_conditioned_reconstriction_sequence_(inds_keep_these_atoms_cartesian, 
                                                                               ABCD = self.ABCD,
                                                                               verbose = True)
        if reconstriction_sequence is not None:
            self.inds_XYZ = np.array(inds_keep_these_atoms_cartesian).flatten()
            self.n_XYZ = self.inds_XYZ.shape[0]
            self.inds_IC = np.array(list( set(np.arange(self.n)) - set(self.inds_XYZ) )).flatten()
            self.n_IC = self.inds_IC.shape[0]
            self.reconstriction_sequence= reconstriction_sequence
            self._set_IC_mask()
            if self.XYZ_ranges is None:
                print('next please run _show_this_object_some_data(..) before using')
            else: pass
        else:
            print('Please try a different set.')
            return None

    def _set_IC_mask(self, make_H_bonds_constant = False):
        """
        3.0    <-- excecution order
            optional if it to make_H_bonds_constant = True
        """
        mask = np.ones([self.n,3]).astype(np.int32) # (n,3)
        if self.n_XYZ == 0:
            A,B,C,D = self.ABCD[self.stA]
            mask[C]       = -1  # the origin                     # - 3 d.o.f. 
            mask[B,[1,2]] = -1  # only need to know the bond [0] # - 2 d.o.f. 
            mask[D,2]     =  2  # torsion from,                  
            mask[A,2]     = -2  # to.                            # - 1 d.o.f. 
        else:
            mask[self.inds_XYZ] = -10
        if make_H_bonds_constant:
            mask[self.inds_H,0] = -3                             # - n_H d.o.f.
            self.H_bonds_are_constant = True
            # therefore user can't select inds_XYZ which are hydrogens.
        else: self.H_bonds_are_constant = False

        self.inforamtion_mask = mask

        mask = mask[self.inds_IC] # -10s removed 

        self.inds_ic_flow = np.where(mask.flatten()>0)[0]
        self.n_ic_flow = self.inds_ic_flow.shape[0]
        self.dim_flow = self.n_ic_flow + self.n_XYZ*3

        if self.n_XYZ == 0:
            self.torsion_donor_index = int(np.where(mask.flatten()[self.inds_ic_flow]==2)[0])
        else: pass 

        self.IC_mask = mask # (n_IC,3)

    @property
    def periodic_mask(self,):
        mask_ics =  np.array([[0,0,1]]*self.n).flatten()[self.inds_ic_flow]
        periodic_mask = np.zeros([self.dim_flow,])
        periodic_mask[:self.n_ic_flow] = mask_ics
        return periodic_mask

##

    def scale_X_tf_(self, X, type=0, forward = True):
        # X ~ (m, n_type, 3)

        if type == 0:
            _range = self.IC_ranges
            _inds = self.inds_IC
            _n = self.n_IC # = n_type
        else:
            _range = self.XYZ_ranges
            _inds = self.inds_XYZ
            _n = self.n_XYZ # = n_type

        X_scaled = [] ; ladJ = 0.0

        if forward:
            for i in range(_n):
                k = _inds[i]
                xi = []
                for j in range(3):
                    x, s = fit_to_range_tf_(X[:,i:i+1,j],
                                            current_range=_range[k,j],
                                            target_range=self.model_range) 
                    xi.append(x) ; ladJ += s
                X_scaled.append(tf.stack(xi, axis=-1))
        else:
            for i in range(_n):
                k = _inds[i]
                xi = []
                for j in range(3):
                    x, s = fit_to_range_tf_(X[:,i:i+1,j], # (m,1)
                                            current_range=self.model_range,
                                            target_range=_range[k,j]) 
                    xi.append(x) ; ladJ += s
                X_scaled.append(tf.stack(xi, axis=-1)) # (m,1,3)

        X_scaled = tf.concat(X_scaled, axis=1) # (m,n_type,3)

        return X_scaled, ladJ # (m,n_type,3), (,)

    def forward_IC_mask_tf_(self, ICs):
        # ICs ~ (m,n_IC,3)

        X_ICs = []

        for i in range(self.n_IC):
            for j in range(3):
                x = ICs[:,i,j] # (m,)
                if self.IC_mask[i,j] > 0:
                    X_ICs.append(x)
                else: pass

        X_ICs = tf.stack(X_ICs, axis=1) # (m,n_ic_flow)
        return X_ICs # (m,n_ic_flow)

    def invserse_IC_mask_tf_(self, X_ICs):
        # X_ICs ~ (m,n_ic_flow)

        zero = tf.zeros([X_ICs.shape[0],]) # (m,) # for 6 d.o.f. which are -1 or -2 
        if self.n_XYZ == 0: torsion = X_ICs[:,self.torsion_donor_index] # (m,1)
        else: torsion = None
        
        ICs = [] ; a = 0 ; b = 0
        for i in range(self.n_IC):
            ICs_i = []
            for j in range(3):
                q = self.IC_mask[i,j]
                if q > 0 : x = X_ICs[:,a] ; ICs_i.append(x) ; a += 1 
                if q ==-3 : ICs_i.append(zero + self.H_bonds[b]) ; b +=1   
                if q ==-2 : ICs_i.append(torsion)
                if q ==-1 : ICs_i.append(zero)
            ICs_i = tf.stack(ICs_i, axis=-1) # (m,3)
            ICs.append(ICs_i)
                
        ICs = tf.stack(ICs, axis=1) # (m,n_IC,3)

        return ICs # (m,n_IC,3)

    def forward(self, R):
        # R ~ (m,n,3)
        m = R.shape[0] ; ladJ_rx = 0.0

        if self.n_XYZ == 0:
            IC, ladJ = R_to_IC_tf_(R, self.ABCD[self.inds_IC]) ; ladJ_rx += ladJ
            ICs, ladJ = self.scale_X_tf_(IC, type=0, forward = True) ; ladJ_rx += ladJ
            X = self.forward_IC_mask_tf_(ICs) # (m,n_ic_flow)
            
        else:
            IC, ladJ = R_to_IC_tf_(R, self.ABCD[self.inds_IC]) ; ladJ_rx += ladJ
            ICs, ladJ = self.scale_X_tf_(IC, type=0, forward = True) ; ladJ_rx += ladJ
            X_ICs = self.forward_IC_mask_tf_(ICs) # (m,n_ic_flow)

            XYZ = tf.gather(R, self.inds_XYZ, axis=1)
            XYZs, ladJ = self.scale_X_tf_(XYZ, type=1, forward = True) ; ladJ_rx += ladJ
            X_XYZs = tf.reshape(XYZs, [m, self.n_XYZ*3])
            X = tf.concat([X_ICs, X_XYZs], axis=1)

        return X, ladJ_rx # (m, n_ic_flow +/- (n_XYZ*3) ), (m,1)

    def inverse(self, X, flatten_output_coordinates=False):
        # X ~(m, n_ic_flow +/- (n_XYZ*3) )
        m = X.shape[0] ; ladJ_xr = 0.0

        if self.n_XYZ == 0:
            ICs = self.invserse_IC_mask_tf_(X) # (m,n_IC,3), here = (m,n,3)
            IC, ladJ = self.scale_X_tf_(ICs, type=0, forward = False) ; ladJ_xr += ladJ
            R, ladJ = IC_to_R_from_origin_tf_(IC,
                                              ABCD = self.ABCD,
                                              reconstriction_sequence = self.reconstriction_sequence,
                                              index_of_starting_distance = self.stBC,
                                              ) ; ladJ_xr += ladJ

        else:
            X_XYZs = X[:,-self.n_XYZ*3:] # (m,n_XYZ*3)
            X_ICs  = X[:,:-self.n_XYZ*3] # (m,n_ic_flow)

            XYZs = tf.reshape(X_XYZs, [m, self.n_XYZ, 3]) # (m,n_XYZ,3)
            XYZ, ladJ = self.scale_X_tf_(XYZs, type=1, forward = False) ; ladJ_xr += ladJ

            ICs = self.invserse_IC_mask_tf_(X_ICs) # (m,n_IC,3)
            IC, ladJ = self.scale_X_tf_(ICs, type=0, forward = False) ; ladJ_xr += ladJ

            R, ladJ = IC_to_R_from_Rseeds_tf_(IC,
                                              R_seeds = XYZ,
                                              ABCD = self.ABCD,
                                              inds_IC = self.inds_IC,
                                              inds_XYZ = self.inds_XYZ,
                                              reconstriction_sequence = self.reconstriction_sequence,
                                              ) ; ladJ_xr += ladJ
            #  # (m,n,3), (m,1)

        if flatten_output_coordinates:
            return tf.reshape(R,[m,self.n*3]), ladJ_xr  # (m,n*3), (m,1)
        else:
            return R, ladJ_xr #  # (m,n,3), (m,1)
