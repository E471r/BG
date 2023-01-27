import numpy as np

import tensorflow as tf

##

def reset_tf_functions_(drop_rate=0.0):
    globals()['_drop_rate_'] = drop_rate

    global foward_function_
    global inverse_function_

    global step_ML_function_
    global step_KL_function_

    @tf.function
    def foward_function_(r):
        x, ladJrx = _xr_map_.forward(r)
        z, ladJxz = _nn_.forward(x)
        log_pr = ladJrx + ladJxz + _nn_.evaluate_log_prior_(z)
        return z, log_pr

    @tf.function
    def inverse_function_(z):
        x, ladJzx = _nn_.inverse(z)
        r, ladJxr = _xr_map_.inverse(x)
        neg_log_pr = ladJzx + ladJxr - _nn_.evaluate_log_prior_(z)
        return r, neg_log_pr

    @tf.function
    def step_ML_function_(r):
        x, ladJrx = _xr_map_.forward(r)
        #one_noise = tf.random.uniform([1, x.shape[1]])*2 - 1
        #x = tf.concat([x,one_noise], axis=0)
        with tf.GradientTape() as tape:
            z, ladJxz = _nn_.forward(x, drop_rate=_drop_rate_)
            log_pr = tf.stop_gradient(ladJrx) + ladJxz + _nn_.evaluate_log_prior_(z) 
            av_ML_loss_batch = - tf.reduce_mean(log_pr)

        grads_ML_batch = tape.gradient(av_ML_loss_batch, _nn_.trainable_variables)
        return z, grads_ML_batch, av_ML_loss_batch, -log_pr
    
    @tf.function
    def step_KL_function_(z):
        x, ladJzx = _nn_.inverse(z, drop_rate=_drop_rate_)
        r, ladJxr = _xr_map_.inverse(x, True)
        neg_log_pr = ladJzx + ladJxr - _nn_.evaluate_log_prior_(z)
        return r, neg_log_pr

class TRAINER(object):
    def __init__(self,
                 _nn_, # object
                 _xr_map_, # object
                 potential_energy_function = None,
                 fixed_parameters_id : str = None,
                 custom_metric = None, 
                 ):
        # global names here are linked to the original instances defined elsewhere:
        globals()['_nn_'] = _nn_
        globals()['_xr_map_'] = _xr_map_
        self.dim_flow = _nn_.dim_flow

        self.potential_energy_function_ = potential_energy_function

        self.latent_noise_ = _nn_.sample_prior_

        self.drop_rate = 0.0
        reset_tf_functions_() # everything which can be wrapped in graph mode.

        if fixed_parameters_id is not None: self.backup_parameters_with_name_(fixed_parameters_id)
        else: self.string_identifying_fixed_parameters = None     
        self.backup_all_parameters_before_training_() # faster instead of reloading model.
        
        self.reset_optimiser_(1e-3)
        self.reset_training_logs_()
        
        self.custom_metric_ = custom_metric
        
    def BG_forward(self, r):
        # graph mode.
        return foward_function_(r)

    def BG_inverse(self, z):
        # graph mode.
        return inverse_function_(z)

    def backup_all_parameters_before_training_(self):
        self.all_parameters_before_training = []
        for i in range(_nn_.n_trainable_tensors):
            self.all_parameters_before_training.append(tf.Variable(_nn_.trainable_variables[i]))

    def reset_model(self):
        for i in range(_nn_.n_trainable_tensors):
            _nn_.trainable_variables[i].assign(self.all_parameters_before_training[i])
        self.reset_optimiser_(self.learning_rate)
        self.reset_training_logs_()
        reset_tf_functions_()
        print('everything was reset.')
        
    def backup_parameters_with_name_(self, name : str = 'shift'):
        self.keep_these_parameters_fixed = []
        self.string_identifying_fixed_parameters = name
        for i in range(_nn_.n_trainable_tensors):
            if name in _nn_.trainable_variables[i].name:
                self.keep_these_parameters_fixed.append(tf.Variable(_nn_.trainable_variables[i]))
            else: pass
    def replace_parameters_with_name_from_backup_(self, a=0):
        for i in range(_nn_.n_trainable_tensors):
            if self.string_identifying_fixed_parameters in _nn_.trainable_variables[i].name:
                _nn_.trainable_variables[i].assign(self.keep_these_parameters_fixed[a]) ; a+=1
            else: pass
        
    def reset_optimiser_(self, learning_rate : float = 0.001):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.learning_rate = learning_rate

    def no_nans_(self, gradients):
        if np.array([np.isnan(x).any() for x in gradients], dtype=int).sum() != 0:
            print('NaN was found in gradient! skipped paremeter update in this batch.')
            return False
        else: return True
        
    @property
    def blank_gradients(self):
        return [0.0 for i in range(_nn_.n_trainable_tensors)]
    
    def step_ML_(self):
        z, grads_ML_batch, av_ML_loss_batch, self.ladJ = step_ML_function_(self.r_batch)
        # logging here:
        self.hist_ML_loss.append(av_ML_loss_batch.numpy())
        return grads_ML_batch

    def reset_training_logs_(self):
        # global training metrics:
        self.training_step = 0
        self.n_skipped = 0
        # specific training histories logged here after these are initialised:
        # from self.step_ML_
        self.hist_ML_loss = [0]
        # from self.step_KL_
        self.hist_U_batch_mean = [0]
        self.hist_acc_rate = [1]
        self.hist_KL_loss = [0]
        # from self.train_batch_mixed_
        self.hist_n_skipped = [0]
        self.hist_CV_loss = [0]
        
        self.hist_u_loss = [0]
        self.hist_S_loss = [0]
        self.hist_custom_metric = [0]

        self.FE_training = []
        self.FE_validation = []
        
    def print_verbose_string(self):
        print(
                'step:', self.training_step, # int
                'ML:', np.array(self.hist_ML_loss[-1]).round(2),
                'KL:', np.array(self.hist_KL_loss[-1]).round(2),
                'u_av:', np.array(self.hist_U_batch_mean[-1]).round(2),
                'acc:', np.array(self.hist_acc_rate[-1]).round(2),
                'skips:', self.hist_n_skipped[-1], # int
                'L(cv):',self.hist_CV_loss[-1],
                'u_:',np.array(self.hist_u_loss[-1]).round(2),
                'S_:',np.array(self.hist_S_loss[-1]).round(2),
              )

    def train_MLonly_(self,
                      data_r,
                      data_U_kT,
                      store_all_info : bool = False,
                      batch_size : int = 500,
                      n_batches_per_step : int = 1,

                      learning_rate : float = 0.001,
                      drop_rate : float = 0.0,

                      n_steps : int = 1000,
                      n_print : int = 200,
                      ):

        self.w_ML, self.w_KL = [1.,0.]
        N = data_r.shape[0]

        self.batch_size = batch_size
        self.n_batches_per_step = max(1, int(n_batches_per_step))

        ####
        # ! training reset if learning_rate is changed:
        if  learning_rate != self.learning_rate:
            self.reset_optimiser_(learning_rate)
            self.reset_training_logs_()
            print('training was reset.')
        else: pass
        if drop_rate != self.drop_rate:
            reset_tf_functions_(drop_rate)
            self.drop_rate = drop_rate
            print('training functions were reset to have new drop_rate:', self.drop_rate)
        else : pass
        ###

        verbose = np.arange(0, n_steps, n_print)
        for step in range(n_steps):

            grads_step = self.blank_gradients
            F_step = 0.0
            for _ in range(self.n_batches_per_step):

                inds_random = np.random.choice(N, self.batch_size, replace=False)
                grads_batch, av_loss_batch, ladJ = step_ML_function_(tf.constant(data_r[inds_random], dtype=tf.float32))[1:]

                #grads_batch, av_loss_batch = step_ML_function_(tf.constant(data_r[inds_random], dtype=tf.float32))[1:]
                grads_step = [x + y/self.n_batches_per_step for x, y in zip(grads_step, grads_batch)]
                F_step = F_step + (data_U_kT[0][inds_random].mean() - av_loss_batch.numpy()*data_U_kT[1])/self.n_batches_per_step

                if store_all_info:
                    kT = data_U_kT[1]
                    #f = -kT*np.log(np.exp(-(data_U_kT[0][inds_random]/kT - ladJ)).mean())
                    f = np.array(data_U_kT[0][inds_random] - kT*ladJ).mean()
                    self.FE_training.append(f)

                    if len(data_U_kT) > 2:
                        inds_random = np.random.choice(data_U_kT[2].shape[0], self.batch_size, replace=True)
                        ladJ = foward_function_(data_U_kT[2][inds_random])[-1]
                        #vf = -kT*np.log(np.exp(-(data_U_kT[3][inds_random]/kT + ladJ)).mean())
                        vf = np.array(data_U_kT[3][inds_random] + kT*ladJ).mean()
                        self.FE_validation.append(vf)
                        #print('train:',f,'val:',vf)
                    else: pass

                else: pass

            if self.no_nans_(grads_step):
                self.optimizer.apply_gradients(zip(grads_step, _nn_.trainable_variables))
            else: pass
            self.hist_ML_loss.append(F_step)
            if step in verbose: self.print_verbose_string()
            else: pass
            self.training_step += 1

        return None

    def train_batch_mixed_(self):
        if self.w_ML > 0.0: grads_ML_batch = self.step_ML_()
        else: grads_ML_batch = self.blank_gradients
        if self.w_KL > 0.0: grads_KL_batch = self.step_KL_()
        else: grads_KL_batch = self.blank_gradients

        # KL training checks to allow KL loss to climb if needed but not sporadically:
        if self.w_KL > 0.0:
            if self.training_step == 0: delta_KL_loss = 0
            else: delta_KL_loss = self.hist_KL_loss[-1] - self.hist_KL_loss[-2]
            
            if not self.accept_all:
                _t = max(1.0, float(self.n_skipped)*self.update_tol)
                accept_parameter_update = delta_KL_loss <= 0 or np.random.rand(1) < np.exp( - delta_KL_loss/_t)
            else: accept_parameter_update = True

            if accept_parameter_update:
                self.n_skipped = 0
            else:
                self.n_skipped += 1
                self.hist_KL_loss = self.hist_KL_loss[:-1] + [self.hist_KL_loss[-2]] # batch didn't count.
        else:
            accept_parameter_update = True # ML-only training (i.e., weights = [1,0]) doesn't need these checks.
 
        # generic checks, and specific paramter replacements:
        if accept_parameter_update:
            grads_batch = [(x*self.w_ML + y*self.w_KL) for x, y in zip(grads_ML_batch, grads_KL_batch)]

            if self.no_nans_(grads_batch):
                self.optimizer.apply_gradients(zip(grads_batch, _nn_.trainable_variables))

                if self.string_identifying_fixed_parameters is not None: self.replace_parameters_with_name_from_backup_() # crude way to keep certain functions constant. Not used anymore.
                else: pass
                    
            else:
                pass # nan warning printed.
        else:
            pass # no warning printed, but self.n_skipped and other information is printed in verbose string.

        # logging here:
        self.hist_n_skipped.append(self.n_skipped)
        return None

    def train(self,

              weights : list = [0.5,0.5], # [ML,KL]
              data_r = None, # np.ndarray or tf.constant
              data_U_kT : list = None,

              learning_rate : float = 0.001, # ! changes here reset the optimiser.
              drop_rate : float = 0.0, # ! changes here reset the graphed training functions.

              batch_size : int = 500,
              n_oversample : int = 1,

              n_steps : int = 1000,
              n_print : int = 20,

              u_margin : float = 50.0,
              temperature_factor : float = 1.0,
            
              accept_all_KL_batches = True,
              KL_update_tol : float = 1.0,
              
              ):
        
        reset_tf_functions_()
                   
        if data_r is None and weights[0] > 0.0:
            print('Warning: no training data provided to train ML. Training KL only instead.')
            weights = [0.0,1.0]
        else: pass
            
        if data_r is not None:
            N = data_r.shape[0]
            trainign_data_given = True
        else: trainign_data_given = False

        if self.potential_energy_function_ is None and weights[1] > 0:
            print(' No potential_energy_function provided to train KL. Training ML only instead.')
            weights = [1.0,0.0]
        else: pass

        self.w_ML = weights[0]/sum(weights)
        self.w_KL = weights[1]/sum(weights)

        # ! training reset if learning_rate is changed:
        if  learning_rate != self.learning_rate:
            self.reset_optimiser_(learning_rate)
            self.reset_training_logs_()
            print('training was reset.')
        else: pass
        if drop_rate != self.drop_rate:
            reset_tf_functions_(drop_rate)
            self.drop_rate = drop_rate
            print('training functions were reset to have new drop_rate:', self.drop_rate)
        else : pass

        # things that can be paused and changed freely at any time:
        self.batch_size = batch_size
        self.n_oversample = max(1, int(n_oversample))
        self.u_margin = u_margin
        self.beta_factor = 1.0/temperature_factor
        self.accept_all = accept_all_KL_batches
        self.update_tol = KL_update_tol

        self.list_Us = []
        self.list_ladJs = []
        verbose = np.arange(0, n_steps, n_print)
        for step in range(n_steps):
            
            if trainign_data_given: 
                inds_random = np.random.choice(N, self.batch_size, replace=False)
                self.r_batch = tf.constant(data_r[inds_random], dtype=tf.float32)
            else: pass

            self.train_batch_mixed_()

            if self.w_ML > 0.0 and data_U_kT is not None: # dont use
                # F = <U> - <-T*ln_pr>
                self.hist_ML_loss[-1] = data_U_kT[0][inds_random].mean() - self.hist_ML_loss[-1]*data_U_kT[1]
                self.list_Us.append(data_U_kT[0][inds_random])
                self.list_ladJs.append(self.ladJ.numpy())
            else: pass

            if step in verbose: self.print_verbose_string()
            else: pass

            self.training_step += 1
            
        return None # logs are in self.hist_* for plotting.

    def step_KL_(self):
        ''
        grads_KL = self.blank_gradients

        U_batch_expected = 0
        acc_rate = 0
        KL_loss_expected = 0
        for _ in range(self.n_oversample):
            z = self.latent_noise_(self.batch_size)

            with tf.GradientTape() as tape:
                r_batch, ladJzr_batch = step_KL_function_(z)
                S_batch = ladJzr_batch[:,0] # (m,)
                u_batch = self.potential_energy_function_(r_batch)[:,0] * self.beta_factor
                
                Unp = np.array(u_batch)
                Snp = np.array(S_batch)
                Fnp = Unp - Snp

                inds_acc = np.where(Unp < self.u_margin)[0] ; acc_rate += (len(inds_acc)/self.batch_size) / self.n_oversample

                u_batch = tf.gather(u_batch, inds_acc)
                S_batch = tf.gather(S_batch, inds_acc)    
                # r_batch_valid = tf.gather(r_batch, inds_acc)

                Unp = np.array(u_batch)
                Snp = np.array(S_batch)
                Fnp = Unp - Snp

                #pacc = np.exp( - (Fnp[:-1,np.newaxis]-Fnp[np.newaxis,1:])/3. ).mean(1)
                #pacc = np.exp( - (Fnp[:-1]-Fnp[1:])/3. )
                
                #pacc = np.exp( - (Unp[:-1,np.newaxis]-Unp[np.newaxis,1:]).mean(1) )
                #pacc = np.exp( - (Unp[:-1]-Unp[1:]) )
                
                #inds_acc = np.where(np.random.rand(len(pacc)) < pacc)[0] ; acc_rate += (len(inds_acc)/self.batch_size) / self.n_oversample
                
                #inds_acc = np.where(Unp < 1000)[0] 

                #u_batch = tf.gather(u_batch, inds_acc)
                #S_batch = tf.gather(S_batch, inds_acc)    
                # r_batch_valid = tf.gather(r_batch, inds_acc)

                #Unp = np.array(u_batch)
                #Snp = np.array(S_batch)
                #Fnp = Unp - Snp

                '''
                U_batch_mean = Unp[np.where(Unp < self.u_margin)[0]].mean()
                U_batch_expected += U_batch_mean / self.n_oversample
                if self.training_step == 0 : self.U_batch_excepted_previous = U_batch_mean
                else: pass
                pacc = np.exp( - (Unp - (self.U_batch_excepted_previous + 40))) # heuristic margin self.u_margin
                inds_acc = np.where(np.random.rand(len(pacc)) < pacc)[0]
                '''

                u_batch_mean = tf.reduce_mean(u_batch)
                S_batch_mean = - tf.reduce_mean(S_batch)
                
                KL_batch_mean  = u_batch_mean + S_batch_mean
                
                '''
                KL_batch = (u_batch - S_batch)
                
                Es = KL_batch
                E0 = Es[0]
                E = E0
                for mc_step in range(1,len(Es)-1):
                    E_prop = Es[mc_step]
                    tf_acc = tf.cast(tf.random.uniform([1]) < np.exp(-(E_prop-E)), tf.float32)
                    E = (1.0 - tf_acc)*E + tf_acc*E_prop 
                KL_batch = - (E-E0)[0]
                '''

                KL_loss_expected += KL_batch_mean.numpy() / self.n_oversample
                
            grads_KL_batch = tape.gradient(KL_batch_mean, _nn_.trainable_variables)
            grads_KL = [x + y/self.n_oversample for x, y in zip(grads_KL, grads_KL_batch)]
        
        #self.U_batch_excepted_previous = U_batch_expected
        
        # logging here:
        #self.hist_U_batch_mean.append( U_batch_expected )
        self.hist_acc_rate.append( acc_rate )
        self.hist_KL_loss.append( KL_loss_expected )
        self.hist_CV_loss.append( 0.0 )

        self.hist_u_loss.append( u_batch_mean.numpy() )
        self.hist_S_loss.append( S_batch_mean.numpy() )
        
        if self.custom_metric_ is not None:
            self.hist_custom_metric.append(self.custom_metric_(np.array(r_batch)))
        else: pass

        return grads_KL
