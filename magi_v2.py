import numpy as np
from scipy.special import kvp, gamma
from sklearn.model_selection import KFold
from scipy.interpolate import splrep, splev
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from typing import Union, Callable
import time
from tqdm.autonotebook import tqdm


# main class for TFP-powered Manifold-Constrained Gaussian Processes Inference (PNAS, Yang, Wong, & Kou 2020).
# source: https://www.pnas.org/doi/10.1073/pnas.2020397118
class MAGI_v2:
    
    '''
    Inputs:
    1. D_thetas: number of parameters governing our system.
    2. ts_obs: timesteps in our data
    3. X_obs: matrix of observed + missing values in our data.
    4. bandsize: are we doing band-matrix approximations? If None, then no. Else, positive integer bandsize.
    5. f_vec: batchable function. Given timesteps t, observations X, & parameters theta, return dX/dt at said timesteps.
    '''
    
    # constructor - let's encode the ODE information
    def __init__(self, D_thetas : int, ts_obs: np.ndarray, X_obs : np.ndarray, 
                 bandsize : Union[int, None], f_vec: Callable):
        
        # variables from constructor
        self.D_thetas = D_thetas
        self.BANDSIZE = bandsize
        
        # our data (not accounting for any forecasting)
        self.ts_obs = ts_obs
        self.X_obs = X_obs
        self.N, self.D = self.X_obs.shape
        
        # parameters governing observed vs. completely-unobserved components
        self.observed_indicators = (~np.isnan(X_obs)).mean(axis=0) > 0
        self.observed_components = np.arange(self.D)[self.observed_indicators]
        self.D_observed = len(self.observed_components)
        self.unobserved_components = np.setdiff1d(np.arange(self.D), self.observed_components)
        self.D_unobserved = len(self.unobserved_components)
        self.proper_order = np.argsort(np.concatenate([self.observed_components, self.unobserved_components]))
        
        # let's also observe how many not-NaN entries do we have in our data x_obs?
        self.N_ds = (~tf.math.is_nan(self.X_obs).numpy()).sum(axis=0)
        
        # placeholder variables to be filled when fitting
        self.I, self.X_obs_discret = None, None
        self.beta, self.mag_I = None, None
        self.not_nan_idxs, self.not_nan_cols = None, None
        self.y_tau_ds_observed = None
        self.X_interp_obs, self.X_interp_unobs = None, None
        
        # full set of kernel hyperparameters + initial values for sampling
        self.phi1s = np.full(shape=(self.D,), fill_value=np.nan)
        self.phi2s = np.full(shape=(self.D,), fill_value=np.nan)
        self.sigma_sqs_init = np.full(shape=(self.D,), fill_value=np.nan)
        self.Xhat_init, self.thetas_init = None, None
        self.mu_ds = np.full(shape=(self.D,), fill_value=np.nan)
        
        # 3x |I| x |I| x D tensor-arrays to store kernel-based matrices for GP.
        self.C_d_invs, self.m_ds, self.K_d_invs = None, None, None
        
        # what is the ODE-system governing our data?
        self.f_vec = f_vec
        
        
    '''
    Notes: 
    1. We will automatically fit the phi1, phi2, and sigma_sq_init hyperparameters.
    2. User can overwrite the fitted values with exogenous values on their own.
    '''
    # function for modifying state variables + progressing with the fitting process.
    def initial_fit(self, discretization : int, verbose=False, use_fourier_prior=True, phi_exo=None):
        self.use_fourier_prior = use_fourier_prior
        
        # discretize our data
        self.I, self.X_obs_discret = self._discretize(self.ts_obs, self.X_obs, discretization)
        
        # compute |I| and beta (used for tempering prior vs. likelihood)
        self.mag_I = self.I.shape[0]
        self.beta = tf.cast((self.D * self.mag_I) / self.N_ds.sum(), tf.float64)
        
        '''
        Record where our NaNs are in the X_obs_discret in XLA-compatible format.
        - Idea is to be ready for masking operations that don't violate tf.function + XLA.
        '''
        # flattened indices of where X_obs_discret is NOT NAN, also what columns they correspond to
        self.not_nan_idxs = tf.convert_to_tensor(np.where(~np.isnan(self.X_obs_discret).flatten())[0])
        self.not_nan_cols = self.not_nan_idxs % self.D

        # get flattened entries of not-NaN values in y_tau_ds = X_obs_discret (following PNAS paper notation)
        self.y_tau_ds_observed = tf.gather(tf.reshape(self.X_obs_discret, [-1]), self.not_nan_idxs)
        
        #### FITTING KERNEL HYPERPARAMETERS FOR OBSERVED COMPONENTS
        
        # interpolate our fully/partially-observed components + fit (phi1, phi2, sigma_sq)
        self.X_interp_obs = self._linear_interpolate(self.X_obs_discret[:, self.observed_indicators])
        if phi_exo is None:
            hparams_obs = self._fit_kernel_hparams(I=self.I, X_filled=self.X_interp_obs, verbose=verbose)
        else:
            hparams_obs = {
                'phi1s': phi_exo['phi1s'][self.observed_indicators],
                'phi2s': phi_exo['phi2s'][self.observed_indicators],
                'sigma_sqs': phi_exo['sigma_sqs'][self.observed_indicators]
            }

        # populate our hparams + initialize Xhat for the observed components
        self.phi1s[self.observed_indicators] = hparams_obs["phi1s"]
        self.phi2s[self.observed_indicators] = hparams_obs["phi2s"]
        self.sigma_sqs_init[self.observed_indicators] = hparams_obs["sigma_sqs"]
        self.Xhat_init = self.X_obs_discret.copy()
        self.Xhat_init[:,self.observed_indicators] = self.X_interp_obs
        self.mu_ds[self.observed_indicators] = self.X_interp_obs.mean(axis=0)
        
        # construct our matrices for GP kernel
        self.C_d_invs = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        self.m_ds = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        self.K_d_invs = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        
        # populate for observed components
        for i, d in enumerate(self.observed_components):

            # Eqn. 6 of PNAS paper
            C_d, m_d, K_d = self._build_matrices(self.I, hparams_obs["phi1s"][i], hparams_obs["phi2s"][i], v=2.01)
            self.C_d_invs[d] = tf.linalg.pinv(C_d) # pinv equal to inv if invertible!
            self.m_ds[d] = m_d
            self.K_d_invs[d] = tf.linalg.pinv(K_d)
        
        #### FIT THETAS_INIT IF ALL COMPONENTS OBSERVED, ELSE POINT-ESTIMATE (UNOBSERVED COMPONENTS, THETA) JOINTLY
        
        # if all components observed, fit thetas_init via ADAM.
        if np.all(self.observed_indicators):
            
            # initialize starting theta based on optimizing routine
            thetas_var = tf.Variable(initial_value=np.ones(shape=(self.D_thetas,)), name="thetas_var", dtype=np.float64)
            
            # pre-compute our centered differences + reshape as needed for below function.
            X_cent = tf.reshape(self.Xhat_init - self.mu_ds, 
                                shape=(self.Xhat_init.shape[0], 1, self.Xhat_init.shape[1]))
            m_ds_prod_X_cent = self.m_ds @ tf.transpose(X_cent, perm=[2, 0, 1])
            
            # create local version of f_vec, and assorted 
            f_vec_local, Xhat_init_local, I_local, K_d_invs_local = self.f_vec, self.Xhat_init, self.I, self.K_d_invs
            
            # loss function - log posterior as a function of ONLY THETAS! (want to MINIMIZE!)
            def theta_objective(thetas):
                '''
                See unnormalized_log_prob for the full posterior + input variables.
                - We can ignore the terms that don't involve theta! (i.e. first, third, and fourth terms)
                - We just need to deal with the 2nd term, which is the only place where theta appears!
                '''
                # second term: ||f_{d, I} - \mudot_d(I) - m_d{ x_d(I) - \mu_d(I) }||_{K_d^-1}^2
                f_vals = tf.reshape(f_vec_local(I_local, Xhat_init_local, thetas), 
                                    shape=(Xhat_init_local.shape[1], Xhat_init_local.shape[0], 1))
                toNorm = f_vals - m_ds_prod_X_cent
                return tf.reduce_sum( tf.transpose(toNorm, perm=[0, 2, 1]) @ (K_d_invs_local @ toNorm) )
            
            # setup our optimizer + our one-step Adam function
            num_iters = 10000; optimizer = tf_keras.optimizers.Adam(learning_rate=.01)
            @tf.function(autograph=True, jit_compile=True)
            def fit_theta_step():
                with tf.GradientTape() as tape:
                    loss = theta_objective(thetas_var)
                grads = tape.gradient(loss, thetas_var)
                optimizer.apply_gradients([(grads, thetas_var)])
                return loss
            
            # run our parameter initialization for theta!
            if verbose:
                for i in tqdm(range(num_iters), desc="Initializing theta"):
                  loss = fit_theta_step()
            else:
                for i in range(num_iters):
                    loss = fit_theta_step()

            # extract out our thetas_init, which should make us ready for sampling!
            self.thetas_init = thetas_var.numpy().copy()
        
        # if we have missing components ...
        else:
            
            '''
            If some components unobs,
            1. Fit (thetas_init, X_unobs) jointly via gradient-matching. 
            2. Then, fit kernel hyperparameters.
            
            *11/26/2024: fix observed components at their cv-smoothed values!
            '''
            # fix the observed components at their interpolated values
            X_smoothed_obs = self.cv_cubic_smoother(self.I, self.X_interp_obs)
            Xhat_init_obs_tf = tf.convert_to_tensor(X_smoothed_obs)
            
            # create local version of f_vec, and assorted variables to ensure tf.function scope functionality.
            proper_order_local, I_local, f_vec_local = self.proper_order, self.I, self.f_vec
            
            # loss function for finite-differences-based gradient-matching.
            def unobserved_objective(X_unobs, thetas):

                # create our full implied observations by filling with X_unobs
                X_full = tf.gather(tf.concat([Xhat_init_obs_tf, X_unobs], axis=1), 
                                   indices=proper_order_local, axis=1)

                # compute implied derivatives on all components.
                f_vals = f_vec_local(I_local, X_full, thetas)

                '''
                Compute 2nd-order finite-diff approx:
                - f'(x) ~ (f(x+dx) - f(x-dx)) / (2dx) (assuming equally spaced I!)
                '''
                # can assume fixed dx stepsize looking at I
                f_diff = (X_full[2:,:] - X_full[:-2,:]) / (2*(self.I[1,0] - self.I[0,0]))

                # return L2 error of f_vals and f_diff
                return tf.reduce_sum((f_vals[1:-1] - f_diff) ** 2)
            
            # initialize our guesses + setup as autograd variables
            mu_unobs_init = self.X_interp_obs.mean() # heuristic - avg of all interpolated values in obs. components.
            sd_unobs_init = (self.X_interp_obs.std(axis=0) ** 2).mean() ** 0.5 # avg of all interpolated obs. comps. variances.
            
            # note that we are optimizing X_unobs & thetas jointly in one-pass!
            X_unobs_var = tf.Variable(initial_value=np.random.normal(loc=mu_unobs_init, 
                                                                     scale=sd_unobs_init, 
                                                                     size=(self.mag_I, self.D_unobserved)), 
                                      name="X_unobs_var", dtype=np.float64)
            thetas_var = tf.Variable(initial_value=np.ones((self.D_thetas,)), name="thetas_var", dtype=np.float64)
            
            # setup our optimizer + our one-step Adam function + run
            num_iters = 10000; optimizer = tf_keras.optimizers.Adam(learning_rate=.01)
            @tf.function(autograph=True, jit_compile=True)
            def fit_unobserved_step():
                with tf.GradientTape() as tape:
                    loss = unobserved_objective(X_unobs_var, thetas_var)
                grads = tape.gradient(loss, [X_unobs_var, thetas_var])
                optimizer.apply_gradients(zip(grads, [X_unobs_var, thetas_var]))
                return loss
            
            # run our ADAM-based optimization for initializing X_unobs and thetas.
            if verbose:
                for i in tqdm(range(num_iters), desc="Fitting X_unobs and theta"):
                  loss = fit_unobserved_step()
            else:
                for i in range(num_iters):
                    loss = fit_unobserved_step()
            
            # get our fitted values for thetas_init and X_unobs_init
            self.X_interp_unobs = X_unobs_var.numpy().copy()
            self.thetas_init = thetas_var.numpy().copy()
            
            # fit kernel + sigma_sq hparams for unobserved components. Populate our model variables.
            if phi_exo is None:
                hparams_unobs = self._fit_kernel_hparams(I=self.I, X_filled=self.X_interp_unobs, verbose=verbose)
            else:
                hparams_unobs = {
                    'phi1s': phi_exo['phi1s'][self.unobserved_components],
                    'phi2s': phi_exo['phi2s'][self.unobserved_components],
                    'sigma_sqs': phi_exo['sigma_sqs'][self.unobserved_components]
                }

            self.phi1s[self.unobserved_components] = hparams_unobs["phi1s"]
            self.phi2s[self.unobserved_components] = hparams_unobs["phi2s"]
            self.sigma_sqs_init[self.unobserved_components] = hparams_unobs["sigma_sqs"]
            self.Xhat_init[:,self.unobserved_components] = self.X_interp_unobs

            # need to update self.mu_ds for unobserved components!
            self.mu_ds[self.unobserved_components] = self.X_interp_unobs.mean(axis=0)
            
            # populate the kernel matrices for the unobserved components.
            for i, d in enumerate(self.unobserved_components):
        
                # Eqn. 6 of PNAS paper
                C_d, m_d, K_d = self._build_matrices(self.I, hparams_unobs["phi1s"][i], hparams_unobs["phi2s"][i], v=2.01)
                self.C_d_invs[d] = tf.linalg.pinv(C_d) # pinv equal to inv if invertible!
                self.m_ds[d] = m_d
                self.K_d_invs[d] = tf.linalg.pinv(K_d)
            
        # finally, create banded matrix approximations of C_d, m_d, K_d before going into sampling
        if self.BANDSIZE is not None:
            self.C_d_invs = tf.linalg.band_part(input=self.C_d_invs, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
            self.K_d_invs = tf.linalg.band_part(input=self.K_d_invs, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
            self.m_ds = tf.linalg.band_part(input=self.m_ds, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
        
        # before we are ready to proceed with sampling, let's smooth our initial values for Xhat.
        self.Xhat_init = self.cv_cubic_smoother(self.I, self.Xhat_init)
        
    
    '''
    Function for sampling from posterior distribution to perform inference:
    1. Make sure that there are no NaNs in self.{Xhat_init, thetas_init, sigma_sqs_init}.
    2. Can specify number of burn-in + actual samples.
    '''
    # note that tqdm is not permissible because violates XLA-environment + tf.function.
    def predict(self, num_results: int = 1000, num_burnin_steps: int = 1000, sigma_sqs_LB=None, tempering=False, verbose=False):
        
        # make sure we are ready to do inference (i.e., no NaNs in initializations)
        assert ~np.any(np.isnan(self.Xhat_init)), "Please make sure Xhat_init does not have NaNs."
        assert ~np.any(np.isnan(self.sigma_sqs_init)), "Please make sure sigma_sqs_init does not have NaNs."
        assert ~np.any(np.isnan(self.thetas_init)), "Please make sure thetas_init does not have NaNs."
        
        # to ensure tf.function compatibility, need to create local versions of some variables
        mu_ds, C_d_invs, f_vec, I = self.mu_ds, self.C_d_invs, self.f_vec, self.I
        m_ds, K_d_invs, N_ds, not_nan_idxs = self.m_ds, self.K_d_invs, self.N_ds, self.not_nan_idxs
        y_tau_ds_observed, not_nan_cols, beta = self.y_tau_ds_observed, self.not_nan_cols, self.beta
        
        # compute a lower bound on what sigma_sq should be for each component for numerical stability
        if sigma_sqs_LB is None:
            sigma_sqs_LB = ((self.Xhat_init.std(axis=0)) * 0.01) ** 2
        
        '''
        As of 11/25/2024, we will use optional temperature annealing to encourage more initial exploration.
        - Will also constrain sigma_sq values to be at least sigma_sq_LB using softplus bijector.
        - Will also constrain theta values to be positive for now using softplus bijector.
        '''
        # the FULL posterior distribution that we are sampling from (optimized for XLA, yay!)
        def unnormalized_log_prob(X, sigma_sqs_pre, thetas_pre, beta_temp):
            '''
            Takes in as input the following, and returns the unnormalized log-posterior
            1. Our samples of the trajectory components X with dimensions |I| x D
            2. sigma_sqs - a (|D|, ) vector of the noises on each component d.
            3. thetas - the (d_thetas,) vector-type sample of the parameters governing our system
            '''
            
            # compute what the actual sigma_sqs is, after softplus transformation
            sigma_sqs = tf.math.log(1.0 + tf.math.exp(sigma_sqs_pre)) + sigma_sqs_LB
            thetas = tf.math.log(1.0 + tf.math.exp(thetas_pre))
            
            # also need to account for the change-of-variables via log-Jacobian for both sigma^2 and thetas
            log_jacobian_sigma_sqs = tf.reduce_sum(sigma_sqs_pre - tf.math.log(1.0 + tf.math.exp(sigma_sqs_pre)))
            log_jacobian_thetas = tf.reduce_sum(thetas_pre - tf.math.log(1.0 + tf.math.exp(thetas_pre)))
            
            # need to tell TensorFlow to not track gradients on beta_temp
            beta_temp = tf.stop_gradient(beta_temp)
            
            # pre-compute our centered differences + reshape as needed
            X_cent = tf.reshape(X - mu_ds, shape=(X.shape[0], 1, X.shape[1]))

            # first term: ||x_D(I) - \mu_d(I)||_{C_d^-1}^2
            t1 = tf.reduce_sum( (tf.transpose(X_cent) @ C_d_invs) @ tf.transpose(X_cent, perm=[2, 0, 1]) )

            # second term: ||f_{d, I} - \mudot_d(I) - m_d{ x_d(I) - \mu_d(I) }||_{K_d^-1}^2
            f_vals = tf.transpose(f_vec(I, X, thetas)[:,None], perm=[2, 0, 1])
            toNorm = f_vals - (m_ds @ tf.transpose(X_cent, perm=[2, 0, 1]))
            t2 = tf.reduce_sum( tf.transpose(toNorm, perm=[0, 2, 1]) @ (K_d_invs @ toNorm) )

            # third term: N_d log(2\pi \sigma_d^2)
            t3 = tf.reduce_sum(N_ds * tf.math.log(2.0*np.pi * sigma_sqs))

            # fourth term on ONLY THE ACTUAL OBSERVED VALUES!: ||x_d(\tau_d) - y_d(\tau_d)||_{\sigma_d^{-2}}^2
            X_observed = tf.gather(tf.reshape(X, [-1]), not_nan_idxs)
            t4 = tf.reduce_sum(tf.math.multiply(tf.square(X_observed - y_tau_ds_observed), 
                                                tf.gather(1.0 / sigma_sqs, not_nan_cols)))
            
            # prior-temper using 1/beta: -0.5 * ( ((1/beta) * (t1 + t2)) + (t3 + t4) ), then annealing-tempering.
            return beta_temp * ( -0.5 * ( ((1.0 / beta) * (t1 + t2)) + (t3 + t4)) + log_jacobian_sigma_sqs + log_jacobian_thetas)
        
        
        # wrapper function if not doing any log-posterior tempering
        def unnormalized_log_prob_no_temp(X, sigma_sqs_pre, thetas_pre):
            return unnormalized_log_prob(X, sigma_sqs_pre, thetas_pre, beta_temp=1.0)
        
        # need to create a helper function to let NUTS know initially it's a 3-variable input.
        def init_tempered_log_prob(X, sigma_sqs_pre, thetas, beta_temp):
            return unnormalized_log_prob(X, sigma_sqs_pre, thetas, beta_temp=beta_temp)
        
        # what should the initial beta_temp be?
        beta_temp_init = logarithmic_temperature_schedule(step=0, min_temp=0.1)
                                
        # create our NUTS sampler with dual step-size adaptation (as the base)
        adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=lambda X, sigma_sqs_pre, thetas : init_tempered_log_prob(X, sigma_sqs_pre, thetas,
                                                                                            beta_temp=beta_temp_init), 
                step_size=0.1),
            num_adaptation_steps=int(0.8 * num_burnin_steps),
            target_accept_prob=0.75)
        
        # initialize our final sampler based on whether we are doing log-tempering or not
        if tempering:
        
            # wrap the above with our temperature-controlled custom kernel setup.
            final_sampler = LogAnnealedNUTS(base_kernel=adaptive_sampler, 
                                            num_steps=num_burnin_steps + num_results,
                                            unnormalized_log_prob=unnormalized_log_prob)  
        
        # just use standard NUTS + DualAveraging Step-Size Adaptation
        else:
            
            # keep it as the same
            final_sampler = adaptive_sampler
        
        # set up our initial state, i.e., our current state (note the softplus inverse bijection)
        sigma_sqs_pre_init = np.full_like(a=self.sigma_sqs_init, fill_value=-5.0) # filling with something small by default.
        sigma_sqs_pre_init[self.sigma_sqs_init > sigma_sqs_LB] = np.log(np.exp( (self.sigma_sqs_init - sigma_sqs_LB)
                                                                                 [self.sigma_sqs_init > sigma_sqs_LB] ) - 1.0)
        
        # repeat the process for the theta_pre_inits
        thetas_pre_init = np.full_like(a=self.thetas_init, fill_value=-5.0) # filling with something small by default.
        thetas_pre_init[self.thetas_init > 0.0] = np.log(np.exp( (self.thetas_init - 0.0)[self.thetas_init > 0.0] ) - 1.0)
        
        # fill our initial state with our pre-transformed values
        initial_state = [self.Xhat_init, sigma_sqs_pre_init, thetas_pre_init]
        
        # accelerated sampling.
        @tf.function(autograph=True, jit_compile=True)
        def run_nuts():
            samples, kernel_results = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=final_sampler,
                trace_fn=lambda _, pkr: pkr
            )
            return samples, kernel_results
        
        if verbose:
            print("Starting NUTS posterior sampling ...")
        
        # run our samples
        start = time.time()
        samples, kernel_results = run_nuts()
        end = time.time()

        # how much time did it take?
        minutes = np.round((end - start) / 60, 2)
        if verbose: 
            print(f"Finished sampling in {minutes} minutes.")
            
        # package everything into a dictionary as output
        results = {"phi1s" : self.phi1s, "phi2s" : self.phi2s, 
                   "Xhat_init" : self.Xhat_init, 
                   "sigma_sqs_LB" : sigma_sqs_LB,
                   "sigma_sqs_init" : self.sigma_sqs_init,
                   "sigma_sqs_init_constrained" : tf.math.log(1.0 + tf.math.exp(sigma_sqs_pre_init)) + sigma_sqs_LB, 
                   "thetas_init" : self.thetas_init,
                   "thetas_init_constrained" : tf.math.log(1.0 + tf.math.exp(thetas_pre_init)).numpy(), 
                   "I" : self.I, 
                   "X_samps" : samples[0].numpy(), 
                   "sigma_sqs_samps" : np.log(np.exp(samples[1].numpy()) + 1.0) + sigma_sqs_LB, 
                   "thetas_samps" : np.log(np.exp(samples[2].numpy()) + 1.0),
                   "kernel_results" : kernel_results,
                   "sample_results" : samples,
                   "minutes_elapsed" : minutes}
        
        # return our results package
        return results
    
    
    '''
    When forecasting, we do not need to update self.not_nan_idxs & self.not_nan_cols because padding NaNs has no effect!
    - There will be a separate function for updating X_obs_discret & mu_ds.
    '''
    # function for updating our kernel matrices, to facilitate forecasting (i.e., not adding new observations).
    def update_kernel_matrices(self, I_new, phi1s_new, phi2s_new):
        
        # Internalize new phi1s, phi2s + recalculate beta, mag_I
        self.I = I_new.reshape(-1, 1)
        self.phi1s, self.phi2s = phi1s_new.copy(), phi2s_new.copy()
        self.mag_I = self.I.shape[0]
        self.beta = tf.cast((self.D * self.mag_I) / self.N_ds.sum(), tf.float64)
        
        # create a temporary storage for the matrices to avoid EagerTensor errors
        C_d_invs = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        m_ds = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        K_d_invs = np.zeros(shape=(self.D, self.mag_I, self.mag_I))
        
        # go component-by-component
        for d in range(self.D):
            C_d, m_d, K_d = self._build_matrices(self.I, self.phi1s[d], self.phi2s[d], v=2.01)
            C_d_invs[d] = tf.linalg.pinv(C_d) # pinv equal to inv if invertible!
            m_ds[d] = m_d
            K_d_invs[d] = tf.linalg.pinv(K_d)
            
        # update the instance variables
        self.C_d_invs = C_d_invs
        self.m_ds = m_ds
        self.K_d_invs = K_d_invs
            
        # update banded matrix approximations too
        if self.BANDSIZE is not None:
            self.C_d_invs = tf.linalg.band_part(input=self.C_d_invs, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
            self.K_d_invs = tf.linalg.band_part(input=self.K_d_invs, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
            self.m_ds = tf.linalg.band_part(input=self.m_ds, num_lower=self.BANDSIZE, num_upper=self.BANDSIZE)
        
        
        
    #####################################################
    ############## Helper Functions #####################
    #####################################################
    
    '''
    Adds 2^discretization - 1 evenly-spcaed timesteps between consecutive observations in X_obs
    - Returns I (vector of discretized timesteps) and X_obs_discret.
    '''
    # adds in 2^discret - 1 evenly-spaced timesteps between consecutive observations
    def _discretize(self, ts_obs, X_obs, discretization):
        
        # making sure dimensions work out
        ts_obs = ts_obs.flatten()
        assert ts_obs.shape[0] == X_obs.shape[0],\
        "Please make sure there are equal numbers of observations in ts_obs and X_obs."
        
        # how many points do we want to insert?
        N, D = X_obs.shape
        N_discret = (2 ** discretization) * (N-1) + 1
        I = np.full(shape=(N_discret,), fill_value=np.nan)
        X_obs_discret = np.full(shape=(N_discret, D), fill_value=np.nan)
        
        # put in evenly-spaced timesteps for I
        I[::(2 ** discretization)] = ts_obs
        indices = np.arange(len(I))
        I = np.interp(x=indices, xp=indices[~np.isnan(I)], fp=I[~np.isnan(I)])
        I = I.reshape(-1, 1) # force into column for TFP compatibility
        
        # fill with our observed values
        X_obs_discret[::(2 ** discretization)] = X_obs
        
        # return both the discretized timesteps + data matrix.
        return I, X_obs_discret
        
        
    '''
    - Takes in data matrix X and linearly-interpolates the NaNs.
    - Assumes no column of X is completely missing.
    - Returns: interpolated X_interp with same shape as X.
    
    *Columns that were completely missing will still be completely missing!
    '''
    # linearly interpolating any NaNs in columns that are NOT completely NaNs.
    def _linear_interpolate(self, X_partial):
        
        # get dimensions of our X_partial
        N_partial, D_partial = X_partial.shape
        
        # will be eventually returning X_interp
        X_interp = X_partial.copy() # includes nans for completely missing components still!
        indices = np.arange(N_partial)
        
        # start by doing linear interpolation + getting priors on phi2
        for d in range(D_partial):

            # linear interpolation if we need to (i.e., there are NaNs in this column)
            if np.any(np.isnan(X_interp[:,d])):
                X_interp[:,d] = np.interp(x=indices, xp=indices[~np.isnan(X_partial[:,d])], 
                                          fp=X_partial[~np.isnan(X_partial[:,d]),d])
                
        # return the matrix with the interpolated values in the partially/fully-observed columns
        return X_interp
        
        
    '''
    Fits phi1, phi2, and sigma_sq_init hyperparameters given:
    1. I - discretized timesteps.
    2. X_filled - data matrix assuming each column is completely observed (i.e., interpolated).
    
    Returns: our (D_filled,) phi1s, phi2s, and sigma_sqs, as well as X_filled.shape smoothed values
    ''' 
    # fit (phi1, phi2, sigma_sq) for components in X_filled.
    def _fit_kernel_hparams(self, I, X_filled, verbose=False):
        
        #### 1. DATA-DRIVEN FOURIER-INFORMED PRIOR HYPERPARAMETERS
        
        # get dimensions of our data.
        N_filled, D_filled = X_filled.shape
        
        # data structures to store our prior-means, mu_phi2s, and sd_phi2s
        mu_ds, mu_phi2s, sd_phi2s = [], [], []
        
        # fourier-transforms to get prior hyperparameters for observed components.
        for d in range(D_filled):
            
            # phi2 priors: mean + SD
            z = np.fft.fft(X_filled[:,d]); zmod = np.abs(z)
            zmod_effective = zmod[1:(len(zmod) - 1) // 2 + 1]; zmod_effective_sq = zmod_effective ** 2
            idxs = np.linspace(1, len(zmod_effective), len(zmod_effective))
            freq = np.sum(idxs * zmod_effective_sq) / np.sum(zmod_effective_sq)
            mu_phi2 = 0.5 / freq; sd_phi2 = (1 - mu_phi2) / 3

            # get prior mean for our data, too
            mu_d = X_filled[:,d].mean() # CHECK WITH PROF. YANG ABOUT THIS ONE!

            # add to our lists
            mu_ds.append(mu_d); mu_phi2s.append(mu_phi2); sd_phi2s.append(sd_phi2)
            
        # convert to arrays
        mu_ds, mu_phi2s, sd_phi2s = np.array(mu_ds), np.array(mu_phi2s), np.array(sd_phi2s)
        
        #### 2. SETTING UP TFP-POWERED GP OBJECTS FOR OPTIMIZATION.
        
        '''
        Note: keeping this function as inner function to have access to mu_ds as exogenous.
        - TFP has different parameterizations of Matern kernel vs. PNAS paper.
        '''
        # defining tensorflow-probability-kernel-based GPs for each component in X_filled
        def build_gps(phi1s, sigma_sqs, phi2s):

            # broadcast across components!
            if D_filled != 1:
                kernel = tfk.GeneralizedMatern(df=2.01, 
                                               amplitude=tf.sqrt(phi1s)[:,None], 
                                               length_scale=phi2s[:,None])
            
            # need to treat single-component systems separately!
            else:
                kernel = tfk.GeneralizedMatern(df=2.01, 
                                               amplitude=tf.sqrt(phi1s), 
                                               length_scale=phi2s)

            # custom mean function
            def mean_fn(x):
                mu_reshaped = tf.reshape(mu_ds, (D_filled, 1, 1))
                return tf.broadcast_to(mu_reshaped, (D_filled, 1, x.shape[-1]))

            # no need for a separate mean function -- just return a scalar! Directly build GP.    
            gps = tfd.GaussianProcess(kernel=kernel, 
                                      index_points=I,
                                      mean_fn=mean_fn,
                                      observation_noise_variance=sigma_sqs[:,None])

            # everything combined
            return gps
        
        #### 3. PERFORMING ADAM-BASED OPTIMIZATION TO FIT KERNEL HYPERPARAMETERS
        '''
        8/20/2024:
        1. By vectorizing the Gaussian Process, we are overly computing our prior contributions x D_observed times.
        2. Resolve by noting that for Normal / TruncatedNormal, scaling LLH by 1/D ~ scaling Normal variance by D.
        3. gpjm.log_prob() is a D_observed x D_observed matrix of partial derivatives. We can take TRACE!
        '''

        #### 3. CHOOSE PRIOR FOR PHI2 BASED ON FLAG
        if not self.use_fourier_prior:
            # Flat prior for phi2 (mean = 1.0, large variance)
            mu_phi2s = np.full(D_filled, 1.0)
            sd_phi2s = np.full(D_filled, 1000.0)

        # constructing a sampleable-object to pass into TFP optimization
        gpjm = tfd.JointDistributionNamed(
            {"phi1s" :
                 tfd.TruncatedNormal(loc=np.float64([1e-4] * D_filled),
                                     low=np.float64([1e-6] * D_filled),
                                     high=np.float64([np.inf] * D_filled),
                                     scale=np.float64([1000.0 * np.sqrt(D_filled)] * D_filled)), # flat prior
             "sigma_sqs" :
                 tfd.TruncatedNormal(loc=np.float64( ((X_filled.std(axis=0)) * 0.1 ) ** 2),
                                     low=np.float64([1e-6] * D_filled),
                                     high=np.float64([np.inf] * D_filled),
                                     scale=np.float64([1000.0 * np.sqrt(D_filled)] * D_filled)), # flat prior
             "phi2s" :
                 tfd.TruncatedNormal(loc=np.float64(mu_phi2s),
                                     low=np.float64([1e-6] * D_filled),
                                     high=np.float64([np.inf] * D_filled),
                                     scale=np.float64(sd_phi2s * np.sqrt(D_filled))),
             "observations" : build_gps})

        # define our TO-BE-TRAINABLE variables + constrain them to be positive, and then make them positive.
        phi1s_var = tfp.util.TransformedVariable(initial_value=X_filled.std(axis=0) ** 2, 
                                         bijector=tfp.bijectors.Softplus(), 
                                         name="phi1s",
                                         dtype=np.float64) # overall variance
        phi2s_var = tfp.util.TransformedVariable(initial_value=mu_phi2s, 
                                                 bijector=tfp.bijectors.Softplus(), 
                                                 name="phi2s",
                                                 dtype=np.float64) # bandwidth
        sigma_sqs_var = tfp.util.TransformedVariable(initial_value=( (X_filled.std(axis=0)) * 0.1 ) ** 2, 
                                                     bijector=tfp.bijectors.Softplus(), 
                                                     name="sigma_sqs",
                                                     dtype=np.float64) # noise

        # which variables are we attempting to fit?
        trainable_variables = [v.trainable_variables[0] for v in [phi1s_var, phi2s_var, sigma_sqs_var]]

        # optimization function + Adam routine initialize
        X_filled_bcst = X_filled.T[:,np.newaxis,:] # broadcast + adding axis for TFP compatibility.
        def target_log_prob(phi1s, sigma_sqs, phi2s):
          return gpjm.log_prob({"phi1s": phi1s, 
                                "sigma_sqs": sigma_sqs, 
                                "phi2s": phi2s, 
                                "observations": X_filled_bcst})
        num_iters = 100; optimizer = tf_keras.optimizers.Adam(learning_rate=.01)

        # taking one step of Adam + scaling up to train our model
        @tf.function(autograph=True, jit_compile=True)
        def train_model():
          with tf.GradientTape() as tape:
            loss = -target_log_prob(phi1s_var, 
                                    sigma_sqs_var, 
                                    phi2s_var)
          grads = tape.gradient(loss, trainable_variables)
          optimizer.apply_gradients(zip(grads, trainable_variables))
          return loss
        
        # status-updates if verbose mode.
        if verbose:
            
            # get a description + use TQDM
            desc=f"Fitting hparams for {D_filled} components"
            for i in tqdm(range(num_iters), desc=desc):
                loss = train_model()
        else:
            
            # directly train for num_iters iterations.
            for i in range(num_iters):
                loss = train_model()
        
        '''
        # generate smoothed trajectories of this GP using the fitted phi1, phi2, sigma_sq values
        - Note: 11/26/2024. GP smoothing without ODE information gets really poor looking trajectories ...
        gps = build_gps(phi1s=phi1s_var._value().numpy(), 
                        sigma_sqs=sigma_sqs_var._value().numpy(), 
                        phi2s=phi2s_var._value().numpy())
        X_filled_smoothed = gps.sample(1000)
        '''
        # return as outputs our (D_filled,) phi1s, phi2s, and sigma_sqs
        return {"phi1s" : phi1s_var._value().numpy(),
                "phi2s" : phi2s_var._value().numpy(),
                "sigma_sqs" : sigma_sqs_var._value().numpy()}
        
    
    # for doing cross-validated cubic splines for smoother initial values (assumes no NaNs)
    def cv_cubic_smoother(self, I, X_filled):
        
        # only do this procedure if we have 10 or more observations
        I = I.flatten()
        if I.shape[0] < 10:
            return X_filled
        else:
            return np.stack([self.single_cv_cubic_smoother(I, X_filled[:,i]) 
                             for i in range(X_filled.shape[1])], axis=1)
        
        
    # smoothing 1 component of our system.
    def single_cv_cubic_smoother(self, I, x):
        
        # only do this procedure if we have 10 or more observations
        I = I.flatten()
        if I.shape[0] < 10:
            return x

        # create our cross-validated splits
        kf = KFold(n_splits = 5, shuffle=True, random_state=1)

        # how many possible knots do we have? (either no knots, or max. of 10 obs. per knot)
        knot_nums = np.arange(0, (I.shape[0] // 10) +1)

        # create a list to store error vectors for each split
        split_errs = []

        # do k-fold cross-validation on the number of knots
        for train_idx, val_idx in kf.split(np.arange(I.shape[0])):

            # create a list to store all of the errors for this fold.
            knot_errs = []

            # iterate thru our knots
            for knot_num in knot_nums:

                # where are we placing the knots?
                if knot_num == 0:
                    knot_positions = np.array([])
                else:
                    knot_positions = np.linspace(start=I[0], stop=I[-1], num=knot_num+2)[1:-1]

                # fit our spline + evaluate the error on the validation set
                tck = splrep(I[train_idx], x[train_idx], t=knot_positions, s=0)
                preds = splev(I[val_idx], tck)
                err = ((preds - x[val_idx]) ** 2).mean(); knot_errs.append(err)

            # add to split_errs
            split_errs.append(knot_errs)

        # get our optimal number of knots
        optimal_knot_num = knot_nums[np.array(split_errs).mean(axis=0).argmin()]

        # fit cubic spline using these knots
        if knot_num == 0:
            knot_positions = np.array([])
        else:
            knot_positions = np.linspace(start=I[0], stop=I[-1], num=knot_num+2)[1:-1]

        # fit our cubic spline on the full data    
        tck = splrep(I, x, t=knot_positions, s=0)
        smoothed = splev(I, tck)

        # fit cubic spline using these knots
        if knot_num == 0:
            knot_positions = np.array([])
        else:
            knot_positions = np.linspace(start=I[0], stop=I[-1], num=knot_num+2)[1:-1]

        # fit our cubic spline on the full data    
        tck = splrep(I, x, t=knot_positions, s=0)
        x_smoothed = splev(I, tck)

        # return the smoothed trajectory
        return x_smoothed
    
    
    # take in timesteps I + hparams (phi1, phi2, v) and returns (C_d, m_d, K_d) for a given component dim
    def _build_matrices(self, I, phi1, phi2, v=2.01):
        '''
        Takes in discretized timesteps I and hparams (phi1, phi2, v). Returns (C_d, m_d, K_d) for component d.
        - I is an np.array of discretized timesteps, phi1 & phi2 are floats.
        '''

        # tile appropriately to facilitate vectorization
        s = np.tile(A=I.reshape(-1, 1), reps=I.shape[0]); t = s.T

        # l = |s-t|, u = sqrt(2*nu) * l / phi2 - let's nan out diagonals to avoid imprecision errors.
        l = np.abs(s - t); u = np.sqrt(2*v) * l / phi2; np.fill_diagonal(a=u, val=np.nan)

        # pre-compute Bessel function + derivatives
        Bv0, Bv1, Bv2 = kvp(v=v, z=u, n=0), kvp(v=v, z=u, n=1), kvp(v=v, z=u, n=2)

        # 1. Kappa itself, but we need to correct everywhere with l=|s-t|=0 to have value exp(0.0) = 1.0
        Kappa = (phi1/gamma(v)) * (2 ** (1 - (v/2))) * ((np.sqrt(v) / phi2) ** v)
        Kappa *= Bv0
        Kappa *= (l ** v)

        # https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        np.fill_diagonal(Kappa, val=phi1 + phi1 * 1e-6) # behavior as |s-t| \to 0^+

        # 2. p_Kappa, but need to replace everywhere with l=|s-t|=0 to have value 0.0.
        p_Kappa = (2 ** (1 - (v/2)))
        p_Kappa *= phi1 * ((u / np.sqrt(2)) ** v)
        p_Kappa *= ( (u * phi2 * Bv1) + (v*phi2*Bv0) )
        p_Kappa /= (phi2 * (s-t) * gamma(v))
        np.fill_diagonal(p_Kappa, val=0.0) # behavior as |s-t| \to 0^+

        # 3. Kappa_p (by symmetry)
        Kappa_p = p_Kappa * -1

        # 4. Kappa_pp - let's proceed term-by-term (save multiplier terms at the end)
        Kappa_pp = 2 * np.sqrt(2) * (v ** 1.5) * phi2 * l * Bv1
        Kappa_pp += ( ( (v ** 2) * (phi2 ** 2) ) - ( v * (phi2 ** 2) ) ) * Bv0
        Kappa_pp += ( (2 * v * (s ** 2)) - (4 * v * s * t) + (2 * v * (t ** 2)) ) * Bv2
        Kappa_pp *= ( -1.0 * (2 ** (1 - (v/2))) * phi1 * ((u / np.sqrt(2)) ** v) )
        Kappa_pp /= ( (phi2 ** 2) * (l ** 2) * gamma(v) )

        # CHECK WITH PROF. YANG ABOUT THIS ONE! SHOULD THERE BE A NEGATIVE HERE?
        np.fill_diagonal(Kappa_pp, val=v*phi1/( (phi2 ** 2) * (v-1) ) * (1 + 1e-6)) # behavior as |s-t| \to 0^+

        # 5. form our C, m, and K matrices (let's not do any band approximations yet!)
        C_d, Kappa_inv = Kappa.copy(), np.linalg.pinv(Kappa)
        m_d = p_Kappa @ Kappa_inv
        K_d = Kappa_pp - (p_Kappa @ Kappa_inv @ Kappa_p)

        # 6. return our three matrices
        return C_d, m_d, K_d
    
'''
Inputs:
1. step - i.e., which step of sampling are we on?
2. min_temp - what is the minimum temperature?

*Returns the multiplier we will place on our log-posterior for temperature control ("beta_temp")
'''
# helper functions for implementing logarithmic decay tempering of log-posterior
def logarithmic_temperature_schedule(step : Union[int, tf.Tensor], min_temp : float = 0.1):
    step, min_temp = tf.cast(step, tf.float64), tf.cast(min_temp, tf.float64)
    return tf.maximum(1.0 / tf.math.log(step + 2.0), min_temp)

# custom kernel based on NUTS that allows for logarithm annealing of log-posterior.
class LogAnnealedNUTS(tfp.mcmc.TransitionKernel):
    
    # standard constructor
    def __init__(self, base_kernel, num_steps, unnormalized_log_prob, min_temp=0.1):
        
        # will be inheriting NUTS + DualAveragingStepSizeAdaptation combination.
        self.base_kernel = base_kernel # will end up being adaptive_sampler
        self.num_steps = num_steps
        self.unnormalized_log_prob = unnormalized_log_prob
        self.min_temp = min_temp
        self.current_step = tf.Variable(0, dtype=tf.int32)  # Keep track of steps

        
    # custom function to perform one step of the sampling process
    def one_step(self, current_state, previous_kernel_results):
        
        # update beta_temp dynamically using our log-temp scheduler function
        beta_temp = logarithmic_temperature_schedule(self.current_step, self.min_temp)
        self.current_step.assign_add(1)  # Increment the step counter

        # make the function accessible
        unnormalized_log_prob = self.unnormalized_log_prob
        
        # wrapper for the log-posterior function to account for sampling parameters + temperature.
        def tempered_log_prob(*args):
            return unnormalized_log_prob(*args, beta_temp=beta_temp)

        # need to update the base inner NUTS sampler with this modified log-prob sampler to account for temperature
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=tempered_log_prob,
            step_size=self.base_kernel.inner_kernel.step_size  # Using the current step size
        )

        # create new DualAveraging wrapper preserving the old settings
        self.base_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=self.base_kernel.num_adaptation_steps,
            target_accept_prob=self.base_kernel.parameters["target_accept_prob"]
        )
        
        # take one step of this new base kernel.
        return self.base_kernel.one_step(current_state, previous_kernel_results)

    
    # need to define this function get the kernel results for compatibility with DualStepSizeAdaptation, etc.
    def bootstrap_results(self, current_state):
        return self.base_kernel.bootstrap_results(current_state)

    
    # inherited function that checks if chain has converged.
    def is_calibrated(self):
        return self.base_kernel.is_calibrated()
