import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import pickle

# core MAGI-TFP class
import magi_v2


# 3-component model governing the SEIR system, appropriate for tensorflow vectorization
def f_vec(t, X, thetas):
    '''
    1. X - array containing (E, I, R) components. Suppose it is (N x D) for vectorization.
    2. theta - array containing (beta, gamma, sigma) components.
    3. Note that N_pop = 1.0, and that S is deterministic: S = 1 - (E + I + R)
    '''
    # implicitly compute S
    S = 1.0 - tf.reshape(tf.reduce_sum(X, axis=1), shape=(-1, 1))
    return tf.concat([(thetas[0] * S * X[:,1:2]) - (thetas[2] * X[:,0:1]), # dE/dt = bSI - sE
                      (thetas[2] * X[:,0:1]) - (thetas[1] * X[:,1:2]), # dI/dt = sE - gI
                      (thetas[1] * X[:,1:2])], # dR/dt = g*I
                     axis=1)


# create our list of settings (480x settings)
settings = []
for d_obs in [20, 40]:
    for discretization in [1, 2]:
        for t_max in [2.0, 4.0, 8.0]:
            for comp_obs in [[True, True, True], [False, True, True], 
                             [True, False, True], [True, True, False]]:
                for seed in range(10):
                    settings.append((d_obs, discretization, t_max, comp_obs, seed))
                
# which command-line argument setting are we running?
d_obs, discretization, t_max, comp_obs, seed = settings[int(sys.argv[1])]


# load in our data, thinning based on density of observations
raw_data = pd.read_csv(f'data/SEIR_beta=6_gamma=0.6_sigma=1.8_alpha=0.05_seed={seed}.csv').query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = obs_data.t.values.astype(np.float64)

# 11/18/2024 - let's try using the truth instead of _obs
X_obs = obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64) # S is implicit!

# make certain components missing if necessary
for i, comp_obs_val in enumerate(comp_obs):
    if comp_obs_val != True:
        X_obs[:,i] = np.nan
        
# create our model - f_vec is the ODE governing equations function defined earlier.
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# fit Matern kernel hyperparameters (phi1, phi2) as well as (Xhat_init, sigma_sqs_init, thetas_init)
model.initial_fit(discretization=discretization, verbose=False)

# collect our samples from NUTS posterior sampling
results = model.predict(num_results=3000, num_burnin_steps=3000, verbose=False)

# what are our discretized timesteps?
I = model.I.flatten()

# checking derivatives (i.e., physics fidelity)
raw_data["t"] = np.round(raw_data["t"].values, 3)
raw_data.set_index("t", inplace=True)

# get our true values
X_true = raw_data.loc[np.round(I, 3)][["E_true", "I_true", "R_true"]].values
thetas_true = np.array([6.0, 0.6, 1.8])

# compute GP-implied derivatives at truth
X_cent = tf.reshape(X_true - model.mu_ds, shape=(X_true.shape[0], 1, X_true.shape[1]))
f_gp = model.m_ds @ tf.transpose(X_cent, perm=[2, 0, 1])

# compute the true derivatives at truth
f_ode = tf.transpose(f_vec(I, X_true, thetas_true)[:,None], perm=[2, 0, 1])

# save these true-X derivatives
results["f_gp_true"] = np.transpose(f_gp.numpy(), axes=[2, 1, 0])[0]
results["f_ode_true"] = np.transpose(f_ode.numpy(), axes=[2, 1, 0])[0]

# name our trial
obs_desc = ""
for i, comp in enumerate(["E", "I", "R"]):
    obs_desc += f"{comp}={comp_obs[i]}_"
fname = f"DOBS={d_obs}_DISC={discretization}_TM={t_max}_{obs_desc}seed={seed}"

# save our results
with open(f"results/{fname}.pickle", "wb") as file:
    pickle.dump(results, file)