import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tfpigp.magi_v2 as magi_v2  # MAGI-TFP class for Bayesian inference
from tfpigp.visualization import *
from scipy.integrate import solve_ivp
from tfpigp.mle import mle

# Define the EIR representation ODE on the log scale
def f_vec(t, X, thetas):
    '''
    Log-scale SEIR model with E implicitly represented.

    Parameters:
    1. X - array containing (logE, logI, logR) components. Shape (N x 3).
    2. thetas - array containing (beta, gamma, sigma) parameters.

    Returns:
    Derivatives (dlogE/dt, dlogI/dt, dlogR/dt) as a tensor of shape (N x 3).
    '''
    logE, logI, logR = tf.unstack(X, axis=1)
    beta = thetas[0]
    gamma = thetas[1]
    sigma = thetas[2]

    # Convert log variables back to original scale
    E = tf.exp(logE)
    I = tf.exp(logI)
    R = tf.exp(logR)

    # Compute S implicitly (EIR representation)
    S = 1.0 - E - I - R

    # Ensure stability for S
    S = tf.maximum(S, 1e-10)

    # Derivatives in the original scale
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Derivatives in the log scale (chain rule)
    dlogEdt = dEdt / E
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    # Return derivatives as a tensor
    return tf.stack([dlogEdt, dlogIdt, dlogRdt], axis=1)

# Initial settings
d_obs = 20  # Observations per unit time
t_max = 2.0  # Observation interval length

# Load data and select observations
raw_data = pd.read_csv('tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.15_seed=2.csv').query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# Extract observation times and log-transformed noisy observations
ts_obs = obs_data.t.values.astype(np.float64)
X_obs = np.log(obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64))

X_obs[:,0] = np.nan

# benchmark MLE
final_thetas, X0_final, sigma_obs_est, loss, intervel_est = mle(ts_obs, X_obs, maxiter=1000)


# Create the MAGI-TFP model
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# Fit initial hyperparameters
phi_exo = None
model.initial_fit(discretization=2, verbose=True, use_fourier_prior=False, phi_exo=phi_exo)
model.phi1s[0] = 2
model.phi2s[0] = 0.44
model.update_kernel_matrices(I_new=model.I, phi1s_new=model.phi1s, phi2s_new=model.phi2s)

clear_output(wait=True)

# Collect samples using NUTS posterior sampling
results = model.predict(num_results=5000, num_burnin_steps=10000, tempering=False, verbose=True)

ts_true = raw_data.t.values
x_true = raw_data[["E_true", "I_true", "R_true"]]
x_true = np.log(x_true)
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs)
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, trans_func=np.exp)
print_parameter_estimates(results, [6.0, 0.6, 1.8])
plot_trace(results["thetas_samps"], [6.0, 0.6, 1.8], ["beta", "gamma", "sigma"])


# 'results' contains posterior samples from the in-sample fit, e.g. up to t_max=2.0
# Now we want to predict out-of-sample beyond t=2.0, say up to t=4.0

t_step_prev_end = 2.0  # end of the in-sample period used in the first script
t_forecast_end = 4.0   # new forecast horizon
t_stepsize = 2.0       # length of the new interval we want to forecast

# We assume a similar density of discretization points as the in-sample fit.
# The first script used something like 20 observations per unit time.
# The model.I attribute contains the discretization grid used internally.
# We will extend this grid to t=4.0

I_append = np.linspace(start=model.I[-1, 0],
                       stop=model.I[-1, 0] + t_stepsize,
                       num=int(80 * t_stepsize + 1))[1:].reshape(-1,1)
I_new = np.vstack([model.I, I_append])

# Update kernel matrices for the extended interval
model.update_kernel_matrices(I_new=I_new, phi1s_new=model.phi1s, phi2s_new=model.phi2s)

# Use posterior means for sigma_sqs as a starting point
model.sigma_sqs_init = results["sigma_sqs_samps"].mean(axis=0)
# Use the posterior means for thetas and X_samps as starting points
model.thetas_init = results["thetas_samps"].mean(axis=0)
Xhat_init_in = results["X_samps"].mean(axis=0)

# The states in Xhat_init_in are [logE, logI, logR]
# ODE in original (linear) scale to integrate forward
# Create a wrapper for solve_ivp in log-scale
def ODE_log_scale(t, y, theta):
    # y is [logE, logI, logR]
    y_tf = tf.convert_to_tensor(y.reshape(1,-1), dtype=tf.float64)
    theta_tf = tf.convert_to_tensor(theta, dtype=tf.float64)
    dYdt_tf = f_vec(t, y_tf, theta_tf)
    return dYdt_tf[0].numpy()  # return numpy array of shape (3,)

# Integrate forward in log-scale from t=2.0 to t=4.0
sol = solve_ivp(fun=lambda t, y: ODE_log_scale(t, y, model.thetas_init),
                t_span=(t_step_prev_end, I_append[-1,0]),
                y0=Xhat_init_in[-1],  # last known log-state
                t_eval=np.concatenate(([t_step_prev_end], I_append.flatten())),
                rtol=1e-10, atol=1e-10)

# Extract the solution beyond t=2.0
Xhat_init_out_log = sol.y.T[1:]  # shape (#new_points, 3)

# Combine old and new
Xhat_init_combined = np.vstack([Xhat_init_in, Xhat_init_out_log])
model.Xhat_init = Xhat_init_combined

# Now run prediction again for the extended time period
results_forecast = model.predict(num_results=1000, num_burnin_steps=5000, tempering=False, verbose=True)

# plot
raw_data = pd.read_csv('tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.15_seed=2.csv').query(f"t <= {t_forecast_end}")
ts_true = raw_data.t.values
x_true = raw_data[["E_true", "I_true", "R_true"]]
x_true = np.log(x_true)

# results_forecast now contains posterior samples for the entire time range [0,4], including the forecasted portion.

# Optionally, we can visualize the forecast:
plot_trajectories(ts_true, x_true, results_forecast, ts_obs, X_obs)
plot_trajectories(ts_true, x_true, results_forecast, ts_obs, X_obs, trans_func=np.exp)
plot_trace(results_forecast["thetas_samps"], [6.0, 0.6, 1.8], ["beta", "gamma", "sigma"])

sol_mle = solve_ivp(fun=lambda t, y: ODE_log_scale(t, y, final_thetas),
                    t_span=(model.I[0], model.I[-1]),
                    y0=X0_final,  # last known log-state
                    t_eval=model.I.flatten(),
                    rtol=1e-10, atol=1e-10)
results_forecast["Xhat_mle"] = sol_mle.y.T
plot_trajectories(ts_true, x_true, results_forecast, ts_obs, X_obs, trans_func=np.exp)
