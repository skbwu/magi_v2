import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from IPython.display import clear_output
import tfpigp.magi_v2 as magi_v2  # MAGI-TFP class for Bayesian inference
from tfpigp.visualization import *
from scipy.integrate import solve_ivp

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

# Add command-line argument for the seed
parser = argparse.ArgumentParser(description="Run SEIR model and save results.")
parser.add_argument("--seed", type=int, required=True, help="Seed for the simulation")
args = parser.parse_args()

# Seed value from command-line argument
seed = args.seed

# Specify output directory and create it if it doesn't exist
output_dir = f"results_fully_observed_seed_{seed}"
os.makedirs(output_dir, exist_ok=True)

# Initial settings
d_obs = 20  # Observations per unit time
t_max = 2.0  # Observation interval length

# Load data and select observations
orig_data = pd.read_csv(f'tfpigp/data/logSEIR_beta=6.0_gamma=0.6_sigma=1.8_alpha=0.15_seed={seed}.csv')
raw_data = orig_data.query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# Extract observation times and log-transformed noisy observations
ts_obs = obs_data.t.values.astype(np.float64)
X_obs = np.log(obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64))

# Create the MAGI-TFP model
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# Fit initial hyperparameters
phi_exo = None
model.initial_fit(discretization=2, verbose=True, use_fourier_prior=False, phi_exo=phi_exo)

clear_output(wait=True)

# Collect samples using NUTS posterior sampling
results = model.predict(num_results=1000, num_burnin_steps=1000, tempering=False, verbose=True)

ts_true = raw_data.t.values
x_true = raw_data[["E_true", "I_true", "R_true"]]
x_true = np.log(x_true)

# Visualization and saving results
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, caption_text="MAGI on log-scale SEIR", output_dir=output_dir)
plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, trans_func=np.exp, caption_text="MAGI on original-scale SEIR", output_dir=output_dir)
print_parameter_estimates(results, [6.0, 0.6, 1.8])
plot_trace(results["thetas_samps"], [6.0, 0.6, 1.8], ["beta", "gamma", "sigma"], "trace plot for theta in MAGI", output_dir=output_dir)

# 'results' contains posterior samples from the in-sample fit, e.g., up to t_max=2.0
# Extend model for forecasting
t_step_prev_end = 2.0  # end of the in-sample period used in the first script
t_forecast_end = 4.0   # new forecast horizon
t_stepsize = 2.0       # length of the new interval we want to forecast

I_append = np.linspace(start=model.I[-1, 0],
                       stop=model.I[-1, 0] + t_stepsize,
                       num=int(80 * t_stepsize + 1))[1:].reshape(-1, 1)
I_new = np.vstack([model.I, I_append])

model.update_kernel_matrices(I_new=I_new, phi1s_new=model.phi1s, phi2s_new=model.phi2s)

model.sigma_sqs_init = results["sigma_sqs_samps"].mean(axis=0)
model.thetas_init = results["thetas_samps"].mean(axis=0)
Xhat_init_in = results["X_samps"].mean(axis=0)

def ODE_log_scale(t, y, theta):
    y_tf = tf.convert_to_tensor(y.reshape(1, -1), dtype=tf.float64)
    theta_tf = tf.convert_to_tensor(theta, dtype=tf.float64)
    dYdt_tf = f_vec(t, y_tf, theta_tf)
    return dYdt_tf[0].numpy()

sol = solve_ivp(fun=lambda t, y: ODE_log_scale(t, y, model.thetas_init),
                t_span=(t_step_prev_end, I_append[-1, 0]),
                y0=Xhat_init_in[-1],
                t_eval=np.concatenate(([t_step_prev_end], I_append.flatten())),
                rtol=1e-10, atol=1e-10)

Xhat_init_out_log = sol.y.T[1:]
Xhat_init_combined = np.vstack([Xhat_init_in, Xhat_init_out_log])
model.Xhat_init = Xhat_init_combined

results_forecast = model.predict(num_results=1000, num_burnin_steps=1000, tempering=False, verbose=True)

# Visualization and saving forecast results
raw_data = orig_data.query(f"t <= {t_forecast_end}")
ts_true = raw_data.t.values
x_true = raw_data[["E_true", "I_true", "R_true"]]
x_true = np.log(x_true)

plot_trajectories(ts_true, x_true, results_forecast, ts_obs, X_obs, caption_text="MAGI forecast", output_dir=output_dir)
plot_trace(results_forecast["thetas_samps"], [6.0, 0.6, 1.8], ["beta", "gamma", "sigma"], "trace plot for theta in MAGI forecast", output_dir=output_dir)

# Save results and data as pickle files
output_data = {
    "ts_true": ts_true,
    "x_true": x_true,
    "results_forecast": results_forecast,
    "X_obs": X_obs,
    "results": results
}

pickle_file_path = os.path.join(output_dir, "simulation_results.pkl")
with open(pickle_file_path, "wb") as f:
    pickle.dump(output_data, f)

print(f"Results saved to {pickle_file_path}")
