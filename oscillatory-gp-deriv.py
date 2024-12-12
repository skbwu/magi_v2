import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import clear_output

# core MAGI-TFP class
import tfpigp.magi_v2 as magi_v2

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

# initial data settings
d_obs = 20 # no. of observations per unit time
t_max = 2.0 # length of observation interval

# load in our data, thinning based on density of observations
raw_data = pd.read_csv('tfpigp/data/SEIR_beta=6_gamma=0.6_sigma=1.8_alpha=0.05_seed=2.csv').query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = obs_data.t.values.astype(np.float64)

# get the noisy observations
X_obs = obs_data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64) # S is implicit!

# create our model - f_vec is the ODE governing equations function defined earlier.
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# fit Matern kernel hyperparameters (phi1, phi2) as well as (Xhat_init, sigma_sqs_init, thetas_init)
# phi_exo = {
#     'phi1s': np.array([0.00630768, 0.00512528, 0.00489537]),
#     'phi2s': np.array([0.50114429, 0.50701198, 0.46386214]),
#     'sigma_sqs': np.array([1e-4, 1e-4, 1e-4]),
# }
phi_exo = None

model.initial_fit(discretization=2, verbose=True, use_fourier_prior=False, phi_exo=phi_exo)
model.phi1s
model.phi2s
model.sigma_sqs_init

# clear console for pretty output
clear_output(wait=True)

# collect our samples from NUTS posterior sampling - toggle tempering=True to use log-tempering.
results = model.predict(num_results=1000, num_burnin_steps=1000, tempering=False, verbose=True)

# Visualize our trajectories
fig, ax = plt.subplots(1, 3, dpi=200, figsize=(12, 6))

# Get timesteps, mean trajectory predictions, and 2.5% + 97.5% predictive intervals
I = results["I"].flatten()
Xhat_means = results["X_samps"].mean(axis=0)
Xhat_intervals = np.quantile(results["X_samps"], q=[0.025, 0.975], axis=0)
Xhat_init = results["Xhat_init"]

# Loop through each component and plot
for i, comp in enumerate(["$E$", "$I$", "$R$"]):

    # Plot ground truth trajectory
    ax[i].plot(raw_data["t"], raw_data[["E_true", "I_true", "R_true"][i]], color="black", label="Ground Truth")

    # Plot mean trajectory and 95% predictive interval
    ax[i].plot(I, Xhat_means[:, i], color="blue", label="Mean Prediction")
    ax[i].fill_between(I, Xhat_intervals[0, :, i], Xhat_intervals[1, :, i], color="blue", alpha=0.3, label="95% Predictive Interval")
    ax[i].plot(I, Xhat_init[:, i], linestyle="--", color="green", label="Initialization")

    # Plot noisy observations
    ax[i].scatter(ts_obs, X_obs[:, i], color="grey", s=20, zorder=5, label="Noisy Observations")

    # Titles and labels
    ax[i].set_title(f"Component {comp}")
    ax[i].set_xlabel("$t$")
    ax[i].set_ylabel(f"{comp}")
    ax[i].grid()

# Set shared legend at a better position to avoid overlap
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10)

# Beautify the layout
# plt.suptitle("Predicted Trajectories vs. Ground Truth", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# check our parameters too
mean_thetas_pred = results["thetas_samps"].mean(axis=0)

print("Estimated Parameters:")
print(f"- Beta: {np.round(mean_thetas_pred[0], 3)} (Predicted) vs. 6.0 (Actual).")
print(f"- Gamma: {np.round(mean_thetas_pred[1], 3)} (Predicted) vs. 0.6 (Actual).")
print(f"- Sigma: {np.round(mean_thetas_pred[2], 3)} (Predicted) vs. 1.8 (Actual).")

import matplotlib.pyplot as plt

# Extract the parameter samples
thetas_samps = results["thetas_samps"]
beta_samples = thetas_samps[:, 0]
gamma_samples = thetas_samps[:, 1]
sigma_samples = thetas_samps[:, 2]

# Create trace plots for each parameter
fig, ax = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True)

# Beta trace plot
ax[0].plot(beta_samples, color='blue', alpha=0.7, lw=1)
ax[0].set_title("Trace Plot for $\\beta$ (Beta)", fontsize=12)
ax[0].set_ylabel("Value")
ax[0].grid()

# Gamma trace plot
ax[1].plot(gamma_samples, color='green', alpha=0.7, lw=1)
ax[1].set_title("Trace Plot for $\\gamma$ (Gamma)", fontsize=12)
ax[1].set_ylabel("Value")
ax[1].grid()

# Sigma trace plot
ax[2].plot(sigma_samples, color='red', alpha=0.7, lw=1)
ax[2].set_title("Trace Plot for $\\sigma$ (Sigma)", fontsize=12)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Value")
ax[2].grid()

# Adjust layout and show
plt.tight_layout()
plt.show()
