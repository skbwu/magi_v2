import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

##############################
# Define the ODE system
##############################
def f_vec(t, X, thetas):
    '''
    Log-scale EIR model with E implicitly represented.
    ODE:
      dE/dt = beta * S * I - sigma * E
      dI/dt = sigma * E - gamma * I
      dR/dt = gamma * I
    where S = 1 - E - I - R

    On the log scale: X = [logE, logI, logR]

    Parameters
    ----------
    t : float
        Time
    X : np.array (3,)
        [logE, logI, logR]
    thetas : array-like
        [beta, gamma, sigma]

    Returns
    -------
    dXdt : np.array (3,)
        Derivatives in log scale
    '''
    logE, logI, logR = X
    beta, gamma, sigma = thetas

    E = np.exp(logE)
    I = np.exp(logI)
    R = np.exp(logR)

    S = 1.0 - E - I - R
    if S <= 0.0:
        S = 1e-10  # small positive value to avoid instability

    # Original scale derivatives
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Log scale derivatives
    dlogEdt = dEdt / E
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    return np.array([dlogEdt, dlogIdt, dlogRdt])

def ODE_log_scale(t, y, theta):
    return f_vec(t, y, theta)

##############################
# Negative log-likelihood function
##############################
# Negative log-likelihood function
def negative_log_likelihood(params, ts_obs, logI_obs, logR_obs):
    # params = [log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs]
    log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs = params

    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    sigma = np.exp(log_sigma)
    sigma_obs = np.exp(log_sigma_obs)

    thetas = [beta, gamma, sigma]
    X0 = np.array([logE0, logI0, logR0])

    # Solve ODE
    sol = solve_ivp(lambda t, y: ODE_log_scale(t, y, thetas),
                    t_span=(ts_obs[0], ts_obs[-1]),
                    y0=X0, t_eval=ts_obs, rtol=1e-10, atol=1e-10)

    if not sol.success:
        return np.inf

    X_pred = sol.y.T
    logI_pred = X_pred[:, 1]
    logR_pred = X_pred[:, 2]

    mask_I = ~np.isnan(logI_obs)
    mask_R = ~np.isnan(logR_obs)

    residuals_I = (logI_obs[mask_I] - logI_pred[mask_I])**2
    residuals_R = (logR_obs[mask_R] - logR_pred[mask_R])**2

    SSE = np.sum(residuals_I) + np.sum(residuals_R)
    N = np.sum(mask_I) + np.sum(mask_R)

    # Negative log-likelihood with unknown sigma_obs:
    # NLL = N*log(sigma_obs) + SSE/(2*sigma_obs^2) + constant
    # The constant doesn't affect optimization.
    NLL = N * log_sigma_obs + SSE / (2.0 * sigma_obs**2)

    return NLL

# Main function for running optimization and visualization
def mle(ts_obs, X_obs_full, maxiter=1000):
    logI_obs = X_obs_full[:, 1]
    logR_obs = X_obs_full[:, 2]

    # Initial guesses for parameters
    init_log_beta = 0.5 * np.random.randn()
    init_log_gamma = 0.5 * np.random.randn()
    init_log_sigma = 0.5 * np.random.randn()
    init_logI0 = logI_obs[0] if not np.isnan(logI_obs[0]) else -4.0
    init_logR0 = logR_obs[0] if not np.isnan(logR_obs[0]) else -4.0
    init_logE0 = -4.0
    init_log_sigma_obs = np.log(0.1)  # starting guess for sigma_obs

    init_guess = [init_log_beta, init_log_gamma, init_log_sigma,
                  init_logE0, init_logI0, init_logR0, init_log_sigma_obs]

    # Adjust bounds if needed. For sigma_obs, let's allow it to vary widely:
    bounds = [(-5, 5), (-3, 3), (-3, 3), (-10, 0), (-10, 5), (-10, 5), (-5, 0)]
    # Here, (-5,0) for log_sigma_obs means sigma_obs is between exp(-5)~0.0067 and exp(0)=1.0. Adjust as needed.

    res = minimize(negative_log_likelihood, init_guess,
                   args=(ts_obs, logI_obs, logR_obs),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})

    log_beta_est, log_gamma_est, log_sigma_est, logE0_est, logI0_est, logR0_est, log_sigma_obs_est = res.x

    beta_est = np.exp(log_beta_est)
    gamma_est = np.exp(log_gamma_est)
    sigma_est = np.exp(log_sigma_est)
    sigma_obs_est = np.exp(log_sigma_obs_est)

    print("Best fit parameters (on natural scale):")
    print("beta:         ", beta_est)
    print("gamma:        ", gamma_est)
    print("sigma:        ", sigma_est)
    print("E0:           ", np.exp(logE0_est))
    print("I0:           ", np.exp(logI0_est))
    print("R0:           ", np.exp(logR0_est))
    print("sigma_obs:    ", sigma_obs_est)

    final_thetas = [beta_est, gamma_est, sigma_est]
    X0_final = np.array([logE0_est, logI0_est, logR0_est])

    # Compute approximate uncertainty
    hess_inv_mat = res.hess_inv.todense()  # inverse Hessian approximation
    se = np.sqrt(np.diag(hess_inv_mat))  # standard errors on log-scale
    z_val = 1.96
    lower_log = res.x - z_val * se
    upper_log = res.x + z_val * se

    param_names = ["beta", "gamma", "sigma", "E0", "I0", "R0", "sigma_obs"]
    print("\nApproximate 95% CI (on log scale):")
    for i, p in enumerate(param_names):
        print(f"{p}: [{lower_log[i]:.4f}, {upper_log[i]:.4f}]")

    lower_nat = np.exp(lower_log)
    upper_nat = np.exp(upper_log)
    print("\nApproximate 95% CI (on natural scale):")
    for i, p in enumerate(param_names):
        print(f"{p}: [{lower_nat[i]:.4f}, {upper_nat[i]:.4f}]")

    return final_thetas, X0_final, sigma_obs_est, res.fun, (res.x, se, lower_nat, upper_nat), res

##############################
# Metropolis-Hastings Sampler
##############################
def metropolis_hastings(initial_params, n_samples, proposal_cov, ts_obs, logI_obs, logR_obs):
    # log_posterior is proportional to -NLL, so we can just use negative_log_likelihood and convert
    # For MH, we actually only need NLL and use exp(-NLL)
    # We'll store samples in a chain:
    chain = np.zeros((n_samples, len(initial_params)))
    chain[0] = initial_params

    current_params = initial_params.copy()
    current_nll = negative_log_likelihood(current_params, ts_obs, logI_obs, logR_obs)
    accepted = 0

    for i in range(1, n_samples):
        # Propose new parameters
        proposal = np.random.multivariate_normal(current_params, proposal_cov)
        # Compute NLL for proposal
        proposal_nll = negative_log_likelihood(proposal, ts_obs, logI_obs, logR_obs)

        # Metropolis criterion
        # alpha = exp(-(proposal_nll - current_nll)) = exp(-proposal_nll)/exp(-current_nll)
        # If alpha >= 1, accept. If alpha < 1, accept with probability alpha.
        alpha = np.exp(-(proposal_nll - current_nll))

        if np.random.rand() < alpha:
            # Accept
            current_params = proposal
            current_nll = proposal_nll
            accepted += 1

        chain[i] = current_params

    acceptance_rate = accepted / (n_samples - 1)
    return chain, acceptance_rate


def simulate_trajectory(params, t_eval):
    """
    Given a set of log-scale parameters and initial conditions, simulate the ODE trajectory.
    params: [log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs]
    t_eval: array of times to evaluate solution
    Returns: E(t), I(t), R(t) on the original (not log) scale.
    """
    log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs = params
    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    sigma = np.exp(log_sigma)
    thetas = np.array([beta, gamma, sigma])
    X0 = np.array([logE0, logI0, logR0])
    sol = solve_ivp(lambda t, y: ODE_log_scale(t, y, thetas), (t_eval[0], t_eval[-1]), X0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    E = np.exp(sol.y[0])
    I = np.exp(sol.y[1])
    R = np.exp(sol.y[2])
    return E, I, R


def plot_mcmc(samples, raw_data, X_obs, ts_obs, final_thetas, X0_final, t_max=2.0, n_pred_samples=1000,
              caption_text="MCMC on MLE", output_dir=None):
    # Extract parameter names and indices
    # Define a fine time grid for plotting
    t_fine = np.linspace(ts_obs[0], t_max, 200)

    # Sample a subset of posterior samples (to reduce computational cost if needed)
    idxs = np.random.choice(samples.shape[0], size=n_pred_samples, replace=False)
    E_pred_samples = []
    I_pred_samples = []
    R_pred_samples = []

    for i in idxs:
        E_, I_, R_ = simulate_trajectory(samples[i], t_fine)
        E_pred_samples.append(E_)
        I_pred_samples.append(I_)
        R_pred_samples.append(R_)

    E_pred_samples = np.array(E_pred_samples)  # shape: (n_pred_samples, len(t_fine))
    I_pred_samples = np.array(I_pred_samples)
    R_pred_samples = np.array(R_pred_samples)

    # Compute posterior mean and 95% credible intervals
    E_mean = np.mean(E_pred_samples, axis=0)
    I_mean = np.mean(I_pred_samples, axis=0)
    R_mean = np.mean(R_pred_samples, axis=0)

    E_lower = np.percentile(E_pred_samples, 2.5, axis=0)
    E_upper = np.percentile(E_pred_samples, 97.5, axis=0)

    I_lower = np.percentile(I_pred_samples, 2.5, axis=0)
    I_upper = np.percentile(I_pred_samples, 97.5, axis=0)

    R_lower = np.percentile(R_pred_samples, 2.5, axis=0)
    R_upper = np.percentile(R_pred_samples, 97.5, axis=0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Ground truth (if available)
    if 'E_true' in raw_data.columns:
        E_true = raw_data.query("t <= @t_max")['E_true'].values
        I_true = raw_data.query("t <= @t_max")['I_true'].values
        R_true = raw_data.query("t <= @t_max")['R_true'].values
        t_true = raw_data.query("t <= @t_max")['t'].values
        axes[0].plot(t_true, E_true, 'k-', label='Ground Truth')
        axes[1].plot(t_true, I_true, 'k-', label='Ground Truth')
        axes[2].plot(t_true, R_true, 'k-', label='Ground Truth')

    # Posterior Mean and intervals
    axes[0].plot(t_fine, E_mean, color='blue', label='Mean Prediction')
    axes[0].fill_between(t_fine, E_lower, E_upper, color='blue', alpha=0.3, label='95% Predictive Interval')

    axes[1].plot(t_fine, I_mean, color='blue', label='Mean Prediction')
    axes[1].fill_between(t_fine, I_lower, I_upper, color='blue', alpha=0.3, label='95% Predictive Interval')

    axes[2].plot(t_fine, R_mean, color='blue', label='Mean Prediction')
    axes[2].fill_between(t_fine, R_lower, R_upper, color='blue', alpha=0.3, label='95% Predictive Interval')

    # Observations
    axes[0].scatter(ts_obs, X_obs[:, 0], color='gray', alpha=0.8, label='Noisy Observations')
    axes[1].scatter(ts_obs, X_obs[:, 1], color='gray', alpha=0.8, label='Noisy Observations')
    axes[2].scatter(ts_obs, X_obs[:, 2], color='gray', alpha=0.8, label='Noisy Observations')

    # Initialization (e.g., from MLE solution)
    beta_est, gamma_est, sigma_est = final_thetas
    X0_est = X0_final
    thetas_init = [beta_est, gamma_est, sigma_est]
    sol_init = solve_ivp(lambda t, y: ODE_log_scale(t, y, thetas_init), (t_fine[0], t_fine[-1]), X0_est, t_eval=t_fine,
                         rtol=1e-10, atol=1e-10)
    E_init = np.exp(sol_init.y[0])
    I_init = np.exp(sol_init.y[1])
    R_init = np.exp(sol_init.y[2])

    axes[0].plot(t_fine, E_init, '--', color='green', label='Initialization')
    axes[1].plot(t_fine, I_init, '--', color='green', label='Initialization')
    axes[2].plot(t_fine, R_init, '--', color='green', label='Initialization')

    axes[0].set_title("Component E_true")
    axes[1].set_title("Component I_true")
    axes[2].set_title("Component R_true")

    axes[0].set_xlabel("t")
    axes[1].set_xlabel("t")
    axes[2].set_xlabel("t")

    axes[0].set_ylabel("E_true")
    axes[1].set_ylabel("I_true")
    axes[2].set_ylabel("R_true")

    # Combine legends
    handles, labels = axes[0].get_legend_handles_labels()
    # Just ensure we get all unique labels
    handles2, labels2 = axes[1].get_legend_handles_labels()
    handles += handles2
    labels += labels2
    handles3, labels3 = axes[2].get_legend_handles_labels()
    handles += handles3
    labels += labels3

    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=5)

    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if output_dir:
        plt.savefig(f"{output_dir}/f{caption_text.replace(' ', '_')}.png")
    else:
        plt.show()
