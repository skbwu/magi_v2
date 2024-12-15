import numpy as np
import os
import matplotlib.pyplot as plt

def plot_trajectories(ts_true, x_true, results, ts_obs, X_obs, trans_func=lambda x: x, caption_text="", output_dir=None):
    """
    Plot the predicted trajectories against ground truth and noisy observations.

    Parameters:
    - ts_true: Array of time steps for the ground truth data.
    - x_true: DataFrame or array of ground truth data (columns for each component, e.g., ["E", "I", "R"]).
    - results: Dictionary containing prediction results from MAGI-TFP.
    - ts_obs: Array of time steps corresponding to noisy observations.
    - X_obs: Array of noisy observations.
    - trans_func: Transformation function to apply to all plotted data (default: identity function).
    """
    fig, ax = plt.subplots(1, x_true.shape[1], dpi=200, figsize=(12, 6))
    I = results["I"].flatten()
    Xhat_means = trans_func(results["X_samps"].mean(axis=0))
    Xhat_intervals = np.quantile(trans_func(results["X_samps"]), q=[0.025, 0.975], axis=0)
    Xhat_init = trans_func(results["Xhat_init"])

    for i in range(x_true.shape[1]):
        comp = x_true.columns[i]
        true_values = trans_func(x_true[comp].values)

        # Plot ground truth trajectory
        ax[i].plot(ts_true, true_values, color="black", label="Ground Truth")
        # Plot mean trajectory and predictive interval
        ax[i].plot(I, Xhat_means[:, i], color="blue", label="Mean Prediction")
        ax[i].fill_between(I, Xhat_intervals[0, :, i], Xhat_intervals[1, :, i], color="blue", alpha=0.3, label="95% Predictive Interval")
        ax[i].plot(I, Xhat_init[:, i], linestyle="--", color="green", label="Initialization")
        if "Xhat_mle" in results:
            ax[i].plot(I, trans_func(results["Xhat_mle"])[:, i], linestyle="--", color="red", label="MLE")
        # Plot noisy observations
        ax[i].scatter(ts_obs, trans_func(X_obs[:, i]), color="grey", s=20, zorder=5, label="Noisy Observations")
        # Titles and labels
        ax[i].set_title(f"Component {comp}")
        ax[i].set_xlabel("$t$")
        ax[i].set_ylabel(f"{comp}")
        ax[i].grid()

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10)
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, color='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_dir:
        plt.savefig(f"{output_dir}/f{caption_text.replace(' ', '_')}.png")
    else:
        plt.show()


def plot_trace(thetas_samps, true_values=None, param_names=["$\\beta$", "$\\gamma$", "$\\sigma$"],
               caption_text="", output_dir=None):
    """
    Plot trace plots and histograms for sampled parameters.

    Parameters:
    - thetas_samps: Array of sampled parameters with shape (num_samples, num_params).
    - true_values: List of true parameter values for comparison (default: None).
    - param_names: List of parameter names for labeling (default: ["$\\beta$", "$\\gamma$", "$\\sigma$"]).
    """
    num_params = len(param_names)
    fig, ax = plt.subplots(num_params, 2, figsize=(12, 3 * num_params), dpi=200)

    for i, param in enumerate(param_names):
        # Trace plot
        ax[i, 0].plot(thetas_samps[:, i], alpha=0.7, lw=1, color="blue")
        ax[i, 0].set_title(f"Trace Plot for {param}", fontsize=12)
        ax[i, 0].set_ylabel("Value")
        ax[i, 0].grid()

        # Add true value as a horizontal line if provided
        if true_values is not None:
            ax[i, 0].axhline(y=true_values[i], color="red", linestyle="--", label="True Value")
            ax[i, 0].legend(loc="upper right")

        # Histogram
        ax[i, 1].hist(thetas_samps[:, i], bins=30, density=True, orientation="horizontal", color="green", alpha=0.7)
        ax[i, 1].set_title(f"Histogram for {param}", fontsize=12)
        ax[i, 1].set_xlabel("Density")
        ax[i, 1].grid()

        # Add a vertical red line for the true value
        if true_values is not None:
            ax[i, 1].axhline(y=true_values[i], color="red", linestyle="--", label="True Value")
            ax[i, 1].legend(loc="upper right")

        # Match the y-axis of the histogram to the trace plot for visual consistency
        ax[i, 1].set_ylim(ax[i, 0].get_ylim())

    # Label the x-axis of the trace plots at the bottom row
    ax[-1, 0].set_xlabel("Iteration")
    fig.text(0.5, 0.01, caption_text, ha='center', fontsize=10, color='gray')
    plt.tight_layout()

    if output_dir:
        os.makedirs(f"{output_dir}", exist_ok=True)
        plt.savefig(f"{output_dir}/{caption_text.replace(' ', '_')}.png")
    else:
        plt.show()


def print_parameter_estimates(results, true_values):
    """
    Print the estimated parameters against the true values.

    Parameters:
    - results: Dictionary containing prediction results from MAGI-TFP.
    - true_values: List of true parameter values [beta, gamma, sigma].
    """
    mean_thetas_pred = results["thetas_samps"].mean(axis=0)
    print("Estimated Parameters:")
    for i, (predicted, actual) in enumerate(zip(mean_thetas_pred, true_values)):
        print(f"- Parameter {i + 1}: {np.round(predicted, 3)} (Predicted) vs. {actual} (Actual).")

