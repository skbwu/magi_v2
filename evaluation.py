import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_rmse(true_values, inferred_samples, observed_indices):
    """
    Compute the Root Mean Square Error (RMSE) between the true values and inferred samples.

    Parameters:
    - true_values (np.array): Shape (T, D), true trajectories.
    - inferred_samples (np.array): Shape (num_samples, T, D), posterior samples.
    - observed_indices (np.array): Indices of observed time points.

    Returns:
    - float: RMSE value.
    """
    # Extract true values at observed points
    true_obs = true_values[observed_indices]

    # Compute the mean inferred trajectory
    inferred_mean = inferred_samples.mean(axis=0)[observed_indices]

    # Calculate RMSE
    mse = np.mean((true_obs - inferred_mean) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def compute_log_rmse(true_values_log, inferred_samples):
    """
    Compute RMSE on the log scale.

    Parameters:
    - true_values_log (np.array): Shape (T, D), true log-transformed trajectories.
    - inferred_samples (np.array): Shape (num_samples, T, D), posterior samples.
    - observed_indices (np.array): Indices of observed time points.

    Returns:
    - float: RMSE on log scale.
    """
    # Extract true log values at observed points
    true_obs_log = true_values_log

    # Compute the mean inferred log trajectory
    inferred_mean_log = inferred_samples.mean(axis=0)

    # Calculate RMSE
    mse_log = np.mean((true_obs_log - inferred_mean_log) ** 2, axis=0)
    rmse_log = np.sqrt(mse_log)
    return rmse_log


def compute_parameter_error(true_params, inferred_samples):
    """
    Compute parameter estimation errors.

    Parameters:
    - true_params (list or np.array): True parameter values.
    - inferred_samples (np.array): Shape (num_samples, P), posterior samples of parameters.

    Returns:
    - np.array: Absolute errors for each parameter.
    """
    inferred_mean = inferred_samples.mean(axis=0)
    errors = inferred_mean - np.array(true_params)
    return errors


def compute_parameter_rmse(true_params, inferred_samples):
    """
    Compute RMSE for parameter estimation.

    Parameters:
    - true_params (list or np.array): True parameter values.
    - inferred_samples (np.array): Shape (num_samples, P), posterior samples of parameters.

    Returns:
    - np.array: RMSE for each parameter.
    """
    inferred_mean = inferred_samples.mean(axis=0)
    mse = np.mean((inferred_mean - np.array(true_params)) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def summarize_simulation_results(results_dir, true_params, observed_time_points):
    """
    Summarize simulation results across multiple pickle files.

    Parameters:
    - results_dir (str): Path to the directory containing simulation result subdirectories.
    - true_params (list or np.array): True parameter values [beta, gamma, sigma].
    - observed_time_points (np.array): Array of observed time points.

    Returns:
    - pd.DataFrame: Summary table with RMSE and parameter errors for each simulation.
    """
    summary = []
    true_R0 = true_params[0] / (true_params[1] + true_params[2])  # R0 = beta / (gamma + sigma)
    true_params = np.concatenate([true_params, [true_R0]])

    # Iterate through each simulation directory
    for sim_dir in os.listdir(results_dir):
        sim_path = os.path.join(results_dir, sim_dir)
        if os.path.isdir(sim_path):
            pickle_file = os.path.join(sim_path, "simulation_results.pkl")
            if os.path.exists(pickle_file):
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)

                # Extract necessary data
                ts_true = data["ts_true"]  # Shape (T,)
                x_true = data["x_true"]  # Shape (T, D)
                results_forecast = data["results_forecast"]
                X_obs = data["X_obs"]  # Shape (observed_T, D)
                results = data["results"]  # Contains "thetas_samps", "X_samps", etc.

                # Identify observed time indices
                observed_indices_in_true = np.isin(np.round(ts_true, 5), np.round(observed_time_points, 5)).nonzero()[0]
                observed_indices_in_I = np.isin(np.round(results['I'], 5), np.round(observed_time_points, 5)).nonzero()[0]

                # Compute RMSE on log scale
                X_samps = results["X_samps"]  # Shape (num_samples, T, D)
                rmse_log = compute_log_rmse(x_true.loc[observed_indices_in_true, :], X_samps[:, observed_indices_in_I, :])

                # Compute parameter estimation errors
                thetas_samps = results["thetas_samps"]  # Shape (num_samples, P)

                beta_samples = thetas_samps[:, 0]
                gamma_samples = thetas_samps[:, 1]
                sigma_samples = thetas_samps[:, 2]
                R0_samples = beta_samples / (gamma_samples + sigma_samples)
                thetas_samps = np.hstack([thetas_samps, R0_samples.reshape(-1, 1)])

                param_errors = compute_parameter_error(true_params, thetas_samps)

                # Compute parameter coverage
                coverage = compute_coverage(true_params, thetas_samps, confidence_level=95)

                # Append results to summary
                summary.append({
                    "Simulation": sim_dir,
                    "RMSE_logE": rmse_log[0],
                    "RMSE_logI": rmse_log[1],
                    "RMSE_logR": rmse_log[2],
                    "Beta_Error": param_errors[0],
                    "Gamma_Error": param_errors[1],
                    "Sigma_Error": param_errors[2],
                    "R0_Error": param_errors[3],
                    "Beta_Coverage": coverage["Beta_Coverage"],
                    "Gamma_Coverage": coverage["Gamma_Coverage"],
                    "Sigma_Coverage": coverage["Sigma_Coverage"],
                    "R0_Coverage": coverage["R0_Coverage"]
                })

    # Convert summary to DataFrame
    summary_df = pd.DataFrame(summary)
    return summary_df


def generate_latex_table(summary_df, output_path="summary_table.tex"):
    """
    Generate a LaTeX table from the summary DataFrame.

    Parameters:
    - summary_df (pd.DataFrame): Summary DataFrame with error metrics.
    - output_path (str): Path to save the LaTeX table.

    Returns:
    - None
    """
    # Select columns excluding 'Simulation'
    metrics = summary_df.columns.drop("Simulation")

    # Compute mean and std of absolute values
    mean_values = np.abs(summary_df[metrics]).mean()
    std_values = np.abs(summary_df[metrics]).std()

    # Create a new DataFrame for the table
    table_df = pd.DataFrame({
        "Metric": metrics,
        "Mean Absolute Error": mean_values.values,
        "Std Dev": std_values.values
    })

    # Round the values for better presentation
    table_df["Mean Absolute Error"] = table_df["Mean Absolute Error"].round(4)
    table_df["Std Dev"] = table_df["Std Dev"].round(4)

    # Generate LaTeX table
    latex_table = table_df.to_latex(index=False, caption="Summary of Simulation Results", label="tab:summary_simulation_results", column_format="lcc", float_format="%.4f")

    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_path} as \n{latex_table}")

def visualize_forecast_means(results_dir, observed_time_points, output_dir="visualizations"):
    """
    Visualize the aggregated forecasted trajectories across all simulations.

    Parameters:
    - results_dir (str): Path to the directory containing simulation result subdirectories.
    - observed_time_points (np.array): Array of observed time points.
    - output_dir (str): Directory to save the plots.

    Returns:
    - None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to collect posterior mean trajectories from each dataset
    all_posterior_means = []

    # Iterate through each simulation directory
    for sim_dir in os.listdir(results_dir):
        sim_path = os.path.join(results_dir, sim_dir)
        if os.path.isdir(sim_path):
            pickle_file = os.path.join(sim_path, "simulation_results.pkl")
            if os.path.exists(pickle_file):
                try:
                    with open(pickle_file, "rb") as f:
                        data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {pickle_file}: {e}")
                    continue

                results_forecast = data["results_forecast"]
                X_samps = results_forecast["X_samps"]  # Shape (num_samples, T_forecast, D)

                # Compute posterior mean for this dataset
                posterior_mean = X_samps.mean(axis=0)  # Shape (T_forecast, D)
                all_posterior_means.append(posterior_mean)

    if not all_posterior_means:
        print("No forecast data found.")
        return

    # Convert list to numpy array
    # Shape: (num_datasets, T_forecast, D)
    all_posterior_means = np.array(all_posterior_means)

    num_datasets, T_forecast, D = all_posterior_means.shape

    # Compute aggregated statistics across datasets
    # Mean across datasets: Shape (T_forecast, D)
    aggregated_mean = np.mean(all_posterior_means, axis=0)

    # 2.5th and 97.5th percentiles across datasets: Shape (T_forecast, D)
    lower_quantile = np.percentile(all_posterior_means, 2.5, axis=0)
    upper_quantile = np.percentile(all_posterior_means, 97.5, axis=0)

    # Assume the time points are the same across simulations
    # Extract from one of the simulations
    sample_sim_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not sample_sim_dirs:
        print("No simulation directories found for extracting time points.")
        return

    sample_sim_path = os.path.join(results_dir, sample_sim_dirs[0])
    sample_pickle_file = os.path.join(sample_sim_path, "simulation_results.pkl")
    try:
        with open(sample_pickle_file, "rb") as f:
            sample_data = pickle.load(f)
        ts_forecast = sample_data["results_forecast"]["I"].flatten()  # Shape (T_forecast,)
    except Exception as e:
        print(f"Error loading {sample_pickle_file} for time points: {e}")
        return

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        "Time": ts_forecast,
        "E_mean": aggregated_mean[:, 0],
        "E_lower": lower_quantile[:, 0],
        "E_upper": upper_quantile[:, 0],
        "I_mean": aggregated_mean[:, 1],
        "I_lower": lower_quantile[:, 1],
        "I_upper": upper_quantile[:, 1],
        "R_mean": aggregated_mean[:, 2],
        "R_lower": lower_quantile[:, 2],
        "R_upper": upper_quantile[:, 2],
    })

    # Plot settings
    compartments = ["E", "I", "R"]
    colors = {"E": "blue", "I": "green", "R": "red"}

    for compartment in compartments:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_df["Time"], plot_df[f"{compartment}_mean"], label=f"Mean {compartment}", color=colors[compartment])
        plt.fill_between(plot_df["Time"], plot_df[f"{compartment}_lower"], plot_df[f"{compartment}_upper"], color=colors[compartment], alpha=0.3, label="95% CI")
        plt.title(f"Aggregated Forecasted Trajectory for {compartment}")
        plt.xlabel("Time")
        plt.ylabel(f"log({compartment})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{compartment}_aggregated_forecast.png"))
        plt.close()
        print(f"Saved aggregated forecast plot for {compartment} to {output_dir}")

    # Optionally, plot the average of the posterior means across datasets with true trajectories
    # Uncomment the following block if you wish to include true trajectories in the plots

    # Load true trajectories from one simulation (assuming all have the same ts_true)
    try:
        with open(sample_pickle_file, "rb") as f:
            sample_data = pickle.load(f)
        ts_true = sample_data["ts_true"]
        x_true = sample_data["x_true"]
        if not isinstance(x_true, pd.DataFrame):
            x_true = pd.DataFrame(x_true, columns=["E_true", "I_true", "R_true"])
        true_forecast = x_true.copy()
        true_forecast.index = ts_true

        X_obs = data["X_obs"]  # Shape (observed_T, D)
        ts_obs = observed_time_points  # Assuming observed_time_points align with X_obs
        example_observations = X_obs  # Shape (observed_T, D)

        # Extract true trajectory
        x_true = data["x_true"]  # pd.DataFrame with ['E_true', 'I_true', 'R_true']
        if not isinstance(x_true, pd.DataFrame):
            x_true = pd.DataFrame(x_true, columns=["E_true", "I_true", "R_true"])
        true_trajectory = x_true.values  # Shape (T, D)

    except Exception as e:
        print(f"Error loading true trajectories: {e}")
        true_forecast = None

    for compartment in compartments:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_df["Time"], plot_df[f"{compartment}_mean"], label=f"Mean {compartment}", color=colors[compartment])
        plt.fill_between(plot_df["Time"], plot_df[f"{compartment}_lower"], plot_df[f"{compartment}_upper"], color=colors[compartment], alpha=0.3, label="95% CI")
        if true_forecast is not None:
            plt.plot(true_forecast.index, true_forecast[f"{compartment}_true"], label=f"True {compartment}", color="black", linestyle="-")
        plt.title(f"Aggregated Forecasted Trajectory for {compartment}")
        plt.xlabel("Time")
        plt.ylabel(f"log({compartment})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{compartment}_aggregated_forecast_with_true.png"))
        plt.close()
        print(f"Saved aggregated forecast plot with true trajectory for {compartment} to {output_dir}")


    # Initialize the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for idx, compartment in enumerate(compartments):
        ax = axes[idx]
        # Plot aggregated mean trajectory
        ax.plot(plot_df["Time"], plot_df[f"{compartment}_mean"], label=f"Mean {compartment}", color=colors[compartment])

        # Shade the 95% confidence interval
        ax.fill_between(plot_df["Time"], plot_df[f"{compartment}_lower"], plot_df[f"{compartment}_upper"],
                        color=colors[compartment], alpha=0.3, label="95% CI")

        # Plot true trajectory
        if true_forecast is not None:
            ax.plot(true_forecast.index, true_forecast[f"{compartment}_true"], label=f"True {compartment}", color="black", linestyle="-")

        # Plot example dataset observations
        if example_observations is not None:
            # Find the indices in ts_forecast that correspond to observations
            # Assuming observed_time_points are a subset of ts_forecast
            observed_indices = np.isin(np.round(ts_forecast, 5), np.round(observed_time_points, 5))
            ax.scatter(ts_obs, example_observations[:, idx], label="Example Observations", color="orange", marker='o', s=100, edgecolors='k')

        ax.set_title(f"Forecasted Trajectory for {compartment}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"log({compartment})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"log_aggregated_forecast_with_true_and_sample_obs.png"))
    plt.close()
    print(f"Aggregated forecast figure saved to {output_dir}")

    plot_df.iloc[:,1:] = np.exp(plot_df.iloc[:,1:])
    true_forecast = np.exp(true_forecast)
    example_observations = np.exp(example_observations)

    # Initialize the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for idx, compartment in enumerate(compartments):
        ax = axes[idx]
        # Plot aggregated mean trajectory
        ax.plot(plot_df["Time"], plot_df[f"{compartment}_mean"], label=f"Mean {compartment}", color=colors[compartment])

        # Shade the 95% confidence interval
        ax.fill_between(plot_df["Time"], plot_df[f"{compartment}_lower"], plot_df[f"{compartment}_upper"],
                        color=colors[compartment], alpha=0.3, label="95% CI")

        # Plot true trajectory
        if true_forecast is not None:
            ax.plot(true_forecast.index, true_forecast[f"{compartment}_true"], label=f"True {compartment}", color="black", linestyle="-")

        # Plot example dataset observations
        if example_observations is not None:
            # Find the indices in ts_forecast that correspond to observations
            # Assuming observed_time_points are a subset of ts_forecast
            observed_indices = np.isin(np.round(ts_forecast, 5), np.round(observed_time_points, 5))
            ax.scatter(ts_obs, example_observations[:, idx], label="Example Observations", color="orange", marker='o', s=100, edgecolors='k')

        ax.set_title(f"Forecasted Trajectory for {compartment}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"log({compartment})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"orig_aggregated_forecast_with_true_and_sample_obs.png"))
    plt.close()
    print(f"Aggregated forecast figure saved to {output_dir}")


def compute_coverage(true_params, inferred_samples, confidence_level=95):
    """
    Compute the coverage of the true parameters within the posterior intervals.

    Parameters:
    - true_params (list or np.array): True parameter values [beta, gamma, sigma].
    - inferred_samples (np.array): Shape (num_samples, P), posterior samples of parameters.
    - confidence_level (float): The confidence level for the interval (default is 95).

    Returns:
    - dict: Coverage information for each parameter.
    """
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    coverage = {}
    for i, param in enumerate(['Beta', 'Gamma', 'Sigma', 'R0']):
        lower = np.percentile(inferred_samples[:, i], lower_percentile)
        upper = np.percentile(inferred_samples[:, i], upper_percentile)
        is_covered = lower <= true_params[i] <= upper
        coverage[f"{param}_Coverage"] = is_covered
    return coverage
