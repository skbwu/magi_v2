import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# General function to plot and save results
def plot_magi_pinn_comparison(magi_data, pinn_data, output_name='partial'):
    """
    Plot MAGI and PINN forecasts for E, I, R compartments in a single figure.

    Parameters:
    - magi_data (dict): MAGI data containing plot_df, true_forecast, etc.
    - pinn_data (dict): PINN data containing plot_df_true, plot_df_partial, etc.
    - output_name (str): Output file name to save the plot.
    """
    # Extract relevant data
    pinn_plot_df_true = pinn_data['plot_df_true']
    pinn_plot_df_partial = pinn_data['plot_df_partial']
    pinn_truth = pinn_data['truth']

    magi_plot_df = magi_data['plot_df']
    magi_true_forecast = magi_data['true_forecast']
    magi_example_observations = magi_data['example_observations']
    magi_ts_obs = magi_data['ts_obs']

    # Compartments and colors
    compartments = ["E", "I", "R"]
    colors = {"E": "blue", "I": "green", "R": "red"}

    # Initialize combined figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Top row: MAGI plots
    for idx, compartment in enumerate(compartments):
        ax = axes[0, idx]
        ax.plot(magi_plot_df["Time"], magi_plot_df[f"{compartment}_mean"], label="MAGI Mean", color=colors[compartment])
        ax.fill_between(magi_plot_df["Time"], magi_plot_df[f"{compartment}_lower"], magi_plot_df[f"{compartment}_upper"],
                        color=colors[compartment], alpha=0.3, label="MAGI 95% CI")
        if magi_true_forecast is not None:
            ax.plot(magi_true_forecast.index, magi_true_forecast[f"{compartment}_true"], color="black", linestyle="--", label="True")
        if magi_example_observations is not None:
            ax.scatter(magi_ts_obs, magi_example_observations[:, idx], color="orange", marker="o", s=50, edgecolors='k', label="Example Observations")
        ax.set_title(f"MAGI {compartment} Forecast")
        if idx == 0:
            ax.set_ylabel("Trajectory Values")
        ax.legend()

    if output_name == 'full':
        pinn_plot_df = pinn_plot_df_true
    else:
        pinn_plot_df = pinn_plot_df_partial

    # Bottom row: PINN plots
    for idx, compartment in enumerate(compartments):
        ax = axes[1, idx]
        ax.plot(pinn_plot_df["Time"], pinn_plot_df[f"{compartment}_mean"], label="PINN True Mean", color=colors[compartment])
        ax.fill_between(pinn_plot_df["Time"], pinn_plot_df[f"{compartment}_lower"], pinn_plot_df[f"{compartment}_upper"],
                        color=colors[compartment], alpha=0.3, label="PINN Partial 95% CI")
        ax.plot(magi_true_forecast.index, magi_true_forecast[f"{compartment}_true"], color="black", linestyle="--",
                label="True")
        ax.set_title(f"PINN {compartment} Forecast")
        if idx == 0:
            ax.set_ylabel("Trajectory Values")
        ax.legend()

    # Add common x and y labels
    fig.text(0.5, 0.04, "Time", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "Trajectory Values (log scale)", va='center', rotation='vertical', fontsize=12)

    # Save the combined figure
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(f"seir_magi_pinn_2x3_{output_name}.png", dpi=300)
    plt.show()
    print(f"Figure saved to {output_name}")

# Load pickle files
pinn_data = load_pickle('results/seir_pinn_plot_data.pkl')
magi_data_full = load_pickle('fully_observed/seir_magi_plot_data.pkl')
magi_data_partial = load_pickle('large run/seir_magi_plot_data.pkl')

# Generate Full Observation Case
plot_magi_pinn_comparison(magi_data_full, pinn_data, "full")

# Generate Partial Observation Case
plot_magi_pinn_comparison(magi_data_partial, pinn_data, "partial")


import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_magi_pinn_comparison(magi_data, pinn_data, output_name='partial'):
    """
    Plot MAGI and PINN forecasts for E, I, R compartments in a single figure.
    Both MAGI and PINN forecasts with their confidence intervals are plotted
    on the same axes for each compartment.

    Parameters:
    - magi_data (dict): MAGI data containing plot_df, true_forecast, etc.
    - pinn_data (dict): PINN data containing plot_df_true, plot_df_partial, etc.
    - output_name (str): Output file name to save the plot.
    """
    # Extract relevant data
    pinn_plot_df_true = pinn_data['plot_df_true']
    pinn_plot_df_partial = pinn_data['plot_df_partial']

    magi_plot_df = magi_data['plot_df']
    magi_true_forecast = magi_data['true_forecast']
    magi_example_observations = magi_data['example_observations']
    magi_ts_obs = magi_data['ts_obs']

    # Determine which PINN dataframe to use
    if output_name == 'full':
        pinn_plot_df = pinn_plot_df_true
    else:
        pinn_plot_df = pinn_plot_df_partial

    # Compartments we are plotting
    compartments = ["E", "I", "R"]

    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for idx, compartment in enumerate(compartments):
        ax = axes[idx]

        # Plot MAGI mean and CI
        ax.plot(magi_plot_df["Time"], magi_plot_df[f"{compartment}_mean"],
                label="MAGI Mean", color="red", zorder=2)

        ax.fill_between(magi_plot_df["Time"], magi_plot_df[f"{compartment}_lower"], magi_plot_df[f"{compartment}_upper"],
                        facecolor="none", edgecolor="red", hatch='/', alpha=0.5,
                        label="MAGI 95% CI", zorder=1)

        # Plot PINN mean and CI
        ax.plot(pinn_plot_df["Time"], pinn_plot_df[f"{compartment}_mean"],
                label="PINN Mean", color="blue", zorder=2)

        ax.fill_between(pinn_plot_df["Time"], pinn_plot_df[f"{compartment}_lower"], pinn_plot_df[f"{compartment}_upper"],
                        facecolor="none", edgecolor="blue", hatch='\\', alpha=0.5,
                        label="PINN 95% CI", zorder=1)

        # Plot observations if any
        if magi_example_observations is not None:
            ax.scatter(magi_ts_obs, magi_example_observations[:, idx], color="orange",
                       marker="o", s=50, edgecolors='k', label="Example Observations", zorder=3)

        # Plot the true trajectory last, with a thicker line and higher zorder
        if magi_true_forecast is not None:
            ax.plot(magi_true_forecast.index, magi_true_forecast[f"{compartment}_true"],
                    color="black", linestyle="-", linewidth=3, label="True", zorder=4)

        ax.set_title(f"{compartment} Forecast Comparison")
        if idx == 0:
            ax.set_ylabel("Trajectory Values")
        ax.set_xlabel("Time")
        ax.legend()

    # Add common labels
    fig.text(0.5, 0.0, "Time", ha="center", fontsize=12)
    fig.text(0.0, 0.5, "Trajectory Values (log scale)", va='center', rotation='vertical', fontsize=12)

    # Save the combined figure
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(f"seir_magi_pinn_1x3_{output_name}.png", dpi=300)
    plt.show()
    print(f"Figure saved to {output_name}")

# Load pickle files
pinn_data = load_pickle('results/seir_pinn_plot_data.pkl')
magi_data_full = load_pickle('fully_observed/seir_magi_plot_data.pkl')
magi_data_partial = load_pickle('large run/seir_magi_plot_data.pkl')

# Generate Full Observation Case
plot_magi_pinn_comparison(magi_data_full, pinn_data, "full")

# Generate Partial Observation Case
plot_magi_pinn_comparison(magi_data_partial, pinn_data, "partial")
