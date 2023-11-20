# Script name: visualize_physio_respiratory.py

import os
import gzip
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is non-interactive but stable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bids import BIDSLayout
import logging
from statsmodels import robust

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Smooth data using a rolling window for visualization.
def smooth_data(data, window_size=11):
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

# Plot respiratory data across all participants.
def plot_all_respiratory(respiratory_files_derivatives):
    """
    Plot respiratory data for all subjects and runs with a Median Absolute Deviation (MAD) band.

    This function reads respiratory data from provided file paths, calculates the mean and MAD,
    and plots these on a single figure. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - respiratory_files_derivatives: List of file paths to the respiratory derivative data files.
    """
    logging.info("Plotting respiratory data for all subjects and runs.")

    all_respiratory_data = []  # List to hold all respiratory data across subjects and runs

    # Aggregate respiratory data from all files
    for file in respiratory_files_derivatives:
        try:
            with gzip.open(file, 'rt') as f:
                respiratory_data = pd.read_csv(f, delimiter='\t')
            if 'respiratory' in respiratory_data.columns:
                # Assuming each file represents a run, append the respiratory data
                all_respiratory_data.append(respiratory_data['respiratory'])
            else:
                logging.warning(f"Column 'respiratory' not found in file: {file}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")

    # Process and plot data if available
    if all_respiratory_data:
        # Concatenate all respiratory data into a single DataFrame
        all_respiratory_data_df = pd.concat(all_respiratory_data, axis=1)
        # Calculate the mean and MAD
        mean_respiratory = all_respiratory_data_df.mean(axis=1)
        mad_respiratory = robust.mad(all_respiratory_data_df, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(mean_respiratory, label='All Subjects Mean respiratory')
        plt.fill_between(range(len(mean_respiratory)), mean_respiratory - mad_respiratory, mean_respiratory + mad_respiratory, color='gray', alpha=0.25, label='Median Absolute Deviation')
        plt.title('Mean respiratory with Median Absolute Deviation')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (V)')
        plt.legend()
        plt.show()  # Display the plot.
    else:
        logging.warning("No respiratory data found to plot.")

# Plot respiratory data seperately for stroke and control groups.
def plot_group_respiratory(all_respiratory_data_df, group_name, color, ax):
    """
    Plot respiratory data for a specific group with a Median Absolute Deviation (MAD) band.

    This function takes pre-processed and concatenated respiratory data, calculates the mean and MAD,
    and plots these on the provided axes object. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - all_respiratory_data_df: DataFrame containing the concatenated respiratory data.
    - group_name: Name of the group (e.g., 'stroke', 'control') for labeling.
    - color: Color to use for the mean respiratory plot of the group.
    - ax: Matplotlib axes object for plotting.
    """
    logging.info(f"Plotting respiratory data for group: {group_name}")

    # Calculate the mean and MAD for all aggregated data
    mean_respiratory = all_respiratory_data_df.mean(axis=1)
    mad_respiratory = robust.mad(all_respiratory_data_df, axis=1)

    # Plot the MAD as a shaded band around the mean
    ax.fill_between(range(len(mean_respiratory)), mean_respiratory - mad_respiratory, mean_respiratory + mad_respiratory, color=color, alpha=0.3, label=f'{group_name} MAD', zorder=1)
   
    # Plot the mean respiratory line on top
    ax.plot(range(len(mean_respiratory)), mean_respiratory, label=f'{group_name} Mean', color=color, linewidth=1.5, alpha=0.7,zorder=2)

# Main script logic
def main():
    """
    Main function to visualize respiratory data from a BIDS dataset.

    The function performs the following steps:
    - Validates the presence of necessary directories and files.
    - Reads the participants.tsv file to determine the groups.
    - Plots the respiratory data for each group, including the mean and standard deviation bands.
    """
    logging.info("Starting main function")

    # Define and check BIDS root directory
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    logging.info(f"Checking BIDS root directory: {bids_root_dir}")
    if not os.path.exists(bids_root_dir):
        logging.warning(f"Directory not found: {bids_root_dir}")
        return

    # Define and check derivatives directory
    derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/learn/events/')
    logging.info(f"Checking derivatives directory: {derivatives_dir}")
    if not os.path.exists(derivatives_dir):
        logging.warning(f"Directory not found: {derivatives_dir}")
        return

    # Define and check participants file
    participants_file = os.path.join(bids_root_dir, 'participants.tsv')
    logging.info(f"Checking participants file: {participants_file}")
    if not os.path.exists(participants_file):
        logging.warning(f"Participants file not found: {participants_file}")
        return

    # Initialize BIDS Layout
    layout = BIDSLayout(bids_root_dir, validate=True)
    
    try: 
        # Read participants file and process groups
        participants_df = pd.read_csv(participants_file, delimiter='\t')
        
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error reading participants file: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading participants file: {e}")
        return    
    
    logging.info(f"Found {len(participants_df)} participants")
    groups = participants_df.groupby('group')  # Adjust 'group' to the actual group column name in your participants.tsv
    logging.info(f"Found {len(groups)} groups")

    group_colors = {'Stroke': 'red', 'Control': 'blue'}  # Define colors for each group

    # Create a figure with subplots for each run
    fig, axs = plt.subplots(8, 2, figsize=(10, 30))  # 8 runs x 2 types of trials

    legend_handles = {}

    for run_number in range(8):  # Processing all 8 runs
        for trial_index, trial_type in enumerate(['sequence', 'random']):
            logging.info(f"Processing run {run_number}, trial type: {trial_type}")
            for group_name, group_df in participants_df.groupby('group'):
                participant_ids = group_df['participant_id'].tolist()

                run_data = []
                for participant_id in participant_ids:
                    respiratory_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-2_task-learn_run-{run_number:02d}_recording-{trial_type}_physio.tsv.gz' in f]
                    logging.info(f"Found {len(respiratory_files)} files for participant {participant_id}")
                    for file in respiratory_files:
                        try:
                            with gzip.open(file, 'rt') as f:
                                respiratory_data = pd.read_csv(f, delimiter='\t')
                            if 'respiratory' in respiratory_data.columns:
                                smoothed_respiratory = smooth_data(respiratory_data['respiratory'])
                                run_data.append(smoothed_respiratory)
                            else:
                                logging.warning(f"Column 'respiratory' not found in file: {file}")
                        except Exception as e:
                            logging.error(f"Error reading file {file}: {e}")

                if run_data:
                    run_data_df = pd.concat(run_data, axis=1)
                    ax = axs[run_number, trial_index]
                    plot_group_respiratory(run_data_df, group_name, group_colors[group_name], ax)

                    # Capture the line handle for the legend
                    line = ax.get_lines()[-1]
                    legend_handles[group_name] = line

    # Customize and show the plot
    for i in range(8):
        axs[i, 0].set_title(f'Run {i+1} Sequence')
        axs[i, 1].set_title(f'Run {i+1} Random')
        axs[i, 0].set_ylabel('Amplitude (V)')
        axs[i, 1].set_ylabel('Amplitude (V)')

   # Add a single legend for all groups at the bottom of the figure
    fig.legend(handles=legend_handles.values(), labels=legend_handles.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(groups))

    fig.tight_layout()
    plt.savefig(f"{derivatives_dir}/plots/all_subs_ses-2_task-learn_all_events_respiratory.png")
    plt.show()

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()