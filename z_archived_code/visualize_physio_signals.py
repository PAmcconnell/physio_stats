# Script name: visualize_physio.py

import os
import gzip
import pandas as pd
import numpy as np
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

# Plot ECG data across all participants.
def plot_all_ecg(ecg_files_derivatives):
    """
    Plot ECG data for all subjects and runs with a Median Absolute Deviation (MAD) band.

    This function reads ECG data from provided file paths, calculates the mean and MAD,
    and plots these on a single figure. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - ecg_files_derivatives: List of file paths to the ECG derivative data files.
    """
    logging.info("Plotting ECG data for all subjects and runs.")

    all_ecg_data = []  # List to hold all ECG data across subjects and runs

    # Aggregate ECG data from all files
    for file in ecg_files_derivatives:
        try:
            with gzip.open(file, 'rt') as f:
                ecg_data = pd.read_csv(f, delimiter='\t')
            if 'cardiac' in ecg_data.columns:
                # Assuming each file represents a run, append the cardiac data
                all_ecg_data.append(ecg_data['cardiac'])
            else:
                logging.warning(f"Column 'cardiac' not found in file: {file}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")

    # Process and plot data if available
    if all_ecg_data:
        # Concatenate all ECG data into a single DataFrame
        all_ecg_data_df = pd.concat(all_ecg_data, axis=1)
        # Calculate the mean and MAD
        mean_ecg = all_ecg_data_df.mean(axis=1)
        mad_ecg = robust.mad(all_ecg_data_df, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(mean_ecg, label='All Subjects Mean ECG')
        plt.fill_between(range(len(mean_ecg)), mean_ecg - mad_ecg, mean_ecg + mad_ecg, color='gray', alpha=0.5, label='Median Absolute Deviation')
        plt.title('Mean ECG with Median Absolute Deviation')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.show()  # Display the plot.
    else:
        logging.warning("No ECG data found to plot.")

# Plot ECG data seperately for stroke and control groups.
def plot_group_ecg(all_ecg_data_df, group_name, color, ax):
    """
    Plot ECG data for a specific group with a Median Absolute Deviation (MAD) band.

    This function takes pre-processed and concatenated ECG data, calculates the mean and MAD,
    and plots these on the provided axes object. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - all_ecg_data_df: DataFrame containing the concatenated ECG data.
    - group_name: Name of the group (e.g., 'stroke', 'control') for labeling.
    - color: Color to use for the mean ECG plot of the group.
    - ax: Matplotlib axes object for plotting.
    """
    logging.info(f"Plotting ECG data for group: {group_name}")

    # Calculate the mean and MAD for all aggregated data
    mean_ecg = all_ecg_data_df.mean(axis=1)
    mad_ecg = robust.mad(all_ecg_data_df, axis=1)

    # Plot the MAD as a shaded band around the mean
    ax.fill_between(range(len(mean_ecg)), mean_ecg - mad_ecg, mean_ecg + mad_ecg, color=color, alpha=0.3, label=f'{group_name} MAD', zorder=1)
   
    # Plot the mean ECG line on top
    ax.plot(range(len(mean_ecg)), mean_ecg, label=f'{group_name} Mean', color=color, linewidth=2, zorder=2)

# Main script logic
def main():
    """
    Main function to visualize ECG data from a BIDS dataset.

    The function performs the following steps:
    - Validates the presence of necessary directories and files.
    - Reads the participants.tsv file to determine the groups.
    - Plots the ECG data for each group, including the mean and standard deviation bands.
    """
    logging.info("Starting main function")

    # Define and check BIDS root directory
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    logging.info(f"Checking BIDS root directory: {bids_root_dir}")
    if not os.path.exists(bids_root_dir):
        logging.warning(f"Directory not found: {bids_root_dir}")
        return

    # Define and check derivatives directory
    derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
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
    fig, axs = plt.subplots(4, 1, figsize=(15, 24))  # Assuming 4 runs
    
    # Adjust the layout of the figure to create space for the legend
    plt.subplots_adjust(bottom=0.125)  # Adjust the bottom parameter as needed

    # Prepare a dictionary to store handles for legend
    legend_handles = {}

    for run_number in range(1, 5):  # Process each run
        logging.info(f"Processing run {run_number}")
        for group_name, group_df in groups:
            participant_ids = group_df['participant_id'].tolist()
            logging.info(f"Processing group {group_name} for run {run_number}")

            # Collect ECG data for all participants in the group for the current run
            run_data = []
            session_id = "ses-1"
            for participant_id in participant_ids:
                # Retrieve ECG files for the current run and participant
                ecg_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_{session_id}_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
                logging.info(f"Filepaths: {ecg_files}")
                logging.info(f"Found {len(ecg_files)} ECG files for participant {participant_id} and run {run_number}")
                for file in ecg_files:
                    try:
                        with gzip.open(file, 'rt') as f:
                            ecg_data = pd.read_csv(f, delimiter='\t')
                        if 'cardiac' in ecg_data.columns:
                            
                            # Apply smoothing to the ECG data
                            smoothed_ecg = smooth_data(ecg_data['cardiac'])
                            run_data.append(smoothed_ecg)
                        else:
                            logging.warning(f"Column 'cardiac' not found in file: {file}")
                    except Exception as e:
                        logging.error(f"Error reading file {file}: {e}")

            # If data is available, plot it
            if run_data:
                run_data_df = pd.concat(run_data, axis=1)
                color = group_colors.get(group_name, 'gray')
                plot_group_ecg(run_data_df, group_name, color, axs[run_number - 1])
                
                # Capture the line handle for the legend from the last plotted line
                line = axs[run_number - 1].get_lines()[-1]
                legend_handles[group_name] = line

    # Customize and show the plot
    for i, ax in enumerate(axs):
        ax.set_title(f'Run {i+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude (mV)')

    # Add a single legend for all groups at the bottom of the figure
    fig.legend(handles=legend_handles.values(), labels=legend_handles.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(groups))

    plt.tight_layout()
    plt.show()

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()