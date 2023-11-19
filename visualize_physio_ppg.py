# Script name: visualize_physio_ppg.py

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

# Plot ppg data across all participants.
def plot_all_ppg(ppg_files_derivatives):
    """
    Plot ppg data for all subjects and runs with a Median Absolute Deviation (MAD) band.

    This function reads ppg data from provided file paths, calculates the mean and MAD,
    and plots these on a single figure. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - ppg_files_derivatives: List of file paths to the ppg derivative data files.
    """
    logging.info("Plotting ppg data for all subjects and runs.")

    all_ppg_data = []  # List to hold all ppg data across subjects and runs

    # Aggregate ppg data from all files
    for file in ppg_files_derivatives:
        try:
            with gzip.open(file, 'rt') as f:
                ppg_data = pd.read_csv(f, delimiter='\t')
            if 'ppg' in ppg_data.columns:
                # Assuming each file represents a run, append the ppg data
                all_ppg_data.append(ppg_data['ppg'])
            else:
                logging.warning(f"Column 'ppg' not found in file: {file}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")

    # Process and plot data if available
    if all_ppg_data:
        # Concatenate all ppg data into a single DataFrame
        all_ppg_data_df = pd.concat(all_ppg_data, axis=1)
        # Calculate the mean and MAD
        mean_ppg = all_ppg_data_df.mean(axis=1)
        mad_ppg = robust.mad(all_ppg_data_df, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(mean_ppg, label='All Subjects Mean ppg')
        plt.fill_between(range(len(mean_ppg)), mean_ppg - mad_ppg, mean_ppg + mad_ppg, color='gray', alpha=0.25, label='Median Absolute Deviation')
        plt.title('Mean ppg with Median Absolute Deviation')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (V)')
        plt.legend()
        plt.show()  # Display the plot.
    else:
        logging.warning("No ppg data found to plot.")

# Plot ppg data seperately for stroke and control groups.
def plot_group_ppg(all_ppg_data_df, group_name, color, ax):
    """
    Plot ppg data for a specific group with a Median Absolute Deviation (MAD) band.

    This function takes pre-processed and concatenated ppg data, calculates the mean and MAD,
    and plots these on the provided axes object. The mean is plotted as a line, and the MAD
    is visualized as a shaded band around the mean.

    Parameters:
    - all_ppg_data_df: DataFrame containing the concatenated ppg data.
    - group_name: Name of the group (e.g., 'stroke', 'control') for labeling.
    - color: Color to use for the mean ppg plot of the group.
    - ax: Matplotlib axes object for plotting.
    """
    logging.info(f"Plotting ppg data for group: {group_name}")

    # Calculate the mean and MAD for all aggregated data
    mean_ppg = all_ppg_data_df.mean(axis=1)
    mad_ppg = robust.mad(all_ppg_data_df, axis=1)

    # Plot the MAD as a shaded band around the mean
    ax.fill_between(range(len(mean_ppg)), mean_ppg - mad_ppg, mean_ppg + mad_ppg, color=color, alpha=0.3, label=f'{group_name} MAD', zorder=1)
   
    # Plot the mean ppg line on top
    ax.plot(range(len(mean_ppg)), mean_ppg, label=f'{group_name} Mean', color=color, linewidth=2, zorder=2)

# Main script logic
def main():
    """
    Main function to visualize ppg data from a BIDS dataset.

    The function performs the following steps:
    - Validates the presence of necessary directories and files.
    - Reads the participants.tsv file to determine the groups.
    - Plots the ppg data for each group, including the mean and standard deviation bands.
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

            # Collect ppg data for all participants in the group for the current run
            run_data = []
            session_id = "ses-1"
            for participant_id in participant_ids:
                # Retrieve ppg files for the current run and participant
                ppg_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_{session_id}_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
                logging.info(f"Filepaths: {ppg_files}")
                logging.info(f"Found {len(ppg_files)} ppg files for participant {participant_id} and run {run_number}")
                for file in ppg_files:
                    try:
                        with gzip.open(file, 'rt') as f:
                            ppg_data = pd.read_csv(f, delimiter='\t')
                        if 'ppg' in ppg_data.columns:
                            
                            # Apply smoothing to the ppg data
                            smoothed_ppg = smooth_data(ppg_data['ppg'])
                            run_data.append(smoothed_ppg)
                        else:
                            logging.warning(f"Column 'ppg' not found in file: {file}")
                    except Exception as e:
                        logging.error(f"Error reading file {file}: {e}")

            # If data is available, plot it
            if run_data:
                run_data_df = pd.concat(run_data, axis=1)
                color = group_colors.get(group_name, 'gray')
                plot_group_ppg(run_data_df, group_name, color, axs[run_number - 1])
                
                # Capture the line handle for the legend from the last plotted line
                line = axs[run_number - 1].get_lines()[-1]
                legend_handles[group_name] = line

    # Customize and show the plot
    for i, ax in enumerate(axs):
        ax.set_title(f'Run {i+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude (V)')

    # Add a single legend for all groups at the bottom of the figure
    fig.legend(handles=legend_handles.values(), labels=legend_handles.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(groups))

    fig.tight_layout(pad=1.0)  # Adjust 'pad' as needed

    plt.show()

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()