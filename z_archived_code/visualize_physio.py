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

    all_ecg_data = []

    for file in ecg_files_derivatives:
        # Read ECG data from a gzipped tsv file
        with gzip.open(file, 'rt') as f:
            ecg_data = pd.read_csv(f, delimiter='\t')
    
        # Check if 'cardiac' column exists and plot the data
        if 'cardiac' in ecg_data.columns:
            all_ecg_data.append(ecg_data['cardiac'])
    
    if all_ecg_data:
        
        # Convert to a DataFrame for easier manipulation
        all_ecg_data_df = pd.concat(all_ecg_data, axis=1)

        # Compute mean and standard deviation
        mean_ecg = all_ecg_data_df.mean(axis=1)
        std_ecg = all_ecg_data_df.std(axis=1)

        # Calculate the MAD (Median Absolute Deviation) for each time point.
        mad_ecg = robust.mad(all_ecg_data_df, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(mean_ecg, label='Group Mean ECG')
        plt.fill_between(mean_ecg.index, mean_ecg - mad_ecg, mean_ecg + mad_ecg, color='gray', alpha=0.5, label='Median Absolute Deviation')
        #plt.fill_between(mean_ecg.index, mean_ecg - std_ecg, mean_ecg + std_ecg, color='gray', alpha=0.5, label='Standard Deviation')
        #plt.title('Group Mean ECG with Standard Deviation')
        plt.title('Mean ECG with Median Absolute Deviation')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        #plt.show() # Comment/uncomment to display/hide the plot.
    else:
        logging.warning("No ECG data found.")

# Plot ECG data seperately for stroke and control groups.
def plot_group_ecg(ecg_files, group_name, color, fig, ax):
    """
    Plot ECG data for a specific group with a standard deviation band.

    This function reads ECG data from provided file paths, calculates the mean and standard deviation,
    and plots these on a shared figure. The mean is plotted as a line, and the standard deviation 
    is visualized as a shaded band around the mean.

    Parameters:
    - ecg_files: List of file paths to the ECG data files.
    - group_name: Name of the group (e.g., 'stroke', 'control') for labeling.
    - color: Color to use for the mean ECG plot of the group.
    - fig, ax: Matplotlib figure and axes objects for plotting.
    """
    logging.info(f"Plotting ECG data for group: {group_name}")

    all_ecg_data = []  # List to hold all ECG data for the group

    # Aggregate ECG data from all files
    for file in ecg_files:
        try:
            with gzip.open(file, 'rt') as f:
                ecg_data = pd.read_csv(f, delimiter='\t')
            if 'cardiac' in ecg_data.columns:
                # Smooth the data and append to the list
                smoothed_ecg = smooth_data(ecg_data['cardiac'])
                all_ecg_data.append(smoothed_ecg)
                #all_ecg_data.append(ecg_data['cardiac']) # Uncomment to plot unsmoothed. 
            else:
                logging.warning(f"Column 'cardiac' not found in file: {file}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")

    # Process and plot data if available
    if all_ecg_data:
        # Concatenate all ECG data into a single DataFrame
        all_ecg_data_df = pd.concat(all_ecg_data, axis=1)
        # Calculate the mean and standard deviation
        mean_ecg = all_ecg_data_df.mean(axis=1)
        #std_ecg = all_ecg_data_df.std(axis=1) # Uncomment to use standard deviation.

        # Calculate the MAD (Median Absolute Deviation) for each time point.
        mad_ecg = robust.mad(all_ecg_data_df, axis=1)

        # Plot the standard deviation as a shaded band
        ax.fill_between(mean_ecg.index, mean_ecg - mad_ecg, mean_ecg + mad_ecg, color=color, alpha=0.3, zorder=1)
        # Plot the mean ECG line on top
        ax.plot(mean_ecg, label=f'{group_name} Mean', color=color, zorder=2)
    else:
        logging.warning(f"No ECG data to plot for group {group_name}")

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
    #group_colors = {'Stroke': '#D3D3D3', 'Control': '#90EE90'}  # Example colors

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Process and plot each group
    for group_name, group_df in groups:
        participant_ids = group_df['participant_id'].tolist()
        logging.info(f"Processing group {group_name}")
        logging.info(f"Found {len(participant_ids)} participants in group {group_name}")
        
        color = group_colors.get(group_name, 'gray')  # Get the color for the group
        
        ecg_files = []

        # Retrieve ECG files for each participant in the group
        for participant_id in participant_ids:
            ecg_file = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if participant_id in f and f.endswith('run-01_physio.tsv.gz')]
            ecg_files.extend(ecg_file)

        if ecg_files:
            logging.info(f"Processing {len(ecg_files)} ECG files for group {group_name}")
            plot_group_ecg(ecg_files, group_name, color, fig, ax) # Plot data for the group
        else:
            logging.warning(f"No ECG files found for group {group_name}")

    # Finalize the plot
    #ax.set_title('Mean ECG with Standard Deviation for Groups') # Uncomment to use standard deviation.
    ax.set_title('Mean ECG with Median Absolute Deviation for Groups')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    plt.show()

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()