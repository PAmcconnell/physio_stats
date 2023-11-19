# Script name: 'plot_physio.py'

# 1. Load the participants data to associate participant_ids with groups.
# 2. Load the physiological data for each participant.
# 3. Compute the mean and standard deviation for each physiological signal across all participants at each timepoint.
# 4. Create subplots to compare the two groups (0 for stroke and 1 for healthy).

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import glob
import os
import json
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define function to load participant data
def load_participants_data(file_path):
    """
    Load participant data from a TSV file.

    Parameters:
    file_path (str): The file path to the participants' data TSV file.

    Returns:
    DataFrame: The loaded participant data.
    """
    try:
        participants_data = pd.read_csv(file_path, sep='\t')
        logging.info(f"Participant data loaded successfully from {file_path}")
        return participants_data
    except Exception as e:
        logging.error(f"Failed to load participant data from {file_path}: {e}")
        return None

# Define function to load physio data for a participant
def load_physio_data(file_path):
    """
    Load physiological data for a participant from a gzipped TSV file.

    Parameters:
    file_path (str): The file path to the gzipped TSV file containing physiological data.

    Returns:
    DataFrame: The loaded physiological data.
    """
    try:
        physio_data = pd.read_csv(file_path, sep='\t', compression='gzip')
        logging.info(f"Physiological data loaded successfully from {file_path}")

        # Load corresponding JSON file
        json_file_path = file_path.replace('.tsv.gz', '.json')
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            logging.info(f"JSON data loaded successfully from {json_file_path}")

        return physio_data, json_data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return None, None

# Define function to calculate mean and standard deviation across all participants
def calculate_group_stats(physio_data):
    """
    Calculate mean and standard deviation for each physiological signal across all participants.

    Parameters:
    physio_data (list): A list of DataFrame objects, each containing physiological data for a participant.

    Returns:
    tuple: Two DataFrames containing the mean and standard deviation of each physiological signal.
    """
    
    try:
        if not physio_data:
            raise ValueError("No physiological data provided for statistics calculation.")

        combined_data = pd.concat(physio_data)
        logging.info(f"Data concatenated successfully: {combined_data.shape}")
        if combined_data.empty:
            raise ValueError("No data available after concatenation.")

        # Calculate mean and standard deviation for each signal across all timepoints
        means = combined_data.groupby(combined_data.index).mean()
        logging.info(f"Mean values shape: {means.shape}")
        logging.info(f"Mean values calculated successfully: {means}")
        stds = combined_data.groupby(combined_data.index).std()
        logging.info(f"Standard deviation values shape: {stds.shape}")
        logging.info(f"Standard deviation values calculated successfully: {stds}"
                     )
        logging.info("Group statistics calculated successfully.")
        return means, stds
    except Exception as e:
        logging.error(f"Error calculating group statistics: {e}")
        return None, None
    
# Define function to plot the data
def plot_data(means, stds, title, signal_name, y_label):
    """
    Plot the means and standard deviations of a single physiological signal.

    Parameters:
    means (Series): Pandas Series containing mean values for the signal.
    stds (Series): Pandas Series containing standard deviation values for the signal.
    title (str): The title for the plot.
    signal_name (str): The name of the physiological signal to be plotted.
    """
    
    try:
        x_values = range(len(means))
        plt.figure(figsize=(10, 5))
        plt.errorbar(x=x_values, y=means, yerr=stds, fmt='-o', label=f'{signal_name} Mean ± SD')
        plt.title(f"{title} - {signal_name}")
        plt.xlabel('Timepoints')
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting data for {title} - {signal_name}: {e}")
        
# Define function to compare two groups
def compare_groups(group0_data, group1_data):
    """
    Compare two groups using a t-test and plot the results.

    This function takes the means and standard deviations for two groups and performs a t-test
    to compare them at each timepoint. It then plots the p-values.

    Parameters:
    group0_data (tuple): A tuple containing two pandas Series for Group 0: the means and standard deviations.
    group1_data (tuple): A tuple containing two pandas Series for Group 1: the means and standard deviations.

    Returns:
    None
    """

    try:
        # Validate input data
        if not all(isinstance(group, tuple) and len(group) == 2 for group in [group0_data, group1_data]):
            raise ValueError("Invalid group data format. Expected tuples of (means, stds).")

        if group0_data[0] is None or group1_data[0] is None:
            logging.error("Invalid group data: None values found in means of one or both groups.")
            return
        
        group0_means, group0_stds = group0_data
        group1_means, group1_stds = group1_data

        # Check that the data for both groups is available and of the same length
        if group0_means is None or group1_means is None or len(group0_means) != len(group1_means):
            raise ValueError("Means of the two groups must be non-None and of the same length.")

        # Perform t-test at each timepoint
        t_stats, p_values = ttest_ind(group0_means, group1_means, equal_var=False)

        # Plot the p-values
        plt.figure(figsize=(10, 5))
        plt.plot(p_values, label='P-values')
        plt.axhline(y=0.05, color='r', linestyle='--', label='P=0.05')
        plt.title('P-values from t-test comparing two groups')
        plt.xlabel('Timepoints')
        plt.ylabel('P-value')
        plt.legend()
        plt.show()

        logging.info("Groups compared successfully, p-values plotted.")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while comparing groups: {e}")

# Load the JSON file content
def load_json_data(file_path):    
    """
    Load data from a JSON file.

    This function reads a JSON file and returns its contents as a dictionary.

    Parameters:
    file_path (str): The file path to the JSON file.

    Returns:
    dict: The content of the JSON file as a dictionary, or None if an error occurs.
    """
    
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            logging.info(f"JSON data loaded successfully from {file_path}")
            return data
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}. Please check the file path and try again.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {file_path}. Please check the file content.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading JSON data: {e}")
        return None
    
# Main script logic
def main():
    """
    Main function to process and compare physiological data for two groups.

    This function loads participant data and physiological data, then calculates 
    statistics for two groups (stroke and healthy), plots these statistics, 
    and compares the groups.

    Parameters:
    participants_file_path (str): The file path to the participants' data.
    physio_data_folder (str): The folder path containing the physiological data files.
    """
    logging.info("Starting main function")

    participants_file_path = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/participants.tsv')
    physio_data_folder = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')

    try:
        participants_data = load_participants_data(participants_file_path)
        if participants_data is None:
            logging.error("Failed to load participants data. Exiting.")
            return

        physio_data_groups = {0: [], 1: []}
        json_data_groups = {0: [], 1: []}  # To store JSON data for each group

        for participant_id, group in participants_data[['participant_id', 'group']].itertuples(index=False):
            tsv_file_pattern = os.path.join(physio_data_folder, f'{participant_id}_*run-01_physio.tsv.gz')
            for physio_file in glob.glob(tsv_file_pattern):
                physio_data, json_data = load_physio_data(physio_file)
                if physio_data is not None and json_data is not None:
                    physio_data_groups[group].append(physio_data)
                    json_data_groups[group].append(json_data)  # Append JSON data


        # Process and plot data for each group
        for group in physio_data_groups:
            group_data = physio_data_groups[group]
            means, stds = calculate_group_stats(group_data)
            if means is None or stds is None:
                continue

            for signal in means.columns:
                if signal == 'trigger':  # Skip 'trigger' signal
                    continue

                unit = json_data.get(signal, {}).get('Units', 'Unknown Unit')
                y_label = f'Value ({unit})'
                plot_data(means[signal], stds[signal], f'Group {group}', signal, y_label)

        # Perform and plot t-tests between groups for each physiological signal
        group0_means, group0_stds = calculate_group_stats(physio_data_groups[0])
        group1_means, group1_stds = calculate_group_stats(physio_data_groups[1])
        if group0_means is None or group1_means is None:
            logging.error("Failed to calculate statistics for one or both groups.")
            return

        for signal in group0_means.columns:
            if signal == 'trigger':  # Skip 'trigger' signal
                continue

            if not np.isfinite(group0_means[signal]).all() or not np.isfinite(group1_means[signal]).all():
                logging.warning(f"Non-finite values found in data for signal {signal}. Skipping t-test.")
                continue

            t_stat, p_val = ttest_ind(group0_means[signal], group1_means[signal], equal_var=False, nan_policy='omit')
            logging.info(f"t-test for {signal}: t-statistic = {t_stat}, p-value = {p_val}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()
