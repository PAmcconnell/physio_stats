import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(file_path, group_data):
    """
    Reads physiological data from a given file, merges it with participant information based on 'participant_id',
    and returns the merged DataFrame.

    Parameters:
    file_path (str): Path to the physiological data file.
    group_data (DataFrame): DataFrame containing participant information.

    Returns:
    DataFrame: Merged DataFrame with physiological and participant information, or None if an error occurs.
    """
    try:
        subject_id_match = re.search(r'sub-(LRN\d+)_', file_path)
        if subject_id_match:
            subject_id = 'sub-' + subject_id_match.group(1)  # Add 'sub-' prefix to match group data
            df = pd.read_csv(file_path, sep='\t', compression='gzip')
            df['participant_id'] = subject_id  # Add the subject ID as a new column
            
            # Debug: Log the IDs from both data sources
            logging.debug(f"Physiological data IDs: {df['participant_id'].unique()}")
            logging.debug(f"Group data IDs: {group_data['participant_id'].unique()}")
            
            merged_df = pd.merge(df, group_data, on='participant_id')
            if merged_df.empty:
                logging.warning(f"Merge resulted in an empty DataFrame for file: {file_path}")
            else:
                logging.debug(f"Merge successful for file: {file_path} with {merged_df.shape[0]} rows.")
            return merged_df
        else:
            logging.error(f"Failed to extract participant ID from filename: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    return None

def calc_mean_std_dev(data, signal):
    """
    Calculates the mean and standard deviation for the specified signal at each time point across subjects in the group.

    Parameters:
    data (DataFrame): DataFrame containing the physiological data.
    signal (str): The column name of the signal in the DataFrame.

    Returns:
    tuple: Two arrays containing the mean and standard deviation of the specified signal at each time point.
    """
    if data[signal].empty:
        logging.warning(f"No data available for signal: {signal}")
        return np.nan, np.nan
    else:
        grouped_data = data.groupby(data.index)[signal]
        
        # Calculate mean and standard deviation
        mean = grouped_data.mean()
        std_dev = grouped_data.std()

        # Ensure we have multiple data points
        count = grouped_data.count()
        if (count <= 1).any():
            logging.warning(f"Insufficient data points for standard deviation calculation in signal: {signal}")

        # Debugging logs
        logging.info(f"Mean for signal {signal}: {mean.head()}")
        logging.info(f"Standard deviation for signal {signal}: {std_dev.head()}")

        return mean, std_dev

def process_and_plot(file_paths, group_file_path):
    """
    Processes the physiological data files and plots group means with standard errors for different signals.

    Parameters:
    file_paths (list): List of paths to the physiological data files.
    group_file_path (str): Path to the group information file.
    """
    # Read the group information
    group_data = pd.read_csv(group_file_path, sep='\t')

    # Check the first few rows to understand its structure
    #print(group_data.head())

    combined_data = pd.DataFrame()

    for file_path in file_paths:
        data = read_data(file_path, group_data)
        
        # Check the first few rows to understand its structure
        #print(data.head())

        if data is not None and not data.empty:
            combined_data = pd.concat([combined_data, data], ignore_index=True)
        else:
            logging.warning(f"No data to merge for file: {file_path}")

    if combined_data.empty:
        logging.error("No data available after merging. Check the 'participant_id' columns in both datasets.")
        return

    signals = ['cardiac'] #, 'respiratory', 'eda', 'ppg']
    groups = combined_data['group'].unique()

    # Check for missing time points
    total_time_points = 1750000
    print("Missing time points:", set(range(total_time_points)) - set(data['time']))

    time_points = range(total_time_points)

    for signal in signals:
        plt.figure(figsize=(12, 6))

        for group in groups:
            group_data = combined_data[combined_data['group'] == group]

            # Check and log the size of the group data
            logging.debug(f"Data size for signal {signal} in group {group}: {group_data.shape}")

            mean_per_time_point = group_data.groupby(group_data.index)[signal].mean()

            # Log if the mean data is all NaN
            if mean_per_time_point.isna().all():
                logging.warning(f"All mean data is NaN for signal {signal} in group {group}")
            elif not mean_per_time_point.isna().any():
                plt.plot(range(len(mean_per_time_point)), mean_per_time_point, label=f'Group {group} - {signal.capitalize()}')
            else:
                logging.warning(f"No valid data to plot for signal {signal} in group {group}")

        plt.title(f'Group Mean - {signal.capitalize()} Signal')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

# Main script execution
if __name__ == '__main__':
    pathname = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    group_file_path = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/participants.tsv')  

    # Validate the existence of the group file path
    if not os.path.exists(group_file_path):
        logging.error(f"The group file path does not exist: {group_file_path}")
    else:
        # Finding all .tsv.gz files in the directory
        file_paths = glob.glob(os.path.join(pathname, '*run-01_physio.tsv.gz'))
        process_and_plot(file_paths, group_file_path)
