"""
Script Name: clean_physio_task-rest_eda.py

Description:
This script is designed for processing and analyzing electrodermal activity (EDA) signals within a BIDS (Brain Imaging Data Structure) dataset. It applies various signal processing techniques to clean and decompose the EDA signals into tonic and phasic components. Additionally, it identifies skin conductance responses (SCRs) using multiple detection methods. The script generates visualizations of these components and saves both the processed data and figures for further analysis.

The script performs the following main functions:
1. Sets up a detailed logging system for monitoring and debugging.
2. Reads EDA signal data from specified BIDS dataset directories.
3. Applies a low-pass Butterworth filter to the raw EDA signals.
4. Decomposes the filtered EDA signal using NeuroKit and other methods.
5. Detects SCRs and characterizes them in terms of onsets, peaks, and amplitudes.
6. Generates and saves plots of the raw, cleaned, tonic, and phasic EDA signals along with identified SCRs.
7. Saves processed data and logs in a structured manner for each participant and session.

Usage:
The script is intended to be run from the command line or an IDE. It requires the specification of dataset directories and participant/session details within the script. The script can be customized to process multiple participants and sessions by modifying the relevant sections.

Requirements:
- Python 3.x
- Libraries: neurokit2, pandas, matplotlib, scipy, numpy
- A BIDS-compliant dataset with EDA signal data.

Note:
- Ensure that the dataset paths and participant/session details are correctly set in the script.
- The script contains several hardcoded paths and parameters which may need to be adjusted based on the dataset structure and analysis needs.

Author: PAMcConnell
Date: 20231122
Version: 1.0

"""

import re
import neurokit2 as nk
import pandas as pd
import gzip
import matplotlib
#matplotlib.use('Agg')  # Use the 'Agg' backend, which is for writing to files, not for rendering in a window.
import matplotlib.pyplot as plt
import logging
import os
import traceback
from datetime import datetime
import numpy as np
import sys
from scipy.signal import iirfilter, sosfreqz, sosfiltfilt, freqz 
import psutil 
import cProfile
import tracemalloc

# Function to log system resource usage
def log_resource_usage():
    """Logs the CPU and Memory usage of the system."""
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    logging.info(f"Current memory usage: {memory_usage}%, CPU usage: {cpu_usage}%")

# Sets up archival logging for the script, directing log output to both a file and the console.
def setup_logging(subject_id, session_id, run_id, dataset_root_dir):
    """

    The function configures logging to capture informational, warning, and error messages. It creates a unique log file for each 
    subject-session combination, stored in a 'logs' directory within the 'doc' folder adjacent to the BIDS root directory.

  
    Parameters:
    - subject_id (str): The identifier for the subject.
    - session_id (str): The identifier for the session.
    - data_root_dir (str): The root directory of the dataset.

    Returns:
    - log_file_path (str): The path to the log file.

    This function sets up a logging system that writes logs to both a file and the console. 
    The log file is named based on the subject ID, session ID, and the script name. 
    It's stored in a 'logs' directory within the 'doc' folder by subject ID, which is located at the same 
    level as the BIDS root directory.

    The logging level is set to INFO, meaning it captures all informational, warning, and error messages.

    Usage Example:
    setup_logging('sub-01', 'ses-1', '/path/to/bids_root_dir')
    """

    try: 
        # Extract the base name of the script without the .py extension.
        script_name = os.path.basename(__file__).replace('.py', '')

        # Construct the log directory path within 'doc/logs'
        log_dir = os.path.join(os.path.dirname(dataset_root_dir), 'doc', 'logs', script_name, subject_id, run_id)
        print(f"Checking log directory: {log_dir}")

        # Create the log directory if it doesn't exist.
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Construct the log file name using subject ID, session ID, and script name.
        log_file_name = f"{subject_id}_{session_id}_task-rest_{run_id}_{script_name}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        # Configure file logging.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file_path,
            filemode='w' # 'w' mode overwrites existing log file
        )

        # If you also want to log to console.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        logging.info(f"Logging setup complete. Log file: {log_file_path}")

        return log_file_path
    
    except Exception as e:
            print(f"Error setting up logging: {e}")
            sys.exit(1) # Exiting the script due to logging setup failure.

def comb_band_stop_filter(data, stop_freq, fs, order=6, visualize=False):
    """
    Apply a comb band-stop filter.

    Parameters:
    - data (array-like or pandas.Series): The input signal.
    - stop_freq (float): The stop frequency of the filter (center of the notch).
    - fs (int): The sampling rate of the signal.
    - order (int, optional): The order of the filter.

    Returns:
    - y (numpy.ndarray): The filtered signal.
    """

    # Convert Pandas Series to NumPy array if necessary
    if isinstance(data, pd.Series):
        data = data.values

    # Ensure data is a 1D array
    if data.ndim != 1:
        raise ValueError("The input data must be a 1D array or Series.")

    # Design the Butterworth filter
    nyquist = 0.5 * fs
    normal_stop_freq = stop_freq / nyquist
    
    # Calculate the stop band frequencies
    stop_band = [0.26, 0.75]

    # Design the bandstop filter
    sos = iirfilter(order, stop_band, btype='bandstop', fs=fs, output='sos')

    # Visualize the frequency response
    if visualize:
        w, h = sosfreqz(sos, worN=8000, fs=fs)
        plt.plot(w, abs(h), label="Frequency Response")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response of the Band-Stop Filter')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Apply the filter
    y = sosfiltfilt(sos, data)
    
    return y

# Function to calculate and log the EDA autocorrelation for a given signal and method.
def log_eda_autocorrelation(eda_signal, method_name, sampling_rate, logfile_path):
    """
    Calculate and log the EDA autocorrelation for a given signal and method.
    
    Parameters:
    - eda_signal: The EDA signal (pandas Series or numpy array).
    - method_name: Name of the method used for EDA signal processing.
    - sampling_rate: Sampling rate of the EDA signal.
    - logfile_path: Path to the logfile where the autocorrelation result will be recorded.
    """
    try:
        autocorrelation = nk.eda_autocor(eda_signal, sampling_rate=sampling_rate)
        with open(logfile_path, 'a') as log_file:
            log_file.write(f"Autocorrelation for {method_name}: {autocorrelation}\n")
        logging.info(f"Autocorrelation for {method_name} calculated and logged.")
    except Exception as e:
        logging.error(f"Error calculating autocorrelation for {method_name}: {e}")

# Main script logic
def main():
    """
    Main function to clean EDA data from a BIDS dataset.
    """
    print(f"Starting main function")
    start_time = datetime.now()
    tracemalloc.start()
    
    # Define and check dataset root directory
    dataset_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/')
    print(f"Checking BIDS root directory: {dataset_root_dir}")
    if not os.path.exists(dataset_root_dir):
        print(f"Directory not found: {dataset_root_dir}")
        return
    
    # Define and check BIDS root directory
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    print(f"Checking BIDS root directory: {bids_root_dir}")
    if not os.path.exists(bids_root_dir):
        print(f"Directory not found: {bids_root_dir}")
        return

    # Define and check derivatives directory
    derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    print(f"Checking derivatives directory: {derivatives_dir}")
    if not os.path.exists(derivatives_dir):
        print(f"Directory not found: {derivatives_dir}")
        return

    # Define and check participants file
    participants_file = os.path.join(bids_root_dir, 'participants.tsv')
    print(f"Checking participants file: {participants_file}")
    if not os.path.exists(participants_file):
        print(f"Participants file not found: {participants_file}")
        return

    try: 
        # Read participants file and process groups
        participants_df = pd.read_csv(participants_file, delimiter='\t')
        
    except pd.errors.EmptyDataError as e:
        print(f"Error reading participants file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading participants file: {e}")
        return    
    
    print(f"Found {len(participants_df)} participants")
    
    # Process each run for each participant
    for i, participant_id in enumerate(participants_df['participant_id']):
        # Record the start time for this participant
        participant_start_time = datetime.now()
        for run_number in range(1, 5):  # Assuming 4 runs
            try:
            
                # Define the processed signals filename for checking
                processed_signals_filename = f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio_unfiltered_cleaned_eda_processed.tsv.gz"
                processed_signals_path = os.path.join(derivatives_dir, processed_signals_filename)

                # Check if processed file exists
                if os.path.exists(processed_signals_path):
                    print(f"Processed EDA files found for {participant_id} for run {run_number}, skipping...")
                    continue  # Skip to the next run

                # Set a higher DPI for better resolution
                dpi_value = 300 

                task_name = 'rest'

                # Process the first run for the selected participant
                run_id = f"run-0{run_number}"

                # Construct the base path
                base_path = os.path.join(derivatives_dir, participant_id, run_id)

                # Make sure the directories exist
                os.makedirs(base_path, exist_ok=True)

                # Setup logging
                session_id = 'ses-1'  # Assuming session ID is known
                log_file_path = setup_logging(participant_id, session_id, run_id, dataset_root_dir)
                logging.info(f"Testing EDA processing for task {task_name} run-0{run_number} for participant {participant_id}")

                # Filter settings
                stop_freq = 0.5  # 0.5 Hz (1/TR with TR = 2s)
                sampling_rate = 5000    # acquisition sampling rate
            
                # Define the frequency band
                frequency_band = (0.04, 0.25) # 0.045 - 0.25 Hz Sympathetic Band

                pattern = re.compile(f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz")
                physio_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
                for file in physio_files:
                    # Record the start time for this run
                    run_start_time = datetime.now()
                    logging.info(f"Processing file: {file}")

                    try:
                        # Generate a base filename by removing the '.tsv.gz' extension
                        #base_filename = os.path.splitext(os.path.splitext(file)[0])[0]
                        base_filename = os.path.basename(file).replace('.tsv.gz', '')

                        # Define the list of sympathetic analysis methods
                        sympathetic_methods = ["posada"] #"ghiasi" for HRV + EDA symapthovagal balance analysis

                        # Open the compressed EDA file and load it into a DataFrame
                        with gzip.open(file, 'rt') as f:
                            physio_data = pd.read_csv(f, delimiter='\t')
                            # Check for the presence of the 'eda' column
                            if 'eda' not in physio_data.columns:
                                logging.error(f"'eda' column not found in {file}. Skipping this file.")
                                continue  # Skip to the next file
                            eda = physio_data['eda']

                            # Calculate the time vector in minutes - Length of EDA data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(len(eda)) / sampling_rate / 60

                            ### Step 1: Process raw unfiltered EDA signal using NeuroKit for examining the raw and cleaned signals ###
                            logging.info(f"Step 1: Process raw unfiltered EDA signal using NeuroKit for examining the raw and cleaned signals.")
                            
                            # Process EDA signal using NeuroKit
                            eda_signals_neurokit, info_eda_neurokit = nk.eda_process(eda, sampling_rate=sampling_rate, method='neurokit')
                            logging.info("Default Neurokit unfiltered raw EDA signal processing complete.")

                            # Create a figure with three subplots
                            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                            # Plot 1: Overlay of Raw and Cleaned Unfiltered EDA
                            axes[0].plot(eda_signals_neurokit['EDA_Raw'], label='Raw EDA', color='green')
                            axes[0].plot(eda_signals_neurokit['EDA_Clean'], label='Cleaned EDA', color='orange')
                            axes[0].set_title('Raw and Cleaned Unfiltered EDA Signal')
                            axes[0].set_ylabel('EDA (µS)')
                            axes[0].legend()

                            # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery
                            axes[1].plot(eda_signals_neurokit['EDA_Phasic'], label='Unfiltered Cleaned Phasic Component', color='green')
                            axes[1].scatter(info_eda_neurokit['SCR_Onsets'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Onsets']], color='blue', label='SCR Onsets')
                            axes[1].scatter(info_eda_neurokit['SCR_Peaks'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Peaks']], color='red', label='SCR Peaks')
                            
                            # Assuming 'SCR_HalfRecovery' is in info_eda_neurokit
                            axes[1].scatter(info_eda_neurokit['SCR_Recovery'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Recovery']], color='purple', label='SCR Half Recovery')
                            axes[1].set_title('Unfiltered Cleaned Phasic EDA with SCR Events')
                            axes[1].set_ylabel('Amplitude (µS)')
                            axes[1].legend()

                            # Plot 3: Tonic Component
                            axes[2].plot(time_vector, eda_signals_neurokit['EDA_Tonic'], label='Tonic Component', color='brown')
                            axes[2].set_title('Tonic EDA')
                            axes[2].set_xlabel('Time (minutes)')
                            axes[2].set_ylabel('Amplitude (µS)')
                            axes[2].legend()

                            plt.tight_layout()
                            plot_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_default_subplots.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            plt.close()
                            logging.info(f"Saved default unfiltered cleaned EDA subplots to {plot_filename}")

                            # Assuming eda_signals_neurokit and info_eda_neurokit are defined and valid
                            phasic_component = eda_signals_neurokit['EDA_Phasic']
                            tonic_component = eda_signals_neurokit['EDA_Tonic']

                            # Basic statistics for Phasic Component
                            phasic_stats = {
                                'Mean': phasic_component.mean(),
                                'Median': phasic_component.median(),
                                'Std Deviation': phasic_component.std(),
                                'Variance': phasic_component.var(),
                                'Skewness': phasic_component.skew(),
                                'Kurtosis': phasic_component.kurtosis()
                            }

                            # SCR-specific metrics
                            scr_onsets = info_eda_neurokit['SCR_Onsets']
                            scr_peaks = info_eda_neurokit['SCR_Peaks']
                            scr_amplitudes = phasic_component[scr_peaks] - phasic_component[scr_onsets]

                            # Calculate additional SCR-specific metrics
                            total_time = (len(phasic_component) / sampling_rate) / 60  # Convert to minutes
                            average_scr_frequency = len(scr_peaks) / total_time
                            amplitude_range = scr_amplitudes.max() - scr_amplitudes.min()
                            inter_scr_intervals = np.diff(scr_onsets) / sampling_rate  # Convert to seconds
                            average_inter_scr_interval = np.mean(inter_scr_intervals)

                            scr_stats = {
                                'SCR Count': len(scr_peaks),
                                'Mean SCR Amplitude': scr_amplitudes.mean(),
                                'Average SCR Frequency': average_scr_frequency,
                                'Amplitude Range of SCRs': amplitude_range,
                                'Average Inter-SCR Interval': average_inter_scr_interval
                                # Add AUC and distribution analysis as needed
                            }

                            # Basic statistics for Tonic Component
                            tonic_stats = {
                                'Mean': tonic_component.mean(),
                                'Median': tonic_component.median(),
                                'Std Deviation': tonic_component.std(),
                                'Variance': tonic_component.var(),
                                'Skewness': tonic_component.skew(),
                                'Kurtosis': tonic_component.kurtosis(),
                                'Range': tonic_component.max() - tonic_component.min(),
                                'Total Absolute Sum': tonic_component.abs().sum(),
                                '25th Percentile': tonic_component.quantile(0.25),
                                '75th Percentile': tonic_component.quantile(0.75),
                                'IQR': tonic_component.quantile(0.75) - tonic_component.quantile(0.25),
                                '10th Percentile': tonic_component.quantile(0.10),
                                '90th Percentile': tonic_component.quantile(0.90)
                            }

                            # Combine all stats in a single dictionary or DataFrame
                            eda_summary_stats = {
                                'Phasic Stats': phasic_stats,
                                'SCR Stats': scr_stats,
                                'Tonic Stats': tonic_stats
                            }

                            # Save default unfiltered processed signals to a TSV file
                            processed_signals_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_processed.tsv")
                            eda_signals_neurokit.to_csv(processed_signals_filename, sep='\t', index=False)
                            logging.info(f"Saving processed unfiltered cleaned signals to TSV file.")
                            
                            logging.info(f"Compressing processed unfiltered cleaned signals to tsv.gz file.")
                            
                            # Define the full path for the gzip file
                            gzip_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_processed.tsv.gz")

                            # Compress the processed signals file
                            with open(processed_signals_filename, 'rb') as f_in:
                                with gzip.open(gzip_filename, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            logging.info(f"Saved and compressed unfiltered cleaned processed signals to {gzip_filename}")

                            # Remove the uncompressed file
                            os.remove(processed_signals_filename)
                            logging.info(f"Removed uncompressed file: {processed_signals_filename}")

                            ### Step 2: Compute PSD of unfiltered cleaned EDA in the 0 - 1 Hz frequency band ###
                            logging.info(f'Step 2: Compute PSD of unfiltered cleaned EDA in the 0 - 1 Hz frequency band.')

                            # Compute Power Spectral Density 0 - 1 Hz
                            logging.info(f"Computing Power Spectral Density (PSD) in 0 - 1 Hz band for unfiltered cleaned EDA using multitapers hann windowing.")
                            eda_psd_clean = nk.signal_psd(eda_signals_neurokit['EDA_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                    silent=False, t=None)
                            
                            # Initialize psd_full_stats dictionary
                            unfiltered_clean_psd_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_clean[(eda_psd_clean['Frequency'] >= low_freq) & (eda_psd_clean['Frequency'] < high_freq)]['Power'].sum()
                                unfiltered_clean_psd_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            unfiltered_clean_psd_full_stats.update({
                                'Mean': eda_psd_clean['Power'].mean(),
                                'Median': eda_psd_clean['Power'].median(),
                                'Total Power': eda_psd_clean['Power'].sum(),
                                'Peak Frequency': eda_psd_clean.loc[eda_psd_clean['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_clean['Power'].std(),
                                'Variance': eda_psd_clean['Power'].var(),
                                'Skewness': eda_psd_clean['Power'].skew(),
                                'Kurtosis': eda_psd_clean['Power'].kurtosis(),
                                'Peak Power': eda_psd_clean['Power'].max(),
                                'Bandwidth': eda_psd_clean['Frequency'].iloc[-1] - eda_psd_clean['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_clean['Power'], eda_psd_clean['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("Unfiltered Cleaned PSD (0 - 1 Hz) Summary Statistics:")
                            for stat, value in unfiltered_clean_psd_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plotting Power Spectral Density
                            logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for unfiltered cleaned EDA using multitapers hann windowing.")
                            plt.figure(figsize=(12, 6))
                            plt.fill_between(eda_psd_clean['Frequency'], eda_psd_clean['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_clean['Frequency'], eda_psd_clean['Power'], color='blue', label='Normalized PSD (Multitapers with Hanning Window)')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for Unfiltered Cleaned EDA')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the full range PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd.tsv")
                            eda_psd_clean.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range PSD data to {full_file_path}")

                            # Save the full range PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_summary_statistics.tsv")
                            pd.DataFrame([unfiltered_clean_psd_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range PSD summary statistics to {full_file_path}")

                            # Assuming 'base_filename' is already defined
                            tonic_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_tonic_summary_statistics.tsv")
                            phasic_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_phasic_summary_statistics.tsv")

                            # Convert the dictionaries to DataFrame for saving
                            tonic_stats_df = pd.DataFrame([tonic_stats])
                            phasic_stats_df = pd.DataFrame([phasic_stats])

                            # Save to TSV files
                            tonic_stats_df.to_csv(tonic_filename, sep='\t', index=False)
                            phasic_stats_df.to_csv(phasic_filename, sep='\t', index=False)

                            # Logging the actions
                            logging.info(f"Saved unfiltered cleaned EDA PSD tonic summary statistics to {tonic_filename}")
                            logging.info(f"Saved unfiltered cleaned EDA PSD phasic summary statistics to {phasic_filename}")

                            ### Step 3: Compute PSD of unfiltered cleaned EDA in the 0.04 - 0.25 Hz sympathetic frequency band ###
                            logging.info(f'Step 3: Compute PSD of unfiltered cleaned EDA in the 0.04 - 0.25 Hz sympathetic frequency band.')

                            # Compute Power Spectral Density in Sympathetic band
                            logging.info(f"Computing Power Spectral Density (PSD) for unfiltered cleaned EDA Sympathetic Band (0.04 - 0.25 Hz) using multitapers hann windowing.")
                            eda_psd_symp_clean = nk.signal_psd(eda_signals_neurokit['EDA_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                    silent=False, t=None)

                            # Initialize psd_full_stats dictionary
                            unfiltered_psd_symp_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_symp_clean[(eda_psd_symp_clean['Frequency'] >= low_freq) & (eda_psd_symp_clean['Frequency'] < high_freq)]['Power'].sum()
                                unfiltered_psd_symp_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            unfiltered_psd_symp_clean_full_stats.update({
                                'Mean': eda_psd_symp_clean['Power'].mean(),
                                'Median': eda_psd_symp_clean['Power'].median(),
                                'Total Power': eda_psd_symp_clean['Power'].sum(),
                                'Peak Frequency': eda_psd_symp_clean.loc[eda_psd_symp_clean['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_symp_clean['Power'].std(),
                                'Variance': eda_psd_symp_clean['Power'].var(),
                                'Skewness': eda_psd_symp_clean['Power'].skew(),
                                'Kurtosis': eda_psd_symp_clean['Power'].kurtosis(),
                                'Peak Power': eda_psd_symp_clean['Power'].max(),
                                'Bandwidth': eda_psd_symp_clean['Frequency'].iloc[-1] - eda_psd_symp_clean['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_symp_clean['Power'], eda_psd_symp_clean['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("Unfiltered Cleaned PSD Symp band (0.04 - 0.25 Hz) Summary Statistics:")
                            for stat, value in unfiltered_psd_symp_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plot the PSD Symphathetic band
                            plt.figure(figsize=(10, 6))
                            plt.fill_between(eda_psd_symp_clean['Frequency'], eda_psd_symp_clean['Power'], color='purple', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_symp_clean['Frequency'], eda_psd_symp_clean['Power'], label='Normalized PSD (Multitapers with Hanning Window)', color='purple')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for Unfiltered EDA Sympathetic Band')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_sympband.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the sympathetic band PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_sympband.tsv")
                            processed_signals_filename = eda_psd_symp_clean.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved unfiltered cleaned sympathetic band PSD data to {full_file_path}")

                            # Save the sympathetic band PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_unfiltered_cleaned_eda_psd_sympband_summary_statistics.tsv")
                            pd.DataFrame([unfiltered_psd_symp_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved unfiltered cleaned sympathetic band PSD summary statistics to {full_file_path}")
                            
                            ### Step 4: Compute sympathetic indicies for unfiltered cleaned phasic and tonic eda using posada method ###
                            logging.info(f"Step 4: Compute sympathetic indicies for unfiltered cleaned phasic and tonic eda using posada method.")

                            # After decomposing and before peak detection, calculate sympathetic indices
                            for symp_method in sympathetic_methods:
                                try:
                                    # Calculate sympathetic indices
                                    eda_symp_phasic = nk.eda_sympathetic(eda_signals_neurokit["EDA_Phasic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                    # Log the sympathetic index for unfiltered EDA
                                    logging.info(f"Posada calculated sympathetic index for unfiltered cleaned Phasic EDA: {eda_symp_phasic['EDA_Sympathetic']}")
                                    logging.info(f"Posada calculated normalized sympathetic index for unfiltered cleaned Phasic EDA: {eda_symp_phasic['EDA_SympatheticN']}")
                                    logging.info(f"Calculated sympathetic indices using {symp_method} method for unfiltered cleaned phasic eda.")

                                    # Calculate sympathetic indices
                                    eda_symp_tonic = nk.eda_sympathetic(eda_signals_neurokit["EDA_Tonic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                    # Log the sympathetic index for unfiltered EDA
                                    logging.info(f"Posada calculated sympathetic index for unfiltered cleaned tonic EDA: {eda_symp_tonic['EDA_Sympathetic']}")
                                    logging.info(f"Posada calculated normalized sympathetic index for unfiltered cleaned Phasic EDA: {eda_symp_tonic['EDA_SympatheticN']}")
                                    logging.info(f"Calculated sympathetic indices using {symp_method} method for unfiltered cleaned tonic eda.")

                                except Exception as e:
                                    logging.error(f"Error in sympathetic index calculation (unfiltered eda, {symp_method}): {e}")
                            
                            ### Step 5: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz ###
                            logging.info(f"Step 5: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz.")

                            # Pre-filter gradient artifact.
                            eda_filtered_array = comb_band_stop_filter(eda, stop_freq, sampling_rate, visualize=False)

                            # Downsample the filtered data.
                            eda_filtered_ds = nk.signal_resample(eda_filtered_array, desired_length=None, sampling_rate=sampling_rate, desired_sampling_rate=50, method='pandas')

                            # Hanlde the index for the downsampled data
                            new_length = len(eda_filtered_ds)
                            new_index = pd.RangeIndex(start=0, stop=new_length, step=1)  # or np.arange(new_length)

                            # Convert the filtered data back into a Pandas Series
                            eda_filtered = pd.Series(eda_filtered_ds, index=new_index)

                            if eda_filtered.empty:
                                logging.error("Error: 'eda_filtered' is empty.")
                                # Log stack trace for debugging purposes
                                logging.error(traceback.format_exc())
                            else:
                                logging.info(f"'eda_filtered' is not empty, length: {len(eda_filtered)}")

                            sampling_rate = 50    # downsampled sampling rate
                            
                            # Calculate the time vector in minutes - Length of EDA data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(new_length) / sampling_rate / 60

                            ### Step 6: Process filtered cleaned EDA signal using NeuroKit for examining the raw and cleaned filtered signals ###
                            logging.info(f"Step 6: Process filtered cleaned EDA signal using NeuroKit for examining the raw and cleaned filtered signals.")

                            # Process EDA signal using NeuroKit
                            eda_signals_neurokit_filt, info_eda_neurokit_filt = nk.eda_process(eda_filtered, sampling_rate=sampling_rate, method='neurokit')
                            logging.info("Default Neurokit filtered EDA signal processing complete.")

                            # Create a figure with three subplots
                            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                            # Plot 1: Overlay of Filtered Raw and Cleaned EDA
                            #axes[0].plot(eda, label='Raw EDA', color='blue', linewidth=4.0)
                            axes[0].plot(eda_signals_neurokit_filt['EDA_Raw'], label='Raw (Prefiltered) EDA', color='green')
                            axes[0].plot(eda_signals_neurokit_filt['EDA_Clean'], label='Cleaned EDA', color='orange')
                            axes[0].set_title('Raw (Prefiltered) and Cleaned EDA Signal')
                            axes[0].set_ylabel('EDA (µS)')
                            axes[0].legend()

                            # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery
                            axes[1].plot(eda_signals_neurokit_filt['EDA_Phasic'], label='Phasic Component', color='green')
                            axes[1].scatter(info_eda_neurokit_filt['SCR_Onsets'], eda_signals_neurokit_filt['EDA_Phasic'][info_eda_neurokit_filt['SCR_Onsets']], color='blue', label='SCR Onsets')
                            axes[1].scatter(info_eda_neurokit_filt['SCR_Peaks'], eda_signals_neurokit_filt['EDA_Phasic'][info_eda_neurokit_filt['SCR_Peaks']], color='red', label='SCR Peaks')
                            
                            # Assuming 'SCR_HalfRecovery' is in info_eda_neurokit
                            axes[1].scatter(info_eda_neurokit_filt['SCR_Recovery'], eda_signals_neurokit_filt['EDA_Phasic'][info_eda_neurokit_filt['SCR_Recovery']], color='purple', label='SCR Half Recovery')
                            axes[1].set_title('Phasic EDA with SCR Events')
                            axes[1].set_ylabel('Amplitude (µS)')
                            axes[1].legend()

                            # Plot 3: Tonic Component
                            axes[2].plot(time_vector, eda_signals_neurokit_filt['EDA_Tonic'], label='Tonic Component', color='brown')
                            axes[2].set_title('Tonic EDA')
                            axes[2].set_xlabel('Time (minutes)')
                            axes[2].set_ylabel('Amplitude (µS)')
                            axes[2].legend()

                            plt.tight_layout()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_default_subplots.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            plt.close()
                            logging.info(f"Saved default filtered cleaned EDA subplots to {plot_filename}")
                        
                            # Assuming eda_signals_neurokit_filt and info_eda_neurokit_filt are defined and valid
                            phasic_component = eda_signals_neurokit_filt['EDA_Phasic']
                            tonic_component = eda_signals_neurokit_filt['EDA_Tonic']

                            # Basic statistics for Phasic Component
                            phasic_stats = {
                                'Mean': phasic_component.mean(),
                                'Median': phasic_component.median(),
                                'Std Deviation': phasic_component.std(),
                                'Variance': phasic_component.var(),
                                'Skewness': phasic_component.skew(),
                                'Kurtosis': phasic_component.kurtosis()
                            }

                            # SCR-specific metrics
                            scr_onsets = info_eda_neurokit_filt['SCR_Onsets']
                            scr_peaks = info_eda_neurokit_filt['SCR_Peaks']
                            scr_amplitudes = phasic_component[scr_peaks] - phasic_component[scr_onsets]

                            # Calculate additional SCR-specific metrics
                            total_time = (len(phasic_component) / sampling_rate) / 60  # Convert to minutes
                            average_scr_frequency = len(scr_peaks) / total_time
                            amplitude_range = scr_amplitudes.max() - scr_amplitudes.min()
                            inter_scr_intervals = np.diff(scr_onsets) / sampling_rate  # Convert to seconds
                            average_inter_scr_interval = np.mean(inter_scr_intervals)

                            scr_stats = {
                                'SCR Count': len(scr_peaks),
                                'Mean SCR Amplitude': scr_amplitudes.mean(),
                                'Average SCR Frequency': average_scr_frequency,
                                'Amplitude Range of SCRs': amplitude_range,
                                'Average Inter-SCR Interval': average_inter_scr_interval
                                # Add AUC and distribution analysis as needed
                            }

                            # Basic statistics for Tonic Component
                            tonic_stats = {
                                'Mean': tonic_component.mean(),
                                'Median': tonic_component.median(),
                                'Std Deviation': tonic_component.std(),
                                'Variance': tonic_component.var(),
                                'Skewness': tonic_component.skew(),
                                'Kurtosis': tonic_component.kurtosis(),
                                'Range': tonic_component.max() - tonic_component.min(),
                                'Total Absolute Sum': tonic_component.abs().sum(),
                                '25th Percentile': tonic_component.quantile(0.25),
                                '75th Percentile': tonic_component.quantile(0.75),
                                'IQR': tonic_component.quantile(0.75) - tonic_component.quantile(0.25),
                                '10th Percentile': tonic_component.quantile(0.10),
                                '90th Percentile': tonic_component.quantile(0.90)
                            }

                            # Combine all stats in a single dictionary or DataFrame
                            eda_summary_stats = {
                                'Phasic Stats': phasic_stats,
                                'SCR Stats': scr_stats,
                                'Tonic Stats': tonic_stats
                            }

                            
                            # Save default filtered processed signals to a TSV file
                            processed_signals_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_processed.tsv")
                            eda_signals_neurokit_filt.to_csv(processed_signals_filename, sep='\t', index=False)
                            logging.info(f"Saving processed filtered cleaned signals to TSV file.")
                            
                            logging.info(f"Compressing processed filtered cleaned signals to tsv.gz file.")
                            
                            # Define the full path for the gzip file
                            gzip_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_processed.tsv.gz")

                            # Compress the processed signals file
                            with open(processed_signals_filename, 'rb') as f_in:
                                with gzip.open(gzip_filename, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            logging.info(f"Saved and compressed unfiltered cleaned processed signals to {gzip_filename}")

                            # Remove the uncompressed file
                            os.remove(processed_signals_filename)
                            logging.info(f"Removed uncompressed file: {processed_signals_filename}")

          
                            ### Step 7: Compute PSD of filtered cleaned EDA in the 0 - 1 Hz frequency band ###
                            logging.info(f'Step 2: Compute PSD of filtered cleaned EDA in the 0 - 1 Hz frequency band.')

                            # Compute Power Spectral Density 0 - 1 Hz
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned EDA using multitapers hann windowing.")

                            eda_psd_filt_clean = nk.signal_psd(eda_signals_neurokit_filt['EDA_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                    silent=False, t=None)
                            
                            # Initialize psd_full_stats dictionary
                            filtered_psd_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_filt_clean[(eda_psd_filt_clean['Frequency'] >= low_freq) & (eda_psd_filt_clean['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_clean_full_stats.update({
                                'Mean': eda_psd_filt_clean['Power'].mean(),
                                'Median': eda_psd_filt_clean['Power'].median(),
                                'Total Power': eda_psd_filt_clean['Power'].sum(),
                                'Peak Frequency': eda_psd_filt_clean.loc[eda_psd_filt_clean['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_filt_clean['Power'].std(),
                                'Variance': eda_psd_filt_clean['Power'].var(),
                                'Skewness': eda_psd_filt_clean['Power'].skew(),
                                'Kurtosis': eda_psd_filt_clean['Power'].kurtosis(),
                                'Peak Power': eda_psd_filt_clean['Power'].max(),
                                'Bandwidth': eda_psd_filt_clean['Frequency'].iloc[-1] - eda_psd_filt_clean['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_filt_clean['Power'], eda_psd_filt_clean['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered cleaned PSD (0 - 1 Hz) Summary Statistics:")
                            for stat, value in filtered_psd_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plotting Power Spectral Density
                            logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned EDA using multitapers hann windowing.")
                            plt.figure(figsize=(12, 6))
                            plt.fill_between(eda_psd_filt_clean['Frequency'], eda_psd_filt_clean['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_filt_clean['Frequency'], eda_psd_filt_clean['Power'], color='blue', label='Normalized PSD (Multitapers with Hanning Window)')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned EDA')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Assuming 'base_filename' is already defined
                            tonic_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_tonic_summary_statistics.tsv")
                            phasic_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_phasic_summary_statistics.tsv")

                            # Convert the dictionaries to DataFrame for saving
                            tonic_stats_df = pd.DataFrame([tonic_stats])
                            phasic_stats_df = pd.DataFrame([phasic_stats])

                            # Save to TSV files
                            tonic_stats_df.to_csv(tonic_filename, sep='\t', index=False)
                            phasic_stats_df.to_csv(phasic_filename, sep='\t', index=False)

                            # Logging the actions
                            logging.info(f"Saved filtered cleaned EDA PSD tonic summary statistics to {tonic_filename}")
                            logging.info(f"Saved filtered cleaned EDA PSD phasic summary statistics to {phasic_filename}")
                            
                            # Save the full range PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd.tsv")
                            eda_psd_filt_clean.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                            # Save the full range PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_sympband_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned PSD summary statistics to {full_file_path}")

        
                            ### Step 8: Compute PSD of filtered cleaned EDA in the 0.04 - 0.25 Hz sympathetic frequency band ###
                            logging.info(f'Step 8: Compute PSD of filtered cleaned EDA in the 0.04 - 0.25 Hz sympathetic frequency band.')

                            # Compute Power Spectral Density in Sympathetic band
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered EDA Sympathetic band using multitapers hann windowing.")
                            eda_psd_symp_filt_clean = nk.signal_psd(eda_signals_neurokit_filt['EDA_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                    silent=False, t=None)

                            # Initialize psd_full_stats dictionary
                            filtered_psd_symp_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_symp_filt_clean[(eda_psd_symp_filt_clean['Frequency'] >= low_freq) & (eda_psd_symp_filt_clean['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_symp_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_symp_clean_full_stats.update({
                                'Mean': eda_psd_symp_filt_clean['Power'].mean(),
                                'Median': eda_psd_symp_filt_clean['Power'].median(),
                                'Total Power': eda_psd_symp_filt_clean['Power'].sum(),
                                'Peak Frequency': eda_psd_symp_filt_clean.loc[eda_psd_symp_filt_clean['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_symp_filt_clean['Power'].std(),
                                'Variance': eda_psd_symp_filt_clean['Power'].var(),
                                'Skewness': eda_psd_symp_filt_clean['Power'].skew(),
                                'Kurtosis': eda_psd_symp_filt_clean['Power'].kurtosis(),
                                'Peak Power': eda_psd_symp_filt_clean['Power'].max(),
                                'Bandwidth': eda_psd_symp_filt_clean['Frequency'].iloc[-1] - eda_psd_symp_filt_clean['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_symp_filt_clean['Power'], eda_psd_symp_filt_clean['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered cleaned 0.04 - 0.25 Hz PSD Symp Band Summary Statistics:")
                            for stat, value in filtered_psd_symp_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plot the PSD Symphathetic band
                            plt.figure(figsize=(10, 6))
                            plt.fill_between(eda_psd_symp_filt_clean['Frequency'], eda_psd_symp_filt_clean['Power'], color='purple', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_symp_filt_clean['Frequency'], eda_psd_symp_filt_clean['Power'], label='Normalized PSD (Multitapers with Hanning Window)', color='purple')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned EDA Sympathetic Band')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_sympband.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the sympathetic band PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_sympband.tsv")
                            eda_psd_symp_filt_clean.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered cleaned EDA sympathetic band PSD data to {full_file_path}")

                            # Save the sympathetic band PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_sympband_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_symp_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered cleaned EDA sympathetic band PSD summary statistics to {full_file_path}")

                            ### Step 9: Compute PSD of filtered cleaned Phasic EDA in the 0 - 1 Hz frequency band ###
                            logging.info(f'Step 9: Compute PSD of filtered cleaned Phasic EDA in the 0 - 1 Hz frequency band.')

                            # Compute Power Spectral Density 0 - 1 Hz
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned Phasic EDA using multitapers hann windowing.")
                            eda_psd_filt_phasic = nk.signal_psd(eda_signals_neurokit_filt['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                    silent=False, t=None)
                            
                            # Initialize psd_full_stats dictionary
                            filtered_psd_phasic_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_filt_phasic[(eda_psd_filt_phasic['Frequency'] >= low_freq) & (eda_psd_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_phasic_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_phasic_clean_full_stats.update({
                                'Mean': eda_psd_filt_phasic['Power'].mean(),
                                'Median': eda_psd_filt_phasic['Power'].median(),
                                'Total Power': eda_psd_filt_phasic['Power'].sum(),
                                'Peak Frequency': eda_psd_filt_phasic.loc[eda_psd_filt_phasic['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_filt_phasic['Power'].std(),
                                'Variance': eda_psd_filt_phasic['Power'].var(),
                                'Skewness': eda_psd_filt_phasic['Power'].skew(),
                                'Kurtosis': eda_psd_filt_phasic['Power'].kurtosis(),
                                'Peak Power': eda_psd_filt_phasic['Power'].max(),
                                'Bandwidth': eda_psd_filt_phasic['Frequency'].iloc[-1] - eda_psd_filt_phasic['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_filt_phasic['Power'], eda_psd_filt_phasic['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered, cleaned Phasic EDA PSD Summary Statistics:")
                            for stat, value in filtered_psd_phasic_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plotting Power Spectral Density
                            logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered, cleaned Phasic EDA using multitapers hann windowing.")
                            plt.figure(figsize=(12, 6))
                            plt.fill_between(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', label='Normalized Phasic PSD (Multitapers with Hanning Window)')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) (0 - 1 Hz) for filtered cleaned Phasic EDA')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the full range PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic.tsv")
                            eda_psd_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned Phasic PSD data to {full_file_path}")

                            # Save the full range PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_sympband_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_phasic_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned Phasic PSD summary statistics to {full_file_path}")

                            ### Step 10: Compute PSD of filtered cleaned Phasic EDA in the 0.04 - 0.25 Hz sympathetic frequency band ###
                            logging.info(f'Step 10: Compute PSD of filtered cleaned Phasic EDA in the 0.04 - 0.25 Hz sympathetic frequency band.')

                            # Compute Power Spectral Density in Sympathetic band
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered Cleaned Phasic EDA Sympathetic band using multitapers hann windowing.")
                            eda_psd_symp_filt_phasic = nk.signal_psd(eda_signals_neurokit_filt['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                    silent=False, t=None)

                            # Initialize psd_full_stats dictionary
                            filtered_psd_symp_filt_phasic_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_symp_filt_phasic[(eda_psd_symp_filt_phasic['Frequency'] >= low_freq) & (eda_psd_symp_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_symp_filt_phasic_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_symp_filt_phasic_clean_full_stats.update({
                                'Mean': eda_psd_symp_filt_phasic['Power'].mean(),
                                'Median': eda_psd_symp_filt_phasic['Power'].median(),
                                'Total Power': eda_psd_symp_filt_phasic['Power'].sum(),
                                'Peak Frequency': eda_psd_symp_filt_phasic.loc[eda_psd_symp_filt_phasic['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_symp_filt_phasic['Power'].std(),
                                'Variance': eda_psd_symp_filt_phasic['Power'].var(),
                                'Skewness': eda_psd_symp_filt_phasic['Power'].skew(),
                                'Kurtosis': eda_psd_symp_filt_phasic['Power'].kurtosis(),
                                'Peak Power': eda_psd_symp_filt_phasic['Power'].max(),
                                'Bandwidth': eda_psd_symp_filt_phasic['Frequency'].iloc[-1] - eda_psd_symp_filt_phasic['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_symp_filt_phasic['Power'], eda_psd_symp_filt_phasic['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered cleaned Phasic PSD Symp Summary Statistics:")
                            for stat, value in filtered_psd_symp_filt_phasic_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plot the PSD Symphathetic band
                            plt.figure(figsize=(10, 6))
                            plt.fill_between(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], color='purple', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], label='Normalized Phasic PSD (Multitapers with Hanning Window)', color='purple')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered, cleaned Phasic EDA Sympathetic Band')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_sympband.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the sympathetic band PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_sympband.tsv")
                            eda_psd_symp_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered, cleaned Phasic EDA sympathetic band PSD data to {full_file_path}")

                            # Save the sympathetic band PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_sympband_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_symp_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered, cleaned Phasic EDA sympathetic band PSD summary statistics to {full_file_path}")
           
                            ### Step 11: Compute PSD of filtered cleaned Tonic EDA in the 0 - 1 Hz frequency band ###
                            logging.info(f'Step 11: Compute PSD of filtered cleaned Tonic EDA in the 0 - 1 Hz frequency band.')

                            # Compute Power Spectral Density 0 - 1 Hz
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned Tonic EDA using multitapers hann windowing.")
                            eda_psd_filt_tonic = nk.signal_psd(eda_signals_neurokit_filt['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                    silent=False, t=None)
                            
                            # Initialize psd_full_stats dictionary
                            filtered_psd_tonic_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_filt_tonic[(eda_psd_filt_tonic['Frequency'] >= low_freq) & (eda_psd_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_tonic_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_tonic_clean_full_stats.update({
                                'Mean': eda_psd_filt_tonic['Power'].mean(),
                                'Median': eda_psd_filt_tonic['Power'].median(),
                                'Total Power': eda_psd_filt_tonic['Power'].sum(),
                                'Peak Frequency': eda_psd_filt_tonic.loc[eda_psd_filt_tonic['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_filt_tonic['Power'].std(),
                                'Variance': eda_psd_filt_tonic['Power'].var(),
                                'Skewness': eda_psd_filt_tonic['Power'].skew(),
                                'Kurtosis': eda_psd_filt_tonic['Power'].kurtosis(),
                                'Peak Power': eda_psd_filt_tonic['Power'].max(),
                                'Bandwidth': eda_psd_filt_tonic['Frequency'].iloc[-1] - eda_psd_filt_tonic['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_filt_tonic['Power'], eda_psd_filt_tonic['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered, cleaned tonic EDA PSD Summary Statistics:")
                            for stat, value in filtered_psd_tonic_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plotting Power Spectral Density
                            logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered, cleaned tonic EDA using multitapers hann windowing.")
                            plt.figure(figsize=(12, 6))
                            plt.fill_between(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', label='Normalized tonic PSD (Multitapers with Hanning Window)')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned tonic EDA')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Assuming 'base_filename' is already defined
                            tonic_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_summary_statistics.tsv")
                            phasic_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_summary_statistics.tsv")

                            # Save the full range PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic.tsv")
                            eda_psd_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned tonic PSD data to {full_file_path}")

                            # Save the full range PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_tonic_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved full range Filtered Cleaned tonic PSD summary statistics to {full_file_path}")

                            ### Step 12: Compute PSD of filtered cleaned Tonic EDA in the 0.04 - 0.25 Hz sympathetic frequency band ###
                            logging.info(f'Step 12: Compute PSD of filtered cleaned Tonic EDA in the 0.04 - 0.25 Hz sympathetic frequency band.')

                            # Compute Power Spectral Density in Sympathetic band
                            logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA Sympathetic band using multitapers hann windowing.")
                            eda_psd_symp_filt_tonic = nk.signal_psd(eda_signals_neurokit_filt['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                    min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                    silent=False, t=None)

                            # Initialize psd_full_stats dictionary
                            filtered_psd_symp_tonic_clean_full_stats = {}    

                            # Define the frequency bands
                            frequency_bands = {
                                'VLF': (0, 0.045),
                                'LF': (0.045, 0.15),
                                'HF1': (0.15, 0.25),
                                'HF2': (0.25, 0.4),
                                'VHF': (0.4, 0.5)
                            }

                            # Calculate the power in each band
                            for band, (low_freq, high_freq) in frequency_bands.items():
                                band_power = eda_psd_symp_filt_tonic[(eda_psd_symp_filt_tonic['Frequency'] >= low_freq) & (eda_psd_symp_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                filtered_psd_symp_tonic_clean_full_stats[f'{band} Power'] = band_power

                            # Calculate and save summary statistics for the full range PSD
                            filtered_psd_symp_tonic_clean_full_stats.update({
                                'Mean': eda_psd_symp_filt_tonic['Power'].mean(),
                                'Median': eda_psd_symp_filt_tonic['Power'].median(),
                                'Total Power': eda_psd_symp_filt_tonic['Power'].sum(),
                                'Peak Frequency': eda_psd_symp_filt_tonic.loc[eda_psd_symp_filt_tonic['Power'].idxmax(), 'Frequency'],
                                'Standard Deviation': eda_psd_symp_filt_tonic['Power'].std(),
                                'Variance': eda_psd_symp_filt_tonic['Power'].var(),
                                'Skewness': eda_psd_symp_filt_tonic['Power'].skew(),
                                'Kurtosis': eda_psd_symp_filt_tonic['Power'].kurtosis(),
                                'Peak Power': eda_psd_symp_filt_tonic['Power'].max(),
                                'Bandwidth': eda_psd_symp_filt_tonic['Frequency'].iloc[-1] - eda_psd_symp_filt_tonic['Frequency'].iloc[0],
                                'PSD Area': np.trapz(eda_psd_symp_filt_tonic['Power'], eda_psd_symp_filt_tonic['Frequency'])
                            })

                            # Log the summary statistics
                            logging.info("filtered cleaned tonic PSD Symp Summary Statistics:")
                            for stat, value in filtered_psd_symp_tonic_clean_full_stats.items():
                                logging.info(f"{stat}: {value}")

                            # Plot the PSD Symphathetic band
                            plt.figure(figsize=(10, 6))
                            plt.fill_between(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], color='purple', alpha=0.3)  # alpha controls the transparency
                            plt.plot(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], label='Normalized tonic PSD (Multitapers with Hanning Window)', color='purple')
                            plt.title('Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered, cleaned tonic EDA Sympathetic Band')
                            plt.xlabel('Frequency (Hz)')
                            plt.ylabel('Normalized Power')
                            plt.legend()
                            plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_sympband.png")
                            plt.savefig(plot_filename, dpi=dpi_value)
                            #plt.show()

                            # Save the sympathetic band PSD data to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_sympband.tsv")
                            eda_psd_symp_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered, cleaned tonic EDA sympathetic band PSD data to {full_file_path}")

                            # Save the sympathetic band PSD summary statistics to a TSV file
                            full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_sympband_summary_statistics.tsv")
                            pd.DataFrame([filtered_psd_symp_tonic_clean_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                            logging.info(f"Saved filtered, cleaned tonic EDA sympathetic band PSD summary statistics to {full_file_path}")

                            ### Step 13: Compute sympathetic indicies for filtered cleaned phasic and tonic eda using posada method ###
                            logging.info(f"Step 13: Compute sympathetic indicies for filtered cleaned phasic and tonic eda using posada method.")

                            # After decomposing and before peak detection, calculate sympathetic indices
                            for symp_method in sympathetic_methods:
                                try:
                                    # Calculate phasic sympathetic indices
                                    eda_symp_filt_phasic = nk.eda_sympathetic(eda_signals_neurokit_filt["EDA_Phasic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                    # Log the phasic sympathetic index for filtered EDA
                                    logging.info(f"Posada calculated sympathetic index for filtered, cleaned Phasic EDA: {eda_symp_filt_phasic['EDA_Sympathetic']}")
                                    logging.info(f"Posada calculated normalized sympathetic index for filtered, cleaned Phasic EDA: {eda_symp_filt_phasic['EDA_SympatheticN']}")
                                    logging.info(f"Calculated sympathetic indices using {symp_method} method for filtered, cleaned Phasic eda.")

                                    # Calculate tonic sympathetic indices
                                    eda_symp_filt_tonic = nk.eda_sympathetic(eda_signals_neurokit_filt["EDA_Tonic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                    # Log the tonic sympathetic index for filtered EDA
                                    logging.info(f"Posada calculated sympathetic index for filtered, cleaned tonic EDA: {eda_symp_filt_tonic['EDA_Sympathetic']}")
                                    logging.info(f"Posada calculated normalized sympathetic index for filtered, cleaned Phasic EDA: {eda_symp_filt_tonic['EDA_SympatheticN']}")
                                    logging.info(f"Calculated sympathetic indices using {symp_method} method for filtered, cleaned tonic eda.")

                                except Exception as e:
                                    logging.error(f"Error in sympathetic index calculation (filtered cleaned eda, {symp_method}): {e}")
                            
                            ### Step 14: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods. ###
                            logging.info(f"Step 14: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods.")

                            # First, clean the EDA signal
                            eda_cleaned = nk.eda_clean(eda_filtered, sampling_rate=sampling_rate)
                            logging.info(f"Prefiltered EDA signal cleaned using NeuroKit's eda_clean.")
                            
                            logging.info(f"Starting phasic decomposition and peak detection for prefiltered EDA signal.")
                            logging.info(f"Sampling rate: {sampling_rate} Hz")
                            logging.info(f"Size of prefiltered EDA signal: {eda_cleaned.size}")

                            # Define methods for phasic decomposition and peak detection.
                            methods = ['cvxEDA', 'smoothmedian', 'highpass', 'sparse'] #sparsEDA ?
                            logging.info(f"Using the following methods for phasic decomposition: {methods}")

                            peak_methods = ["kim2004", "neurokit", "gamboa2008", "vanhalem2020", "nabian2018"]
                            logging.info(f"Using the following methods for peak detection: {peak_methods}")

                            # Initialize psd_full_stats dictionary
                            filtered_psd_full_stats_by_method = {}    
                            filtered_psd_symp_full_stats_by_method = {}    
                            
                            # Process phasic decomposition of EDA signal using NeuroKit
                            for method in methods:
                                logging.info(f"Starting processing for method: {method}")

                                # Decompose EDA signal using the specified method
                                try:
                                    decomposed = nk.eda_phasic(eda_cleaned, method=method, sampling_rate=sampling_rate)
                                    
                                    # Ensure 'decomposed' is a DataFrame with the required column.
                                    if not isinstance(decomposed, pd.DataFrame) or 'EDA_Phasic' not in decomposed.columns:
                                        logging.error(f"'decomposed' is not a DataFrame or 'EDA_Phasic' column is missing for method {method}.")
                                        # Log stack trace for debugging purposes
                                        logging.error(traceback.format_exc())
                                        continue
                                    
                                    logging.info(f"Decomposed EDA using {method} method. Size of decomposed data: {decomposed.size}")

                                except ValueError as e:
                                    logging.error(f"ValueError encountered: {e}")
                                    logging.error(f"Method: {method}, Sampling Rate: {sampling_rate}")
                                    logging.error(f"EDA Cleaned details: Range: {eda_cleaned.min()} - {eda_cleaned.max()}, NaNs: {eda_cleaned.isna().sum()}, Infs: {np.isinf(eda_cleaned).sum()}")
                                    raise  # Optionally re-raise the exception if you want the program to stop
                                
                                except Exception as e:
                                    logging.error(f"Error in EDA decomposition with method {method}: {e}")
                                    # Log stack trace for debugging purposes
                                    logging.error(traceback.format_exc())                        
                                    continue

                                # Compute Power Spectral Density 0 - 1 Hz for Phasic EDA
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered Phasic EDA {method} using multitapers hann windowing.")
                                eda_psd_filt_phasic = nk.signal_psd(decomposed['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_filt_phasic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_filt_phasic[(eda_psd_filt_phasic['Frequency'] >= low_freq) & (eda_psd_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_filt_phasic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_filt_phasic_full_stats.update({
                                    'Mean': eda_psd_filt_phasic['Power'].mean(),
                                    'Median': eda_psd_filt_phasic['Power'].median(),
                                    'Total Power': eda_psd_filt_phasic['Power'].sum(),
                                    'Peak Frequency': eda_psd_filt_phasic.loc[eda_psd_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Standard Deviation': eda_psd_filt_phasic['Power'].std(),
                                    'Variance': eda_psd_filt_phasic['Power'].var(),
                                    'Skewness': eda_psd_filt_phasic['Power'].skew(),
                                    'Kurtosis': eda_psd_filt_phasic['Power'].kurtosis(),
                                    'Peak Power': eda_psd_filt_phasic['Power'].max(),
                                    'Bandwidth': eda_psd_filt_phasic['Frequency'].iloc[-1] - eda_psd_filt_phasic['Frequency'].iloc[0],
                                    'PSD Area': np.trapz(eda_psd_filt_phasic['Power'], eda_psd_filt_phasic['Frequency'])
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned Phasic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_filt_phasic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned Phasic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', label='Normalized Phasic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned Phasic EDA from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}.tsv")
                                eda_psd_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                                # Save the full range PSD summary statistics to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}_summary_statistics.tsv")
                                pd.DataFrame([eda_psd_filt_phasic_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD summary statistics to {full_file_path}")

                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned Phasic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_phasic = nk.signal_psd(decomposed['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_symp_filt_phasic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_symp_filt_phasic[(eda_psd_filt_phasic['Frequency'] >= low_freq) & (eda_psd_symp_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_phasic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_phasic_full_stats.update({
                                    'Mean': eda_psd_symp_filt_phasic['Power'].mean(),
                                    'Median': eda_psd_symp_filt_phasic['Power'].median(),
                                    'Total Power': eda_psd_symp_filt_phasic['Power'].sum(),
                                    'Peak Frequency': eda_psd_symp_filt_phasic.loc[eda_psd_symp_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Standard Deviation': eda_psd_symp_filt_phasic['Power'].std(),
                                    'Variance': eda_psd_symp_filt_phasic['Power'].var(),
                                    'Skewness': eda_psd_symp_filt_phasic['Power'].skew(),
                                    'Kurtosis': eda_psd_symp_filt_phasic['Power'].kurtosis(),
                                    'Peak Power': eda_psd_symp_filt_phasic['Power'].max(),
                                    'Bandwidth': eda_psd_symp_filt_phasic['Frequency'].iloc[-1] - eda_psd_symp_filt_phasic['Frequency'].iloc[0],
                                    'PSD Area': np.trapz(eda_psd_symp_filt_phasic['Power'], eda_psd_symp_filt_phasic['Frequency'])
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned Phasic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_symp_filt_phasic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned Phasic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], color='blue', label='Normalized Phasic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned Phasic EDA in Symp band from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}.tsv")
                                eda_psd_symp_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Phasic PSD data to {full_file_path}")

                                # Save the full range PSD summary statistics to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}_summary_statistics.tsv")
                                pd.DataFrame([eda_psd_symp_filt_phasic_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Phasic PSD summary statistics to {full_file_path}")

                                # Compute Power Spectral Density 0 - 1 Hz
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} using multitapers hann windowing.")
                                eda_psd_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_filt_tonic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_filt_tonic[(eda_psd_filt_tonic['Frequency'] >= low_freq) & (eda_psd_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_filt_tonic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_filt_tonic_full_stats.update({
                                    'Mean': eda_psd_filt_tonic['Power'].mean(),
                                    'Median': eda_psd_filt_tonic['Power'].median(),
                                    'Total Power': eda_psd_filt_tonic['Power'].sum(),
                                    'Peak Frequency': eda_psd_filt_tonic.loc[eda_psd_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Standard Deviation': eda_psd_filt_tonic['Power'].std(),
                                    'Variance': eda_psd_filt_tonic['Power'].var(),
                                    'Skewness': eda_psd_filt_tonic['Power'].skew(),
                                    'Kurtosis': eda_psd_filt_tonic['Power'].kurtosis(),
                                    'Peak Power': eda_psd_filt_tonic['Power'].max(),
                                    'Bandwidth': eda_psd_filt_tonic['Frequency'].iloc[-1] - eda_psd_filt_tonic['Frequency'].iloc[0],
                                    'PSD Area': np.trapz(eda_psd_filt_tonic['Power'], eda_psd_filt_tonic['Frequency'])
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned tonic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_filt_tonic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned tonic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', label='Normalized tonic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned tonic EDA from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}.tsv")
                                eda_psd_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned Tonic PSD data to {full_file_path}")

                                # Save the full range PSD summary statistics to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}_summary_statistics.tsv")
                                pd.DataFrame([eda_psd_filt_tonic_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")

                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.04, max_frequency=0.25, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_symp_filt_tonic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_symp_filt_tonic[(eda_psd_filt_tonic['Frequency'] >= low_freq) & (eda_psd_symp_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_tonic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_tonic_full_stats.update({
                                    'Mean': eda_psd_symp_filt_tonic['Power'].mean(),
                                    'Median': eda_psd_symp_filt_tonic['Power'].median(),
                                    'Total Power': eda_psd_symp_filt_tonic['Power'].sum(),
                                    'Peak Frequency': eda_psd_symp_filt_tonic.loc[eda_psd_symp_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Standard Deviation': eda_psd_symp_filt_tonic['Power'].std(),
                                    'Variance': eda_psd_symp_filt_tonic['Power'].var(),
                                    'Skewness': eda_psd_symp_filt_tonic['Power'].skew(),
                                    'Kurtosis': eda_psd_symp_filt_tonic['Power'].kurtosis(),
                                    'Peak Power': eda_psd_symp_filt_tonic['Power'].max(),
                                    'Bandwidth': eda_psd_symp_filt_tonic['Frequency'].iloc[-1] - eda_psd_symp_filt_tonic['Frequency'].iloc[0],
                                    'PSD Area': np.trapz(eda_psd_symp_filt_tonic['Power'], eda_psd_symp_filt_tonic['Frequency'])
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned tonic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_symp_filt_tonic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned tonic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], color='blue', label='Normalized tonic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned tonic EDA in Symp band from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}.tsv")
                                eda_psd_symp_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Tonic PSD data to {full_file_path}")

                                # Save the full range PSD summary statistics to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}_summary_statistics.tsv")
                                pd.DataFrame([eda_psd_symp_filt_tonic_full_stats]).to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")
        
                                # After decomposing and before peak detection, calculate sympathetic indices
                                for symp_method in sympathetic_methods:
                                    try:
                                        # Calculate phasic sympathetic indices
                                        eda_symp_decomposed_phasic = nk.eda_sympathetic(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=symp_method, show=False)
            
                                        # Log the phasic sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned phasic sympathetic indices using {symp_method} method for {method}.")

                                        # Calculate tonic sympathetic indices
                                        eda_symp_decomposed_tonic = nk.eda_sympathetic(decomposed["EDA_Tonic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                        # Log the sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned tonic sympathetic indices using {symp_method} method for {method}.")

                                    except Exception as e:
                                        logging.error(f"Error in sympathetic index calculation ({method}, {symp_method}): {e}")
                                
                                # Check DataFrame size before adding new columns for SCR events
                                expected_columns_count = len(peak_methods) * 3  # SCR_Amplitude, SCR_Onsets, SCR_Peaks for each peak method
                                if decomposed.shape[1] + expected_columns_count > decomposed.shape[1]:
                                    
                                    # Initialize columns for SCR events in the DataFrame
                                    for peak_method in peak_methods:
                                        decomposed[f"SCR_Amplitude_{peak_method}"] = np.nan
                                        decomposed[f"SCR_Onsets_{peak_method}"] = np.nan
                                        decomposed[f"SCR_Peaks_{peak_method}"] = np.nan
                                else:
                                    logging.error(f"Insufficient size to add new columns for method {method}.")
                                    # Log stack trace for debugging purposes
                                    logging.error(traceback.format_exc())
                                    continue

                                # Calculate amplitude_min from the phasic component for peak detection
                                amplitude_min = 0.01 * decomposed["EDA_Phasic"].max()
                                logging.info(f"Calculated amplitude_min: {amplitude_min} for {method} method.")

                                # Peak detection and plotting
                                for peak_method in peak_methods:
                                    try:
                                        # Check if the phasic component is not empty
                                        if decomposed["EDA_Phasic"].size == 0:
                                            logging.warning(f"The phasic component is empty for {method}. Skipping peak detection.")
                                            continue
                                        
                                        # Only add amplitude_min for applicable methods
                                        peak_detection_kwargs = {"sampling_rate": sampling_rate, "method": peak_method}

                                        # Apply amplitude_min only for the "neurokit" or "kim2004" methods
                                        if peak_method in ["neurokit", "kim2004"]:
                                            peak_detection_kwargs["amplitude_min"] = amplitude_min
                                
                                        # Detect peaks using the specified method
                                        logging.info(f"Detecting peaks using method: {peak_method}")    
                                        _, peaks = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=peak_method)
                                        print("SCR Onsets:", peaks['SCR_Onsets'])
                                        print("SCR Recovery:", peaks['SCR_Recovery'])
                                        # Check if SCR_Onsets is not empty and does not contain NaN values
                                        if 'SCR_Onsets' in peaks and not np.isnan(peaks['SCR_Onsets']).any():
                                            decomposed.loc[peaks['SCR_Onsets'], f"SCR_Onsets_{peak_method}"] = 1
                                        else:
                                            logging.warning(f"No valid SCR onsets found for {peak_method} method.")

                                        # Add SCR Amplitude, Onsets, and Peaks to the DataFrame
                                        if peaks['SCR_Peaks'].size > 0:
                                            for i in range(len(peaks['SCR_Peaks'])):
                                                peak_index = int(peaks['SCR_Peaks'][i])
                                                recovery_index = int(peaks['SCR_Recovery'][i]) if not np.isnan(peaks['SCR_Recovery'][i]) else None
                                                half_amplitude = decomposed['EDA_Phasic'][peak_index] / 2

                                                # Find the half-recovery point
                                                if recovery_index is not None:
                                                    recovery_phase = decomposed['EDA_Phasic'][peak_index:recovery_index]
                                                    half_recovery_indices = recovery_phase[recovery_phase <= half_amplitude].index
                                                    if len(half_recovery_indices) > 0:
                                                        half_recovery_index = half_recovery_indices[0]
                                                        decomposed.loc[half_recovery_index, f"SCR_HalfRecovery_{peak_method}"] = 1

                                            # Add other SCR information to the DataFrame
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_Peaks_{peak_method}"] = 1
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_Height_{peak_method}"] = peaks['SCR_Height']
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_Amplitude_{peak_method}"] = peaks['SCR_Amplitude']
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_RiseTime_{peak_method}"] = peaks['SCR_RiseTime']
                                            
                                            # Check if SCR_Recovery is not empty and does not contain NaN values
                                            if 'SCR_Recovery' in peaks and not np.isnan(peaks['SCR_Recovery']).any():
                                                decomposed.loc[peaks['SCR_Recovery'], f"SCR_Recovery_{peak_method}"] = 1
                                                decomposed.loc[peaks['SCR_Recovery'], f"SCR_RecoveryTime_{peak_method}"] = peaks['SCR_RecoveryTime']
                                            else:
                                                logging.warning(f"No valid SCR recovery found for {peak_method} method.")
                                            
                                            logging.info(f"SCR events detected and added to DataFrame for {method} using {peak_method}")
                                        else:
                                            logging.warning(f"No SCR events detected for method {method} using peak detection method {peak_method}.")

                                        logging.info(f"Plotting peaks for {method} using {peak_method} method.")
                                        
                                        # Create a figure with three subplots for each combination
                                        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                                        # Plot 1: Overlay of Raw and Cleaned EDA (common)
                                        axes[0].plot(eda_filtered, label='Raw Filtered EDA')
                                        axes[0].plot(eda_cleaned, label='Filtered Cleaned EDA', color='orange')
                                        axes[0].set_title('Filtered Raw and Cleaned EDA Signal')
                                        axes[0].set_ylabel('EDA (µS)')
                                        axes[0].legend()

                                        # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery (specific)
                                        axes[1].plot(decomposed["EDA_Phasic"], label='Phasic Component', color='green')
                                        
                                        # Check if SCR_Onsets is not empty and does not contain NaN values
                                        if 'SCR_Onsets' in peaks and not np.isnan(peaks['SCR_Onsets']).any():
                                            # Plot the SCR Onsets
                                            axes[1].scatter(peaks['SCR_Onsets'], decomposed.loc[peaks['SCR_Onsets'], "EDA_Phasic"], color='blue', label='SCR Onsets')
                                        else:
                                            logging.warning(f"No valid SCR onsets found for plotting {peak_method} method.")
                                        axes[1].scatter(peaks['SCR_Peaks'], decomposed["EDA_Phasic"][peaks['SCR_Peaks']], color='red', label='SCR Peaks')
                                        
                                        # Check if the SCR Half Recovery column exists for the current peak_method
                                        half_recovery_column = f"SCR_HalfRecovery_{peak_method}"
                                        if half_recovery_column in decomposed.columns:
                                            # Extract indices where Half Recovery points are marked
                                            half_recovery_indices = decomposed[decomposed[half_recovery_column] == 1].index

                                            # Plot the Half Recovery points
                                            if not half_recovery_indices.empty:
                                                axes[1].scatter(half_recovery_indices, decomposed.loc[half_recovery_indices, "EDA_Phasic"], color='purple', label='SCR Half Recovery')
                                        
                                        axes[1].set_title(f'Phasic EDA ({method}) with {peak_method} Peaks')
                                        axes[1].set_ylabel('Amplitude (µS)')
                                        axes[1].legend()

                                        # Plot 3: Tonic Component (common)
                                        axes[2].plot(time_vector, decomposed["EDA_Tonic"], label='Tonic Component', color='brown')
                                        axes[2].set_title(f'Tonic EDA ({method})')
                                        axes[2].set_xlabel('Time (minutes)')
                                        axes[2].set_ylabel('Amplitude (µS)')
                                        axes[2].legend()

                                        plt.tight_layout()
                                        combo_plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_{method}_{peak_method}_subplots.png")
                                        plt.savefig(combo_plot_filename, dpi=dpi_value)
                                        plt.close()
                                        logging.info(f"Saved EDA subplots for {method} with {peak_method} to {combo_plot_filename}")

                                        # Assuming eda_signals_neurokit and info_eda_neurokit are defined and valid
                                        phasic_component = decomposed['EDA_Phasic']
                                        tonic_component = decomposed['EDA_Tonic']

                                        # Basic statistics for Phasic Component
                                        phasic_stats = {
                                            'Mean': phasic_component.mean(),
                                            'Median': phasic_component.median(),
                                            'Std Deviation': phasic_component.std(),
                                            'Variance': phasic_component.var(),
                                            'Skewness': phasic_component.skew(),
                                            'Kurtosis': phasic_component.kurtosis()
                                        }

                                        # SCR-specific metrics
                                        scr_onsets = peaks['SCR_Onsets']
                                        scr_peaks = peaks['SCR_Peaks']
                                        scr_amplitudes = phasic_component[scr_peaks] - phasic_component[scr_onsets]

                                        # Calculate additional SCR-specific metrics
                                        total_time = (len(phasic_component) / sampling_rate) / 60  # Convert to minutes
                                        average_scr_frequency = len(scr_peaks) / total_time
                                        amplitude_range = scr_amplitudes.max() - scr_amplitudes.min()
                                        inter_scr_intervals = np.diff(scr_onsets) / sampling_rate  # Convert to seconds
                                        average_inter_scr_interval = np.mean(inter_scr_intervals)

                                        scr_stats = {
                                            'SCR Count': len(scr_peaks),
                                            'Mean SCR Amplitude': scr_amplitudes.mean(),
                                            'Average SCR Frequency': average_scr_frequency,
                                            'Amplitude Range of SCRs': amplitude_range,
                                            'Average Inter-SCR Interval': average_inter_scr_interval
                                            # Add AUC and distribution analysis as needed
                                        }

                                        # Basic statistics for Tonic Component
                                        tonic_stats = {
                                            'Mean': tonic_component.mean(),
                                            'Median': tonic_component.median(),
                                            'Std Deviation': tonic_component.std(),
                                            'Variance': tonic_component.var(),
                                            'Skewness': tonic_component.skew(),
                                            'Kurtosis': tonic_component.kurtosis(),
                                            'Range': tonic_component.max() - tonic_component.min(),
                                            'Total Absolute Sum': tonic_component.abs().sum(),
                                            '25th Percentile': tonic_component.quantile(0.25),
                                            '75th Percentile': tonic_component.quantile(0.75),
                                            'IQR': tonic_component.quantile(0.75) - tonic_component.quantile(0.25),
                                            '10th Percentile': tonic_component.quantile(0.10),
                                            '90th Percentile': tonic_component.quantile(0.90)
                                        }

                                        # Combine all stats in a single dictionary or DataFrame
                                        eda_summary_stats = {
                                            'Phasic Stats': phasic_stats,
                                            'SCR Stats': scr_stats,
                                            'Tonic Stats': tonic_stats
                                        }
                                    
                                        # Convert the combined stats to a DataFrame
                                        eda_summary_df = pd.DataFrame(eda_summary_stats)

                                        # Save the summary statistics to a TSV file
                                        summary_stats_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_{method}_{peak_method}_summary_statistics.tsv")
                                        eda_summary_df.to_csv(summary_stats_filename, sep='\t', index=False)
                                        logging.info(f"Saved summary statistics to {summary_stats_filename}")
                                    
                                    except Exception as e:
                                        logging.error(f"Error in peak detection ({method}, {peak_method}): {e}")
                                        # Log stack trace for debugging purposes
                                        logging.error(traceback.format_exc())
                                        continue

                                # Save decomposed data to a TSV file
                                logging.info(f"Saving decomposed data to TSV file for {method}.")
                                decomposed_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_processed_{method}.tsv")
                                decomposed.to_csv(decomposed_filename, sep='\t', index=False)
                                
                                # Compress the TSV file
                                with open(decomposed_filename, 'rb') as f_in:
                                    with gzip.open(f"{decomposed_filename}.gz", 'wb') as f_out:
                                        f_out.writelines(f_in)
                                
                                # Remove the uncompressed file
                                os.remove(decomposed_filename)
                                logging.info(f"Saved and compressed decomposed data to {decomposed_filename}.gz")

                    except Exception as e:
                        logging.error(f"Error processing file {file}: {e}")
                        traceback.print_exc()
                        continue # Continue to the next file
                
                # Log autocorrelation for the raw and filtered EDA signals
                log_eda_autocorrelation(eda, "Raw Unfiltered EDA", sampling_rate, log_file_path)
                log_eda_autocorrelation(eda_filtered, "Filtered EDA", sampling_rate, log_file_path)
                
                # Log autocorrelation for the raw, cleaned, decomposed EDA signals
                log_eda_autocorrelation(eda_signals_neurokit['EDA_Clean'], "Unfiltered EDA (Clean)", sampling_rate, log_file_path)                
                log_eda_autocorrelation(eda_signals_neurokit['EDA_Phasic'], "Unfiltered Phasic EDA (Clean)", sampling_rate, log_file_path)
                log_eda_autocorrelation(eda_signals_neurokit['EDA_Tonic'], "Unfiltered Tonic EDA (Clean)", sampling_rate, log_file_path)

                # Log autocorrelation for the filtered, cleaned, decomposed EDA signals
                log_eda_autocorrelation(eda_signals_neurokit_filt['EDA_Clean'], "Filtered EDA (Clean)", sampling_rate, log_file_path)
                log_eda_autocorrelation(eda_signals_neurokit_filt['EDA_Phasic'], "Filtered Phasic EDA (Clean)", sampling_rate, log_file_path)
                log_eda_autocorrelation(eda_signals_neurokit_filt['EDA_Tonic'], "Filtered Tonic EDA (Clean)", sampling_rate, log_file_path)

                # For each phasic decomposition method
                for method in methods:
                    # Ensure the EDA phasic and tonic components exist in the DataFrame
                    if f"EDA_Phasic_{method}" in decomposed.columns and f"EDA_Tonic_{method}" in decomposed.columns:
                        # Calculate and log autocorrelation for the phasic component
                        log_eda_autocorrelation(decomposed[f"EDA_Phasic_{method}"], f"Phasic EDA ({method})", sampling_rate, log_file_path)

                        # Calculate and log autocorrelation for the tonic component
                        log_eda_autocorrelation(decomposed[f"EDA_Tonic_{method}"], f"Tonic EDA ({method})", sampling_rate, log_file_path)

                # Record the end time for this run and calculate runtime
                run_end_time = datetime.now()
                run_runtime = (run_end_time - run_start_time).total_seconds() / 60
                logging.info(f"Run {run_number} completed. Runtime: {run_runtime} minutes.")
                log_resource_usage()  # Log resource usage at the end of each run
            
            except Exception as e:
            # Log the error
                logging.error(f"Error processing run {run_number} for participant {participant_id}: {e}")
                traceback.print_exc()  # Print the traceback for debugging purposes
                continue  # Continue to the next run
        
        # Record the end time for this participant and calculate runtime
        participant_end_time = datetime.now()
        participant_runtime = (participant_end_time - participant_start_time).total_seconds() / 60
        print(f"Participant {participant_id} completed. Runtime: {participant_runtime} minutes.")
                
    # Record the script end time and calculate runtime
    end_time = datetime.now()
    script_runtime = (end_time - start_time).total_seconds() / 60
    print(f"Main function completed. Script runtime: {script_runtime} minutes. Processing complete for participant {participant_id}.")
        

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    cProfile.run('main()')