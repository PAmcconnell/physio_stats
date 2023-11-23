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

import neurokit2 as nk
import pandas as pd
import gzip
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is for writing to files, not for rendering in a window.
import matplotlib.pyplot as plt
import logging
import os
import traceback
from datetime import datetime
import numpy as np
import sys
from scipy.signal import iirfilter, sosfreqz, sosfiltfilt
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

def comb_band_stop_filter(data, stop_freq, fs, order=5):
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
    stop_band = [0.4, 0.6]

    # Design the bandstop filter
    sos = iirfilter(order, stop_band, btype='bandstop', fs=fs, output='sos')

    # Apply the filter
    y = sosfiltfilt(sos, data)
    
    return y

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
                
                # Set a higher DPI for better resolution
                dpi_value = 300 

                # Select the first participant for testing
                #participant_id = participants_df['participant_id'].iloc[2]

                task_name = 'rest'

                # Process the first run for the selected participant
                #run_number = 1
                run_id = f"run-0{run_number}"

                # Setup logging
                session_id = 'ses-1'  # Assuming session ID is known
                log_file_path = setup_logging(participant_id, session_id, run_id, dataset_root_dir)
                log_resource_usage()  # Log resource usage at the start of participant processing
                logging.info(f"Testing EDA processing for task {task_name} run-0{run_number} for participant {participant_id}")

                # Filter settings
                stop_freq = 0.5  # 0.5 Hz (1/TR with TR = 2s)
                sampling_rate = 5000    # Your sampling rate

                physio_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
                for file in physio_files:
                    # Record the start time for this run
                    run_start_time = datetime.now()
                    logging.info(f"Processing file: {file}")
                    log_resource_usage()  # Log resource usage at the start of run processing
                    
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    logging.info(f"[Memory Usage] Top 10 lines")
                    for stat in top_stats[:10]:
                        logging.info(stat)
                    
                    try:
                        # Generate a base filename by removing the '.tsv.gz' extension
                        base_filename = os.path.splitext(os.path.splitext(file)[0])[0]

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

                            # Process EDA signal using NeuroKit
                            eda_signals_neurokit, info_eda_neurokit = nk.eda_process(eda, sampling_rate=5000, method='neurokit')
                            logging.info("Default EDA signal processing complete.")

                            # Create a figure with three subplots
                            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                            # Plot 1: Overlay of Raw and Cleaned EDA
                            axes[0].plot(eda_signals_neurokit['EDA_Raw'], label='Raw EDA', color='green')
                            axes[0].plot(eda_signals_neurokit['EDA_Clean'], label='Cleaned EDA', color='orange')
                            axes[0].set_title('Raw and Cleaned EDA Signal')
                            axes[0].set_ylabel('EDA (µS)')
                            axes[0].legend()

                            # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery
                            axes[1].plot(eda_signals_neurokit['EDA_Phasic'], label='Phasic Component', color='green')
                            axes[1].scatter(info_eda_neurokit['SCR_Onsets'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Onsets']], color='blue', label='SCR Onsets')
                            axes[1].scatter(info_eda_neurokit['SCR_Peaks'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Peaks']], color='red', label='SCR Peaks')
                            
                            # Assuming 'SCR_HalfRecovery' is in info_eda_neurokit
                            axes[1].scatter(info_eda_neurokit['SCR_Recovery'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Recovery']], color='purple', label='SCR Half Recovery')
                            axes[1].set_title('Phasic EDA with SCR Events')
                            axes[1].set_ylabel('Amplitude (µS)')
                            axes[1].legend()

                            # Plot 3: Tonic Component
                            axes[2].plot(time_vector, eda_signals_neurokit['EDA_Tonic'], label='Tonic Component', color='brown')
                            axes[2].set_title('Tonic EDA')
                            axes[2].set_xlabel('Time (minutes)')
                            axes[2].set_ylabel('Amplitude (µS)')
                            axes[2].legend()

                            plt.tight_layout()
                            plot_filename = f"{base_filename}_eda_unfiltered.png"
                            plt.savefig(plot_filename, dpi=dpi_value)
                            plt.close()
                            logging.info(f"Saved default EDA subplots to {plot_filename}")

                            # Pre-filter gradient artifact.
                            eda_filtered_array = comb_band_stop_filter(eda, stop_freq, sampling_rate)

                            # Convert the filtered data back into a Pandas Series
                            eda_filtered = pd.Series(eda_filtered_array, index=eda.index)

                            if eda_filtered.empty:
                                logging.error("Error: 'eda_filtered' is empty.")
                            else:
                                logging.info(f"'eda_filtered' is not empty, length: {len(eda_filtered)}")

                            # Process EDA signal using NeuroKit
                            eda_signals_neurokit, info_eda_neurokit = nk.eda_process(eda_filtered, sampling_rate=5000, method='neurokit')
                            logging.info("Default EDA signal processing complete.")

                            # Create a figure with three subplots
                            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                            # Plot 1: Overlay of Raw and Cleaned EDA
                            axes[0].plot(eda, label='Raw EDA', color='blue', linewidth=4.0)
                            axes[0].plot(eda_signals_neurokit['EDA_Raw'], label='Raw (Prefiltered) EDA', color='green')
                            axes[0].plot(eda_signals_neurokit['EDA_Clean'], label='Cleaned EDA', color='orange')
                            axes[0].set_title('Raw (Prefiltered) and Cleaned EDA Signal')
                            axes[0].set_ylabel('EDA (µS)')
                            axes[0].legend()

                            # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery
                            axes[1].plot(eda_signals_neurokit['EDA_Phasic'], label='Phasic Component', color='green')
                            axes[1].scatter(info_eda_neurokit['SCR_Onsets'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Onsets']], color='blue', label='SCR Onsets')
                            axes[1].scatter(info_eda_neurokit['SCR_Peaks'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Peaks']], color='red', label='SCR Peaks')
                            
                            # Assuming 'SCR_HalfRecovery' is in info_eda_neurokit
                            axes[1].scatter(info_eda_neurokit['SCR_Recovery'], eda_signals_neurokit['EDA_Phasic'][info_eda_neurokit['SCR_Recovery']], color='purple', label='SCR Half Recovery')
                            axes[1].set_title('Phasic EDA with SCR Events')
                            axes[1].set_ylabel('Amplitude (µS)')
                            axes[1].legend()

                            # Plot 3: Tonic Component
                            axes[2].plot(time_vector, eda_signals_neurokit['EDA_Tonic'], label='Tonic Component', color='brown')
                            axes[2].set_title('Tonic EDA')
                            axes[2].set_xlabel('Time (minutes)')
                            axes[2].set_ylabel('Amplitude (µS)')
                            axes[2].legend()

                            plt.tight_layout()
                            plot_filename = f"{base_filename}_eda_default_subplots.png"
                            plt.savefig(plot_filename, dpi=dpi_value)
                            plt.close()
                            logging.info(f"Saved default EDA subplots to {plot_filename}")
                            
                            # First, clean the EDA signal
                            eda_cleaned = nk.eda_clean(eda_filtered, sampling_rate=5000)
                            logging.info(f"Prefiltered EDA signal cleaned using NeuroKit's eda_clean.")
                            
                            # Define methods for phasic decomposition and peak detection.
                            methods = ['highpass', 'cvxEDA', 'smoothmedian']
                            peak_methods = ["kim2004", "neurokit", "gamboa2008", "vanhalem2020", "nabian2018"]
                                
                            # Process phasic decomposition of EDA signal using NeuroKit
                            for method in methods:
                                logging.info(f"Starting processing for method: {method}")
                                log_resource_usage()  # Log resource usage at the start of method processing
                                
                                snapshot = tracemalloc.take_snapshot()
                                top_stats = snapshot.statistics('lineno')
                                logging.info(f"[Memory Usage] Top 10 lines")
                                for stat in top_stats[:10]:
                                    logging.info(stat)
                                
                                # Decompose EDA signal using the specified method
                                try:
                                    decomposed = nk.eda_phasic(eda_cleaned, method=method, sampling_rate=5000)
                                    logging.info(f"Decomposed EDA using {method} method. Size of decomposed data: {decomposed.size}")
                                except Exception as e:
                                    logging.error(f"Error in EDA decomposition with method {method}: {e}")
                                    continue


                                # Initialize columns for SCR events in the DataFrame
                                for peak_method in peak_methods:
                                    decomposed[f"SCR_Amplitude_{peak_method}"] = np.nan
                                    decomposed[f"SCR_Onsets_{peak_method}"] = np.nan
                                    decomposed[f"SCR_Peaks_{peak_method}"] = np.nan

                                # Calculate amplitude_min from the phasic component for peak detection
                                amplitude_min = 0.01 * decomposed["EDA_Phasic"].max()
                                logging.info(f"Calculated amplitude_min: {amplitude_min} for {method} method.")

                                # Plotting for Tonic and Phasic components
                                for component in ["EDA_Tonic", "EDA_Phasic"]:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(decomposed[component], label=f'{component}')
                                    ax.set_title(f'{component} - {method}')
                                    ax.set_xlabel('Time (seconds)')
                                    ax.set_ylabel('Amplitude')
                                    ax.legend()
                                    
                                    # Save each component's plot
                                    component_plot_filename = f"{base_filename}_{method}_{component}_plot.png"
                                    plt.savefig(component_plot_filename, dpi=dpi_value)
                                    plt.close()
                                    logging.info(f"Saved {component} plot for {method} to {component_plot_filename}")

                                # Peak detection and plotting
                                for peak_method in peak_methods:
                                    log_resource_usage() # Log resource usage at the start of peak method processing
                                    logging.info(f"Processing peak detection for {method} using {peak_method} method.")
                                    
                                    snapshot = tracemalloc.take_snapshot()
                                    top_stats = snapshot.statistics('lineno')
                                    logging.info(f"[Memory Usage] Top 10 lines")
                                    for stat in top_stats[:10]:
                                        logging.info(stat)
                                    
                                    try:
                                        # Check if the phasic component is not empty
                                        if decomposed["EDA_Phasic"].size == 0:
                                            logging.warning(f"The phasic component is empty for {method}. Skipping peak detection.")
                                            continue
                                        
                                        # Only add amplitude_min for applicable methods
                                        peak_detection_kwargs = {"sampling_rate": 5000, "method": peak_method}

                                        # Apply amplitude_min only for the "neurokit" or "kim2004" methods
                                        if peak_method in ["neurokit", "kim2004"]:
                                            peak_detection_kwargs["amplitude_min"] = amplitude_min
                                
                                        # Detect peaks using the specified method
                                        logging.info(f"Detecting peaks using method: {peak_method}")    
                                        _, peaks = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=5000, method=peak_method)
                                        
                                        # Add SCR Amplitude, Onsets, and Peaks to the DataFrame
                                        if peaks['SCR_Peaks'].size > 0:
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_Peaks_{peak_method}"] = 1
                                            decomposed.loc[peaks['SCR_Peaks'], f"SCR_Amplitude_{peak_method}"] = peaks['SCR_Amplitude']

                                            # Optional: If you also want to include SCR onsets
                                            if 'SCR_Onsets' in peaks:
                                                decomposed.loc[peaks['SCR_Onsets'], f"SCR_Onsets_{peak_method}"] = 1

                                            logging.info(f"SCR events detected and added to DataFrame for {method} using {peak_method}")

                                        else:
                                            logging.warning(f"No SCR events detected for method {method} using peak detection method {peak_method}.")
                                        
                                        logging.info(f"Plotting peaks for {method} using {peak_method} method.")
                                        # Create a figure with three subplots for each combination
                                        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                                        # Plot 1: Overlay of Raw and Cleaned EDA (common)
                                        axes[0].plot(eda_signals_neurokit['EDA_Raw'], label='Raw EDA')
                                        axes[0].plot(eda_signals_neurokit['EDA_Clean'], label='Cleaned EDA', color='orange')
                                        axes[0].set_title('Raw and Cleaned EDA Signal')
                                        axes[0].set_ylabel('EDA (µS)')
                                        axes[0].legend()

                                        # Plot 2: Phasic Component with SCR Onsets, Peaks, and Half Recovery (specific)
                                        axes[1].plot(decomposed["EDA_Phasic"], label='Phasic Component', color='green')
                                        axes[1].scatter(peaks['SCR_Onsets'], decomposed["EDA_Phasic"][peaks['SCR_Onsets']], color='blue', label='SCR Onsets')
                                        axes[1].scatter(peaks['SCR_Peaks'], decomposed["EDA_Phasic"][peaks['SCR_Peaks']], color='red', label='SCR Peaks')
                                        # If SCR Half Recovery data is available, include it here
                                        # axes[1].scatter(peaks['SCR_HalfRecovery'], decomposed["EDA_Phasic"][peaks['SCR_HalfRecovery']], color='purple', label='SCR Half Recovery')
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
                                        combo_plot_filename = f"{base_filename}_{method}_{peak_method}_subplots.png"
                                        plt.savefig(combo_plot_filename, dpi=dpi_value)
                                        plt.close()
                                        logging.info(f"Saved EDA subplots for {method} with {peak_method} to {combo_plot_filename}")

                                    except Exception as e:
                                        logging.error(f"Error in peak detection ({method}, {peak_method}): {e}")

                                # Save decomposed data to a TSV file
                                logging.info(f"Saving decomposed data to TSV file for {method}.")
                                decomposed_filename = f"{base_filename}_{method}_decomposed.tsv"
                                decomposed.to_csv(decomposed_filename, sep='\t', index=False)
                                
                                # Compress the TSV file
                                with open(decomposed_filename, 'rb') as f_in:
                                    with gzip.open(f"{decomposed_filename}.gz", 'wb') as f_out:
                                        f_out.writelines(f_in)
                                
                                # Remove the uncompressed file
                                os.remove(decomposed_filename)
                                logging.info(f"Saved and compressed decomposed data to {decomposed_filename}.gz")

                            # Save processed signals to a TSV file
                            
                            logging.info(f"Saving processed signals to TSV file.")
                            processed_signals_filename = f"{base_filename}_processed_eda.tsv"
                            eda_signals_neurokit.to_csv(processed_signals_filename, sep='\t', index=False)
                            
                            # Compress the processed signals file
                            with open(processed_signals_filename, 'rb') as f_in:
                                with gzip.open(f"{base_filename}_processed_eda.tsv.gz", 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            # Remove the uncompressed file
                            os.remove(processed_signals_filename)
                            logging.info(f"Saved and compressed processed signals to {processed_signals_filename}.gz")

                    except Exception as e:
                        logging.error(f"Error processing file {file}: {e}")
                        traceback.print_exc()
                        continue # Continue to the next file
                
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