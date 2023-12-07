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

# Standard library imports
import os
import re
import sys
import logging
import traceback
import gzip
import glob
from datetime import datetime

# Data handling and numerical computation
import numpy as np
import pandas as pd
from scipy.signal import iirfilter, sosfreqz, sosfiltfilt, freqz
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA

# Performance profiling and resource management
import psutil
import cProfile
import tracemalloc

# Neuroimaging data processing
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MCFLIRT
import shutil  # For file operations such as moving or deleting files

# Data visualization
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for file-based plotting
import matplotlib.pyplot as plt

# Neurophysiological data analysis
import neurokit2 as nk

# conda activate fmri (Python 3.9)

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
        # Calculate autocorrelation
        with open(logfile_path, 'a') as log_file:
            logging.info(f"Autocorrelation for {eda_signal} {method_name}: {autocorrelation}\n")
        logging.info(f"Autocorrelation for {method_name} calculated and logged.")
    except Exception as e:
        logging.error(f"Error calculating autocorrelation for {method_name}: {e}")

# Function to calculate framewise displacement
def calculate_fd(motion_params_file):
    """
    Calculate framewise displacement from motion parameters.

    This function reads motion parameters from a file, computes translations and rotations,
    and calculates the framewise displacement. The rotations are converted to mm using a 
    specified radius (default is 50 mm) before summing with translations for FD calculation.

    Parameters:
    motion_params_file (str): Path to the file containing motion parameters.

    Returns:
    numpy.ndarray: Array of framewise displacement values, or None if an error occurs.
    """
    try:
        # Load motion parameters
        motion_params = np.loadtxt(motion_params_file)

        # Calculate translations and rotations
        translations = np.abs(np.diff(motion_params[:, :3], axis=0))
        rotations = np.abs(np.diff(motion_params[:, 3:], axis=0)) * np.pi / 180 * 50  # Convert to mm

        # Sum translations and rotations for FD
        fd = np.sum(translations, axis=1) + np.sum(rotations, axis=1)
        logging.info("Framewise displacement calculated successfully.")

        return fd
    except Exception as e:
        logging.error(f"Error in calculating framewise displacement: {e}")
        return None

# Function to calculate the correlation between FD and EDA
def calculate_fd_eda_correlation(fd, eda):
    """
    Calculate the Pearson correlation coefficient and p-value between framewise displacement (FD) and electrodermal activity (EDA).

    Parameters:
    fd (array-like): The framewise displacement timeseries.
    eda (array-like): The electrodermal activity timeseries.

    Returns:
    float: The Pearson correlation coefficient.
    float: The p-value indicating statistical significance.

    The function assumes both inputs are numpy arrays of the same length.
    The Pearson correlation coefficient measures the linear correlation between the two arrays,
    returning a value between -1 and 1, where 1 means total positive linear correlation,
    0 means no linear correlation, and -1 means total negative linear correlation.
    The p-value roughly indicates the probability of an uncorrelated system producing datasets
    that have a Pearson correlation at least as extreme as the one computed from these datasets.
    """

    # Ensure that FD and EDA are of the same length
    if len(fd) != len(eda):
        logging.error("FD and EDA timeseries must be of the same length.")
        return None, None

    try:
        # Calculate Pearson correlation
        r_value, p_value = pearsonr(fd, eda)
        logging.info(f"Calculated Pearson correlation: {r_value}, p-value: {p_value}")
        return r_value, p_value
    except Exception as e:
        logging.error(f"Error in calculating correlation: {e}")
        return None, None
    
# Save framewise displacement data to a TSV file
def save_fd_to_tsv(fd_timeseries, output_dir, filename):
    """
    Save framewise displacement data to a TSV file.

    This function takes framewise displacement data, converts it to a pandas DataFrame, 
    and then saves it as a tab-separated values (TSV) file in the specified directory.

    Parameters:
    fd_timeseries (list or array-like): The framewise displacement timeseries data.
    output_dir (str): The directory where the TSV file will be saved.
    filename (str): The name of the TSV file.

    Returns:
    bool: True if the file is saved successfully, False otherwise.
    """
    
    try:
        # Ensuring output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Converting the timeseries data to a DataFrame
        fd_df = pd.DataFrame(fd_timeseries, columns=['Framewise_Displacement'])

        # Constructing the full file path
        file_path = os.path.join(output_dir, filename)

        # Saving the DataFrame to a TSV file
        fd_df.to_csv(file_path, sep='\t', index=False)
        logging.info(f"Framewise displacement data saved to {file_path}")

        return True
    except Exception as e:
        logging.error(f"Error in saving framewise displacement data: {e}")
        return False

# Function to update the DataFrame for given feature columns based on valid indices
def update_decomposed_for_feature(df, feature_data, feature_name, method):
    valid_indices = np.where(~np.isnan(feature_data))[0]
    valid_values = feature_data[~np.isnan(feature_data)]

    # Update the DataFrame with valid feature data
    for i, value in zip(valid_indices, valid_values):
        if i < len(df):
            df.at[i, f"{feature_name}_{method}"] = value
        else:
            logging.warning(f"Index {i} is out of bounds for feature {feature_name}.")

# Function to run MCFLIRT motion correction on a 4D fMRI dataset and generate motion parameters
def run_mcflirt_motion_correction(original_data_path, output_dir, working_dir):
    """
    Run MCFLIRT motion correction on a 4D fMRI dataset.

    This function applies MCFLIRT, an FSL tool for motion correction, to a 4D fMRI dataset. It saves
    the motion-corrected output, motion plots, and transformation matrices to specified directories.

    Parameters:
    original_data_path (str): Path to the original 4D fMRI data file.
    output_dir (str): Directory where the motion-corrected file and related outputs will be saved.
    working_dir (str): Working directory for intermediate files.

    Returns:
    str: The path of the motion-corrected output file, or None if an error occurs.

    Raises:
    Exception: Propagates any exceptions that occur during processing.
    """
    try:
        logging.info("Running MCFLIRT motion correction...")

        # Configure MCFLIRT
        mcflirt = MCFLIRT()
        mcflirt.inputs.in_file = original_data_path
        out_filename = 'mcf_' + os.path.basename(original_data_path).replace('.gz', '')  # Handles both .nii and .nii.gz
        mcflirt.inputs.out_file = os.path.join(output_dir, out_filename)
        
        logging.info(f"Output file: {mcflirt.inputs.out_file}")

        # Set MCFLIRT options
        mcflirt.inputs.save_plots = True  # Save motion plots
        mcflirt.inputs.save_mats = True   # Save transformation matrices
        mcflirt.base_dir = working_dir    # Set working directory

        # Run MCFLIRT
        mcflirt.run()

        logging.info("MCFLIRT motion correction completed successfully.")
        return mcflirt.inputs.out_file

    except Exception as e:
        logging.error(f"Error during motion correction with MCFLIRT: {e}")
        raise

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
    
    # Define and check derivatives directory
    derivatives_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    print(f"Checking derivatives directory: {derivatives_root_dir}")
    if not os.path.exists(derivatives_root_dir):
        print(f"Directory not found: {derivatives_root_dir}")
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
#    for i, participant_id in enumerate([participants_df['participant_id'].iloc[0]]):  # For testing

        # Record the start time for this participant
        participant_start_time = datetime.now()
#        for run_number in range(1, 5):  # Assuming 4 runs
        for run_number in range(1, 2):  # Testing with 1 run
            try:
        
                # Set a higher DPI for better resolution
                dpi_value = 300 

                task_name = 'rest'

                # Process the first run for the selected participant
                run_id = f"run-0{run_number}"

                # Construct the base path
                base_path = os.path.join(derivatives_dir, 'eda', participant_id, run_id)

                # Make sure the directories exist
                os.makedirs(base_path, exist_ok=True)
                
                # Define the processed signals filename for checking
                processed_signals_filename = f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio_unfiltered_cleaned_eda_processed.tsv.gz"
                processed_signals_path = os.path.join(base_path, processed_signals_filename)

                # Check if processed file exists
                if os.path.exists(processed_signals_path):
                    print(f"Processed EDA files found for {participant_id} for run {run_number}, skipping...")
                    continue  # Skip to the next run

                # Setup logging
                session_id = 'ses-1'  # Assuming session ID is known
                
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                log_file_path = setup_logging(participant_id, session_id, run_id, dataset_root_dir)
                logging.info(f"Testing EDA processing for task {task_name} run-0{run_number} for participant {participant_id}")

                # Filter settings
                stop_freq = 0.5  # 0.5 Hz (1/TR with TR = 2s)
                sampling_rate = 5000    # acquisition sampling rate
            
                # Define the frequency band
                frequency_band = (0.045, 0.25) # 0.045 - 0.25 Hz Sympathetic Band

                pattern = re.compile(f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz")
                physio_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
                
                bids_subject_dir = os.path.join(bids_root_dir, participant_id, session_id, 'func')
                func_files = [os.path.join(bids_subject_dir, f) for f in os.listdir(bids_subject_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_bold.nii' in f]
                
                for file in physio_files:
                    # Record the start time for this run
                    run_start_time = datetime.now()
                    logging.info(f"Processing file: {file}")
                    
                    try:
                        
                        # BIDS location for original 4D nifti data
                        func_file = os.path.join(bids_subject_dir, f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_bold.nii')
                        original_nii_data_path = func_file
                        logging.info("Original data path: {}".format(original_nii_data_path))
                        
                        # Generate a base filename by removing the '.tsv.gz' extension
                        base_filename_func = os.path.basename(file).replace('.nii', '')

                        # Temp directory for intermediate files
                        working_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/temp')     
                        if not os.path.exists(working_dir):
                            os.makedirs(working_dir)
                        logging.info("Working directory: {}".format(working_dir))
                        
                        # Output directory
                        output_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/_motion')         
                        if not os.path.exists(output_root_dir):
                            os.makedirs(output_root_dir)

                        output_subject_dir = os.path.join(output_root_dir, participant_id, run_id)
                        if not os.path.exists(output_subject_dir):
                            os.makedirs(output_subject_dir)
                        logging.info("Output subject directory: {}".format(output_subject_dir))

                        # Correctly forming the file name for the FD data
                        fd_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_framewise_displacement.tsv"

                        # Joining the output directory with the new file name
                        fd_file_path = os.path.join(output_subject_dir, fd_filename)

                        # Check if FD files already exist
                        if not os.path.exists(fd_file_path):
                            logging.info(f"Processed FD files not found for {participant_id} for run {run_number}, proceeding with mcflirt...")
                            
                            # Run MCFLIRT motion correction
                            try:
                                output_file = run_mcflirt_motion_correction(original_nii_data_path, output_subject_dir, working_dir)
                                logging.info(f"Motion corrected file saved at {output_file}")
                            except Exception as e:
                                logging.error(f"Failed to complete motion correction: {e}")

                            # Load motion parameters and calculate FD
                            try:
                                logging.info("Calculating framewise displacement...")
                                
                                # Create FD tsv from mcflirt motion parameters
                                motion_params_file = os.path.join(output_subject_dir, 'mcf_' + os.path.basename(original_nii_data_path) + '.par')
                                fd = calculate_fd(motion_params_file)

                                # Saving the FD data to the correct path
                                save_fd_to_tsv(fd, output_subject_dir, fd_file_path)
                                logging.info(f"Framewise displacement data saved successfully to {fd_file_path}.")

                            except Exception as e:
                                logging.error("Error in calculating framewise displacement: {}".format(e))
                                raise
                        else: 
                            logging.info(f"Processed FD files found for {participant_id} for run {run_number}, skipping mcflirt...")
                            
                            # Read the existing FD data from the TSV
                            try:
                                fd = pd.read_csv(fd_file_path, delimiter='\t')
                                # Assuming the FD values are in the first column, if not, adjust the index accordingly.
                                fd = fd.iloc[:, 0].values
                            except Exception as e:
                                logging.error(f"Error in reading existing FD data: {e}")
                                raise   

                    except Exception as e:
                        logging.error(f"Error finding functional file for {participant_id} for run {run_number}: {e}")
                        return 
                    
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

                            ### Step 1: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz ###
                            logging.info(f"Step 1: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz.")

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
                                logging.error(f"Error: 'eda_filtered' is empty.")
                                # Log stack trace for debugging purposes
                                logging.error(traceback.format_exc())
                            else:
                                logging.info(f"'eda_filtered' is not empty, length: {len(eda_filtered)}")

                            sampling_rate = 50    # downsampled sampling rate
                            
                            # Calculate the time vector in minutes - Length of EDA data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(new_length) / sampling_rate / 60

                            ### Step 2: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods. ###
                            logging.info(f"Step 2: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods.")

                            # First, clean the EDA signal
                            eda_cleaned = nk.eda_clean(eda_filtered, sampling_rate=sampling_rate)
                            logging.info(f"Prefiltered EDA signal cleaned using NeuroKit's eda_clean.")
                            
                            logging.info(f"Starting phasic decomposition and peak detection for prefiltered EDA signal.")
                            logging.info(f"Sampling rate: {sampling_rate} Hz")
                            logging.info(f"Size of prefiltered EDA signal: {eda_cleaned.size}")

                            # Define methods for phasic decomposition and peak detection.
                            methods = ['cvxEDA'] #sparsEDA 'sparse'? 'smoothmedian', 'highpass'
                            logging.info(f"Using the following methods for phasic decomposition: {methods}")

                            peak_methods = ["gamboa2008"] # "kim2004", "neurokit", "vanhalem2020", "nabian2018"
                            logging.info(f"Using the following methods for peak detection: {peak_methods}")

                            # Initialize psd_full_stats dictionary
                            filtered_psd_full_stats_by_method = {}    
                            filtered_psd_symp_full_stats_by_method = {}    
                            
                            ### Step 3: Perform cvxEDA phasic decomposition with Gamboa 2008 peak detection. ###
                            logging.info(f"Step 3: Perform cvxEDA phasic decomposition with Gamboa 2008 peak detection.")

                            # Process phasic decomposition of EDA signal using NeuroKit2
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
                                    logging.error(f"EDA Cleaned details: Range: {eda_cleaned.min()} - {eda_cleaned.max()}, NaNs: {np.isnan(eda_cleaned).sum()}, Infs: {np.isinf(eda_cleaned).sum()}")
                                    raise  # Optionally re-raise the exception if you want the program to stop
                                
                                except Exception as e:
                                    logging.error(f"Error in EDA decomposition with method {method}: {e}")
                                    # Log stack trace for debugging purposes
                                    logging.error(traceback.format_exc())                        
                                    continue
                                
                                # Process FD

                                fd_timeseries = fd  # Assuming 'fd' is your framewise displacement timeseries
                                eda_phasic = decomposed['EDA_Phasic'].values
                                
                                eda_sampling_rate = sampling_rate
                                eda_duration = len(eda_phasic) / eda_sampling_rate  # Total duration in seconds

                                # Create a time array for the FD timeseries
                                fd_time = np.linspace(0, eda_duration, len(fd_timeseries))

                                # The new time array for the upsampled FD timeseries should match the EDA timeseries length
                                # Make sure the last time point in upsampled_time does not exceed the last point in fd_time
                                upsampled_time = np.linspace(0, eda_duration, len(eda_phasic))

                                # Use linear interpolation with bounds_error set to False to prevent extrapolation
                                fd_interpolator = interp1d(fd_time, fd_timeseries, kind='linear', bounds_error=False, fill_value='extrapolate')
                                fd_upsampled = fd_interpolator(upsampled_time)

                                # Handle any NaN values that might have been introduced due to the bounds_error setting
                                fd_upsampled[np.isnan(fd_upsampled)] = fd_timeseries[-1]  # Replace NaNs with the last valid FD value

                                # Calculate FD-EDA correlation
                                r_value, p_value = calculate_fd_eda_correlation(fd_upsampled, eda_phasic)
                                logging.info(f"Correlation between FD and filtered cleaned EDA timeseries: {r_value}, p-value: {p_value}")
                            
                                # After decomposing and before peak detection, calculate sympathetic indices by Posada method.
                                for symp_method in sympathetic_methods:
                                    try:
                                        # Calculate phasic sympathetic indices
                                        eda_symp_decomposed_phasic = nk.eda_sympathetic(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=symp_method, show=False)
            
                                        # Log the phasic sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned phasic sympathetic indices using {symp_method} method for {method}.")

                                    except Exception as e:
                                        logging.error(f"Error in sympathetic index calculation ({method}, {symp_method}): {e}")
                                
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
                                    'Phasic Mean (Normalized Power)': eda_psd_filt_phasic['Power'].mean(),
                                    'Phasic Median (Normalized Power)': eda_psd_filt_phasic['Power'].median(),
                                    'Phasic Total Power (Normalized Power)': eda_psd_filt_phasic['Power'].sum(),
                                    'Phasic Peak Frequency (Hz)': eda_psd_filt_phasic.loc[eda_psd_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Phasic Standard Deviation (Normalized Power)': eda_psd_filt_phasic['Power'].std(),
                                    'Phasic Variance (Normalized Power)': eda_psd_filt_phasic['Power'].var(),
                                    'Phasic Skewness': eda_psd_filt_phasic['Power'].skew(),
                                    'Phasic Kurtosis': eda_psd_filt_phasic['Power'].kurtosis(),
                                    'Phasic Peak Power (Normalized Power)': eda_psd_filt_phasic['Power'].max(),
                                    'Phasic Bandwidth (Hz)': eda_psd_filt_phasic['Frequency'].iloc[-1] - eda_psd_filt_phasic['Frequency'].iloc[0],
                                    'Phasic PSD Area (Normalized Power)': np.trapz(eda_psd_filt_phasic['Power'], eda_psd_filt_phasic['Frequency']),
                                    'Phasic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_phasic['EDA_Sympathetic'],
                                    'Phasic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_phasic['EDA_SympatheticN']
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
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}.tsv")
                                eda_psd_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")
                            
                                # Assuming eda_psd_filt_phasic_full_stats is your dictionary of statistics
                                eda_psd_filt_phasic_stats_df = pd.DataFrame(eda_psd_filt_phasic_full_stats, index=[0])

                                # Transpose the DataFrame
                                eda_psd_filt_phasic_stats_df = eda_psd_filt_phasic_stats_df.transpose().reset_index()
                                eda_psd_filt_phasic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}_summary_statistics.tsv")
                                eda_psd_filt_phasic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD summary statistics to {full_file_path}")
                                
                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned Phasic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_phasic = nk.signal_psd(decomposed['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.045, max_frequency=0.25, window=None, window_type='hann',
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
                                    band_power = eda_psd_symp_filt_phasic[(eda_psd_symp_filt_phasic['Frequency'] >= low_freq) & (eda_psd_symp_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_phasic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_phasic_full_stats.update({
                                    'Phasic Mean (Normalized Power)': eda_psd_symp_filt_phasic['Power'].mean(),
                                    'Phasic Median (Normalized Power)': eda_psd_symp_filt_phasic['Power'].median(),
                                    'Phasic Total Power (Normalized Power)': eda_psd_symp_filt_phasic['Power'].sum(),
                                    'Phasic Peak Frequency (Hz)': eda_psd_symp_filt_phasic.loc[eda_psd_symp_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Phasic Standard Deviation (Normalized Power)': eda_psd_symp_filt_phasic['Power'].std(),
                                    'Phasic Variance (Normalized Power)': eda_psd_symp_filt_phasic['Power'].var(),
                                    'Phasic Skewness': eda_psd_symp_filt_phasic['Power'].skew(),
                                    'Phasic Kurtosis': eda_psd_symp_filt_phasic['Power'].kurtosis(),
                                    'Phasic Peak Power (Normalized Power)': eda_psd_symp_filt_phasic['Power'].max(),
                                    'Phasic Bandwidth (Hz)': eda_psd_symp_filt_phasic['Frequency'].iloc[-1] - eda_psd_symp_filt_phasic['Frequency'].iloc[0],
                                    'Phasic PSD Area (Normalized Power)': np.trapz(eda_psd_symp_filt_phasic['Power'], eda_psd_symp_filt_phasic['Frequency']),
                                    'Phasic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_phasic['EDA_Sympathetic'],
                                    'Phasic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_phasic['EDA_SympatheticN']
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
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}.tsv")
                                eda_psd_symp_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD Symp data to {full_file_path}")
                            
                                # Assuming eda_psd_symp_filt_phasic_full_stats is your dictionary of statistics
                                eda_psd_symp_filt_phasic_stats_df = pd.DataFrame(eda_psd_symp_filt_phasic_full_stats, index=[0])

                                # Transpose the DataFrame
                                eda_psd_symp_filt_phasic_stats_df = eda_psd_symp_filt_phasic_stats_df.transpose().reset_index()
                                eda_psd_symp_filt_phasic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the correctly formatted DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}_summary_statistics.tsv")
                                eda_psd_symp_filt_phasic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Phasic PSD summary statistics to {full_file_path}")

                                # Compute Power Spectral Density 0 - 1 Hz
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} using multitapers hann windowing.")
                                eda_psd_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_filt_tonic_full_stats = {}  
                                
                                # After decomposing and before peak detection, calculate sympathetic indices
                                for symp_method in sympathetic_methods:
                                    try:
                                        # Calculate tonic sympathetic indices
                                        eda_symp_decomposed_tonic = nk.eda_sympathetic(decomposed["EDA_Tonic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                        # Log the sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned tonic sympathetic indices using {symp_method} method for {method}.")

                                    except Exception as e:
                                        logging.error(f"Error in sympathetic index calculation ({method}, {symp_method}): {e}")
                                
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
                                    'Tonic Mean (Normalized Power)': eda_psd_filt_tonic['Power'].mean(),
                                    'Tonic Median (Normalized Power)': eda_psd_filt_tonic['Power'].median(),
                                    'Tonic Total Power (Normalized Power)': eda_psd_filt_tonic['Power'].sum(),
                                    'Tonic Peak Frequency (Hz)': eda_psd_filt_tonic.loc[eda_psd_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Tonic Standard Deviation (Normalized Power)': eda_psd_filt_tonic['Power'].std(),
                                    'Tonic Variance (Normalized Power)': eda_psd_filt_tonic['Power'].var(),
                                    'Tonic Skewness': eda_psd_filt_tonic['Power'].skew(),
                                    'Tonic Kurtosis': eda_psd_filt_tonic['Power'].kurtosis(),
                                    'Tonic Peak Power (Normalized Power)': eda_psd_filt_tonic['Power'].max(),
                                    'Tonic Bandwidth (Hz)': eda_psd_filt_tonic['Frequency'].iloc[-1] - eda_psd_filt_tonic['Frequency'].iloc[0],
                                    'Tonic PSD Area (Normalized Power)': np.trapz(eda_psd_filt_tonic['Power'], eda_psd_filt_tonic['Frequency']),
                                    'Tonic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_tonic['EDA_Sympathetic'],
                                    'Tonic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_tonic['EDA_SympatheticN']
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
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}.tsv")
                                eda_psd_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                                # Assuming eda_psd_filt_tonic_full_stats is your dictionary of statistics
                                eda_psd_filt_tonic_stats_df = pd.DataFrame(eda_psd_filt_tonic_full_stats, index=[0])

                                # Transpose the DataFrame to get labels as rows and values in the second column
                                eda_psd_filt_tonic_stats_df = eda_psd_filt_tonic_stats_df.transpose().reset_index()
                                eda_psd_filt_tonic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the transposed DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}_summary_statistics.tsv")
                                eda_psd_filt_tonic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved full range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")
                                
                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.045, max_frequency=0.25, window=None, window_type='hann',
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
                                    # Corrected line of code
                                    band_power = eda_psd_symp_filt_tonic[(eda_psd_symp_filt_tonic['Frequency'] >= low_freq) & (eda_psd_symp_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_tonic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_tonic_full_stats.update({
                                    'Tonic Mean (Normalized Power)': eda_psd_symp_filt_tonic['Power'].mean(),
                                    'Tonic Median (Normalized Power)': eda_psd_symp_filt_tonic['Power'].median(),
                                    'Tonic Total Power (Normalized Power)': eda_psd_symp_filt_tonic['Power'].sum(),
                                    'Tonic Peak Frequency (Hz)': eda_psd_symp_filt_tonic.loc[eda_psd_symp_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Tonic Standard Deviation (Normalized Power)': eda_psd_symp_filt_tonic['Power'].std(),
                                    'Tonic Variance (Normalized Power)': eda_psd_symp_filt_tonic['Power'].var(),
                                    'Tonic Skewness': eda_psd_symp_filt_tonic['Power'].skew(),
                                    'Tonic Kurtosis': eda_psd_symp_filt_tonic['Power'].kurtosis(),
                                    'Tonic Peak Power (Normalized Power)': eda_psd_symp_filt_tonic['Power'].max(),
                                    'Tonic Bandwidth (Hz)': eda_psd_symp_filt_tonic['Frequency'].iloc[-1] - eda_psd_symp_filt_tonic['Frequency'].iloc[0],
                                    'Tonic PSD Area (Normalized Power)': np.trapz(eda_psd_symp_filt_tonic['Power'], eda_psd_symp_filt_tonic['Frequency']),
                                    'Tonic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_tonic['EDA_Sympathetic'],
                                    'Tonic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_tonic['EDA_SympatheticN']
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
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}.tsv")
                                eda_psd_symp_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                                # Assuming eda_psd_symp_filt_tonic_full_stats is your dictionary of statistics
                                eda_psd_symp_filt_tonic_stats_df = pd.DataFrame(eda_psd_symp_filt_tonic_full_stats, index=[0])

                                # Transpose the DataFrame to get labels as rows and values in the second column
                                eda_psd_symp_filt_tonic_stats_df = eda_psd_symp_filt_tonic_stats_df.transpose().reset_index()
                                eda_psd_symp_filt_tonic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the transposed DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}_summary_statistics.tsv")
                                eda_psd_symp_filt_tonic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")

                                # # Calculate amplitude_min from the phasic component for peak detection
                                # amplitude_min = 0.001 * decomposed["EDA_Phasic"].max()
                                # logging.info(f"Calculated amplitude_min: {amplitude_min} for {method} method.")

                                for peak_method in peak_methods:
                                    try:
                                       
                                        if not pd.to_numeric(decomposed["EDA_Phasic"], errors='coerce').notnull().all():
                                            logging.warning("Non-numeric values found in EDA_Phasic. Attempting to clean...")
                                            # Implement appropriate cleaning or transformation here

                                        # Check if the phasic component is not empty
                                        if decomposed["EDA_Phasic"].size == 0:
                                            logging.warning(f"The phasic component is empty for {method}. Skipping peak detection.")
                                            continue
                                        
                                        if len(decomposed["EDA_Phasic"]) > 0:
                                            logging.info(f"Array is not empty")
                                        else:
                                            logging.warning(f"Array is empty")

                                        # Additional check for non-empty and valid data
                                        if decomposed["EDA_Phasic"].isnull().all() or decomposed["EDA_Phasic"].empty:
                                            logging.warning(f"No valid data in phasic component for {method} using {peak_method}. Skipping peak detection.")
                                            continue
                                            
                                        # Check if there are NaN values
                                        if np.isnan(decomposed["EDA_Phasic"]).any():
                                            logging.warning(f"Array contains NaN values")

                                            # Fill NaN values with a specific value, for example, 0
                                            logging.info(f"Filling NaN values with 0")
                                            decomposed["EDA_Phasic"].fillna(0, inplace=True)
                                        
                                        if not pd.to_numeric(decomposed["EDA_Phasic"], errors='coerce').notnull().all():
                                            logging.warning(f"Array contains non-numeric values")

                                        if decomposed["EDA_Phasic"].nunique() == 1:
                                            logging.warning(f"Array contains constant values")

                                        # Detect peaks using the specified method
                                        logging.info(f"Detecting peaks using method: {peak_method}")    
                                        _, peaks = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=peak_method)
                                        
                                        # Log the detected SCR onsets and recoveries for inspection
                                        logging.info(f"SCR Onsets: {peaks['SCR_Onsets']}")
                                        logging.info(f"SCR Rise Time: {peaks['SCR_RiseTime']}")
                                        logging.info(f"SCR Peaks: {peaks['SCR_Peaks']}")
                                        logging.info(f"SCR Amplitude: {peaks['SCR_Amplitude']}")
                                        logging.info(f"SCR Height: {peaks['SCR_Height']}")
                                        logging.info(f"SCR Recovery: {peaks['SCR_Recovery']}")
                                        logging.info(f"SCR Recovery Time: {peaks['SCR_RecoveryTime']}")
                                        
                                        # Add SCR Amplitude, Onsets, and Peaks to the DataFrame
                                        if peaks['SCR_Peaks'].size > 0:

                                            # Add SCR onsets to the data frame if valid
                                            valid_onset_indices = peaks['SCR_Onsets'][~np.isnan(peaks['SCR_Onsets'])]
                                            if valid_onset_indices.size > 0:
                                                decomposed.loc[valid_onset_indices, f"SCR_Onsets_{peak_method}"] = 1

                                            valid_peaks_indices = peaks['SCR_Peaks'][~np.isnan(peaks['SCR_Peaks'])]
                                            if valid_peaks_indices.size > 0:
                                                decomposed.loc[valid_peaks_indices, f"SCR_Peaks_{peak_method}"] = 1
                                                decomposed.loc[valid_peaks_indices, f"SCR_Amplitude_{peak_method}"] = peaks['SCR_Amplitude'][~np.isnan(peaks['SCR_Peaks'])]
                                                
                                                # Add other SCR information to the DataFrame
                                                decomposed.loc[valid_peaks_indices, f"SCR_Height_{peak_method}"] = peaks['SCR_Height']
                                                decomposed.loc[valid_peaks_indices, f"SCR_RiseTime_{peak_method}"] = peaks['SCR_RiseTime']
                                                
                                            # Add SCR Recovery information if valid
                                            valid_recovery_indices = peaks['SCR_Recovery'][~np.isnan(peaks['SCR_Recovery'])]
                                            if valid_recovery_indices.size > 0:
                                                decomposed.loc[valid_recovery_indices, f"SCR_Recovery_{peak_method}"] = 1
                                                decomposed.loc[valid_recovery_indices, f"SCR_RecoveryTime_{peak_method}"] = peaks['SCR_RecoveryTime'][~np.isnan(peaks['SCR_Recovery'])]
                                            else:
                                                logging.warning(f"No valid SCR recovery found for {peak_method} method.")
                                            
                                            logging.info(f"SCR events detected and added to DataFrame for {method} using {peak_method}")
                                        else:
                                            logging.warning(f"No SCR events detected for method {method} using peak detection method {peak_method}.")
                                        
                                        logging.info(f"Plotting peaks for {method} using {peak_method} method.")
                                        
                                        # Create and configure plots
                                        fig, axes = plt.subplots(4, 1, figsize=(20, 10))

                                        # Plot 1: Overlay of Raw and Cleaned EDA
                                        axes[0].plot(eda_filtered, label='Raw Filtered EDA')
                                        axes[0].plot(eda_cleaned, label='Filtered Cleaned EDA', color='orange')
                                        axes[0].set_title('Filtered Raw and Cleaned EDA Signal')
                                        axes[0].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                        axes[0].set_ylabel('Amplitude (µS)')
                                        axes[0].legend()

                                        # Plot 2: Phasic Component with SCR Events
                                        axes[1].plot(decomposed["EDA_Phasic"], label='Phasic Component', color='green')
                                        axes[1].scatter(valid_onset_indices, decomposed.loc[valid_onset_indices, "EDA_Phasic"], color='blue', label='SCR Onsets')
                                        axes[1].scatter(valid_peaks_indices, decomposed.loc[valid_peaks_indices, "EDA_Phasic"], color='red', label='SCR Peaks')
                                        axes[1].scatter(valid_recovery_indices, decomposed.loc[valid_recovery_indices, "EDA_Phasic"], color='purple', label='SCR Recovery')

                                        # Mark invalid recoveries with gray x marks
                                        #axes[1].scatter(invalid_recovery_indices, decomposed.loc[invalid_recovery_indices, "EDA_Phasic"], color='black', marker='x', label='Invalid Recoveries')

                                        # Add legend and set titles
                                        axes[1].set_title(f'Phasic EDA ({method}) with {peak_method} Peaks')
                                        axes[1].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                        axes[1].set_ylabel('Amplitude (µS)')
                                        axes[1].legend()

                                        # Plot 3: Tonic Component
                                        axes[2].plot(time_vector, decomposed["EDA_Tonic"], label='Tonic Component', color='brown')
                                        axes[2].set_title(f'Tonic EDA ({method})')
                                        axes[2].set_xlabel('Time (minutes)')
                                        axes[2].set_ylabel('Amplitude (µS)')
                                        axes[2].legend()

                                        # Plot 4: Framewise Displacement
                                        voxel_threshold = 0.5 # mm
                                        axes[3].plot(fd_upsampled, label='Framewise Displacement', color='blue')
                                        axes[3].axhline(y=voxel_threshold, color='r', linestyle='--')
                                        axes[3].set_title('Framewise Displacement')
                                        axes[3].set_xlabel('Volume Number (2 sec TR)')
                                        axes[3].set_ylabel('FD (mm)')

                                        # Add shading where FD is above threshold across all subplots
                                        for ax in axes[:-1]: # Exclude the last axis which is for FD plot
                                            ax.fill_between(time_vector, 0, voxel_threshold, where=fd_upsampled > voxel_threshold, color='red', alpha=0.3)

                                        # Save the combined plot
                                        plt.tight_layout()
                                        combo_plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_{method}_{peak_method}_subplots.png")
                                        plt.savefig(combo_plot_filename, dpi=dpi_value)
                                        plt.close()
                                        logging.info(f"Saved EDA subplots for {method} with {peak_method} to {combo_plot_filename}")

                                        # Assuming decomposed is a DataFrame containing the decomposed EDA components
                                        phasic_component = decomposed['EDA_Phasic']
                                        tonic_component = decomposed['EDA_Tonic']

                                        # Basic statistics for Phasic Component
                                        phasic_stats = {
                                            'Phasic Mean (µS)': phasic_component.mean(),
                                            'Phasic Median (µS)': phasic_component.median(),
                                            'Phasic Std Deviation (µS)': phasic_component.std(),
                                            'Phasic Variance (µS)': phasic_component.var(),
                                            'Phasic Skewness': phasic_component.skew(),
                                            'Phasic Kurtosis': phasic_component.kurtosis(),
                                            'Phasic Range (µS)': phasic_component.max() - phasic_component.min(),
                                            'Phasic Total Absolute Sum (µS)': phasic_component.abs().sum(),
                                            'Phasic 25th Percentile (µS)': phasic_component.quantile(0.25),
                                            'Phasic 75th Percentile (µS)': phasic_component.quantile(0.75),
                                            'Phasic IQR': phasic_component.quantile(0.75) - phasic_component.quantile(0.25),
                                            'Phasic 10th Percentile (µS)': phasic_component.quantile(0.10),
                                            'Phasic 90th Percentile (µS)': phasic_component.quantile(0.90)
                                        }

                                        # Basic statistics for Tonic Component
                                        tonic_stats = {
                                            'Tonic Mean (µS)': tonic_component.mean(),
                                            'Tonic Median (µS)': tonic_component.median(),
                                            'Tonic Std Deviation (µS)': tonic_component.std(),
                                            'Tonic Variance (µS)': tonic_component.var(),
                                            'Tonic Skewness': tonic_component.skew(),
                                            'Tonic Kurtosis': tonic_component.kurtosis(),
                                            'Tonic Range (µS)': tonic_component.max() - tonic_component.min(),
                                            'Tonic Total Absolute Sum (µS)': tonic_component.abs().sum(),
                                            'Tonic 25th Percentile (µS)': tonic_component.quantile(0.25),
                                            'Tonic 75th Percentile (µS)': tonic_component.quantile(0.75),
                                            'Tonic IQR': tonic_component.quantile(0.75) - tonic_component.quantile(0.25),
                                            'Tonic 10th Percentile (µS)': tonic_component.quantile(0.10),
                                            'Tonic 90th Percentile (µS)': tonic_component.quantile(0.90)
                                        }

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
                        traceback_info = traceback.format_exc()
                        logging.error(f"Traceback Information: \n{traceback_info}")
                        return 
                        #continue # Continue to the next file
                
                # Remove the working directory
                try:
                    shutil.rmtree(working_dir)
                    logging.info("Working directory removed successfully.")
                    
                except Exception as e:
                    logging.error(f"Error removing working directory: {e}")
                    raise e
                
                # Record the end time for this run and calculate runtime
                run_end_time = datetime.now()
                run_runtime = (run_end_time - run_start_time).total_seconds() / 60
                logging.info(f"Run {run_number} completed. Runtime: {run_runtime} minutes.")
                log_resource_usage()  # Log resource usage at the end of each run
            
            except Exception as e:
            # Log the error
                logging.error(f"Error processing run {run_number} for participant {participant_id}: {e}")
                traceback.print_exc()  # Print the traceback for debugging purposes
                traceback_info = traceback.format_exc()
                logging.error(f"Traceback Information: \n{traceback_info}")
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
    
    # # Call the main function.
    # cProfile.run('main()')

    main()