"""
Script Name: clean_physio_task-rest_ppg_rev1.py

Description:
This script is designed for processing and analyzing pulse wave (PPG) signals within a BIDS (Brain Imaging Data Structure) dataset. 

Usage:
The script is intended to be run from the command line or an IDE. It requires the specification of dataset directories and participant/session details within the script. The script can be customized to process multiple participants and sessions by modifying the relevant sections.

Requirements:
- Python 3.x
- Libraries: neurokit2, pandas, matplotlib, scipy, numpy
- A BIDS-compliant dataset with PPG signal data.

Note:
- Ensure that the dataset paths and participant/session details are correctly set in the script.
- The script contains several hardcoded paths and parameters which may need to be adjusted based on the dataset structure and analysis needs.

Author: PAMcConnell
Date: 20231215
Version: 1.0

"""
#%% Import libraries
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
from scipy.stats import linregress
from scipy.stats import t
import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
import pywt
from statsmodels.tsa.stattools import adfuller
from scipy.interpolate import CubicSpline

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
#matplotlib.use('Agg')  # Use 'Agg' backend for file-based plotting
import matplotlib.pyplot as plt
import plotly # For interactive plotting
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# Neurophysiological data analysis
import neurokit2 as nk

# conda activate nipype (Python 3.9)

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

def comb_band_stop_filter(data, fs, order=4, visualize=False):
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
    #nyquist = 11.5 * fs  # 11.5 Hz -> nSlices/(MBF*TR) = 69/(2*3) = 11.5 Hz ? 
    #normal_stop_freq = stop_freq / nyquist
    
    # Calculate the stop band frequencies
    stop_band = [0.45, 0.55]

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

# Function to calculate the correlation between FD and PPG
def calculate_fd_ppg_correlation(fd, ppg):
    """
    Calculate the Pearson correlation coefficient and p-value between framewise displacement (FD) and PPG.

    Parameters:
    fd (array-like): The framewise displacement timeseries.
    ppg (array-like): The ppg activity timeseries.

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

    # Ensure that FD and PPG are of the same length
    if len(fd) != len(ppg):
        logging.error("FD and PPG timeseries must be of the same length.")
        return None, None

    try:
        # Calculate Pearson correlation
        r_value, p_value = pearsonr(fd, ppg)
        logging.info(f"Calculated Pearson correlation: {r_value}, p-value: {p_value}")
        
        # Round p-value to 3 decimal places
        p_value_rounded = round(p_value, 3)
        p_value = p_value_rounded

        # Check for very large sample sizes which might render p-value as 0
        if p_value == 0:
            logging.info("P-value returned as 0, possibly due to large sample size and high precision of correlation.")

        return r_value, p_value
    except Exception as e:
        logging.error(f"Error in calculating correlation: {e}")
        return None, None

# Function to plot the correlation between FD and PPG
def plot_fd_ppg_correlation(fd, ppg, file_name):
    
    if len(fd) != len(ppg):
        logging.warning("Error: FD and PPG timeseries must be of the same length.")
        return
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(fd, ppg)
        fit_line = slope * fd + intercept
        r_squared = r_value**2
        
        # Calculate the confidence interval of the fit line
        t_val = t.ppf(1-0.05/2, len(fd)-2)  # t-score for 95% confidence interval & degrees of freedom
        conf_interval = t_val * std_err * np.sqrt(1/len(fd) + (fd - np.mean(fd))**2 / np.sum((fd - np.mean(fd))**2))

        # Upper and lower bounds of the confidence interval
        lower_bound = fit_line - conf_interval
        upper_bound = fit_line + conf_interval

        plt.figure(figsize=(10, 6))
        plt.scatter(fd, ppg, alpha=0.5, label='Data Points')
        plt.plot(fd, fit_line, color='red', label=f'Fit Line (R = {r_value:.3f}, p = {p_value:.3f})')
        plt.fill_between(fd, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.ylabel('PPG (Volts)')
        plt.xlabel('Framewise Displacement (mm)')
        plt.title('Correlation between FD and PPG with Linear Fit and Confidence Interval')
        plt.legend(loc='upper left')
        plt.savefig(file_name, dpi=300)  # Save the figure with increased DPI for higher resolution
        plt.close()

    except Exception as e:
        logging.warning(f"An error occurred: {e}")
   
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

# Function to estimate the noise level (using Median Absolute Deviation)
def estimate_noise(signal):
    return np.median(np.abs(signal - np.median(signal))) / 0.6745

def plot_wavelet_coeffs(coeffs, title, sampling_rate):
    num_levels = len(coeffs)
    nyquist_freq = sampling_rate / 2
    fig_height_per_level = 150  # Adjust for spacing
    total_fig_height = num_levels * fig_height_per_level

    fig = make_subplots(rows=num_levels, cols=1)

    for i, coeff in enumerate(coeffs):
        # Calculate frequency range for this level
        high_freq = nyquist_freq / (2 ** i)
        low_freq = high_freq / 2
        # Convert frequency range to BPM
        low_bpm, high_bpm = low_freq * 60, high_freq * 60

        # Create subplot title with frequency and BPM range
        subplot_title = f"{title} - Level {i+1} (Freq: {low_freq:.2f}-{high_freq:.2f} Hz, BPM: {low_bpm:.0f}-{high_bpm:.0f})"
        fig.add_trace(go.Scatter(y=coeff, mode='lines', name=f'Level {i+1}'), row=i+1, col=1)

        # Check if annotations exist for this subplot index
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = subplot_title
        else:
            # If not, add a new annotation
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.5, y=1 - (i+1) / num_levels,  # Position annotation
                xanchor='center', yanchor='bottom',
                text=subplot_title,
                showarrow=False
            )

    fig.update_layout(height=total_fig_height, title=title)
    return fig

def check_stationarity(hrv_data):
    # Assuming `hrv_data` is your HRV time series data as a pandas Series or numpy array
    result = adfuller(hrv_data)

    logging.info(f'ADF Statistic: %f' % result[0])
    logging.info(f'p-value: %f' % result[1])
    logging.info(f'Critical Values:')
    for key, value in result[4].items():
        logging.info(f'\t%s: %.3f' % (key, value))

    # Interpretation
    if result[0] < result[4]["5%"]:
        logging.info(f"The series is stationary at 5% level")
    else:
        logging.info(f"The series is not stationary at 5% level")
    
#%% Main script logic 
def main():
    """
    Main function to clean PPG data from a BIDS dataset.
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
    
    #%% Start looping through participants by run
    for i, participant_id in enumerate(participants_df['participant_id']):
#    for i, participant_id in enumerate([participants_df['participant_id'].iloc[9]]):  # For testing  LRN010 first with ppg data

        # Record the start time for this participant
        participant_start_time = datetime.now()
        for run_number in range(1, 5):  # Assuming 4 runs
#        for run_number in range(1, 2):  # Testing with 1 run
            try:
        
                # Set a higher DPI for better resolution
                dpi_value = 300 

                task_name = 'rest'

                # Process the first run for the selected participant
                run_id = f"run-0{run_number}"

                # Construct the base path
                base_path = os.path.join(derivatives_dir, 'ppg', participant_id, run_id)

                # Make sure the directories exist
                os.makedirs(base_path, exist_ok=True)
                
                # Define the processed signals filename for checking
                processed_signals_filename = f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio_filtered_cleaned_ppg_processed.tsv.gz"
                processed_signals_path = os.path.join(base_path, processed_signals_filename)

                # Check if processed file exists
                if os.path.exists(processed_signals_path):
                    print(f"Processed PPG files found for {participant_id} for run {run_number}, skipping...")
                    continue  # Skip to the next run

                # Setup logging
                session_id = 'ses-1'  # Assuming session ID is known
                
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                log_file_path = setup_logging(participant_id, session_id, run_id, dataset_root_dir)
                logging.info(f"Testing PPG processing for task {task_name} run-0{run_number} for participant {participant_id}")

                sampling_rate = 5000    # acquisition sampling rate
            
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
                        base_filename = os.path.basename(file).replace('.tsv.gz', '')

                        # Open the compressed PPG file and load it into a DataFrame
                        with gzip.open(file, 'rt') as f:
                            physio_data = pd.read_csv(f, delimiter='\t')
                            # Check for the presence of the 'ppg' column
                            if 'ppg' not in physio_data.columns:
                                logging.error(f"'ppg' column not found in {file}. Skipping this file.")
                                
                                # Check if the directory is empty before removing it
                                if not os.listdir(base_path):
                                    os.rmdir(base_path)
                                    logging.info(f"Removed empty directory: {base_path}")

                                continue  # Skip to the next file
                            
                            ppg = physio_data['ppg']

                            # Calculate the time vector in minutes - Length of PPG data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(len(ppg)) / sampling_rate / 60
                            
                            ### Step 1: Prefilter gradient artifact from PPG using wavelets and downsample ###
                            logging.info(f"Step 1: Prefilter gradient artifact from PPG using wavelets and downsample.")

                            # Perform wavelet decomposition
                            coeffs = pywt.wavedec(ppg.values, 'db4', mode='per')
                            sigma = estimate_noise(coeffs[-2])

                            threshold_multiplier = 1.5  # Adjust this multiplier to increase the threshold
                            threshold = sigma * np.sqrt(2 * np.log(len(ppg))) * threshold_multiplier
                            logging.info(f"Estimated noise: {sigma}, threshold: {threshold}")

                            # Apply selective thresholding to the detail coefficients
                            thresholded_coeffs = [coeffs[0]]
                            for i, coeff in enumerate(coeffs[1:]):
                                if 9 <= i <= 11:
                                    # For levels 10-12, apply minimal or no thresholding
                                    adjusted_threshold = 0  # No thresholding
                                    # Alternatively, use a very minimal threshold, e.g., threshold * 0.1
                                else:
                                    # For other levels, apply the standard or a more aggressive threshold
                                    adjusted_threshold = threshold  # Standard threshold
                                
                                thresholded_coeff = pywt.threshold(coeff, value=adjusted_threshold, mode='soft')
                                thresholded_coeffs.append(thresholded_coeff)
                                
                            # Plot original wavelet coefficients with frequency and BPM ranges
                            fig_coeffs_original = plot_wavelet_coeffs(coeffs, 'Original Wavelet Coefficients', sampling_rate=5000)

                            # Save the original coefficients plot
                            original_plot_filename = os.path.join(base_path, f"{base_filename}_original_wavelet_ppg_subplots.html")
                            plotly.offline.plot(fig_coeffs_original, filename=original_plot_filename, auto_open=False)

                            # Plot thresholded wavelet coefficients with frequency and BPM ranges
                            fig_coeffs_thresholded = plot_wavelet_coeffs(thresholded_coeffs, 'Thresholded Wavelet Coefficients', sampling_rate=5000)

                            # Save the thresholded coefficients plot
                            thresholded_plot_filename = os.path.join(base_path, f"{base_filename}_thresholded_wavelet_ppg_subplots.html")
                            plotly.offline.plot(fig_coeffs_thresholded, filename=thresholded_plot_filename, auto_open=False)

                            # Reconstruct the denoised signal
                            ppg_filtered_array = pywt.waverec(thresholded_coeffs, 'db4', mode='per')
                            ppg_unfiltered_array = ppg.values
                            
                            # Plot original and denoised signal
                            fig = make_subplots(rows=2, cols=1, subplot_titles=('Original PPG Signal', 'Denoised PPG Signal'), vertical_spacing=0.3)
                            fig.add_trace(go.Scatter(y=ppg_unfiltered_array, mode='lines', name='Original PPG'), row=1, col=1)
                            fig.add_trace(go.Scatter(y=ppg_filtered_array, mode='lines', name='Denoised PPG'), row=2, col=1)
                            fig.update_layout(height=800, title_text='PPG Signal - Original vs Denoised')
                            fig.update_xaxes(row=1, col=1, matches='x')
                            fig.update_xaxes(row=2, col=1, matches='x')
                            
                            # Disable y-axis zooming for all subplots
                            fig.update_yaxes(fixedrange=True)
                            
                            # Save the plot
                            combo_plot_filename = os.path.join(base_path, f"{base_filename}_wavelet_filtered_ppg_subplots.html")
                            plotly.offline.plot(fig, filename=combo_plot_filename, auto_open=False)
                            logging.info("Plot generated and saved.")

                            # Pre-filter gradient artifact.
                            #ppg_filtered_array = comb_band_stop_filter(ppg, sampling_rate, visualize=False)

                            # Downsample the filtered data.
                            ppg_filtered_ds = nk.signal_resample(ppg_filtered_array, desired_length=None, sampling_rate=sampling_rate, desired_sampling_rate=100, method='pandas')
                            ppg_unfiltered_ds = nk.signal_resample(ppg_unfiltered_array, desired_length=None, sampling_rate=sampling_rate, desired_sampling_rate=100, method='pandas')

                            # Hanlde the index for the downsampled data
                            new_length = len(ppg_filtered_ds)
                            new_index = pd.RangeIndex(start=0, stop=new_length, step=1)  # or np.arange(new_length)

                            # Convert the filtered data back into a Pandas Series
                            ppg_filtered = pd.Series(ppg_filtered_ds, index=new_index)
                            ppg_unfiltered = pd.Series(ppg_unfiltered_ds, index=new_index)

                            if ppg_filtered.empty:
                                logging.error(f"Error: 'ppg_filtered' is empty.")
                                # Log stack trace for debugging purposes
                                logging.error(traceback.format_exc())
                            else:
                                logging.info(f"'ppg_filtered' is not empty, length: {len(ppg_filtered)}")

                            sampling_rate = 100   # downsampled sampling rate
                            logging.info(f"Downsampled PPG sampling rate: {sampling_rate} Hz")
                            
                            # Calculate the time vector in minutes - Length of PPG data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(new_length) / sampling_rate / 60

                            ### Step 2: Clean prefiltered PPG for non-default phasic decomposition and peak detection methods. ###
                            logging.info(f"Step 2: Clean prefiltered PPG for peak detection.")

                            # First, clean the PG signal
                            ppg_cleaned = nk.ppg_clean(ppg_filtered, sampling_rate=sampling_rate, heart_rate=None, method="elgendi") #"nabian2018"
                            logging.info(f"Prefiltered PPG signal cleaned using NeuroKit's ppg_clean.")
                            
                            logging.info(f"Starting peak detection for prefiltered cleaned PPG signal.")
                            logging.info(f"Sampling rate: {sampling_rate} Hz")
                            logging.info(f"Size of prefiltered cleaned PPG signal: {ppg_cleaned.size}")
                            
                            peak_methods = ["elgendi"] #"bishop"
                            logging.info(f"Using the following methods for peak detection: {peak_methods}")
                            
                            # Process FD
                            fd_timeseries = fd  # Assuming 'fd' is your framewise displacement timeseries
                            
                             # Create a time array for the FD timeseries
                            ppg_duration= len(ppg_cleaned) / sampling_rate  # Total duration in seconds
                            fd_time_ppg = np.linspace(0, ppg_duration, len(fd_timeseries))
                            
                            # The new time array for the upsampled FD timeseries should match the PPG timeseries length
                            # Make sure the last time point in upsampled_time does not exceed the last point in fd_time
                            upsampled_time_ppg = np.linspace(0, ppg_duration, len(ppg_cleaned))
                            
                            # Use linear interpolation with bounds_error set to False to prevent extrapolation
                            fd_interpolator_ppg = interp1d(fd_time_ppg, fd_timeseries, kind='linear', bounds_error=False, fill_value='extrapolate')
                            fd_upsampled_ppg = fd_interpolator_ppg(upsampled_time_ppg)

                            # Handle any NaN values that might have been introduced due to the bounds_error setting
                            fd_upsampled_ppg[np.isnan(fd_upsampled_ppg)] = fd_timeseries[-1]  # Replace NaNs with the last valid FD value
                            
                            # Calculate FD-PPG correlation
                            r_value_ppg, p_value_ppg = calculate_fd_ppg_correlation(fd_upsampled_ppg, ppg_cleaned)
                            logging.info(f"Correlation between FD and filtered cleaned PPG timeseries: {r_value_ppg}, p-value: {p_value_ppg}")
                            
                            plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_ppg_correlation.png"
                            plot_filepath = os.path.join(base_path, plot_filename)
                            plot_fd_ppg_correlation(fd_upsampled_ppg, ppg_cleaned, plot_filepath)
                            logging.info(f"FD-PPG correlation plot saved to {plot_filepath}")

                            # Assuming ppg_cleaned is a Pandas Series or a NumPy array
                            ppg_cleaned_df = pd.DataFrame()
                            ppg_cleaned_df['PPG_Unfiltered'] = ppg_unfiltered
                            ppg_cleaned_df['PPG_Filtered'] = ppg_filtered
                            ppg_cleaned_df['PPG_Clean'] = ppg_cleaned
                            
                            for peak_method in peak_methods:
                                try:
                                    # Detect peaks using the specified method
                                    logging.info(f"Detecting peaks using method: {peak_method}")    
                                    #*_, peaks = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate, correct_artifacts=True, method=peak_method)
                                    #*logging.info(f"Note that nk.ppg_peaks artifact correction method is active.")
                                    
                                    _, peaks = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate, correct_artifacts=False, method=peak_method)
                                    logging.info(f"Note that nk.ppg_peaks artifact correction method is NOT active.")
                                    
                                    # Log the detected R-Peaks for inspection
                                    logging.info(f"R Peaks via {peak_method}: {peaks['PPG_Peaks']}")
                                    
                                    # Initialize the columns for PPG in the DataFrame
                                    ppg_cleaned_df[f'PPG_Peaks_{peak_method}'] = 0

                                    #! This is likely an error and oversight that needs to be corrected...
                                    """
                                    # Ensure valid peaks are within the range of the DataFrame
                                    valid_peaks = [p-1 for p in peaks['PPG_Peaks'] if 0 < p <= len(ppg_cleaned_df)]
                                    logging.info(f"Checking indexing and correcting to 0-based indexing if necessary.")
                                    
                                    logging.info(f"Original peak indices: {peaks['PPG_Peaks']}")
                                    logging.info(f"Adjusted peak indices: {valid_peaks}")
                                    #// Convert to 0-based indexing if your data is 1-based indexed
                                    #//valid_peaks = peaks[f'PPG_Peaks'] - 1
                                    """
                                    # Set the valid peaks to the detected peaks
                                    valid_peaks = peaks[f'PPG_Peaks']

                                    # Update R Peaks, in the DataFrame
                                    ppg_cleaned_df.loc[valid_peaks, f'PPG_Peaks_{peak_method}'] = 1
                                    
                                    num_samples = len(ppg_cleaned_df)

                                    # Create a time axis based on the sampling rate
                                    time_axis = np.arange(num_samples) / sampling_rate

                                    # Add the time axis as a column to your DataFrame
                                    ppg_cleaned_df['Time'] = time_axis
                                    
                                    voxel_threshold = 0.5 # mm
                                    
                                    # Calculate the number of volumes (assuming 2 sec TR and given sampling rate)
                                    num_volumes = len(fd_upsampled_ppg) / (sampling_rate * 2)

                                    # Generate the volume numbers for the x-axis
                                    volume_numbers = np.arange(0, num_volumes)
                                    
                                    # Assuming valid_peaks are the indices where the R peaks occur in the downsampled data
                                    # Calculate R-R intervals in milliseconds
                                    rr_intervals = np.diff(valid_peaks) / sampling_rate * 1000  # in ms

                                    # Calculate midpoints between R peaks in terms of the sample indices
                                    midpoint_samples = (valid_peaks[:-1] + valid_peaks[1:]) // 2

                                    # Generate a regular time axis for interpolation
                                    regular_time_axis = np.linspace(midpoint_samples.min(), midpoint_samples.max(), num=len(ppg_cleaned_df))

                                    # Create a cubic spline interpolator
                                    cs = CubicSpline(midpoint_samples, rr_intervals)

                                    # Interpolate over the regular time axis
                                    interpolated_rr = cs(regular_time_axis)

                                    # Add R-R interval data to the PPG dataframe
                                    #ppg_cleaned_df['RR_interval'] = rr_intervals
                                    logging.info(f"Interpolated R-R intervals: {rr_intervals} in ms")

                                    ppg_cleaned_df['RR_interval_interpolated'] = interpolated_rr
                                    logging.info(f"Interpolated R-R intervals: {interpolated_rr} in ms")
                                    
                                    #ppg_cleaned_df['RR_midpoints'] = midpoint_samples
                                    logging.info(f"R-R midpoints: {midpoint_samples} in sample indices")


                                    #%% Plotly subplots
                                    # Create a plotly figure with independent x-axes for each subplot
                                    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, shared_yaxes=False, subplot_titles=('Filtered Raw and Cleaned PPG Signal', f'PPG with {peak_method} R Peaks', 'R-R Intervals Tachogram', 'Framewise Displacement'), vertical_spacing=0.065)

                                    # Add traces to the first subplot (Filtered Raw and Cleaned PPG Signal)
                                    fig.add_trace(go.Scatter(y=ppg_unfiltered, mode='lines', name='Raw Unfiltered PPG', line=dict(color='red', width=3)), row=1, col=1)
                                    fig.add_trace(go.Scatter(y=ppg_filtered, mode='lines', name='Raw Filtered PPG', line=dict(color='blue', width=3, dash='dot')), row=1, col=1)
                                    fig.add_trace(go.Scatter(y=ppg_cleaned, mode='lines', name='Filtered Cleaned PPG', line=dict(color='green', width=3)), row=1, col=1)

                                    # Add traces to the second subplot (PPG with R Peaks)
                                    fig.add_trace(go.Scatter(y=ppg_cleaned, mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')), row=2, col=1)
                                    
                                    # Ensure valid_peaks are within the range of the DataFrame
                                    valid_peaks = valid_peaks[valid_peaks < len(ppg_cleaned_df)]
                                    y_values = ppg_cleaned_df.loc[valid_peaks, 'PPG_Clean'].tolist()  # Make sure this line executes before the next one

                                    # Add Scatter Plot for R Peaks
                                    fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks', marker=dict(color='red')), row=2, col=1)
                                    
                                    # Third Subplot: R-R Intervals Midpoints
                                    fig.add_trace(go.Scatter(x=midpoint_samples, y=rr_intervals, mode='markers', name='R-R Midpoints', marker=dict(color='red')), row=3, col=1)
                                    fig.add_trace(go.Scatter(x=regular_time_axis, y=interpolated_rr, mode='lines', name='Interpolated R-R Intervals', line=dict(color='blue')), row=3, col=1)
                                    
                                    # Add traces to the fourth subplot (Framewise Displacement)
                                    fig.add_trace(go.Scatter(y=fd_upsampled_ppg, mode='lines', name='Framewise Displacement', line=dict(color='blue')), row=4, col=1)
                                    fig.add_hline(y=voxel_threshold, line=dict(color='red', dash='dash'), row=4, col=1)

                                    # Update layout and size
                                    fig.update_layout(height=1200, width=1800, title_text=f'PPG Analysis - {peak_method}')

                                    # Update y-axis labels for each subplot
                                    fig.update_yaxes(title_text='Amplitude (Volts)', row=1, col=1)
                                    fig.update_yaxes(title_text='Amplitude (Volts)', row=2, col=1)
                                    fig.update_yaxes(title_text='R-R Interval (ms)', row=3, col=1)
                                    fig.update_yaxes(title_text='FD (mm)', row=4, col=1)

                                    # Calculate the tick positions for the fourth subplot
                                    tick_interval_fd = 5  # Adjust this value as needed
                                    tick_positions_fd = np.arange(0, len(fd_upsampled_ppg), tick_interval_fd * sampling_rate * 2)
                                    tick_labels_fd = [f"{int(vol)}" for vol in volume_numbers[::tick_interval_fd]]
                                    
                                    # Update x-axis labels for each subplot
                                    fig.update_xaxes(title_text='Samples', row=1, col=1, matches='x')
                                    fig.update_xaxes(title_text='Samples', row=2, col=1, matches='x')
                                    fig.update_xaxes(title_text='Samples', row=3, col=1, matches='x')
                                    fig.update_xaxes(title_text='Volume Number (2 sec TR)', tickvals=tick_positions_fd, ticktext=tick_labels_fd, row=4, col=1, matches='x')

                                    # Disable y-axis zooming for all subplots
                                    fig.update_yaxes(fixedrange=True)

                                    # Save the plot as an HTML file
                                    combo_plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_{peak_method}_subplots.html")
                                    plotly.offline.plot(fig, filename=combo_plot_filename, auto_open=False)

                                    #%% Matplotlib subplots
                                    # Create and configure matplotlib png plots
                                    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

                                    logging.info(f"Unfiltered PPG range: {ppg_unfiltered.min()}, {ppg_unfiltered.max()}")
                                    logging.info(f"Filtered PPG range: {ppg_filtered.min()}, {ppg_filtered.max()}")
                                    logging.info(f"Cleaned PPG range: {ppg_cleaned.min()}, {ppg_cleaned.max()}")

                                    # Plot 1: Overlay of Raw and Cleaned PPG
                                    axes[0].plot(ppg_unfiltered, label='Raw Unfiltered PPG', color='blue', linewidth=1)
                                    axes[0].plot(ppg_filtered, label='Raw Filtered PPG', color='green', linewidth=1)
                                    axes[0].plot(ppg_cleaned, label='Filtered Cleaned PPG', color='orange', linewidth=1)
                                    #axes[0].plot(ppg_unfiltered, label='Raw Unfiltered PPG', linestyle='--')
                                    #axes[0].plot(ppg_filtered, label='Raw Filtered PPG', linestyle='-.')

                                    axes[0].set_title('Filtered Raw and Cleaned PPG Signal')
                                    axes[0].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                    axes[0].set_ylabel('Amplitude (Volts)')
                                    axes[0].legend()

                                    # Plot 2: PPG timeseries with R-Peak Events
                                    axes[1].plot(ppg_cleaned, label='PPG', color='green')
                                    
                                    # Ensure valid_peaks are within the range of the DataFrame
                                    valid_peaks = valid_peaks[valid_peaks < len(ppg_cleaned_df)]

                                    # Scatter Plot for R Peaks
                                    y_values = ppg_cleaned_df.loc[valid_peaks, 'PPG_Clean']
                                    axes[1].scatter(valid_peaks, y_values, color='red', label='R Peaks')

                                    # Add legend and set titles
                                    axes[1].set_title(f'PPG with {peak_method} R Peaks')
                                    axes[1].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                    axes[1].set_ylabel('Amplitude (Volts)')
                                    axes[1].legend()

                                    # Plot 3: Framewise Displacement
                                    axes[2].plot(fd_upsampled_ppg, label='Framewise Displacement', color='blue')
                                    axes[2].axhline(y=voxel_threshold, color='r', linestyle='--')

                                    # Set x-axis ticks to display volume numbers at regular intervals
                                    # The interval for ticks can be adjusted (e.g., every 10 volumes)
                                    tick_interval = 10  # Adjust this value as needed
                                    axes[2].set_xticks(np.arange(0, len(fd_upsampled_ppg), tick_interval * sampling_rate * 2))
                                    axes[2].set_xticklabels([f"{int(vol)}" for vol in volume_numbers[::tick_interval]])

                                    axes[2].set_title('Framewise Displacement')
                                    axes[2].set_xlabel('Volume Number (2 sec TR)')
                                    axes[2].set_ylabel('FD (mm)')

                                    # # Add shading where FD is above threshold across all subplots
                                    # for ax in axes[:-1]: # Exclude the last axis which is for FD plot
                                    #     ax.fill_between(ppg_cleaned.index / sampling_rate / 60, 0, voxel_threshold, where=fd_upsampled_phasic > voxel_threshold, color='red', alpha=0.3)

                                    # Save the combined plot
                                    plt.tight_layout()
                                    combo_plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_{peak_method}_subplots.png")
                                    plt.savefig(combo_plot_filename, dpi=dpi_value)
                                    #plt.show()
                                    plt.close()
                                    logging.info(f"Saved PPG subplots for with {peak_method} to {combo_plot_filename}")

                                    # Check stationarity of signal before PSD
                                    logging.info(f"Checking stationarity of filtered cleaned PPG signal before PSD")
                                    check_stationarity(ppg_cleaned_df['PPG_Clean'])
                                    
                                    logging.info(f"Checking stationarity of RR interval signal before PSD")
                                    check_stationarity(ppg_cleaned_df['RR_interval_interpolated'])
                                    
                                    #%% PSD Plotly Plots
                                    # Compute Power Spectral Density 0 - 8 Hz for PPG
                                    #! Note that we are here using the cleaned PPG signal for PSD QA not the R-R interval timeseries
                                    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG using multitapers hann windowing.")
                                    ppg_filtered_psd = nk.signal_psd(ppg_cleaned_df['PPG_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=8.0, window=None, window_type='hann',
                                                        silent=False, t=None)

                                    # Plotting Power Spectral Density
                                    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 8 Hz for filtered cleaned PPG using multitapers hann windowing.")

                                    # Create a Plotly figure
                                    fig = go.Figure()

                                    # Create a figure with a secondary x-axis using make_subplots
                                    fig = make_subplots(specs=[[{"secondary_y": False}]])

                                    # Add the Power Spectral Density trace to the figure
                                    fig.add_trace(go.Scatter(x=ppg_filtered_psd['Frequency'], y=ppg_filtered_psd['Power'],
                                                            mode='lines', name='Normalized PPG PSD',
                                                            line=dict(color='blue'), fill='tozeroy'))

                                    # Update layout for the primary x-axis (Frequency in Hz)
                                    fig.update_layout(
                                        title='Power Spectral Density (PSD) (Multitapers with Hanning Window) for Filtered Cleaned PPG',
                                        xaxis_title='Frequency (Hz)',
                                        yaxis_title='Normalized Power',
                                        template='plotly_white',
                                        width=1200, height=800
                                    )

                                    # Create a secondary x-axis for Heart Rate in BPM
                                    fig.update_layout(
                                        xaxis=dict(
                                            title='Frequency (Hz)',
                                            range=[0, 8]  # Set the x-axis range from 0 to 8 Hz for frequency
                                        ),
                                        xaxis2=dict(
                                            title='Heart Rate (BPM)',
                                            overlaying='x',
                                            side='top',
                                            range=[0, 8*60],  # Convert frequency to BPM by multiplying by 60 (since 1 Hz = 60 BPM)
                                            scaleanchor='x',
                                            scaleratio=60
                                        )
                                    )

                                    # Save the plot as an HTML file
                                    plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_psd.html")
                                    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)

                                    # Define the frequency bands
                                    frequency_bands = {
                                    'ULF': (0, 0.003), # Associated with: Very slow breathing, vasomotion, myogenic activity, thermoregulation, metabolism, and the renin-angiotensin system; Dominated by: Peak around 0.003 Hz
                                    'VLF': (0.003, 0.04), # Associated with: Thermoregulation, hormonal fluctuations, slow changes in blood pressure, peripheral sympathetic; Motion Artifacts
                                    'LF': (0.04, 0.15), # Associated with: Sympathetic and Parasympathetic nervous system activity, baroreflex function, stress; Dominated by: Peak around 0.1 Hz
                                    'HF': (0.15, 0.4), # Associated with: Parasympathetic nervous system activity, respiratory sinus arrhythmia; Dominated by: Peak around 0.25-0.3 Hz
                                    'VHF': (0.4, 1.0) # Associated with: Very high frequency oscillations; Dominated by: Peak around 0.75 Hz
                                    }
                                    
                                    # Define the frequency bands and their corresponding colors
                                    frequency_bands_plot = {
                                        'ULF': (0, 0.003, 'grey'),  # Ultra Low Frequency
                                        'VLF': (0.003, 0.04, 'red'),  # Very Low Frequency
                                        'LF': (0.04, 0.15, 'green'),  # Low Frequency
                                        'HF': (0.15, 0.4, 'blue'),  # High Frequency
                                        'VHF': (0.4, 1.0, 'purple')  # Very High Frequency
}
                                    #%% PSD Plotly Plots
                                    # Compute Power Spectral Density 0 - 1 Hz for PPG
                                    #! Note that here we are using the corrected R-R interval timeseries for HRV PSD
                                    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG HRV using multitapers hann windowing.")
                                    ppg_filtered_psd_hrv = nk.signal_psd(ppg_cleaned_df['RR_interval_interpolated'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)

                                    # Plotting Power Spectral Density
                                    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned PPG HRV using multitapers hann windowing.")

                                    # Create a Plotly figure
                                    fig = go.Figure()

                                    # Plot the Power Spectral Density for each band with its own color
                                    for band, (low_freq, high_freq, color) in frequency_bands_plot.items():
                                        # Filter the PSD data for the current band
                                        band_mask = (ppg_filtered_psd_hrv['Frequency'] >= low_freq) & (ppg_filtered_psd_hrv['Frequency'] < high_freq)
                                        band_psd = ppg_filtered_psd_hrv[band_mask]
                                        
                                        # Add the PSD trace for the current band
                                        fig.add_trace(go.Scatter(
                                            x=band_psd['Frequency'],
                                            y=band_psd['Power'],
                                            mode='lines',
                                            name=f'{band} Band',
                                            line=dict(color=color, width=2),
                                            fill='tozeroy'
                                        ))

                                    # Update layout for the primary x-axis (Frequency in Hz)
                                    fig.update_layout(
                                        title='Power Spectral Density (PSD) (Multitapers with Hanning Window) for R-R Interval Tachogram',
                                        xaxis_title='Frequency (Hz)',
                                        yaxis_title='Normalized Power',
                                        template='plotly_white',
                                        width=1200, height=800
                                    )

                                    # Create a secondary x-axis for Heart Rate in BPM
                                    fig.update_layout(
                                        xaxis=dict(
                                            title='Frequency (Hz)',
                                            range=[0, 1]  # Set the x-axis range from 0 to 8 Hz for frequency
                                        ),
                                        xaxis2=dict(
                                            title='Heart Rate (BPM)',
                                            overlaying='x',
                                            side='top',
                                            range=[0, 1*60],  # Convert frequency to BPM by multiplying by 60 (since 1 Hz = 60 BPM)
                                            scaleanchor='x',
                                            scaleratio=60
                                        )
                                    )

                                    # Save the plot as an HTML file
                                    plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_psd_hrv.html")
                                    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
                                
                                    #%% Calculate various statistics
                                    
                                    # Create a mask for FD values less than or equal to 0.5
                                    mask_ppg = fd_upsampled_ppg <= 0.5

                                    # Apply the mask to both FD and PPG data
                                    filtered_fd_ppg = fd_upsampled_ppg[mask_ppg]
                                    filtered_ppg = ppg_cleaned[mask_ppg]

                                    # Initialize correlation and above threshold FD variables as NaN
                                    r_value_ppg = np.nan
                                    p_value_ppg = np.nan
                                    r_value_ppg_thresh = np.nan
                                    p_value_ppg_thresh = np.nan

                                    num_samples_above_threshold = np.nan
                                    percent_samples_above_threshold = np.nan
                                    mean_fd_above_threshold = np.nan
                                    std_dev_fd_above_threshold = np.nan
                                    
                                    # Check if there are any FD values above the threshold
                                    if np.any(fd > voxel_threshold):
                                        # Check if filtered data is not empty
                                        if len(filtered_fd_ppg) > 0 and len(filtered_ppg) > 0:
                                            
                                            # Calculate above threshold FD statistics
                                            num_samples_above_threshold = np.sum(fd_upsampled_ppg > voxel_threshold)
                                            percent_samples_above_threshold = num_samples_above_threshold / len(fd_upsampled_ppg) * 100
                                            mean_fd_above_threshold = np.mean(fd_upsampled_ppg[fd_upsampled_ppg > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
                                            std_dev_fd_above_threshold = np.std(fd_upsampled_ppg[fd_upsampled_ppg > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
                                            
                                            # Calculate the correlation between filtered FD and PPG
                                            r_value_ppg, p_value_ppg = calculate_fd_ppg_correlation(filtered_fd_ppg, filtered_ppg)
                                            logging.info(f"Correlation between FD (filtered) and filtered cleaned PPG timeseries < {voxel_threshold} mm: {r_value_ppg}, p-value: {p_value_ppg}")

                                            plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_ppg_correlation_filtered.png"
                                            plot_filepath = os.path.join(base_path, plot_filename)
                                            plot_fd_ppg_correlation(filtered_fd_ppg, filtered_ppg, plot_filepath)
                                            logging.info(f"FD-PPG filtered correlation plot saved to {plot_filepath}")
                                            
                                        else:
                                            # Log a warning if there are no FD values below the threshold after filtering
                                            logging.warning(f"No FD values below {voxel_threshold} mm. Correlation cannot be calculated.")
                                    else:
                                        # Log a warning if there are no FD values above the threshold
                                        logging.warning(f"No FD values above {voxel_threshold} mm. No need to filter and calculate correlation.")

                                     # Calculate statistics related to framewise displacement
                                    mean_fd_below_threshold = np.mean(fd_upsampled_ppg[fd_upsampled_ppg < voxel_threshold])
                                    std_dev_fd_below_threshold = np.std(fd_upsampled_ppg[fd_upsampled_ppg < voxel_threshold])

                                    # Average ppg Frequency (counts/min)
                                    total_time_minutes = len(ppg_cleaned) / (sampling_rate * 60)
                                    average_ppg_frequency = len(valid_peaks) / total_time_minutes

                                    # Average, Std Deviation, Max, Min Inter-ppg Interval (sec)
                                    inter_ppg_intervals = np.diff(valid_peaks) / sampling_rate
                                    average_inter_ppg_interval = inter_ppg_intervals.mean()
                                    std_inter_ppg_interval = inter_ppg_intervals.std()
                                    max_inter_ppg_interval = inter_ppg_intervals.max()
                                    min_inter_ppg_interval = inter_ppg_intervals.min()

                                    logging.info(f"Calculating HRV Indices via Neurokit")
                                    hrv_indices = nk.hrv(valid_peaks, sampling_rate=sampling_rate, show=True)
                                    
                                    # Update the ppg_stats dictionary
                                    ppg_stats = {
                                        'R-Peak Count (# peaks)': len(valid_peaks),
                                        'Average R-Peak Frequency (counts/min)': average_ppg_frequency,
                                        'Average Inter-Peak Interval (sec)': average_inter_ppg_interval,
                                        'Std Deviation Inter-Peak Interval (sec)': std_inter_ppg_interval,
                                        'Max Inter-Peak Interval (sec)': max_inter_ppg_interval,
                                        'Min Inter-Peak Interval (sec)': min_inter_ppg_interval,
                                        'Mean Framewise Displacement (mm)': fd_upsampled_ppg.mean(),
                                        'Std Deviation Framewise Displacement (mm)': fd_upsampled_ppg.std(),
                                        'Max Framewise Displacement (mm)': fd_upsampled_ppg.max(),
                                        'Min Framewise Displacement (mm)': fd_upsampled_ppg.min(),
                                        'Number of samples with FD > 0.5 mm': num_samples_above_threshold,
                                        'Percent of samples with FD > 0.5 mm': percent_samples_above_threshold,
                                        'Mean FD > 0.5 mm': mean_fd_above_threshold,
                                        'Std Deviation FD > 0.5 mm': std_dev_fd_above_threshold,
                                        'Mean FD < 0.5 mm': mean_fd_below_threshold,
                                        'Std Deviation FD < 0.5 mm': std_dev_fd_below_threshold,
                                        'Framewise Displacement - PPG Correlation R-Value': r_value_ppg,
                                        'Framewise Displacement - PPG Correlation P-Value': p_value_ppg,
                                        'Framewise Displacement - PPG Correlation R-Value (FD < 0.5 mm)': r_value_ppg_thresh,
                                        'Framewise Displacement - PPG Correlation P-Value (FD < 0.5 mm)': p_value_ppg_thresh,
                                        }
                                        
                                    # Assume hrv_indices is a dictionary with each value being a Series or a single value
                                    hrv_stats = {}

                                    # Loop through each HRV metric in hrv_indices
                                    for metric, value in hrv_indices.items():
                                        # Check if the value is a Series and take the first value
                                        if isinstance(value, pd.Series):
                                            hrv_stats[metric] = value.iloc[0]
                                        else:
                                            hrv_stats[metric] = value

                                    # Debug: Check the updated PPG statistics
                                    logging.info(f"PPG Stats: {ppg_stats}")

                                    # Assume ppg_stats are dictionaries containing the statistics
                                    ppg_stats_df = pd.DataFrame(ppg_stats.items(), columns=['Statistic', 'Value'])
                                    hrv_stats_df = pd.DataFrame(hrv_stats.items(), columns=['Statistic', 'Value'])
                                    
                                    # Concatenate DataFrames vertically, ensuring the order is maintained
                                    ppg_summary_stats_df = pd.concat([ppg_stats_df, hrv_stats_df], axis=0, ignore_index=True)
                                    
                                    # Add a column to indicate the category of each statistic
                                    ppg_summary_stats_df.insert(0, 'Category', '')
                                    
                                    # Assign 'PPG Stats' to the first part and 'HRV Stats' to the second part
                                    ppg_summary_stats_df.loc[:len(ppg_stats_df) - 1, 'Category'] = 'PPG Stats'
                                    ppg_summary_stats_df.loc[len(ppg_stats_df):, 'Category'] = 'HRV Stats'
                                    
                                    # Save the summary statistics to a TSV file, with headers and without the index
                                    summary_stats_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_{peak_method}_summary_statistics.tsv")
                                    ppg_summary_stats_df.to_csv(summary_stats_filename, sep='\t', header=True, index=False)
                                    logging.info(f"Saving summary statistics to TSV file: {summary_stats_filename}")
                                
                                except Exception as e:
                                    logging.error(f"Error in peak detection {peak_method}): {e}")
                                    # Log stack trace for debugging purposes
                                    logging.error(traceback.format_exc())
                                    continue

                                # Save ppg_cleaned data to a TSV file
                                logging.info(f"Saving ppg_cleaned data to TSV file.")
                                ppg_cleaned_df_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_processed.tsv")
                                ppg_cleaned_df['FD_Upsampled'] = fd_upsampled_ppg
                                ppg_cleaned_df.to_csv(ppg_cleaned_df_filename, sep='\t', index=False)
                                
                                # Compress the TSV file
                                with open(ppg_cleaned_df_filename, 'rb') as f_in:
                                    with gzip.open(f"{ppg_cleaned_df_filename}.gz", 'wb') as f_out:
                                        f_out.writelines(f_in)
                                
                                # Remove the uncompressed file
                                os.remove(ppg_cleaned_df_filename)
                                logging.info(f"Saved and compressed ppg_cleaned data to {ppg_cleaned_df_filename}.gz")

                    except Exception as e:
                        logging.error(f"Error processing ppg file {file}: {e}")
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
    print(f"Main function completed. Script runtime: {script_runtime} minutes. Processing PPG complete for participant {participant_id}.")
        
#%% Main script function
# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # # Call the main function.
    # cProfile.run('main()')

    main()
# %%
