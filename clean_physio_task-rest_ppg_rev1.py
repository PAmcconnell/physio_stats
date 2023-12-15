"""
Script Name: clean_physio_task-rest_ppg.py

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

# Function to plot the correlation between FD and EDA
def plot_fd_eda_correlation(fd, eda, file_name):
    
    if len(fd) != len(eda):
        logging.warning("Error: FD and EDA timeseries must be of the same length.")
        return
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(fd, eda)
        fit_line = slope * fd + intercept
        r_squared = r_value**2
        
        # Calculate the confidence interval of the fit line
        t_val = t.ppf(1-0.05/2, len(fd)-2)  # t-score for 95% confidence interval & degrees of freedom
        conf_interval = t_val * std_err * np.sqrt(1/len(fd) + (fd - np.mean(fd))**2 / np.sum((fd - np.mean(fd))**2))

        # Upper and lower bounds of the confidence interval
        lower_bound = fit_line - conf_interval
        upper_bound = fit_line + conf_interval

        plt.figure(figsize=(10, 6))
        plt.scatter(fd, eda, alpha=0.5, label='Data Points')
        plt.plot(fd, fit_line, color='red', label=f'Fit Line (R = {r_value:.3f}, p = {p_value:.3f})')
        plt.fill_between(fd, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.ylabel('Electrodermal Activity (µS)')
        plt.xlabel('Framewise Displacement (mm)')
        plt.title('Correlation between FD and EDA with Linear Fit and Confidence Interval')
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
#    for i, participant_id in enumerate(participants_df['participant_id']):
    for i, participant_id in enumerate([participants_df['participant_id'].iloc[9]]):  # For testing  LRN010 first with ppg data

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
                base_path = os.path.join(derivatives_dir, 'ppg', participant_id, run_id)

                # Make sure the directories exist
                os.makedirs(base_path, exist_ok=True)
                
                # Define the processed signals filename for checking
                processed_signals_filename = f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio_filtered_cleaned_eda_processed_ppg.tsv.gz"
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
                logging.info(f"Testing EDA processing for task {task_name} run-0{run_number} for participant {participant_id}")

                # Filter settings
                stop_freq = 11.5  # 11.5 Hz -> nSlices/(MBF*TR) = 69/(2*3) = 11.5 Hz ? 
                sampling_rate = 5000    # acquisition sampling rate
            
                # Define the frequency band
#                frequency_band = (0.045, 0.25) # 0.045 - 0.25 Hz Sympathetic Band

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

                        # Open the compressed EDA file and load it into a DataFrame
                        with gzip.open(file, 'rt') as f:
                            physio_data = pd.read_csv(f, delimiter='\t')
                            # Check for the presence of the 'eda' column
                            if 'ppg' not in physio_data.columns:
                                logging.error(f"'ppg' column not found in {file}. Skipping this file.")
                                continue  # Skip to the next file
                            ppg = physio_data['ppg']

                            # Calculate the time vector in minutes - Length of PPG data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(len(ppg)) / sampling_rate / 60

                            
                                
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