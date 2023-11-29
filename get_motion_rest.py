import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MCFLIRT
import shutil
import pandas as pd
import gzip
import glob
from datetime import datetime
import re
import sys

# conda activate nipype (Python 3.9)

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

# Load EDA data from TSV file
def load_eda_data(eda_filepath):
    """
    Load Electrodermal Activity (EDA) data from a TSV file.

    This function reads EDA data from a TSV file compressed with gzip. It expects
    the data to be tab-separated and will return it as a pandas DataFrame.

    Parameters:
    eda_filepath (str): The file path to the gzipped TSV file containing EDA data.

    Returns:
    pandas.DataFrame: A DataFrame containing the EDA data, or None if an error occurs.
    """
    try:
        # Attempting to read the gzipped TSV file
        with gzip.open(eda_filepath, 'rt') as f:
            eda_df = pd.read_csv(f, sep='\t')
        logging.info(f"EDA data loaded successfully from {eda_filepath}")
        return eda_df
    except FileNotFoundError:
        logging.error(f"File not found: {eda_filepath}")
        return None
    except Exception as e:
        logging.error(f"Error in loading EDA data from {eda_filepath}: {e}")
        return None

# Plot EDA and Framewise Displacement
def plot_eda_fd(fd, eda_df, output_dir, voxel_threshold=2.0):
    """
    Plot EDA and Framewise Displacement data.

    This function creates a plot with four subplots: Cleaned EDA, EDA Phasic Component with SCR Events,
    Tonic EDA, and Framewise Displacement. It handles missing data and logs warnings accordingly.

    Parameters:
    fd (array-like): The framewise displacement data.
    eda_df (pandas.DataFrame): DataFrame containing EDA data with columns like 'EDA_Clean', 'EDA_Phasic', etc.
    output_dir (str): The directory where the plots will be saved.
    voxel_threshold (float, optional): Threshold for Framewise Displacement plotting. Default is 2.0.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plots.
    """
    try:
        # Acquisition sampling rate for physiological data
        sampling_rate = 5000   

        # Calculate the time vector in minutes - Length of EDA data divided by the sampling rate gives time in seconds, convert to minutes
        time_vector = np.arange(len(eda_df['EDA_Clean'])) / sampling_rate / 60

        # Create a figure with 4 subplots. 
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))

        # Plot EDA_Clean
        axes[0].plot(eda_df['EDA_Clean'], label='Cleaned EDA', color='orange')
        axes[0].set_title('Filtered and Cleaned EDA Signal')
        axes[0].set_ylabel('Amplitude (µS)')
        axes[0].legend()

        # Plot EDA_Phasic
        axes[1].plot(eda_df['EDA_Phasic'], label='Phasic Component', color='green')

        # Check if 'SCR_Onsets' key exists and contains no NaN values before plotting
        if 'SCR_Onsets' in eda_df and not np.isnan(eda_df['SCR_Onsets']).any():
            axes[1].scatter(eda_df['SCR_Onsets'], eda_df['EDA_Phasic'][eda_df['SCR_Onsets']], color='blue', label='SCR Onsets')
        else:
            logging.warning("SCR_Onsets key not found or contains NaN values in eda_df. Skipping SCR Onsets plotting.")

        # Check if 'SCR_Peaks' key exists and contains no NaN values before plotting
        if 'SCR_Peaks' in eda_df and not np.isnan(eda_df['SCR_Peaks']).any():
            axes[1].scatter(eda_df['SCR_Peaks'], eda_df['EDA_Phasic'][eda_df['SCR_Peaks']], color='red', label='SCR Peaks')
        else:
            logging.warning("SCR_Peaks key not found or contains NaN values in eda_df. Skipping SCR Peaks plotting.")
                
        # Filter out NaN values from SCR_Recovery indices
        valid_recovery_indices = eda_df['SCR_Recovery'][~np.isnan(eda_df['SCR_Recovery'])]

        # Ensure valid_recovery_indices is not empty before plotting
        if len(valid_recovery_indices) > 0:
            axes[1].scatter(valid_recovery_indices, eda_df['EDA_Phasic'][valid_recovery_indices], color='purple', label='SCR Half Recovery')
        else:
            logging.warning("No valid SCR recovery indices found for plotting.")

        axes[1].set_title('EDA Phasic Component with SCR Events')
        axes[1].set_ylabel('Amplitude (µS)')
        axes[1].legend()

        # Plot EDA_Tonic
        axes[2].plot(time_vector, eda_df['EDA_Tonic'], label='Tonic Component', color='brown')
        axes[2].set_title('Tonic EDA')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_ylabel('Amplitude (µS)')
        axes[2].legend()

        # Plot Framewise Displacement with threshold line
        axes[3].plot(fd, label='Framewise Displacement', color='blue')
        axes[3].axhline(y=voxel_threshold, color='r', linestyle='--')
        axes[3].set_title('Framewise Displacement')
        axes[3].set_xlabel('Volume Number')
        axes[3].set_ylabel('FD (mm)')

        plt.tight_layout()

        return fig and axes
    
    except KeyError as e:
        logging.error(f"Missing column in eda_df: {e}")
        return None
    except Exception as e:
        logging.error(f"Error in plotting EDA and FD data: {e}")
        return None

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

# Main function to process each run for each participant and generate plots
def main():
    
    """
    Main function to process each run for each participant in an fMRI study.

    This function orchestrates the processing of fMRI data for each participant. It performs several tasks:
    - Checks the existence of necessary directories (BIDS root, dataset root, derivatives, and output directories).
    - Reads the participant information from a TSV file.
    - Iteratively processes each run for each participant by calling a separate 'process_run' function.

    The function assumes a specific directory structure and file naming convention as per BIDS standards.
    It is designed to handle typical preprocessing tasks in fMRI data analysis, such as motion correction,
    framewise displacement calculation, and generating plots.

    Note: This function does not return any value. It primarily orchestrates the workflow and logs its progress.
    """

    print(f"Starting main function")
    script_start_time = datetime.now()

    # Define and check BIDS root directory
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    print(f"Checking BIDS root directory: {bids_root_dir}")
    if not os.path.exists(bids_root_dir):
        print(f"Directory not found: {bids_root_dir}")
        return

    # Define and check dataset root directory
    dataset_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/')
    print(f"Checking BIDS root directory: {dataset_root_dir}")
    if not os.path.exists(dataset_root_dir):
        print(f"Directory not found: {dataset_root_dir}")
        return
    
    # Define and check derivatives directory
    derivatives_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    print(f"Checking derivatives directory: {derivatives_root_dir}")
    if not os.path.exists(derivatives_root_dir):
        print(f"Directory not found: {derivatives_root_dir}")
        return
    
    # Output directory
    output_root_dir = '/Users/PAM201/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/_motion/'         
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

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
        
        # Loop through each run for this participant
        for run_number in range(1, 5):  # Assuming 4 runs
            try:
                
                run_start_time = datetime.now()

                # Set a higher DPI for better resolution
                dpi_value = 300 

                # Define subject id
                session_id = 'ses-1'  
                
                # Define task name
                task_name = 'rest'

                # Process the first run for the selected participant
                run_id = f"run-0{run_number}"
                
                # Define the processed signals filename for checking
                processed_fd_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_bold_framewise_displacement.png"
                logging.info(f"Checking for processed FD file: {processed_fd_filename}")
                processed_fd_path = os.path.join(derivatives_root_dir, '_motion', participant_id, run_id, processed_fd_filename)

                # Check if processed file exists
                if os.path.exists(processed_fd_path):
                    print(f"Processed FD files found for {participant_id} for run {run_number}, skipping...")
                    continue  # Skip to the next run

                # Reset logging for next participant and run
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                # Configure subject and run specific logging. 
                log_file_path = setup_logging(participant_id, session_id, run_id, dataset_root_dir)
                logging.info(f"Testing FD-EDA processing for task {task_name} run-0{run_number} for participant {participant_id}")

                pattern = re.compile(f"{participant_id}_ses-1_task-rest_run-{run_number:02d}_bold.nii")
                bids_subject_dir = os.path.join(bids_root_dir, participant_id, session_id, 'func')
                func_files = [os.path.join(bids_subject_dir, f) for f in os.listdir(bids_subject_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_bold.nii' in f]
                for file in func_files:
                    
                    # Record the start time for this run
                    run_start_time = datetime.now()
                    logging.info(f"Processing file: {file}")
                        
                    # Generate a base filename by removing the '.tsv.gz' extension
                    base_filename = os.path.basename(file).replace('.nii', '')

                    # Temp directory for intermediate files
                    working_dir = '/Users/PAM201/Documents/MRI/LEARN/BIDS_test/derivatives/physio/temp'     
                    if not os.path.exists(working_dir):
                        os.makedirs(working_dir)
                    logging.info("Working directory: {}".format(working_dir))
                    
                    # Construct the eda path
                    eda_subject_dir = os.path.join(derivatives_root_dir, 'eda', participant_id, run_id)
                    logging.info("EDA subject directory: {}".format(eda_subject_dir))

                    output_subject_dir = os.path.join(output_root_dir, participant_id, run_id)
                    if not os.path.exists(output_subject_dir):
                        os.makedirs(output_subject_dir)
                    logging.info("Output subject directory: {}".format(output_subject_dir))
                    
                    # BIDS location for original 4D nifti data
                    #original_nii_data_path = '/Users/PAM201/Documents/MRI/LEARN/BIDS_test/dataset/sub-LRN001/ses-1/func/sub-LRN001_ses-1_task-rest_run-01_bold.nii' 
                    original_nii_data_path = file
                    logging.info("Original data path: {}".format(original_nii_data_path))
                    
                    # Location of processed EDA data
                    eda_filepath = os.path.join(derivatives_root_dir, 'eda', participant_id, run_id, f"{participant_id}_{session_id}_task-{task_name}_{run_id}_physio_filtered_cleaned_eda_processed.tsv.gz")
                    logging.info("EDA file path: {}".format(eda_filepath))

                    # Check if eda file exists
                    if os.path.exists(eda_filepath):
                        logging.info(f"Processed eda files found for {participant_id} for run {run_number}, proceeding with FD calculation.")
                    else:
                        logging.info(f"Processed eda files not found for {participant_id} for run {run_number}, skipping...")
                        continue
                    
                    try:
                        output_file = run_mcflirt_motion_correction(original_nii_data_path, output_subject_dir, working_dir)
                        logging.info(f"Motion corrected file saved at {output_file}")
                    except Exception as e:
                        logging.error(f"Failed to complete motion correction: {e}")

                    # Load motion parameters and calculate FD
                    try:
                        logging.info("Calculating framewise displacement...")
                        motion_params_file = os.path.join(output_subject_dir, 'mcf_' + os.path.basename(original_nii_data_path) + '.par')
                        fd = calculate_fd(motion_params_file)
                        logging.info("Framewise displacement calculated successfully.")
                    except Exception as e:
                        logging.error("Error in calculating framewise displacement: {}".format(e))
                        raise

                    # Load EDA data
                    eda_df = load_eda_data(eda_filepath)

                    # Correctly forming the file name for the FD data
                    fd_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_framewise_displacement.tsv"

                    # Joining the output directory with the new file name
                    fd_file_path = os.path.join(output_subject_dir, fd_filename)

                    # Saving the FD data to the correct path
                    save_fd_to_tsv(fd, output_subject_dir, fd_file_path)

                    # Plot EDA and FD
                    plot_filename = f"{participant_id}_{session_id}_{task_name}_{run_id}_bold_EDA_FD_plot.png"
                    plot_full_path = os.path.join(output_subject_dir, plot_filename)
                    plt.savefig(plot_full_path, dpi=dpi_value)

            except Exception as e:
                logging.error(f"Error in script: {e}")
                raise

            # Record the end time for this participant and calculate runtime
            run_end_time = datetime.now()
            run_runtime = (run_end_time - run_start_time).total_seconds() / 60
            print(f"Participant {participant_id} completed. Runtime: {run_runtime} minutes.")
            
            # Remove the working directory
            try:
                shutil.rmtree(working_dir)
                logging.info("Working directory removed successfully.")
                
                # Cleanup 'mcf_' prefixed files in the output directory
                mcf_files = glob.glob(os.path.join(output_subject_dir, 'mcf_*'))
                for file_path in mcf_files:
                    try:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                        logging.info(f"Removed '{file_path}'")
                    except Exception as e:
                        logging.error(f"Error removing file or directory: {file_path}: {e}")
                        raise e  
            except Exception as e:
                logging.error(f"Error removing working directory: {e}")
                raise e

        # Record the end time for this participant and calculate runtime
        participant_end_time = datetime.now()
        participant_runtime = (participant_end_time - participant_start_time).total_seconds() / 60
        print(f"Participant {participant_id} completed. Runtime: {participant_runtime} minutes.")
                    
    # Record the script end time and calculate runtime
    end_time = datetime.now()
    script_runtime = (end_time - script_start_time).total_seconds() / 60
    print(f"Main function completed. Script runtime: {script_runtime} minutes. Processing complete for participant {participant_id}.")
    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()