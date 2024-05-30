""" 

Overview:

This script implements a Dash-based web application for correcting peaks, arrhythmias, and artifacts, in photoplethysmogram (PPG) data. 
It provides an interactive interface for uploading PPG data files, visualizing them, manually correcting identified peaks, and saving the corrected data along with
updated HRV statistics. 

Usage:

- Activate the appropriate Python environment that has the required packages installed.
  For Conda environments: `conda activate nipype`
- Run the script from the command line: python correct_peaks_artifacts_revX.x.py --save_dir "/path/to/your/save/directory"
- The script automatically opens a web interface in the default browser where users can upload, correct, and save PPG data.
- Use control-c to stop the server and exit the application when finished with corrections and file saving. 

"""
import base64
import dash
from dash import html, dcc, Input, Output, State
import webbrowser
from threading import Timer
import logging
import pandas as pd
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind, f_oneway
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import t
import os
import argparse
import datetime
import sys
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend for rendering
import matplotlib.pyplot as plt
import neurokit2 as nk
import bisect

# ! This is a functional peak correction interface for PPG data with artifact selection and ([now less] buggy) correction (rev18.1) [- 2024-05-16]) 
# ! Working to finalize save and output (rev 18.x) etc
# ! Working to finalize artifact correction (rev 18.x) etc
# ! Working to integrate nk fixpeaks testing / good, but need to save out statistics (rev 18.x) etc

# NOTE: conda activate nipype
# python correct_peaks_artifacts_rev17.0.py

#%% Completed Tasks
#// TODO: track number of corrections made and save to a file
#// TODO - Fix the save function to handle different directory structures
#// TODO - Add plotly html output here or elsewhere?
#// TODO: implement subject- and run-specific archived logging
#// TODO: implement artifact selection and correction
#// TODO: Handle edge cases for artifact selection and correction
#// TODO: fix the trimming of heartbeats to remove the rise
#// TODO: Fix segmentation and dynamic nadir calculations
#// TODO: Test signal_fixpeaks: https://neuropsychology.github.io/NeuroKit/functions/signal.html#signal-fixpeaks

#%% Higher Priorities

# REVIEW: implement recalculation and saving of PPG and HRV statistics
# - extend this to the other relevant scenarios: fixpeaks (raw, kubios), without corrected artifacts (item below)
# TODO: Implement artifact-corrected and artifact-uncorrected HRV statistics and timeseries save out 
# TODO: Add nk fixpeaks kubios correction to HRV stat output
# TODO: Fix sample tracking for artifact windows
# TODO: Need to add padding of some kind for the edges of the timeseries so that we have full sample (e.g., 35k samples) for ppg and r-r interval timeseries - during final save?

#%% Lesser Priorities

# TODO: smooth the transitions between heartbeats at window boundaries
#? Much better now but still a little jagged at the edges
# TODO: implement amplitude scaling for inserted heartbeats
# Note: The amplitude scaling issue seems resolved with the revised beat template creation
# TODO: add pre and post average heartbeat plotting for QC
#? Is this worth implementing? - It could be useful for visualizing the interpolation process
# TODO: Median and standard deviation for heartbeat shape plots? Pre- and Post- correction?
#? This could be useful for visualizing the variability in the heartbeats before and after correction

# TODO: Implement statistical comparisons of artifact-corrected and artifact-free HRV stats - does interpolation bias findings?
#? Doing this post hoc in R?

# TODO: Fix BUG where points can be click added/removed to r-r interval plot ( & Double Save bug)
# TODO: Implement visual comparison of traditional filtering on r-r interval tachogram
# TODO: Allow artifact window censoring

# Initialize the Dash app
app = dash.Dash(__name__)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run the Dash app for correcting PPG data.")
parser.add_argument('--save_dir', type=str, required=True, help='Directory path to save corrected files to.')
args = parser.parse_args()

# Validate and use the provided directory path
save_directory = args.save_dir
if not os.path.isdir(save_directory):
    print(f"The provided save directory {save_directory} does not exist... Creating it now.")
    os.makedirs(save_directory)

# Sets up archival logging for the script, directing log output to both a file and the console.
def setup_logging(filename):
    """
    The function configures logging (level = INFO) to capture informational, warning, and error messages and writes logs to both a file and the console. 
    It creates a unique log file for each subject-session combination stored in the corrected data save directory. 
    The log file is named based on the script name, filename, and timestamp. 

    Parameters:
    - filename (str): The name of the uploaded file.

    Returns:
    - logging confirmation of successful configuration. 
    
    """

    global save_directory  # Use the global save directory path from the argument parser
    
    try:
        print(f"Setting up logging for file: {filename}")
        
        # Get the current date and time to create a unique timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")   
        
        # Extract the base name of the script without the .py extension.
        script_name = os.path.basename(__file__).replace('.py', '')

        # Extract subject_id and run_id
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"Filename does not contain expected parts.")
            return "Error: Filename structure incorrect."

        subject_id = parts[0]
        session_id = 'ses-1'
        taskName = 'task-rest'
        run_id = parts[3]
        
        # Construct the log file name using timestamp, session ID, and script name.
        log_file_name = f"{script_name}_{timestamp}_{subject_id}_{session_id}_{taskName}_{run_id}_physio_ppg_corrected.log"
        log_file_path = os.path.join(save_directory, log_file_name)
        
        # Clear any existing log handlers to avoid duplicate logs.
        logging.getLogger().handlers = []
        
        # Configure file logging.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file_path,
            filemode='w' # 'w' mode overwrites existing log file.
        )

        # If you also want to log to console.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the console handler to the logger
        logging.getLogger().addHandler(console_handler)

        # Log the successful setup of logging
        logging.info(f"Logging setup complete.")
        logging.info(f"Filename: {filename}")
        logging.info(f"Log file: {log_file_path}")
        logging.info(f"Subject ID: {subject_id}")
        logging.info(f"Session ID: {session_id}")
        logging.info(f"Task Name: {taskName}")
        logging.info(f"Run ID: {run_id}")

    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1) # Exiting the script due to logging setup failure.

# Define the layout of the Dash application
app.layout = html.Div([
    # Store components for holding data in the browser's memory without displaying it
    dcc.Store(id='data-store'),  # To store the main DataFrame after processing
    dcc.Store(id='peaks-store'),  # To store indices of valid peaks identified in the PPG data
    dcc.Store(id='ppg-store'),  # Store for the interpolated PPG data
    dcc.Store(id='filename-store'),  # To store the name of the original file uploaded by the user
    dcc.Store(
        id='peak-change-store', 
        data={
            'original_peaks_count': 0, 
            'peaks_deleted': 0, 
            'peaks_added': 0, 
            'total_corrected_peak_count': 0, 
            'original_total_peak_count': 0, 
            'dash_added_peaks_count': 0, 
            'dash_deleted_peaks_count': 0, 
            'dash_total_corrected_peak_count': 0, 
            'kubios_added_peak_count': 0, 
            'kubios_deleted_peak_count': 0, 
            'kubios_total_corrected_peak_count': 0, 
            'kubios_total_samples_corrected': 0, 
            'artifact_window_censored_total_corrected_peak_count': 0, 
            'artifact_windows_total_count_number': 0, 
            'artifact_windows_total_count_samples': 0, 
            'mean_artifact_window_size_samples': 0, 
            'std_artifact_window_size_samples': 0, 
            'min_artifact_window_size_samples': 0, 
            'max_artifact_window_size_samples': 0,
            'kubios_ectopic_beats': 0,	
            'kubios_missed_beats': 0,	
            'kubios_extra_beats': 0,	
            'kubios_longshort_beats': 0
        }
    ),  # Store for tracking corrections
    dcc.Store(id='artifact-windows-store', data=[]),  # Store for tracking indices of corrected artifact windows
    dcc.Store(id='artifact-interpolation-store', data=[]),  # Store for tracking surrounding ppg timeseries used for artifact interpolation
    dcc.Store(id='rr-store'),  # Store for R-R interval data # REVIEW
    
    # Upload component to allow users to select and upload PPG data files for correction
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select _processed.tsv.gz File to Correct Peaks'),
        #style={},  # OPTIMIZE: Style properties can be added here for customization
        multiple=False  # Restrict to uploading a single file at a time
    ),
    
    # Group artifact selection components with initial style set to hide them
    html.Div(id='artifact-selection-group', children=[
        html.Div(id='artifact-window-output'),
        html.Label(["Start:", dcc.Input(id='artifact-start-input', type='number', value=0)]),
        html.Label(["End:", dcc.Input(id='artifact-end-input', type='number', value=0)]),
        html.Button('Confirm Artifact Selection', id='confirm-artifact-button', n_clicks=0)
    ], style={'display': 'none'}),  # Initially hidden
    
    # Graph component for displaying the PPG data and allowing users to interactively correct peaks
    dcc.Graph(id='ppg-plot'),
    
    # Button to save the corrected data back to a file
    html.Button('Save Corrected Data', id='save-button'),
    
    # Div to display the status of the save operation (e.g., success or error message)
    html.Div(id='save-status'),
])

# Define content parsing function
def parse_contents(contents):
    """
    Parses the contents of an uploaded file encoded in base64 format and returns a DataFrame.

    This function decodes the base64 encoded string of a file uploaded through the Dash interface,
    reads the content using Pandas, assuming the file is a tab-separated value (TSV) file compressed
    with gzip compression, and returns a Pandas DataFrame.

    Parameters:
    - contents (str): A base64 encoded string representing the content of the uploaded file.

    Returns:
    - DataFrame: A Pandas DataFrame containing the data from the uploaded file.

    Raises:
    - ValueError: If the contents cannot be split by a comma, indicating an unexpected format.
    - Exception: For errors encountered during the decoding or Pandas DataFrame creation process.
    """
    
    # Split the content string by the first comma to separate the encoding header from the data
    _, content_string = contents.split(',')
    
    # Decode the base64 encoded string to get the file bytes
    decoded = base64.b64decode(content_string)
    
    # Use Pandas to read the decoded bytes. The file is assumed to be a TSV file compressed with gzip.
    df = pd.read_csv(io.BytesIO(decoded), sep='\t', compression='gzip')
    
    # Return the Pandas DataFrame
    return df

@app.callback(
    [Output('ppg-plot', 'figure'), # Update the figure with the PPG data and peaks
     Output('data-store', 'data'), # Update the data-store with the DataFrame
     Output('peaks-store', 'data'), # Update the peaks-store with the valid peaks
     Output('ppg-store', 'data'), # Update the ppg-store with the interpolated PPG data
     Output('peak-change-store', 'data'),  # Add output for peak change tracking
     Output('artifact-windows-store', 'data'),  # Update the artifact-windows-store with artifact window indices
     Output('artifact-window-output', 'children'),  # Provide feedback on artifact selection
     Output('artifact-start-input', 'value'),  # To reset start input
     Output('artifact-end-input', 'value'),  # To reset end input
     Output('artifact-selection-group', 'style'),  # To show/hide artifact selection group
     Output('artifact-interpolation-store', 'data'),  # Store for tracking surrounding ppg timeseries used for artifact interpolation
     Output('rr-store', 'data')], # Store for R-R interval data
    [Input('upload-data', 'contents'), # Listen to the contents of the uploaded file
     Input('ppg-plot', 'clickData'), # Listen to clicks on the PPG plot
     Input('confirm-artifact-button', 'n_clicks')],  # Listen to artifact confirmation button clicks
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'), # Keep the data-store state
     State('peaks-store', 'data'), # Keep the peaks-store state
     State('ppg-store', 'data'), # Keep the ppg-store state
     State('ppg-plot', 'figure'), # Keep the existing figure state
     State('peak-change-store', 'data'), # Keep the peak-change-store state
     State('artifact-start-input', 'value'),  # Get the start index of artifact window
     State('artifact-end-input', 'value'),  # Get the end index of artifact window
     State('artifact-windows-store', 'data'),  # Keep the artifact-windows-store state
     State('artifact-interpolation-store', 'data')]  # Store for tracking surrounding ppg timeseries used for artifact interpolation
)

# Main callback function to update the PPG plot and peak data
def update_plot_and_peaks(contents, clickData, n_clicks_confirm, filename, data_json, valid_peaks, valid_ppg, existing_figure, peak_changes, artifact_start_idx, artifact_end_idx, artifact_windows, interpolation_windows):

    """
    Updates the PPG plot and peak data in response to user interactions, specifically file uploads and plot clicks.
    
    This callback function handles two primary interactions: uploading a new PPG data file and clicking on the plot to
    correct peaks. It updates the plot with the uploaded data or modified peaks, stores the updated data and peak 
    information, and tracks changes to the peaks throughout the session.

    Parameters:
    - contents (str): Base64 encoded contents of the uploaded file.
    - clickData (dict): Data from plot clicks, used for peak correction.
    - filename (str): The name of the uploaded file.
    - data_json (str): JSON string representation of the DataFrame holding the PPG data.
    - valid_peaks (list): List of indices for valid peaks in the PPG data.
    - existing_figure (dict): The existing figure object before any new updates.
    - peak_changes (dict): A dictionary tracking the number of peaks added, deleted, or originally present, as well as samples corrected.

    Returns:
    - A tuple containing the updated figure, data store JSON, valid peaks list, and peak changes dictionary.

    Raises:
    - dash.exceptions.PreventUpdate: Prevents updating the outputs if there's no new file uploaded or no click interaction on the plot.
    """
    
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
    # Initialize the artifact window output message
    artifact_window_output = "No artifact window confirmed yet." #* or []?

    # Determine if the artifact selection group should be shown
    show_artifact_selection = {'display': 'block'} if contents else {'display': 'none'}
    
    # Initialize the interpolation windows if they are None
    if interpolation_windows is None:
        interpolation_windows = []
    
    try:
        # Handle file uploads and initialize the plot and peak data
        if triggered_id == 'upload-data' and contents:
            setup_logging(filename)  # Set up logging for the current file
            
            # Loading the initial dataframe from the _processed.tsv.gz file
            # NOTE: The df['RR_interval_interpolated'] column computed from the preprocessing script is not correct and needs to be redefined here
            df = parse_contents(contents)
            
            # Clear the existing RR_interval_interpolated column
            df['RR_interval_interpolated'] = np.nan

            # Initialize the Rows and Samples columns - Sanity Check for excel output checking 
            df['Rows'] = range(2, len(df) + 2)  # Starting at 2
            df['Samples'] = range(len(df))  # Starting at 0
            
            # Create initial variables for dcc.Store components and plotting
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            valid_ppg = df['PPG_Clean']
            initial_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.to_numpy()
            logging.info(f"Initial peaks imported from preprocessed dataframe for automated peak correction") 
            
            
            #%% Integration of NeuroKit2 fixpeaks function

            # Methods to apply
            methods = ['Kubios']
            for method in methods:
                logging.info(f"Applying {method} method for peak correction")
                if method == "Kubios":
                    artifacts, corrected_peaks = nk.signal_fixpeaks(initial_peaks, sampling_rate=100, method=method, iterative=True, show=False)
                else:
                    _, corrected_peaks = nk.signal_fixpeaks(initial_peaks, sampling_rate=100, method=method, iterative=True, robust=True, show=False)

                df[f'Peaks_{method}'] = 0
                df.loc[corrected_peaks, f'Peaks_{method}'] = 1

                # Export artifacts and corrections to file and plot
                if method == "Kubios" and artifacts:
                    logging.info("Artifacts detected")
                    # Define artifact types and subspaces based on your information
                    artifact_types = ['ectopic', 'missed', 'extra', 'longshort']
                    subspaces = ['rr', 'drrs', 'mrrs', 's12', 's22']
                    
                    # Get the sample indices for all peaks in 'valid_ppg' timeseries
                    all_peaks_sample_indices = df[df['PPG_Peaks_elgendi'] == 1].index

                    # Initialize columns for artifact types
                    for artifact_type in artifact_types:
                        if artifact_type in artifacts:
                            # Convert artifact peak indices to sample indices
                            # Each peak index from the artifacts dictionary needs to be mapped to the actual sample index
                            artifact_sample_indices = [all_peaks_sample_indices[i-1] for i in artifacts[artifact_type] if i <= len(all_peaks_sample_indices)]

                            # Log the artifact peak indices
                            logging.info(f"Artifact type '{artifact_type}' indices: {artifact_sample_indices}")

                            # Initialize the artifact column to 0
                            df[artifact_type] = 0
                            
                            # Set 1 at the corresponding sample indices to indicate artifacts
                            df.loc[artifact_sample_indices, artifact_type] = 1
                    
                else:
                    logging.info("No artifacts to process.")
            
            sampling_rate = 100  # Hz
            
            # Create subplots with shared x-axes
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                subplot_titles=('Original Peaks', 'R-R Intervals for Original Peaks',
                                                'Kubios Corrected Peaks', 'R-R Intervals for Kubios'
                                                ))
            
            # Function to calculate and plot RR intervals
            def plot_rr_intervals(peaks, valid_ppg, row, markers_color, line_color):
                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / sampling_rate * 1000  # Convert to milliseconds
                    midpoint_samples = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]

                    # Cubic spline interpolation with extrapolation
                    cs = CubicSpline(midpoint_samples, rr_intervals, extrapolate=True)
                    regular_time_axis = np.arange(len(valid_ppg))
                    interpolated_rr = cs(regular_time_axis)

                    # Initialize the RR interval column with NaNs
                    rr_intervals_full = np.full(len(valid_ppg), np.nan)
                    
                    # Assign interpolated RR intervals to the corresponding midpoints
                    for i, midpoint in enumerate(midpoint_samples):
                        rr_intervals_full[midpoint] = rr_intervals[i]

                    # Fill remaining NaN values by interpolation
                    rr_intervals_full = pd.Series(rr_intervals_full).interpolate(method='cubic').to_numpy()

                    # Calculate mean value of the nearest 5 R-R intervals for padding
                    mean_rr_beginning = np.mean(rr_intervals[:5])
                    mean_rr_end = np.mean(rr_intervals[-5:])

                    # Extend interpolation to the beginning of the timeseries with mean value padding
                    rr_intervals_full[:midpoint_samples[0]] = mean_rr_beginning

                    # Extend interpolation to the end of the timeseries with mean value padding
                    rr_intervals_full[midpoint_samples[-1]+1:] = mean_rr_end

                    # Plotting the R-R intervals and the interpolated line
                    fig.add_trace(
                        go.Scatter(x=midpoint_samples, y=rr_intervals, mode='markers', name=f'R-R Intervals (Row {row})', marker=dict(color=markers_color)),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=regular_time_axis, y=rr_intervals_full, mode='lines', name='Interpolated R-R', line=dict(color=line_color)),
                        row=row, col=1
                    )
                    
                    return rr_intervals_full, midpoint_samples  # Return the full interpolated array and midpoint samples
                return np.full(len(valid_ppg), np.nan), []  # Return an array of NaNs if not enough peaks

            # Colors for each type of peaks and their RR intervals
            colors = {
                'PPG_Peaks_elgendi': ('red', '#1f77b4'),  # Original (Blue and Yellow)
                'Peaks_Kubios': ('#e377c2', '#2ca02c'),      # Kubios (Green and Magenta)
            }

            # Define dictionaries for marker shapes and colors
            markers_shapes = {
                'ectopic': 'triangle-up',
                'missed': 'square',
                'extra': 'circle',
                'longshort': 'x'
            }
            markers_colors = {
                'ectopic': 'red',
                'missed': 'blue',
                'extra': 'yellow',
                'longshort': 'purple'
            }

            # Ensure the DataFrame has a column for the interpolated RR intervals
            df['RR_interval_interpolated_Kubios'] = np.nan  # Initialize with NaNs

            # Plot original and corrected PPG data with their peaks and RR intervals
            for i, key in enumerate(['PPG_Peaks_elgendi', 'Peaks_Kubios'], start=1):
                peak_indices = df[df[key] == 1].index
                peak_values = valid_ppg.loc[peak_indices].tolist()

                # Plotting PPG signal and peaks
                fig.add_trace(
                    go.Scatter(x=peak_indices, y=peak_values, mode='markers', name=f'{key} Peaks', marker=dict(color=colors[key][0])),
                    row=i*2-1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=valid_ppg, mode='lines', name=f'{key} PPG', line=dict(color='green'), showlegend=False),
                    row=i*2-1, col=1
                )
                if key == 'PPG_Peaks_elgendi':
                    
                    # Compute and update the RR intervals for PPG_Peaks_elgendi
                    interpolated_rr_elgendi, _ = plot_rr_intervals(valid_peaks, valid_ppg, i*2, 'red', 'blue')
                    df['RR_interval_interpolated'] = interpolated_rr_elgendi
                    logging.info(f"Generated fixed interpolated R-R intervals for PPG_Peaks_elgendi (original peaks, not corrected)")
                    
                elif key == 'Peaks_Kubios':
                    # Compute and update the RR intervals for Peaks_Kubios
                    kubios_peaks = df[df['Peaks_Kubios'] == 1].index.tolist()
                    interpolated_rr_kubios, _ = plot_rr_intervals(kubios_peaks, valid_ppg, i*2, 'green', 'purple')
                    df['RR_interval_interpolated_Kubios'] = interpolated_rr_kubios
                    logging.info(f"Generated fixed interpolated R-R intervals for Peaks_Kubios (Kubios corrected peaks)")
                    
                # Plot artifacts only for Kubios
                if key == 'Peaks_Kubios':
                    
                    # Plot artifacts if they exist and save to the DataFrame
                    artifact_types = ['ectopic', 'missed', 'extra', 'longshort']
                    for artifact_type in artifact_types:
                        if artifact_type in df.columns:
                            artifact_indices = df[df[artifact_type] == 1].index
                            artifact_values = valid_ppg.loc[artifact_indices].tolist()
                            kubios_ectopic_beats = len(df[df['ectopic'] == 1])
                            kubios_missed_beats = len(df[df['missed'] == 1])
                            kubios_extra_beats = len(df[df['extra'] == 1])
                            kubios_longshort_beats = len(df[df['longshort'] == 1])
                            
                            # Add a trace for each artifact type using customized shapes and colors
                            fig.add_trace(
                                go.Scatter(
                                    x=artifact_indices, y=artifact_values,
                                    mode='markers', name=f'{artifact_type} Artifacts',
                                    marker=dict(symbol=markers_shapes[artifact_type], color=markers_colors[artifact_type], size=10)
                                ),
                                row=i*2-1, col=1
                            )

            # Calculate differences for Kubios peaks compared to valid peaks
            kubios_diff_added = set(df[df['Peaks_Kubios'] == 1].index.tolist()) - set(valid_peaks)
            kubios_diff_removed = set(valid_peaks) - set(df[df['Peaks_Kubios'] == 1].index.tolist())
            kubios_total_corrected_peak_count = len(df[df['Peaks_Kubios'] == 1])
            logging.info(f"Kubios peaks added: {kubios_diff_added}")
            logging.info(f"Length of Kubios peaks added: {len(kubios_diff_added)}")
            logging.info(f"Kubios peaks removed: {kubios_diff_removed}")
            logging.info(f"Length of Kubios peaks removed: {len(kubios_diff_removed)}")
            logging.info(f"Total Kubios corrected peaks: {kubios_total_corrected_peak_count}")
            
            # Vertical lines for differing peaks
            vertical_lines = []
            for diff in kubios_diff_added.union(kubios_diff_removed):
                vertical_lines.append(dict(
                    type="line",
                    x0=diff, x1=diff, y0=0, y1=1, yref="paper",
                    line=dict(color="Purple", width=1, dash="dash")
                ))

            # Update axes for all plots
            for i in range(1, 7, 2):
                fig.update_yaxes(title_text="PPG Amplitude", row=i, col=1)
            for i in range(2, 7, 2):
                fig.update_yaxes(title_text="R-R Interval (ms)", row=i, col=1)

            # Calculate the median and IQR for the 'PPG_Clean' column
            median = np.median(df['PPG_Clean'].dropna())
            iqr = np.subtract(*np.percentile(df['PPG_Clean'].dropna(), [75, 25]))

            # Define the button for median range adjustment
            iqr_range = 3 # Number of IQRs to include
            
            adjust_button = dict(
                args=[{
                    "yaxis.range": [median - iqr_range * iqr, median + iqr_range * iqr],
                    #"yaxis2.range": [median - iqr_range * iqr, median + iqr_range * iqr],
                    "yaxis3.range": [median - iqr_range * iqr, median + iqr_range * iqr],
                    #"yaxis4.range": [median - iqr_range * iqr, median + iqr_range * iqr],      
                }],
                label="Median +/- 3 IQR",
                method="relayout"
            )

            # Define the button to reset y-axes to original scale
            reset_button = dict(
                args=[{
                    "yaxis.autorange": True,
                    #"yaxis2.autorange": True,
                    "yaxis3.autorange": True,
                    #"yaxis4.autorange": True,
                }],
                label="Reset Axes",
                method="relayout"
            )

            # Position the button under the legend or at a custom location
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="down",
                        buttons=[adjust_button, reset_button],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=1.05,  # Adjusted to move right
                        xanchor="left",
                        y=0.85,  # Adjusted to move down below the legend
                        yanchor="top"
                    )
                ],
            )
            
            # Update x-axis labels for the bottom plots
            fig.update_xaxes(title_text='Samples', row=4, col=1)

            # Disable y-axis zooming for all subplots
            fig.update_yaxes(fixedrange=False)

            # Update layout and size
            fig.update_layout(
                height=1800, title_text='Comparison of R-Peak Correction Methods',
                shapes=vertical_lines
            )
            
            # Extract subject_id and run_id
            parts = filename.split('_')
            if len(parts) < 4:
                logging.error("Filename does not contain expected parts.")
                return "Error: Filename structure incorrect."

            # Construct the new filename by appending '_corrected' before the file extension
            if filename and not filename.endswith('.tsv.gz'):
                logging.error("Invalid filename format.")
                return "Error: Invalid filename format."
            
            parts = filename.rsplit('.', 2)
            if len(parts) == 3:
                base_name, ext1, ext2 = parts
                ext = f"{ext1}.{ext2}"  # Reassemble the extension
            else:
                # Handle the case where the filename does not have a double extension
                base_name, ext = parts
            
            # Define the new filename for the corrected data and figure
            logging.info(f"Saving nk.signal_fix(peaks) corrected data and figure")
            new_filename = f"{base_name}_corrected_signal_fixpeaks.{ext}"
            figure_filename = f"{base_name}_corrected_subplots_signal_fixpeaks.html"
            full_new_path = os.path.join(save_directory, new_filename)
            figure_filepath = os.path.join(save_directory, figure_filename)

            # Reorder columns to make Rows, Samples, Time, and FD_Upsampled the first four columns
            columns = ['Rows', 'Samples', 'Time', 'FD_Upsampled'] + [col for col in df.columns if col not in ['Rows', 'Samples', 'Time', 'FD_Upsampled']]
            df = df[columns]

            df.to_csv(full_new_path, sep='\t', compression='gzip', index=False)
            pio.write_html(fig, figure_filepath)
            logging.info(f"Saved corrected data to: {full_new_path}")
                        
            # Render initial Plotly figure with the PPG data and peaks
            fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            # Calculate the number of samples modified by Kubios correction
            # here, we want to track the number of samples modfied by kubios correction by comparing the df['RR_interval_interpolated_Kubios'] to df['RR_interval_interpolated'] to see at how many samples do they differ
            kubios_total_samples_corrected = (df['RR_interval_interpolated_Kubios'] != df['RR_interval_interpolated']).sum()
            logging.info(f"Kubios total samples corrected via tachogram interpolation: {kubios_total_samples_corrected}")
            
            # Initialize peak changes data
            peak_changes = {
                'original_total_peak_count': len(initial_peaks), 
                'dash_added_peaks_count': 0, # handled by plot clicks
                'dash_deleted_peaks_count': 0, # handled by plot clicks
                'dash_total_corrected_peak_count': 0, # handled later with len(valid_peaks) final before save out
                'kubios_added_peak_count': len(kubios_diff_added), 
                'kubios_deleted_peak_count': len(kubios_diff_removed), 
                'kubios_total_corrected_peak_count': kubios_total_corrected_peak_count, 
                'kubios_total_samples_corrected': kubios_total_samples_corrected, 
                'artifact_window_censored_total_corrected_peak_count': 0, # handled later before save out by counting the number of peaks outside the artifact windows 
                'artifact_windows_total_count_number': 0, # handled later before save out by counting the number of artifact windows
                'artifact_windows_total_count_samples': 0, # handled later before save out by counting the total number of samples in the artifact windows
                'mean_artifact_window_size_samples': 0, # handled later before save out by calculating the mean size of the artifact windows in samples 
                'std_artifact_window_size_samples': 0, # handled later before save out by calculating the standard deviation of the artifact window sizes in samples
                'min_artifact_window_size_samples': 0, # handled later before save out by calculating the minimum size of the artifact windows in samples
                'max_artifact_window_size_samples': 0, # handled later before save out by calculating the maximum size of the artifact windows in samples
                'kubios_ectopic_beats': kubios_ectopic_beats,	
                'kubios_missed_beats': kubios_missed_beats,	
                'kubios_extra_beats': kubios_extra_beats,	
                'kubios_longshort_beats': kubios_longshort_beats
            }
            
            # BUG: Double peak correction when clicking on plot (R-R interval goes to 0 ms)

            return fig, df.to_json(date_format='iso', orient='split'), valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update, rr_data
        
        # Handling peak correction via plot clicks
        if triggered_id == 'ppg-plot' and clickData:
            clicked_x = clickData['points'][0]['x']
            df = pd.read_json(data_json, orient='split')

            # Handle peak addition
            if clicked_x in valid_peaks:
                logging.info(f"Deleting a peak at sample index: {clicked_x}")
                valid_peaks.remove(clicked_x)
                peak_changes['dash_deleted_peaks_count'] += 1
                
            # Handle peak deletion
            else:
                logging.info(f"Clicked sample index: {clicked_x}")
                peak_indices, _ = find_peaks(valid_ppg)
                nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
                logging.info(f"Adding a new peak at sample index: {nearest_peak}")
                valid_peaks.append(nearest_peak)
                peak_changes['dash_added_peaks_count'] += 1
                valid_peaks.sort()

            # Update the figure with the corrected peaks after each click correction
            fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                ) 
            
            return fig, dash.no_update, valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update, rr_data
        
        # Logic to handle artifact window confirmation
        if triggered_id == 'confirm-artifact-button' and n_clicks_confirm:
            
            # Load the data from JSON
            df = pd.read_json(data_json, orient='split')
            
            # Update the artifact windows store with the new artifact window indices
            new_artifact_window = {'start': artifact_start_idx, 'end': artifact_end_idx}
            artifact_windows.append(new_artifact_window)
            
            # Provide feedback to the user
            artifact_window_output = f"Artifact window confirmed: Start = {artifact_start_idx}, End = {artifact_end_idx}"
            logging.info(f"Artifact window confirmed: Start = {artifact_start_idx}, End = {artifact_end_idx}")
                        
            # Update the figure with the marked artifact window after each confirmation
            if existing_figure:
                fig = go.Figure(existing_figure)
                
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                ) 
                
            # After correcting artifacts
            fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows, artifact_windows = correct_artifacts(df, fig, valid_peaks, valid_ppg, peak_changes, artifact_windows, interpolation_windows)
            
            # Ensure the updated valid_ppg is passed to create_figure
            fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                ) 
            
            # Return updated stores and output & Reset start and end inputs after confirmation
            return fig, dash.no_update, dash.no_update, valid_ppg, dash.no_update, artifact_windows, artifact_window_output, None, None, show_artifact_selection, interpolation_windows, rr_data
        
    # FIXME: More precise handling of errors and error messages. 
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update, dash.no_update]

def correct_artifacts(df, fig, valid_peaks, valid_ppg, peak_changes, artifact_windows, interpolation_windows):
    """
    Corrects artifacts in the PPG data by interpolating over the identified artifact windows,
    using surrounding valid peaks to guide the interpolation process.
    """
    
    logging.info("Starting artifact correction")
   
    # Define sampling rate (down-sampled rate)
    sampling_rate = 100  # Hz
    logging.info(f"Using sampling rate (downsampled): {sampling_rate} Hz")
    
    # Convert valid_ppg back to a pandas Series from a list in dcc.Store
    valid_ppg = pd.Series(valid_ppg, index=range(len(valid_ppg)))

    # The main correction process starts here
    try:
        
        # check if there are any artifact windows to process
        if artifact_windows:
            
            # Focus on the latest specified artifact window
            latest_artifact = artifact_windows[-1]
            
            # check if start and end keys exist
            if 'start' in latest_artifact and 'end' in latest_artifact:
                start, end = latest_artifact['start'], latest_artifact['end']
                logging.info(f"Processing Artifact Window from Boundary Peaks: Start - {start}, End - {end}")
                  
                if start < end:
                    #%% Note: This is where we begin defining the boundaries of the artifact window
                    """
                    This subsection of the correct_artifacts function is responsible for identifying and defining the true interpolation window within an artifact-marked segment of PPG data. 
                    It achieves this by determining the start and end nadirs (lowest points) immediately surrounding an artifact window. 
                    The process involves searching within a limited range before and after the artifact to accurately define the boundaries for interpolation based on these nadirs, 
                    ensuring the artifact correction is based on stable, representative parts of the PPG waveform. Pre and post-artifact windows are also identified for reference.
                    
                    Note variability in the pulse wave morphology with respect to nadirs: 
                    https://www.researchgate.net/publication/317747949_Extended_algorithm_for_real-time_pulse_waveform_segmentation_and_artifact_detection_in_photoplethysmograms/figures?lo=1
                    https://www.jimmynewland.com/wp/research/comparing-ppg-signals-open-vs-closed/
                    
                    """
         
                    # Assuming valid_ppg is a pre-loaded PPG signal array, and start and end are defined
                    num_local_peaks = 5  # Number of peaks to include on either side of the artifact window
                    logging.info(f"Number of local peaks to search: {num_local_peaks}")
                    
                    # Set tolerance for 1st and 3rd derivative crossings (vs. zero)
                    tolerance = 0.00013 # Increase to make more liberal and decrease to make more conservative
                    logging.info(f"Derivative crossing tolerance: {tolerance}") 

                    # Extract the PPG signal segment within the artifact window
                    artifact_segment = valid_ppg[start:end+1]  # Include the endpoint
                    logging.info(f"Artifact segment extracted for interpolation")

                    # Parameters (these need to be tuned based on your specific signal characteristics)
                    # These help prevent the identification of diastolic peaks as P peaks of interest
                    min_height = np.percentile(artifact_segment, 75)  # Only consider peaks above the 75th percentile height
                    min_prominence = np.std(artifact_segment) * 0.5  # Set prominence to be at least half the standard deviation of the signal
                    min_width = 5  # Minimum number of samples over which the peak is wider than at half its prominence

                    # Detect peaks within this segment using scipy's find_peaks for better accuracy
                    peaks_within, properties = find_peaks(artifact_segment, prominence=min_prominence, height=min_height, width=min_width)
                    peaks_within += start  # Adjust indices to match the full signal range
                    logging.info(f"Detected peaks within artifact window: {peaks_within}")
                    
                    # Handle boundaries for pre and post-artifact nadirs (within artifact window)
                    if len(artifact_segment) > 0:
                        logging.info(f"Artifact segment length: {len(artifact_segment)}")
                        
                        # List to store indices of detected potential nadir points and their precise interpolated values
                        nadir_candidates = []
                        interpolated_indices = []

                        # Determine the first peak's index relative to the start of the artifact segment
                        first_peak_index = peaks_within[0] - start if peaks_within.size > 0 else len(artifact_segment)

                        # Ensure the index is within the bounds of the artifact segment
                        first_peak_index = min(first_peak_index, len(artifact_segment) - 1)
                        
                        # Get correct sample index for the first peak within the artifact segment
                        first_peak_sample_index = start + first_peak_index
                        logging.info(f"First peak sample index within artifact segment: {first_peak_sample_index}")
                        
                        first_peak_range = artifact_segment[:first_peak_index]
                        logging.info(f"First peak range: {len(first_peak_range)}")
                        
                        # Calculate first (rate of change) and second (acceleration) derivatives
                        first_derivative = np.gradient(first_peak_range)
                        second_derivative = np.gradient(first_derivative)
                        third_derivative = np.gradient(second_derivative)
                        
                        # Interpolation
                        logging.info(f"Upsampling and interpolating derivatives for the artifact segment")
                        x_original = np.arange(len(first_peak_range))
                        x_interp = np.linspace(0, len(first_peak_range) - 1, num=len(first_peak_range) * 10)  # 10x upsample

                        f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                        first_derivative_interp = f_first_deriv(x_interp)

                        logging.info(f"Searching for derivative crossings up to the first peak within the artifact window at index {first_peak_index}")

                        # Detect local minima in the absolute differences
                        for i in range(1, len(x_interp) - 5):
                            if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
                                
                                # Map the interpolated index back to the original sample index
                                original_index = int(round(x_interp[i])) + start
                                interpolated_indices.append(original_index)

                        # Remove duplicates
                        interpolated_indices = list(set(interpolated_indices))

                        # Determine the closest crossing point to the systolic peak
                        if interpolated_indices:
                            # 'systolic_peak_index' is the index of the systolic peak of interest here called pre_artifact_start
                            pre_peak_nadir = min(interpolated_indices, key=lambda x: abs(x - first_peak_sample_index))
                            logging.info(f"Selected pulse wave end for 'start' peak at index: {pre_peak_nadir}")
                        else:
                            logging.info("No suitable pulse wave start found, fallback to minimum of segment")
                            min_index_in_segment = np.argmin(first_peak_range)
                            logging.info(f"Minimum index in segment: {min_index_in_segment}")
                            pre_peak_nadir = min_index_in_segment + start
                            logging.info(f"Fallback to minimum of segment: Pre-artifact pulse wave start nadir at index: {pre_peak_nadir}")

                        # Create a dataframe from the segment derivatives and original PPG signal
                        segment_derivatives = pd.DataFrame({
                            'PPG_Signal': first_peak_range,
                            'First_Derivative': first_derivative,
                            'Second_Derivative': second_derivative,
                            'Third_Derivative': third_derivative
                        }, index=np.arange(start, start + first_peak_index))  # Setting the index correctly for full data mapping
                        logging.info(f"Created a DataFrame for the segment derivatives")

                        # Save the individual segment derivatives as a raw CSV file
                        segment_derivatives_filename = f'artifact_{start}_{end}_start_{start}_to_nadir_{first_peak_sample_index}.csv'
                        segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                        segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                        logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                        
                        logging.info(f"Plotting the first, second, and third derivatives for the segment before the pre-artifact start peak")
                        
                        # Creating a figure with subplots for PPG waveform and derivative analysis
                        fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                            'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                        ))

                        # Adding traces for derivatives to the first subplot
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                            row=1, col=1
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                            row=1, col=1
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                            row=1, col=1
                        )

                        # Adding scatter plot for the PPG segment to the second subplot
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                            row=2, col=1
                        )

                        # Adding invisible traces for legend entries for crossings
                        fig_derivatives.add_trace(
                            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave Start')
                        )

                        # Adding vertical dashed lines for each crossing point
                        for crossing in interpolated_indices:
                            fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                        # Highlight the crossing closest to the pre_artifact_start
                        closest_crossing = min(interpolated_indices, key=lambda x: abs(x - first_peak_sample_index))
                        fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                        logging.info(f"Added a purple vertical dashed line for the closest crossing to the pre_artifact_start at index: {closest_crossing}")

                        # Adjusting the x-axis ticks
                        fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)
                        
                        # Update layout with specified axis titles
                        fig_derivatives.update_layout(
                            title=f"Analysis of PPG Signal Segment from start {start} to nadir {first_peak_sample_index} and its Derivatives",
                            xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                            xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                            yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                            yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                            legend_title="Trace Types",
                            showlegend=True
                        )

                        # Save the figure as HTML
                        fig_derivatives_filename = f'artifact_{start}_{end}_start_{start}_to_nadir_{first_peak_sample_index}.html'
                        fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                        fig_derivatives.write_html(fig_derivatives_filepath)
                        logging.info(f"Saved the pre_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
        
                        # List to store indices of detected potential nadir points and their precise interpolated values
                        nadir_candidates = []
                        interpolated_indices = []

                        # Determine the last peak's index relative to the start of the artifact segment
                        if peaks_within.size > 0:
                            last_peak_index = peaks_within[-1] - start  # Local index within artifact_segment
                        else:
                            last_peak_index = len(artifact_segment) - 1  # Use the end of the segment if no peaks

                        logging.info(f"Last peak index within artifact segment relative sample index: {last_peak_index}")

                        # Ensure the index is within the bounds of the artifact segment
                        last_peak_index = min(last_peak_index, len(artifact_segment) - 1)

                        # Get correct sample index for the last peak within the artifact segment
                        last_peak_sample_index = start + last_peak_index
                        logging.info(f"Last peak sample index within artifact segment: {last_peak_sample_index}")

                        # Range from the last peak to the end of the segment
                        last_peak_range = artifact_segment[last_peak_index:]
                        logging.info(f"Range from last peak to end of segment: {len(last_peak_range)}")

                        # Calculate first (rate of change) and second (acceleration) derivatives
                        first_derivative = np.gradient(last_peak_range)
                        second_derivative = np.gradient(first_derivative)
                        third_derivative = np.gradient(second_derivative)

                        # Interpolation
                        logging.info(f"Upsampling and interpolating derivatives for the artifact segment")
                        x_original = np.arange(len(last_peak_range))
                        x_interp = np.linspace(0, len(last_peak_range) - 1, num=len(last_peak_range) * 10)  # 10x upsample

                        f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                        first_derivative_interp = f_first_deriv(x_interp)
                        
                        logging.info(f"Searching for derivative crossings from the last peak within the artifact window at index {last_peak_index} to the end of the segment")

                        # Detect local minima in the absolute differences
                        for i in range(5, len(x_interp) - 5):
                            if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
  
                                # Map the interpolated index back to the original sample index
                                original_index = int(round(x_interp[i]))
                                
                                # Ensure the original index is within the bounds of the last_peak_range
                                if original_index < 0 or original_index >= len(last_peak_range):
                                    logging.warning(f"Original index {original_index} is out of bounds, skipping")
                                    continue
                                
                                absolute_index = original_index + last_peak_index  # Translate to the absolute index in the artifact_segment
                                
                                actual_index = absolute_index + start  # Translate to the full data array index

                                interpolated_indices.append(actual_index)
                                
                        # Determine the closest crossing point to the systolic peak
                        if interpolated_indices:
                            # 'systolic_peak_index' is the index of the systolic peak of interest here called pre_artifact_start
                            post_peak_nadir = max(interpolated_indices, key=lambda x: abs(x - last_peak_sample_index))  # Assuming you want the maximum here
                            logging.info(f"Selected pulse wave start for 'end' peak at index: {post_peak_nadir}")
                        else:
                            logging.info("No suitable pulse wave start found, fallback to minimum of segment")
                            min_index_in_segment = np.argmin(last_peak_range)
                            logging.info(f"Minimum index in segment: {min_index_in_segment}")
                            post_peak_nadir = min_index_in_segment + last_peak_sample_index
                            logging.info(f"Fallback to minimum of segment: Post-artifact pulse wave start nadir at index: {post_peak_nadir}")

                        # Correctly setting the index based on the actual length of the last_peak_range
                        index_start = last_peak_sample_index  # This is the correct starting index in the full data array
                        logging.info(f"Correct starting index for the last peak range: {index_start}")
                        index_end = index_start + len(last_peak_range)  # The ending index in the full data array
                        logging.info(f"Correct ending index for the last peak range: {index_end}")

                        # Create a dataframe from the segment derivatives and original PPG signal
                        segment_derivatives = pd.DataFrame({
                            'PPG_Signal': last_peak_range,
                            'First_Derivative': first_derivative,
                            'Second_Derivative': second_derivative,
                            'Third_Derivative': third_derivative
                        }, index=np.arange(index_start, index_end))  # Setting the index correctly for full data mapping
                        logging.info(f"Created a DataFrame for the segment derivatives")

                        # Save the individual segment derivatives as a raw CSV file
                        segment_derivatives_filename = f'artifact_{start}_{end}_start_{start}_to_nadir_{last_peak_sample_index}.csv'
                        segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                        segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                        logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                        
                        logging.info(f"Plotting the first, second, and third derivatives for the segment before the post-artifact start peak")
                        
                        # Creating a figure with subplots for PPG waveform and derivative analysis
                        fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                            'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                        ))

                        # Adding traces for derivatives to the first subplot
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                            row=1, col=1
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                            row=1, col=1
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                            row=1, col=1
                        )

                        # Adding scatter plot for the PPG segment to the second subplot
                        fig_derivatives.add_trace(
                            go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                            row=2, col=1
                        )

                        # Adding invisible traces for legend entries for crossings
                        fig_derivatives.add_trace(
                            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                        )
                        fig_derivatives.add_trace(
                            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave Start')
                        )

                        # Adding vertical dashed lines for each crossing point
                        for crossing in interpolated_indices:
                            fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)
                            
                        # Highlight the crossing closest to the pre_artifact_start
                        closest_crossing = max(interpolated_indices, key=lambda x: abs(x - last_peak_sample_index))
                        fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                        logging.info(f"Added a purple vertical dashed line for the closest crossing to the 'end' at index: {closest_crossing}")

                        # Adjusting the x-axis ticks
                        fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)

                        # Update layout with specified axis titles
                        fig_derivatives.update_layout(
                            title=f"Analysis of PPG Signal Segment from start {start} to nadir {last_peak_sample_index} and its Derivatives",
                            xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                            xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                            yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                            yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                            legend_title="Trace Types",
                            showlegend=True
                        )
                        
                        # Save the figure as HTML
                        fig_derivatives_filename = f'artifact_{start}_{end}_start_{start}_to_nadir_{last_peak_sample_index}.html'
                        fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                        fig_derivatives.write_html(fig_derivatives_filepath)
                        logging.info(f"Saved the pre_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
                    else:
                        # Fallback to using start and end if no peaks are detected
                        pre_peak_nadir = start
                        post_peak_nadir = end
                        logging.warning("No peaks detected within the artifact window, using boundary indices as nadirs.")

                    true_start = pre_peak_nadir
                    logging.info(f"True start of interpolation window (pre-nadir): {true_start}")
                    true_end = post_peak_nadir
                    logging.info(f"True end of interpolation window (post-nadir): {true_end}")

                    # Update the artifact dictionary to record these values for samples corrected logging
                    latest_artifact['start'] = true_start
                    latest_artifact['end'] = true_end
                    
                    # Here we updated the peak_changes dictionary to reflect the current artifact window information
                    #//peak_changes['artifact_windows_total_count_number'] += 1
                    logging.info(f"Added artifact window to the total corrected artifact window count")
                    #//peak_changes['artifact_windows_total_count_samples'] += true_end - true_start + 1
                    logging.info(f"Added artifact window samples to the total corrected artifact window sample count")
                    
                    """Altogether this code above marks the artifact window from ppg waveform nadir to nadir, 
                    derived from the initial manual identification of the artifact window boundaries."""     
                   
                    #%% Note: these are various methods for calculating the expected length of the artifact window taking into account 0-based indexing and 1-based indexing 
                    # Used in logging and debugging
                         
                    # Calculate the expected length of the artifact window
                    interpolated_length = true_end - true_start + 1
                    logging.info(f"Expected interpolated length: {interpolated_length} samples")
                    
                    # Calculate the duration of the artifact window in seconds
                    artifact_duration = (true_end - true_start) / sampling_rate
                    logging.info(f"Artifact window duration: {artifact_duration} seconds")
                    
                    """Some of this code below was rendered redudant in the final implementation,
                    but basically here we are indexing the valid peaks surrounding the artifact window
                    to calculate local r-r interval statistics and begin deriving the average heartbeat template for interpolation."""
                    
                    """
                    Re: start_peak_idx
                    
                    Objective: 
                    The purpose here is to find a reference point in valid_peaks relative to the manually selected start index, which is understood as the index of a peak near the start of an artifact.
                    Here we are idenfying the peak index in valid_peaks that is closest to the manually selected start index (NOT the sample index of the peak itself).
                    
                    Behavior:
                    If start coincides directly with one of the peaks in valid_peaks (which it does), bisect_left returns the index of this exact peak.              
                    """
                    
                    # Find the index of the peak immediately preceding the 'start' of the artifact window in 'valid_peaks'
                    # This is effectively the first peak that was manually selected as the artifact window boundary
                    start_peak_idx = bisect.bisect_left(valid_peaks, start)
                    logging.info(f"Start peak index (via bisect_left): {start_peak_idx}")
                    
                    # Ensure that the start_peak_idx is not less than 0 (start of the data)
                    start_peak_idx = max(start_peak_idx, 0)
                    logging.info(f"Adjusted start peak index (via max(start_peak_idx)): {start_peak_idx}")
                    
                    """
                    Re: end_peak_idx
                    
                    Objective: 
                    The goal is to locate a reference point in valid_peaks relative to the manually selected end index, which is a peak defining the end boundary of the artifact.
                    As with start_peak_idx, we are identifying the peak index in valid_peaks that is closest to the manually selected end index, not the sample index of these reference points.
                    
                    Behavior:
                    If end coincides exactly with a peak in valid_peaks (which it does), bisect_right returns the index right after this peak 
                    (since bisect_right gives the position just after any existing entries of end).
                    Subtracting 1 aligns it back to include end if it is a peak, or the last peak before end if end is between peaks.
                    This approach ensures that end is included in the interpolation range.
                    """
                    
                    # Find the index of the peak immediately following the 'end' of the artifact window in 'valid_peaks'
                    # Subtract 1 to get the last peak that's within the end of the artifact window
                    end_peak_idx = bisect.bisect_right(valid_peaks, end) - 1
                    logging.info(f"End peak index (via bisect_right): {end_peak_idx}")  
                    
                    # Ensure that the end_peak_idx is not beyond the last index of valid_peaks (end of the data)
                    end_peak_idx = min(end_peak_idx, len(valid_peaks) - 1)
                    logging.info(f"Adjusted end peak index (via min(end_peak_idx)): {end_peak_idx}")
                    
                    """ 
                    again note that we are working with peak indices here, not sample indices. 
                    we add 1 to the start_peak_idx and subtract 1 from the end_peak_index in order 
                    to position the index to the peak num_local_peaks away from the artifact start and end peaks,
                    ensuring a total of 5 peaks on either side of artifact window (including the boundary peaks).
                    """
                    
                    # Determine the index for the start of the pre-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact start peak
                    # We add 1 because bisect_left gives us the index of the artifact start peak itself
                    pre_artifact_start_idx = max(0, start_peak_idx - num_local_peaks + 1)
                    logging.info(f"Pre artifact start index (peak index): {pre_artifact_start_idx}")

                    # Determine the actual sample number for the start of the pre-artifact window based on pre_artifact_start_idx
                    pre_artifact_start = valid_peaks[pre_artifact_start_idx] if pre_artifact_start_idx < len(valid_peaks) else 0
                    logging.info(f"Pre artifact start peak (sample index): {pre_artifact_start}")

                    pre_artifact_search_peak_idx = pre_artifact_start_idx - 1
                    logging.info(f"Pre artifact search peak index: {pre_artifact_search_peak_idx}")
                    #FIXME: Make sure this doesn't go out of bounds or does it not matter given conversion below?
                    
                    pre_artifact_search_peak = valid_peaks[pre_artifact_search_peak_idx] if pre_artifact_search_peak_idx >= 0 else 0
                    logging.info(f"Pre artifact search peak (sample index): {pre_artifact_search_peak}")

                    # The end of the pre-artifact window is the same as the start of the artifact window
                    pre_artifact_end = true_start
                    logging.info(f"Pre artifact end peak (sample index) {pre_artifact_end} marked as the start of the artifact window") 

                    # Find the nadir (lowest point) before the pre-artifact window using a robust derivative-based approach
                    if pre_artifact_start_idx > 0:
                        # Handle normal case where a peak is before the pre_artifact_start
                        start_search_segment = valid_ppg[pre_artifact_search_peak:pre_artifact_start]
                        logging.info(f"Normal case: Searching for potential minima before the pre_artifact start peak")
                        logging.info(f"Length of search segment: {len(start_search_segment)}")

                        if len(start_search_segment) > 0:
                            # Calculate first (rate of change), second (acceleration), and third derivatives
                            first_derivative = np.gradient(start_search_segment)
                            second_derivative = np.gradient(first_derivative)
                            third_derivative = np.gradient(second_derivative)

                            # Interpolation
                            x_original = np.arange(len(start_search_segment))
                            x_interp = np.linspace(0, len(start_search_segment) - 1, num=len(start_search_segment) * 10)  # 10x upsample

                            f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                            first_derivative_interp = f_first_deriv(x_interp)

                            # List to store indices of detected potential nadir points and their precise interpolated values
                            nadir_candidates = []
                            interpolated_indices = []

                            logging.info("Starting search for pulse wave start nadir candidates")

                            # Detect local minima in the absolute differences
                            for i in range(1, len(x_interp) - 5):
                                if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
  
                                    # Map the interpolated index back to the original sample index
                                    original_index = int(round(x_interp[i]))
                                    actual_index = original_index + pre_artifact_search_peak
                                    interpolated_indices.append(actual_index)

                            # Remove duplicates
                            interpolated_indices = list(set(interpolated_indices))

                            # Determine the closest crossing point to the systolic peak
                            if interpolated_indices:
                                # 'systolic_peak_index' is the index of the systolic peak of interest here called pre_artifact_start
                                pre_artifact_nadir = min(interpolated_indices, key=lambda x: abs(x - pre_artifact_start))
                                logging.info(f"Selected pulse wave start at index: {pre_artifact_nadir} closest to the systolic peak at index {pre_artifact_start}")
                            else:
                                logging.info("No suitable pulse wave start found, fallback to minimum of segment")
                                min_index_in_segment = np.argmin(start_search_segment)
                                logging.info(f"Minimum index in segment: {min_index_in_segment}")
                                pre_artifact_nadir = min_index_in_segment + pre_artifact_search_peak
                                logging.info(f"Fallback to minimum of segment: Pre-artifact pulse wave start nadir at index: {pre_artifact_nadir}")

                            # Create a dataframe from the segment derivatives and original PPG signal
                            segment_derivatives = pd.DataFrame({
                                'PPG_Signal': start_search_segment,
                                'First_Derivative': first_derivative,
                                'Second_Derivative': second_derivative,
                                'Third_Derivative': third_derivative
                            }, index=np.arange(pre_artifact_search_peak, pre_artifact_start))  # Setting the index correctly for full data mapping
                            logging.info(f"Created a DataFrame for the segment derivatives")

                            # Save the individual segment derivatives as a raw CSV file
                            segment_derivatives_filename = f'artifact_{start}_{end}_pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start}.csv'
                            segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                            segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                            logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                            
                            logging.info(f"Plotting the first, second, and third derivatives for the segment before the pre-artifact start peak")
                            
                            # Creating a figure with subplots for PPG waveform and derivative analysis
                            fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                                'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                            ))

                            # Adding traces for derivatives to the first subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                                row=1, col=1
                            )

                            # Adding scatter plot for the PPG segment to the second subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                                row=2, col=1
                            )

                            # Adding invisible traces for legend entries for crossings
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave Start')
                            )

                            # Adding vertical dashed lines for each crossing point
                            for crossing in interpolated_indices:
                                fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                            # Highlight the crossing closest to the pre_artifact_start
                            closest_crossing = min(interpolated_indices, key=lambda x: abs(x - pre_artifact_start))
                            fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at index: {closest_crossing}")

                            # Adjusting the x-axis ticks
                            fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)

                            # Update layout with specified axis titles
                            fig_derivatives.update_layout(
                                title=f"Analysis of PPG Signal Segment pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start} and its Derivatives",
                                xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                                xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                                yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                                yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                                legend_title="Trace Types",
                                showlegend=True
                            )
                            
                            # Save the figure as HTML
                            fig_derivatives_filename = f'artifact_{start}_{end}_pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start}.html'
                            fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                            fig_derivatives.write_html(fig_derivatives_filepath)
                            logging.info(f"Saved the pre_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
            
                    else:  
                        # Handle edge case where no peak is before the pre_artifact_start
                        start_search_segment = valid_ppg[:pre_artifact_start]
                        logging.info(f"Edge case: Searching for potential minima before the pre_artifact start pulse wave peak")
                        logging.info(f"Length of search segment: {len(start_search_segment)}")
                        if len(start_search_segment) > 0:
                            
                            # Calculate first (rate of change), second (acceleration), and third derivatives
                            first_derivative = np.gradient(start_search_segment)
                            second_derivative = np.gradient(first_derivative)
                            third_derivative = np.gradient(second_derivative)

                            # Interpolation
                            x_original = np.arange(len(start_search_segment))
                            x_interp = np.linspace(0, len(start_search_segment) - 1, num=len(start_search_segment) * 10)  # 10x upsample

                            f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                            first_derivative_interp = f_first_deriv(x_interp)

                            # List to store indices of detected potential nadir points and their precise interpolated values
                            nadir_candidates = []
                            interpolated_indices = []

                            logging.info("Starting search for pulse wave start nadir candidates")

                            # Detect local minima in the absolute differences
                            for i in range(1, len(x_interp) - 5):
                                if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
     
                                    # Map the interpolated index back to the original sample index
                                    original_index = int(round(x_interp[i]))
                                    actual_index = original_index + pre_artifact_search_peak
                                    interpolated_indices.append(actual_index)

                            # Remove duplicates
                            interpolated_indices = list(set(interpolated_indices))

                            # Determine the closest crossing point to the systolic peak
                            if interpolated_indices:
                                # 'systolic_peak_index' is the index of the systolic peak of interest here called pre_artifact_start
                                pre_artifact_nadir = min(interpolated_indices, key=lambda x: abs(x - pre_artifact_start))
                                logging.info(f"Selected pulse wave start at index: {pre_artifact_nadir} closest to the systolic peak at index {pre_artifact_start}")
                            else:
                                logging.info("No suitable pulse wave start found, fallback to minimum of segment")
                                min_index_in_segment = np.argmin(start_search_segment)
                                logging.info(f"Minimum index in segment: {min_index_in_segment}")
                                pre_artifact_nadir = min_index_in_segment + pre_artifact_search_peak
                                logging.info(f"Fallback to minimum of segment: Pre-artifact pulse wave start nadir at index: {pre_artifact_nadir}")

                            # Create a dataframe from the segment derivatives and original PPG signal
                            segment_derivatives = pd.DataFrame({
                                'PPG_Signal': start_search_segment,
                                'First_Derivative': first_derivative,
                                'Second_Derivative': second_derivative,
                                'Third_Derivative': third_derivative
                            }, index=np.arange(pre_artifact_search_peak, pre_artifact_start))  # Setting the index correctly for full data mapping
                            logging.info(f"Created a DataFrame for the segment derivatives")

                            # Save the individual segment derivatives as a raw CSV file
                            segment_derivatives_filename = f'artifact_{start}_{end}_pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start}.csv'
                            segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                            segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                            logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                            
                            logging.info(f"Plotting the first, second, and third derivatives for the segment before the pre-artifact start peak")
                            
                            # Creating a figure with subplots for PPG waveform and derivative analysis
                            fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                                'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                            ))

                            # Adding traces for derivatives to the first subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                                row=1, col=1
                            )

                            # Adding scatter plot for the PPG segment to the second subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                                row=2, col=1
                            )

                            # Adding invisible traces for legend entries for crossings
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave Start')
                            )

                            # Adding vertical dashed lines for each crossing point
                            for crossing in interpolated_indices:
                                fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                            # Highlight the crossing closest to the pre_artifact_start
                            closest_crossing = min(interpolated_indices, key=lambda x: abs(x - pre_artifact_start))
                            fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at index: {closest_crossing}")

                            # Adjusting the x-axis ticks
                            fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)

                            # Update layout with specified axis titles
                            fig_derivatives.update_layout(
                                title=f"Analysis of PPG Signal Segment pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start} and its Derivatives",
                                xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                                xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                                yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                                yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                                legend_title="Trace Types",
                                showlegend=True
                            )

                            # Save the figure as HTML
                            fig_derivatives_filename = f'artifact_{start}_{end}_pre_artifact_window_derivatives_{pre_artifact_start}_to_{true_start}.html'
                            fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                            fig_derivatives.write_html(fig_derivatives_filepath)
                            logging.info(f"Saved the pre_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
                        else:
                            # Default to 0 if no peak is before the pre_artifact_start
                            pre_artifact_nadir = 0
                            logging.warning("Segment is empty, no inflection points to detect, defaulting to 0.")
   
                    # The start of the post-artifact window is the same as the end of the artifact window
                    post_artifact_start = true_end
                    logging.info(f"Post artifact start (sample index) {post_artifact_start} marked as the end of the artifact window {true_end}")  

                    # Calculate the index for the end of the post-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact end peak
                    # We subtract 1 because bisect_right gives us the index of the peak after the artifact end peak
                    post_artifact_end_idx = min(end_peak_idx + num_local_peaks - 1, len(valid_peaks) - 1)
                    logging.info(f"Post artifact end index (peak index): {post_artifact_end_idx}")

                    # Determine the actual sample number for the end of the post-artifact window based on post_artifact_end_idx
                    post_artifact_end = valid_peaks[post_artifact_end_idx] if post_artifact_end_idx >= 0 else len(valid_ppg)
                    logging.info(f"Post artifact end (sample index): {post_artifact_end}")
                    
                    post_artifact_search_peak_idx = post_artifact_end_idx + 1
                    logging.info(f"Post artifact search peak index: {post_artifact_search_peak_idx}")

                    post_artifact_search_peak = valid_peaks[post_artifact_search_peak_idx] if post_artifact_search_peak_idx < len(valid_peaks) else len(valid_ppg) - 1
                    # If post_artifact_end_idx is the last peak index, then post_artifact_search_peak_idx will be out of bounds
                    # In that case, we set post_artifact_search_peak to the last index of the valid PPG signal
                    logging.info(f"Post artifact search peak (sample index): {post_artifact_search_peak}")
                    
                    # Find the nadir (lowest point) after the post-artifact window using a robust derivative-based approach
                    if post_artifact_end_idx < len(valid_peaks) - 1:
                        # Handle normal case where a peak is after the post_artifact_start
                        end_search_segment = valid_ppg[post_artifact_end:post_artifact_search_peak]
                        logging.info(f"Normal case: Searching for potential minima after the post_artifact end peak")
                        logging.info(f"Length of search segment: {len(end_search_segment)}")

                        if len(end_search_segment) > 0:
                            # Calculate first (rate of change), second (acceleration), and third derivatives
                            first_derivative = np.gradient(end_search_segment)
                            second_derivative = np.gradient(first_derivative)
                            third_derivative = np.gradient(second_derivative)

                            # Interpolation
                            x_original = np.arange(len(end_search_segment))
                            x_interp = np.linspace(0, len(end_search_segment) - 1, num=len(end_search_segment) * 10)  # 10x upsample

                            f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                            first_derivative_interp = f_first_deriv(x_interp)

                            # List to store indices of detected potential nadir points and their precise interpolated values
                            nadir_candidates = []
                            interpolated_indices = []

                            logging.info("Starting search for pulse wave start nadir candidates")

                            # Detect local minima in the absolute differences
                            for i in range(1, len(x_interp) - 5):
                                if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
          
                                    # Map the interpolated index back to the original sample index
                                    original_index = int(round(x_interp[i]))
                                    actual_index = original_index + post_artifact_end
                                    interpolated_indices.append(actual_index)

                            # Remove duplicates
                            interpolated_indices = list(set(interpolated_indices))

                            # Determine the closest crossing point to the previous systolic peak
                            if interpolated_indices:
                                # 'systolic_peak_index' is the index of the systolic peak of interest here called post_artifact_end
                                post_artifact_nadir = min(interpolated_indices, key=lambda x: abs(x - post_artifact_search_peak))
                                logging.info(f"Selected pulse wave end at index: {post_artifact_nadir} closest to the systolic peak at index {post_artifact_end}")
                            else:
                                logging.info("No suitable pulse wave end found, fallback to minimum of segment")
                                min_index_in_segment = np.argmin(end_search_segment)
                                logging.info(f"Minimum index in segment: {min_index_in_segment}")
                                post_artifact_nadir = min_index_in_segment + post_artifact_search_peak
                                logging.info(f"Fallback to minimum of segment: Post-artifact pulse wave end nadir at index: {post_artifact_nadir}")

                            # Create a dataframe from the segment derivatives and original PPG signal
                            segment_derivatives = pd.DataFrame({
                                'PPG_Signal': end_search_segment,
                                'First_Derivative': first_derivative,
                                'Second_Derivative': second_derivative,
                                'Third_Derivative': third_derivative
                            }, index=np.arange(post_artifact_end, post_artifact_search_peak))  # Setting the index correctly for full data mapping
                            logging.info(f"Created a DataFrame for the segment derivatives")

                            # Save the individual segment derivatives as a raw CSV file
                            segment_derivatives_filename = f'artifact_{start}_{end}_post_artifact_window_derivatives_{true_end}_to_{post_artifact_end}.csv'
                            segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                            segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                            logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                            
                            logging.info(f"Plotting the first, second, and third derivatives for the segment before the pre-artifact start peak")
                            
                            # Creating a figure with subplots for PPG waveform and derivative analysis
                            fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                                'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                            ))

                            # Adding traces for derivatives to the first subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                                row=1, col=1
                            )

                            # Adding scatter plot for the PPG segment to the second subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                                row=2, col=1
                            )

                            # Adding invisible traces for legend entries for crossings
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave End')
                            )

                            # Adding vertical dashed lines for each crossing point
                            for crossing in interpolated_indices:
                                fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                            # Highlight the crossing closest to the pre_artifact_start
                            closest_crossing = min(interpolated_indices, key=lambda x: abs(x - post_artifact_search_peak))
                            fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at index: {closest_crossing}")

                            # Adjusting the x-axis ticks
                            fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)

                            # Update layout with specified axis titles
                            fig_derivatives.update_layout(
                                title=f"Analysis of PPG Signal Segment post_artifact_window_derivatives_{true_end}_to_{post_artifact_end} and its Derivatives",
                                xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                                xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                                yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                                yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                                legend_title="Trace Types",
                                showlegend=True
                            )
                            
                            # Save the figure as HTML
                            fig_derivatives_filename = f'artifact_{start}_{end}_post_artifact_window_derivatives_{true_end}_to_{post_artifact_end}.html'
                            fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                            fig_derivatives.write_html(fig_derivatives_filepath)
                            logging.info(f"Saved the post_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
            
                    else:  
                        # Handle edge case where no peak is after the post_artifact_end
                        end_search_segment = valid_ppg[post_artifact_end:]
                        logging.info(f"Edge case: Searching for potential minima after the post_artifact_end pulse wave peak")
                        logging.info(f"Length of search segment: {len(end_search_segment)}")
                        
                        """
                        BUG:
                        working but potential bug due to ppg wave form morphology at end, current code could misidentify diacrotic notch in some cases
                        """
                        
                        if len(end_search_segment) > 0:
                            # Calculate first (rate of change), second (acceleration), and third derivatives
                            first_derivative = np.gradient(end_search_segment)
                            second_derivative = np.gradient(first_derivative)
                            third_derivative = np.gradient(second_derivative)

                            # Interpolation
                            x_original = np.arange(len(end_search_segment))
                            x_interp = np.linspace(0, len(end_search_segment) - 1, num=len(end_search_segment) * 10)  # 10x upsample

                            f_first_deriv = interp1d(x_original, first_derivative, kind='cubic')
                            first_derivative_interp = f_first_deriv(x_interp)
                            
                            # List to store indices of detected potential nadir points and their precise interpolated values
                            nadir_candidates = []
                            interpolated_indices = []

                            logging.info("Starting search for pulse wave end nadir candidates")

                            # Detect local minima in the absolute differences
                            for i in range(1, len(x_interp) - 5):
                                if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
                 
                                    # Map the interpolated index back to the original sample index
                                    original_index = int(round(x_interp[i]))
                                    actual_index = original_index + post_artifact_end
                                    interpolated_indices.append(actual_index)

                            # Remove duplicates
                            interpolated_indices = list(set(interpolated_indices))

                            # Determine the closest crossing point to the previous systolic peak
                            if interpolated_indices:
                                # 'systolic_peak_index' is the index of the systolic peak of interest here called post_artifact_end
                                post_artifact_nadir = min(interpolated_indices, key=lambda x: abs(x - post_artifact_end))
                                logging.info(f"Selected pulse wave end at index: {post_artifact_nadir} closest to the systolic peak at index {post_artifact_end}")
                            else:
                                logging.info("No suitable pulse wave end found, fallback to minimum of segment")
                                min_index_in_segment = np.argmin(end_search_segment)
                                logging.info(f"Minimum index in segment: {min_index_in_segment}")
                                post_artifact_nadir = min_index_in_segment + post_artifact_search_peak
                                logging.info(f"Fallback to minimum of segment: Post-artifact pulse wave end nadir at index: {post_artifact_nadir}")

                            # Correcting the DataFrame creation
                            index_range = np.arange(post_artifact_end, post_artifact_search_peak + 1)  # Including the last index
                            if len(index_range) != len(end_search_segment):
                                logging.error(f"Index range length {len(index_range)} does not match segment length {len(end_search_segment)}")
                            else:
                                segment_derivatives = pd.DataFrame({
                                    'PPG_Signal': end_search_segment,
                                    'First_Derivative': first_derivative,
                                    'Second_Derivative': second_derivative,
                                    'Third_Derivative': third_derivative
                                }, index=index_range)  # Correct index setup
                                logging.info("Created a DataFrame for the segment derivatives with correct indexing")

                            # Save the individual segment derivatives as a raw CSV file
                            segment_derivatives_filename = f'artifact_{start}_{end}_post_artifact_window_derivatives_{true_end}_to_{post_artifact_end}.csv'
                            segment_derivatives_filepath = os.path.join(save_directory, segment_derivatives_filename)
                            segment_derivatives.to_csv(segment_derivatives_filepath, index=True, index_label='Sample_Indices')
                            logging.info(f"Saved the individual segment derivatives as a raw CSV file.")
                            
                            logging.info(f"Plotting the first, second, and third derivatives for the segment before the pre-artifact start peak")
                            
                            # Creating a figure with subplots for PPG waveform and derivative analysis
                            fig_derivatives = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
                                'Derivatives of PPG Signal Segment', 'PPG Signal Segment'
                            ))

                            # Adding traces for derivatives to the first subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['First_Derivative'], mode='lines', name='1st Derivative (Rate of Change)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Second_Derivative'], mode='lines', name='2nd Derivative (Acceleration)'),
                                row=1, col=1
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['Third_Derivative'], mode='lines', name='3rd Derivative (Jerk)'),
                                row=1, col=1
                            )

                            # Adding scatter plot for the PPG segment to the second subplot
                            fig_derivatives.add_trace(
                                go.Scatter(x=segment_derivatives.index, y=segment_derivatives['PPG_Signal'], mode='lines', line=dict(color='black'), name='PPG Segment'),
                                row=2, col=1
                            )

                            # Adding invisible traces for legend entries for crossings
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='gray'), name='Zero Crossing')
                            )
                            fig_derivatives.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='purple'), name='Pulse Wave End')
                            )

                            # Adding vertical dashed lines for each crossing point
                            for crossing in interpolated_indices:
                                fig_derivatives.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                            # Highlight the crossing closest to the pre_artifact_start
                            closest_crossing = min(interpolated_indices, key=lambda x: abs(x - post_artifact_end))
                            fig_derivatives.add_vline(x=closest_crossing, line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at index: {closest_crossing}")

                            # Adjusting the x-axis ticks
                            fig_derivatives.update_xaxes(tick0=segment_derivatives.index.min(), dtick=5)

                            # Update layout with specified axis titles
                            fig_derivatives.update_layout(
                                title=f"Analysis of PPG Signal Segment post_artifact_window_derivatives_{true_end}_to_{post_artifact_end} and its Derivatives",
                                xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                                xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                                yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                                yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                                legend_title="Trace Types",
                                showlegend=True
                            )
                            
                            # Save the figure as HTML
                            fig_derivatives_filename = f'artifact_{start}_{end}_post_artifact_window_derivatives_{true_end}_to_{post_artifact_end}.html'
                            fig_derivatives_filepath = os.path.join(save_directory, fig_derivatives_filename)
                            fig_derivatives.write_html(fig_derivatives_filepath)
                            logging.info(f"Saved the pre_artifact_window derivatives plot as an HTML file at {fig_derivatives_filepath}")
               
                    # Adjust the start of the pre-artifact window to the nadir to include the full waveform
                    pre_artifact_start = pre_artifact_nadir
                    logging.info(f"Extended pre_artifact window: Start nadir (sample index) = {pre_artifact_start}, End nadir (i.e., interpolation start point) (sample index) = {pre_artifact_end}")

                    # Adjust the end of the post-artifact window to the nadir to include the full waveform
                    post_artifact_end = post_artifact_nadir
                    logging.info(f"Extended post_artifact window: Start nadir (i.e., interpolation end point) (sample index) = {post_artifact_start}, End nadir (sample index) = {post_artifact_end}")

                    # Update interpolation_windows with pre and post artifact ranges
                    # Note: This is used for displaying the artifact window in red and the surrounding windows in green
                    interpolation_windows.append({'pre_artifact': (pre_artifact_start, pre_artifact_end),
                                                'post_artifact': (post_artifact_start, post_artifact_end)})
                    logging.info(f"Interpolation windows successfully appended: {interpolation_windows}")

                    #%% Here we begin the next phase of artifact correction, which involves sampling heartbeats for creating the average beat template
                    
                    # Ensure valid_peaks is a NumPy array
                    valid_peaks = np.array(valid_peaks)

                    # Ensure pre_artifact_start and post_artifact_end are single integer values of the respective sample indices to avoid type mismatches or errors in indexing
                    pre_artifact_start = int(pre_artifact_start)
                    post_artifact_end = int(post_artifact_end)

                    # Identify and compile into array pre and post artifact peaks based on extended windows including nadirs (peak indices from valid_peaks)
                    pre_artifact_peaks_indices = np.where((valid_peaks >= pre_artifact_start) & (valid_peaks < pre_artifact_end))[0]
                    logging.info(f"Pre artifact peaks indices: {pre_artifact_peaks_indices}")
                    post_artifact_peaks_indices = np.where((valid_peaks > post_artifact_start) & (valid_peaks <= post_artifact_end))[0]
                    logging.info(f"Post artifact peaks indices: {post_artifact_peaks_indices}")

                    # Extract the pre and post artifact peaks sample indices using the calculated indices
                    pre_artifact_peaks = valid_peaks[pre_artifact_peaks_indices]
                    logging.info(f"Pre artifact peaks sample indices: {pre_artifact_peaks}")
                    post_artifact_peaks = valid_peaks[post_artifact_peaks_indices]
                    logging.info(f"Post artifact peaks sample indices: {post_artifact_peaks}")

                    # Ensure 'valid_peaks' is an array of integers as safeguard against type inconsistencies or errors in indexing
                    valid_peaks = np.array(valid_peaks, dtype=int)

                    #%% NOTE: This is where we begin sampling heartbeats for creating the average beat template
                    
                    """Combines the pre- and post-artifact peak sample indices into a single array to get a comprehensive set of clean peaks around the artifact for reference signal analysis."""
                
                    # Concatenate pre- and post-artifact peaks to get the clean peaks around the artifact
                    clean_peaks = np.concatenate((pre_artifact_peaks, post_artifact_peaks))
                    logging.info(f"Clean peaks around the artifact window: {clean_peaks}")
                    
                    # Ensure clean_peaks are sorted; they correspond one-to-one with the heartbeats in the dictionary
                    clean_peaks.sort()
                    logging.info(f"Sorted clean_peaks: {clean_peaks}")

                    """Initially using neurokit2 to segment the PPG signal into heartbeats using the clean peaks, 
                    see https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ppg/ppg_segment.html#ppg_segment for more information.
                    
                    # Segment the PPG signal into heartbeats using the clean peaks, Returns a dictionary containing DataFrames for all segmented heartbeats.
                    heartbeats = nk.ppg_segment(ppg_cleaned=valid_ppg, 
                                                peaks=clean_peaks, 
                                                sampling_rate=sampling_rate, 
                                                show=False)
                    logging.info(f"Segmented heartbeats within clean peaks using nk.ppg_segment function.")
                    
                    There is bug with last segmented heartbeat, so we add an extra valid peak index to the post_artifact_peaks array
                    # Add an extra valid peak index to the post_artifact_peaks array
                    NOTE: The last beat problem likely involves this portion of nk.ppg_segment code:
                    # pad last heartbeat with nan so that segments are equal length
                    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
                    after_last_index = heartbeats[last_heartbeat_key]["Index"] < len(ppg_cleaned)
                    heartbeats[last_heartbeat_key].loc[after_last_index, "Signal"] = np.nan
                    
                    Further, in many cases nk.ppg_segment was not able to segment the heartbeats correctly, 
                    so we decided to implement our own custom functions for this purpose.
                    """
                    def find_nadirs(ppg_signal, peaks, valid_peaks, start, end, true_start, true_end, pre_artifact_nadir, post_artifact_nadir):
                        logging.info("Calculating derivatives for the PPG signal.")
                        first_derivative = np.gradient(ppg_signal)
                        second_derivative = np.gradient(first_derivative)
                        third_derivative = np.gradient(second_derivative)

                        nadirs = []  # Initialize an empty list to store the nadirs
                        all_crossings = []  # To store all crossings for plotting or further analysis
                        closest_crossings = []  # List to store the closest crossings for each peak
                        
                        # Use derivatives to find precise nadir points around peaks
                        for i, peak in enumerate(peaks):  # Iterate over each peak
                            logging.info(f"Processing peak at index {peak}")
                            skip_normal_nadir_processing = False  # Flag to control flow

                            # Handling for peaks at artifact boundaries
                            if peak == start:
                                post_peak_nadir = true_start # Correct usage for artifact start boundary
                                post_crossings = [post_peak_nadir]
                                pre_peak_nadir, pre_crossings = find_derivative_crossing(ppg_signal, peaks[i-1], peak, first_derivative, third_derivative)
                                logging.info(f"Peak index {peak} = Artifact Start, Setting post_peak_nadir to true_start {true_start}.")
                                logging.info(f"Calculated pre-peak nadir {pre_peak_nadir} for peak at index {peak} with post-peak nadir {post_peak_nadir}.")
                                skip_normal_nadir_processing = True  # Set flag
                                
                            if peak == end:
                                pre_peak_nadir = true_end # Correct usage for artifact end boundary
                                pre_crossings = [pre_peak_nadir]
                                post_peak_nadir, post_crossings = find_derivative_crossing(ppg_signal, peak, peaks[i+1], first_derivative, third_derivative)
                                logging.info(f"Peak index {peak} = Artifact End, Setting pre_peak nadir to true_end {true_end}.")
                                logging.info(f"Calculated post-peak nadir {post_peak_nadir} for peak at index {peak} with pre-peak nadir {pre_peak_nadir}.")
                                skip_normal_nadir_processing = True  # Set flag
                                
                            # Handle normal cases if not a special artifact boundary
                            if not skip_normal_nadir_processing:
                                # Special handling for the first peak
                                if i == 0:
                                    pre_peak_nadir = pre_artifact_nadir
                                    pre_crossings = [pre_peak_nadir]
                                    logging.info(f"Using pre-artifact nadir {pre_artifact_nadir} as pre-peak nadir for peak at index {peak}")
                                    post_peak_nadir, post_crossings = find_derivative_crossing(ppg_signal, peak, peaks[i+1], first_derivative, third_derivative)
                                    logging.info(f"Calculated post-peak nadir for peak at index {peak}: {post_peak_nadir}")
                                    skip_normal_nadir_processing = True  # Set flag
                                else:
                                    # Normal case for peaks in the middle
                                    pre_peak_nadir, pre_crossings = find_derivative_crossing(ppg_signal, peaks[i-1], peak, first_derivative, third_derivative)
                                    logging.info(f"Calculated pre-peak nadir for peak at index {peak}: {pre_peak_nadir}")

                                # Special handling for the last peak
                                if i == len(peaks) - 1:
                                    post_peak_nadir = post_artifact_nadir
                                    post_crossings = [post_peak_nadir]
                                    logging.info(f"Using post-artifact nadir as post-peak nadir for peak at index {peak}: {post_peak_nadir}")
                                    pre_peak_nadir, pre_crossings = find_derivative_crossing(ppg_signal, peaks[i-1], peak, first_derivative, third_derivative)
                                    logging.info(f"Calculated pre-peak nadir for peak at index {peak}: {pre_peak_nadir}")
                                    skip_normal_nadir_processing = True  # Set flag
                                else:
                                    # Normal case for peaks in the middle
                                    post_peak_nadir, post_crossings = find_derivative_crossing(ppg_signal, peak, peaks[i+1], first_derivative, third_derivative)
                                    logging.info(f"Calculated post-peak nadir for peak at index {peak}: {post_peak_nadir}")
                                
                            # Append the nadir points if they are outside the artifact window or exactly on the boundary
                            nadirs.append((pre_peak_nadir, peak, post_peak_nadir))
                            logging.info(f"Appended nadir points, pre_peak_nadir {pre_peak_nadir} for peak at index {peak} with post_peak_nadir {post_peak_nadir} to the list.")

                            # Explicitly ensuring both are lists to avoid numpy array operations
                            if pre_crossings is None:
                                pre_crossings = []
                            if post_crossings is None:
                                post_crossings = []

                            logging.info(f"Checking pre-crossings for peak at index {peak}: {pre_crossings}")
                            logging.info(f"Checking post-crossings for peak at index {peak}: {post_crossings}")
                            
                            # Appending crossings as lists to ensure no broadcasting attempts by numpy
                            all_crossings.extend(list(pre_crossings) + list(post_crossings))
                            logging.info(f"Added crossings for peak at index {peak} to the list.")
                            
                            # Calculate closest crossings, handling cases where no crossings are found
                            closest_pre = min(pre_crossings, key=lambda x: abs(x - peaks[i-1]), default=None) if len(pre_crossings) > 0 else None
                            logging.info(f"Calculated closest pre crossings for peak at index {peak}: {closest_pre}")   
                            closest_post = min(post_crossings, key=lambda x: abs(x - peaks[i+1]), default=None) if i < len(peaks) - 1 and len(post_crossings) > 0 else None
                            logging.info(f"Calculated closest post crossings for peak at index {peak}: {closest_post}")
                            closest_crossings.append((closest_pre, closest_post))
                            logging.info(f"Appended closest crossings for peak at index {peak} to the list.")
        
                        return nadirs, all_crossings, closest_crossings

                    def find_derivative_crossing(signal, start_idx, end_idx, first_derivative, third_derivative):
                        """Find the index where first and third derivatives indicate a nadir between two peaks."""
                        # Extract the relevant segment of the signal
                        segment_length = end_idx - start_idx
                        x_original = np.arange(segment_length)

                        # Interpolation
                        logging.info(f"Upsampling and interpolating the first and third derivatives for the segment between peaks {start_idx} and {end_idx}")
                        x_interp = np.linspace(0, segment_length - 1, num=segment_length * 10)  # 10x upsample

                        f_first_deriv = interp1d(x_original, first_derivative[start_idx:end_idx], kind='cubic')
                        first_derivative_interp = f_first_deriv(x_interp)

                        # Detect local minima in the absolute differences
                        crossings = []
                        for i in range(1, len(x_interp) - 5):
                            if first_derivative_interp[i] * first_derivative_interp[i + 1] < 0:  # Check for sign change
         
                                # Map the interpolated index back to the original sample index
                                original_index = int(round(x_interp[i]))
                                actual_index = original_index + start_idx
                                crossings.append(actual_index)

                        # Remove duplicates
                        crossings = list(set(crossings))
                        
                        # Choose the crossing point closest to the end peak as the nadir
                        logging.info(f"Choosing the crossing point closest to the end peak as the nadir.")
                        closest_crossing = crossings[np.argmin(np.abs(np.array(crossings) - end_idx))] if len(crossings) > 0 else np.argmin(signal[start_idx:end_idx]) + start_idx
                        logging.info(f"Selected the closest crossing to the end peak: {closest_crossing}")
                        
                        return closest_crossing, crossings
                        
                    def plot_and_save_heartbeat(heartbeat_segment, all_crossings, segment_label, save_directory, peak, valid_peaks):
                        
                        """Plot and save the derivatives and signal data for a heartbeat segment."""
                        # Extract pre_peak_nadir and post_peak_nadir from segment_label
                        parts = segment_label.split('_')
                        segment_number = parts[0]  # This is 'i+1' part, if neededs
                        pre_peak_nadir = int(parts[1])
                        post_peak_nadir = int(parts[2])
                        start_idx = 0
                        end_idx = 0

                        # Continue with utilizing pre_peak_nadir and post_peak_nadir
                        logging.info(f"Segment number: {segment_number}, Segment label: {segment_label}, Pre-Nadir: {pre_peak_nadir}, Peak: {peak}, Post-Nadir: {post_peak_nadir}")

                        # Find the index of the current peak in the valid peaks list
                        current_peak_index = np.where(valid_peaks == peak)[0][0] if peak in valid_peaks else -1
                        logging.info(f"Processing plot and save for peak at index {current_peak_index} for peak {peak}.")
                        
                        if current_peak_index != -1:
                            
                            # Extend the index ranges to include the peak before and the peak after within the bounds of the valid_ppg signal

                            if peak == start:
                                # Handle edge case of 'start' where we are not interested in next peak across artifact window but
                                # are interested in the last valid peak
                                start_idx = valid_peaks[current_peak_index - 1]
                                end_idx = true_start
                                logging.info(f"Edge case: Start index for the extended segment: {start_idx}")
                                logging.info(f"Edge case: End index for the extended segment is true_start: {end_idx}")

                            elif peak == end:
                                # Handle edge case of 'end' where we are not interested in peak across artifact window but
                                # are interested in the next valid peak
                                start_idx = true_end
                                end_idx = valid_peaks[current_peak_index + 1]
                                logging.info(f"Edge case: Start index for the extended segment set to true_end: {start_idx}")
                                logging.info(f"Edge case: End index for the extended segment: {end_idx}")
                                
                            elif current_peak_index == 0:
                                # If current_peak_index points to the first valid peak, set start index to the first element in valid_ppg
                                start_idx = 0
                                end_idx = valid_peaks[current_peak_index + 1]
                                logging.info(f"Edge case: Start index for the extended segment is start of timeseries: {start_idx}")
                                logging.info(f"Edge case: End index for the extended segment: {end_idx}")
                                
                            elif current_peak_index == len(valid_peaks) - 1:
                                # If current_peak_index points to the last valid peak, set end index to the last element in valid_ppg
                                start_idx = valid_peaks[current_peak_index - 1]
                                end_idx = len(valid_ppg) - 1
                                logging.info(f"Edge case: Start index for the extended segment: {start_idx}")
                                logging.info(f"Edge case: End index for the extended segment is end of timeseries: {end_idx}")

                            elif current_peak_index + 1 < len(valid_peaks):
                                # If not the last peak, set idx to index of the surrounding valid peaks
                                start_idx = valid_peaks[current_peak_index - 1]
                                end_idx = valid_peaks[current_peak_index + 1]
                                logging.info(f"Normal case: Start index for the extended segment: {start_idx}")
                                logging.info(f"Normal case: End index for the extended segment is next peak: {end_idx}")
                            else:
                                # Safety fallback in case none of the conditions are met
                                start_idx = 0
                                end_idx = len(valid_ppg) - 1
                                logging.error(f"Unusual case: Setting fallback start to {start} and end index to end of timeseries: {end_idx}")
    
                            # Filter the PPG signal to the new range
                            extended_segment = valid_ppg[start_idx:end_idx + 1]
                            logging.info(f"Filtered the heartbeat segment to the extended range with length {len(extended_segment)}.")

                            # Calculate derivatives
                            first_derivative = np.gradient(extended_segment)
                            second_derivative = np.gradient(first_derivative)
                            third_derivative = np.gradient(second_derivative)

                            # Filter crossings that are within the current segment's index range
                            relevant_crossings = [cross for cross in all_crossings if start_idx <= cross <= end_idx]
                            logging.info(f"Filtered relevant crossings within the segment range: {relevant_crossings}")

                            # Identify the closest crossing relevant to this segment
                            relevant_closest_crossing = min(relevant_crossings, key=lambda x: abs(x - peak), default=None)
                            logging.info(f"Identified the closest crossing to the peak: {relevant_closest_crossing}")

                            # Create DataFrame for the extended segment
                            index_range = np.arange(start_idx, end_idx + 1)
                            segment_derivatives = pd.DataFrame({
                                'PPG_Signal': extended_segment,
                                'First_Derivative': first_derivative,
                                'Second_Derivative': second_derivative,
                                'Third_Derivative': third_derivative
                            }, index=index_range)
                            
                            logging.info(f"Segment label: {segment_label}")
                            
                            # Save the DataFrame to CSV
                            csv_filename = f'artifact_{start}_{end}_heartbeat_segment_{segment_label}.csv'
                            csv_filepath = os.path.join(save_directory, csv_filename)
                            segment_derivatives.to_csv(csv_filepath, index=True, index_label='Sample Indices')
                            logging.info(f"Saved the segment derivatives as a CSV file at {csv_filepath}")

                            logging.info(f"Plotting the first, second, and third derivatives for the segment around peak {peak}")
                            
                            # Plotting the segment
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                                subplot_titles=('Derivatives of PPG Signal Segment', 'PPG Signal Segment'))

                            # Add derivative traces
                            fig.add_trace(go.Scatter(x=index_range, y=first_derivative, mode='lines', name='1st Derivative'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=index_range, y=second_derivative, mode='lines', name='2nd Derivative'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=index_range, y=third_derivative, mode='lines', name='3rd Derivative'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=index_range, y=extended_segment, mode='lines', name='PPG Segment', line=dict(color='black')), row=2, col=1)

                            # Adding invisible traces for legend entries for crossings and peaks
                            fig.add_trace(
                                go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='gray'), name='Zero Crossing')
                            )
                            fig.add_trace(
                                go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='purple'), name='Pulse Wave')
                            )
                              
                            peak_pos = peak - start_idx
                            logging.info(f"Peak position in the segment: {peak_pos}")
                            pre_peak_nadir_pos = pre_peak_nadir - start_idx
                            logging.info(f"Pre-nadir position in the segment: {pre_peak_nadir_pos}")
                            post_peak_nadir_pos = post_peak_nadir - start_idx
                            logging.info(f"Post-nadir position in the segment: {post_peak_nadir_pos}")
                            
                            # Mark the peak position
                            fig.add_trace(go.Scatter(
                                x=[index_range[peak_pos]], 
                                y=[extended_segment[peak_pos] if isinstance(extended_segment, list) else extended_segment.iloc[peak_pos]], 
                                mode='markers', 
                                marker=dict(color='red', size=8), 
                                name='Peak'),
                                row=2, col=1
                            )
                            logging.info(f"Marked the peak {peak} position in the plot.")
                            
                            # Adding vertical dashed lines for each crossing point
                            for crossing in relevant_crossings:
                                fig.add_vline(x=crossing, line=dict(color="gray", dash="dash"), line_width=1)

                            # Marking pre and post nadirs
                            fig.add_vline(x=index_range[pre_peak_nadir_pos], line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at pre-nadir index: {pre_peak_nadir}")

                            fig.add_vline(x=index_range[post_peak_nadir_pos], line=dict(color="purple", dash="dash"), line_width=2)
                            logging.info(f"Added purple vertical dashed line at post-nadir index: {post_peak_nadir}")
                            
                            # Update layout and save as HTML
                            html_filename = f'artifact_{start}_{end}_heartbeat_segment_{segment_label}.html'
                            html_filepath = os.path.join(save_directory, html_filename)
                            
                            # Update layout with specified axis titles
                            fig.update_layout(
                                title=f'Analysis of Heartbeat Segment {segment_label}',
                                xaxis_title='',  # This sets the X-axis title for the shared x-axis at the bottom
                                xaxis2_title='Sample Index',  # Explicitly setting the X-axis title for the second subplot
                                yaxis_title='Derivative Values',  # Y-axis title for the first subplot
                                yaxis2_title='PPG - Volts',  # Y-axis title for the second subplot (row=2, col=1)
                                legend_title="Trace Types",
                                showlegend=True
                            )
                            
                            fig.write_html(html_filepath)
                            logging.info(f"Saved the plot as an HTML file at {html_filepath}")
                        else:
                            logging.error("Peak not found in valid_peaks array.")

                        return csv_filepath, html_filepath

                    def segment_heartbeats(ppg_signal, peaks, nadirs, save_directory, valid_peaks, all_crossings, closest_crossings):
                        heartbeats = {}
                        logging.info(f"Found all crossings for the heartbeats.")

                        for i, (pre_peak_nadir, peak, post_peak_nadir) in enumerate(nadirs):
                            logging.info(f"Processing heartbeat {i + 1} of {len(nadirs)} around peak {peak}")
                            logging.info(f"Pre-peak nadir: {pre_peak_nadir}, Peak: {peak}, Post-peak nadir: {post_peak_nadir}")
                            if pre_peak_nadir < post_peak_nadir:
                                logging.info(f"Valid nadir indices for segmenting heartbeat {i + 1} around peak {peak}")
                                heartbeat_segment = pd.DataFrame({
                                    'Signal': ppg_signal[pre_peak_nadir:post_peak_nadir + 1],
                                    'Index': np.arange(pre_peak_nadir, post_peak_nadir + 1)
                                })
                                heartbeats[str(i + 1)] = heartbeat_segment
                                logging.info(f"Segmented heartbeat {i + 1} from {pre_peak_nadir} to {post_peak_nadir} around peak {peak}")
                                
                                plot_and_save_heartbeat(heartbeat_segment, all_crossings, f'{i+1}_{pre_peak_nadir}_{post_peak_nadir}', save_directory, peak, valid_peaks)
                            else:
                                logging.warning(f"Skipped segmenting heartbeat at peak {peak} due to invalid nadir indices.")
                                
                        logging.info(f"Segmented {len(heartbeats)} heartbeats.")

                        return heartbeats

                    # Initial call to find_nadirs
                    nadirs, all_crossings, closest_crossings = find_nadirs(valid_ppg, clean_peaks, valid_peaks, start, end, true_start, true_end, pre_artifact_nadir, post_artifact_nadir)
                    # Call to segment_heartbeats with all needed data
                    heartbeats = segment_heartbeats(valid_ppg, clean_peaks, nadirs, save_directory, valid_peaks, all_crossings, closest_crossings)
                    
                    # Check the structure of the heartbeats dictionary
                    logging.info(f"Heartbeats dictionary keys: {list(heartbeats.keys())}")

                    # Initialize a list to hold all segmented heartbeats
                    segmented_heartbeats = []
                    valid_keys = []
                    logging.info("Initialized list to hold segmented heartbeats.")

                    # First, extract and store each heartbeat segment
                    logging.info("Extracting heartbeat segments for averaging.")
                    for i, peak_index in enumerate(clean_peaks):
                        key = str(i + 1)  # 1-based indexing for keys
                        logging.info(f"Accessing the heartbeat using the key: {key}")
                        logging.info(f"Processing peak index: {peak_index}")
                        
                        if key in heartbeats:
                            heartbeat = heartbeats[key].copy()
                            logging.info(f"Copying the heartbeat key {key} for processing.")
                            
                            # Rename 'Signal' to 'PPG_Values'
                            heartbeat.rename(columns={'Signal': 'PPG_Values'}, inplace=True)

                            # Set 'Index' as the DataFrame index
                            heartbeat.set_index('Index', inplace=True)

                            # Check for NaNs in 'PPG_Values' and only proceed if the segment is clean
                            if not heartbeat['PPG_Values'].isna().any():
                                valid_keys.append(key)
                                logging.info(f"Valid heartbeat segment copied.")
                                
                                segmented_heartbeats.append(heartbeat['PPG_Values'].values)
                                logging.info(f"Appended the segmented heartbeat from key {key} to the segmented heartbeat list.")
                                
                                # Save the individual segmented heartbeat as a raw CSV file
                                heartbeat_filename_raw = f'artifact_{start}_{end}_heartbeat_{key}_raw.csv'
                                heartbeat_filepath_raw = os.path.join(save_directory, heartbeat_filename_raw)
                                heartbeat.to_csv(heartbeat_filepath_raw, index=True, index_label='Sample_Indices')
                                logging.info(f"Saved the individual heartbeat segment as a raw CSV file.")
                            else:
                                logging.warning(f"Heartbeat segment {key} contains NaN values and will be skipped.")
                                
                        else:
                            logging.warning(f"Heartbeat segment {key} not found in the data.")

                    # Plotting individual segments
                    for i, key in enumerate(valid_keys):
                        heartbeat_values = segmented_heartbeats[i]  # Get PPG values for current key
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(heartbeat_values)),  # Assuming x-axis is just the index of values
                            y=heartbeat_values,
                            mode='lines',
                            name=f'Heartbeat {key}'
                        ))
                        
                        fig.update_layout(
                            title=f"Individual Heartbeat {key}",
                            xaxis_title='Sample Index',
                            yaxis_title='PPG Amplitude',
                        )
                        # Save the figure as an HTML file
                        heartbeat_plot_filename_raw = f'artifact_{start}_{end}_heartbeat_{key}_raw.html'
                        heartbeat_plot_filepath_raw = os.path.join(save_directory, heartbeat_plot_filename_raw)
                        fig.write_html(heartbeat_plot_filepath_raw)

                    # Function to align heartbeats based on three key points: start, peak, and end
                    def align_heartbeats_by_key_points(heartbeats, nadirs):
                        logging.info("Starting alignment of heartbeats by key points.")
                        
                        aligned_heartbeats = []

                        max_length_before_peak = 0
                        max_length_peak_to_end = 0

                        # Determine the lengths needed for each segment
                        for i, heartbeat in enumerate(heartbeats):
                            beat_start, peak, beat_end = nadirs[i]
                            max_length_before_peak = max(max_length_before_peak, peak - beat_start)
                            max_length_peak_to_end = max(max_length_peak_to_end, beat_end - peak)
                            
                        logging.info(f"Max length before peak: {max_length_before_peak}")
                        logging.info(f"Max length peak to end: {max_length_peak_to_end}")

                        # Align each heartbeat by stretching/compressing and padding
                        for i, heartbeat in enumerate(heartbeats):
                            logging.info(f"Aligning heartbeat {i+1} by key points.")
                            beat_start, peak, beat_end = nadirs[i]
                            logging.info(f"Heartbeat {i+1}: Start={beat_start}, Peak={peak}, End={beat_end}")
                            logging.info(f"Original length of heartbeat {i+1}: {len(heartbeat)}")

                            # Adjust indices to start from zero relative to each segment
                            local_start = beat_start - beat_start
                            local_peak = peak - beat_start
                            local_end = beat_end - beat_start
                            logging.info(f"Aligned start index from {beat_start} to {local_start}")
                            logging.info(f"Aligned peak index from {peak} to {local_peak}")
                            logging.info(f"Aligned end index from {beat_end} to {local_end}")

                            # Extract segments
                            segment_before_peak = heartbeat[:local_peak+1]
                            logging.info(f"Segment before peak: {len(segment_before_peak)} samples")
                            segment_peak_to_end = heartbeat[local_peak:local_end+1]
                            logging.info(f"Segment peak to end: {len(segment_peak_to_end)} samples")

                            # Create new x-axis for interpolation
                            x_before = np.linspace(0, len(segment_before_peak) - 1, max_length_before_peak + 1)
                            logging.info(f"New x-axis for before peak segment: {len(x_before)} samples")
                            
                            if len(segment_peak_to_end) > 1:
                                x_peak_to_end = np.linspace(0, len(segment_peak_to_end) - 1, max_length_peak_to_end + 1)
                                logging.info(f"New x-axis for peak to end segment: {len(x_peak_to_end)} samples")

                            # Interpolate before peak segment to the desired length
                            if len(segment_before_peak) > 1:
                                interp_before_peak = interp1d(np.arange(len(segment_before_peak)), segment_before_peak, kind='cubic')
                                logging.info(f"Interpolating before peak segment.")
                                aligned_before_peak = interp_before_peak(x_before)
                            else:
                                aligned_before_peak = np.pad(segment_before_peak, (0, max_length_before_peak - len(segment_before_peak) + 1), 'edge')
                                logging.warning(f"Segment before peak for heartbeat {i+1} is too short to interpolate, using padding.")

                            # Interpolate peak to end segment to the desired length
                            if len(segment_peak_to_end) > 1:
                                interp_peak_to_end = interp1d(np.arange(len(segment_peak_to_end)), segment_peak_to_end, kind='cubic')
                                logging.info(f"Interpolating peak to end segment.")
                                aligned_peak_to_end = interp_peak_to_end(x_peak_to_end)
                            else:
                                aligned_peak_to_end = np.pad(segment_peak_to_end, (0, max_length_peak_to_end - len(segment_peak_to_end) + 1), 'edge')
                                logging.warning(f"Segment peak to end for heartbeat {i+1} is too short to interpolate, using padding.")
                            
                            # Concatenate aligned segments
                            aligned_heartbeat = np.concatenate((aligned_before_peak, aligned_peak_to_end[1:]))  # Avoid duplicate peak point

                            aligned_heartbeats.append(aligned_heartbeat)
                            logging.info(f"Aligned heartbeat {i+1} to length {len(aligned_heartbeat)}")

                        # Determine the final length for padding/trimming
                        final_length = max([len(heartbeat) for heartbeat in aligned_heartbeats])
                        logging.info(f"Final length for padding/trimming: {final_length}")

                        # Pad or trim the aligned heartbeats to ensure they all have the same length
                        padded_heartbeats = []
                        for i, heartbeat in enumerate(aligned_heartbeats):
                            if len(heartbeat) < final_length:
                                pad_length = final_length - len(heartbeat)
                                logging.info(f"Heartbeat {i+1} needs padding of {pad_length} samples.")
                                pad_after = np.full(pad_length, heartbeat[-1])  # Pad with the last value
                                logging.info(f"Padded after segment: {len(pad_after)} samples")
                                padded_heartbeat = np.concatenate((heartbeat, pad_after))
                                logging.info(f"Padded heartbeat {i+1} to final length {len(padded_heartbeat)}")
                            else:
                                padded_heartbeat = heartbeat[:final_length]
                                logging.info(f"Trimmed heartbeat {i+1} to final length {len(padded_heartbeat)}")
                            
                            padded_heartbeats.append(padded_heartbeat)

                        logging.info("Completed alignment and padding of heartbeats.")
                        return padded_heartbeats
                    
                    aligned_heartbeats = align_heartbeats_by_key_points(segmented_heartbeats, nadirs)
                    logging.info(f"Aligned all heartbeats by key points.")
                    
                    # Replace original list with aligned heartbeats for mean and median calculation
                    segmented_heartbeats = aligned_heartbeats

                    # Calculate the mean and median heartbeat waveforms
                    mean_heartbeat = np.mean(segmented_heartbeats, axis=0)
                    logging.info(f"Mean heartbeat length: {len(mean_heartbeat)}")
                    median_heartbeat = np.median(segmented_heartbeats, axis=0)
                    
                    # Smooth median heartbeat using Savitzky-Golay filter
                    median_heartbeat_smoothed = savgol_filter(median_heartbeat, window_length=10, polyorder=3) # To increase smoothing, increase window_length and polyorder
                    median_heartbeat = median_heartbeat_smoothed
                    logging.info(f"Smoothed median heartbeat length: {len(median_heartbeat)}")
                    
                    # Plot mean heartbeat waveform
                    fig_mean = go.Figure()
                    fig_mean.add_trace(go.Scatter(
                        x=np.arange(len(mean_heartbeat)),
                        y=mean_heartbeat,
                        mode='lines',
                        name='Mean Heartbeat'
                    ))
                    fig_mean.update_layout(
                        title="Mean Heartbeat Waveform",
                        xaxis_title='Sample Index',
                        yaxis_title='PPG Amplitude'
                    )
                    # Save the figure as HTML
                    fig_mean_filename_raw = f'artifact_{start}_{end}_mean_heartbeat_raw.html'
                    fig_mean_filepath_raw = os.path.join(save_directory, fig_mean_filename_raw)
                    fig_mean.write_html(fig_mean_filepath_raw)
                    logging.info(f"Saved the mean heartbeat segment plot as an HTML file.")
                    
                    # Plot median heartbeat waveform
                    fig_median = go.Figure()
                    fig_median.add_trace(go.Scatter(
                        x=np.arange(len(median_heartbeat)),
                        y=median_heartbeat,
                        mode='lines',
                        name='Median Heartbeat'
                    ))
                    fig_median.update_layout(
                        title="Median Heartbeat Waveform (Smoothed)",
                        xaxis_title='Sample Index',
                        yaxis_title='PPG Amplitude'
                    )
                    # Save the figure as HTML
                    fig_median_filename_raw = f'artifact_{start}_{end}_median_heartbeat_raw.html'
                    fig_median_filepath_raw = os.path.join(save_directory, fig_median_filename_raw)
                    fig_median.write_html(fig_median_filepath_raw)
                    logging.info(f"Saved the median heartbeat segment plot as an HTML file.")
                        
                    # Calculate the first derivative of the mean and median heartbeat signal using np.gradient
                    mean_heartbeat_first_derivative = np.gradient(mean_heartbeat)
                    median_heartbeat_first_derivative = np.gradient(median_heartbeat)

                    # Smooth median heartbeat using Savitzky-Golay filter
                    median_heartbeat_first_derivative_smoothed = savgol_filter(median_heartbeat_first_derivative, window_length=10, polyorder=3) # To increase smoothing, increase window_length and polyorder
                    median_heartbeat_first_derivative = median_heartbeat_first_derivative_smoothed

                    # Calculate the first derivative of the mean and median heartbeat signal using np.gradient
                    mean_heartbeat_first_derivative = np.gradient(mean_heartbeat)
                    median_heartbeat_first_derivative = np.gradient(median_heartbeat)

                    # Average the first derivatives
                    average_first_derivative = (mean_heartbeat_first_derivative + median_heartbeat_first_derivative) / 2

                    # Function to find the first sign change before and after the peak with a tolerance
                    def find_sign_change_boundaries(average_first_derivative, peak_idx, tolerance=3):
                        # Initialize start and end indices
                        start_idx = None
                        end_idx = None

                        # Find the first sign change before the peak
                        for i in range(peak_idx - 5, tolerance, -1):
                            if average_first_derivative[i] * average_first_derivative[i - tolerance] < 0:
                                start_idx = i
                                logging.info(f"Found the first sign change before the peak at index {peak_idx}")
                                break

                        # Find the last sign change after the peak
                        for i in range(peak_idx + 5, len(average_first_derivative) - tolerance):
                            if average_first_derivative[i] * average_first_derivative[i + tolerance] < 0:
                                end_idx = i
                                logging.info(f"Found the last sign change after the peak at index {peak_idx}")
                                break

                        # Default to first and last sample if no crossings are found
                        if start_idx is None:
                            start_idx = 0
                            logging.info(f"Defaulting to the first sample for start index.")
                        if end_idx is None:
                            end_idx = len(average_first_derivative) - 1
                            logging.info(f"Defaulting to the last sample for end index.")

                        return start_idx, end_idx

                    # Find the peak index for mean and median heartbeats
                    mean_peak_idx = np.argmax(mean_heartbeat)
                    median_peak_idx = np.argmax(median_heartbeat)
                    combined_peak_idx = (mean_peak_idx + median_peak_idx) // 2  # Approximate combined peak index

                    # Find the start and end indices for the averaged first derivative
                    combined_start_idx, combined_end_idx = find_sign_change_boundaries(average_first_derivative, combined_peak_idx)

                    # Log the results
                    logging.info(f"Combined Heartbeat: Start Index={combined_start_idx}, Peak Index={combined_peak_idx}, End Index={combined_end_idx}")

                    # Trim the mean and median heartbeats based on the detected boundaries
                    trimmed_mean_heartbeat = mean_heartbeat[combined_start_idx:combined_end_idx + 1]
                    trimmed_median_heartbeat = median_heartbeat[combined_start_idx:combined_end_idx + 1]

                    # Log the lengths of the trimmed heartbeats
                    logging.info(f"Trimmed Mean Heartbeat Length: {len(trimmed_mean_heartbeat)}")
                    logging.info(f"Trimmed Median Heartbeat Length: {len(trimmed_median_heartbeat)}")


                    # Plot mean heartbeat derivative waveform
                    fig_mean_derivative = go.Figure()
                    fig_mean_derivative.add_trace(go.Scatter(
                        x=np.arange(len(mean_heartbeat_first_derivative)),
                        y=mean_heartbeat_first_derivative,
                        mode='lines',
                        name='Mean Heartbeat Derivative'
                    ))
                    fig_mean_derivative.update_layout(
                        title="Mean Heartbeat Derivative",
                        xaxis_title='Sample Index',
                        yaxis_title='Rate of PPG Amplitude Change per Sample'
                    )
                    # Save the figure as HTML
                    fig_mean_derivative_filename_raw = f'artifact_{start}_{end}_mean_derivative_heartbeat_raw.html'
                    fig_mean_derivative_filepath_raw = os.path.join(save_directory, fig_mean_derivative_filename_raw)
                    fig_mean_derivative.write_html(fig_mean_derivative_filepath_raw)
                    logging.info(f"Saved the mean derivative heartbeat segment plot as an HTML file.")
                    
                    # Plot median heartbeat derivative waveform
                    fig_median_derivative = go.Figure()
                    fig_median_derivative.add_trace(go.Scatter(
                        x=np.arange(len(median_heartbeat_first_derivative)),
                        y=median_heartbeat_first_derivative,
                        mode='lines',
                        name='Median Heartbeat Derivative'
                    ))
                    fig_median_derivative.update_layout(
                        title="Median Heartbeat Derivative",
                        xaxis_title='Sample Index',
                        yaxis_title='Rate of PPG Amplitude Change per Sample'
                    )
                    # Save the figure as HTML
                    fig_median_derivative_filename_raw = f'artifact_{start}_{end}_median_derivative_heartbeat_raw.html'
                    fig_median_derivative_filepath_raw = os.path.join(save_directory, fig_median_derivative_filename_raw)
                    fig_median_derivative.write_html(fig_median_derivative_filepath_raw)
                    logging.info(f"Saved the median derivative heartbeat segment plot as an HTML file.")
                    
                    # Plot median/median heartbeat average derivative waveform
                    fig_average_first_derivative = go.Figure()
                    fig_average_first_derivative.add_trace(go.Scatter(
                        x=np.arange(len(average_first_derivative)),
                        y=average_first_derivative,
                        mode='lines',
                        name='Average Mean/Median Combined Derivative'
                    ))
                    fig_average_first_derivative.update_layout(
                        title="Average Derivative",
                        xaxis_title='Sample Index',
                        yaxis_title='Rate of PPG Amplitude Change per Sample'
                    )
                    # Save the figure as HTML
                    fig_average_first_derivative_filename_raw = f'artifact_{start}_{end}_average_first_derivative_heartbeat_raw.html'
                    fig_average_first_derivative_filepath_raw = os.path.join(save_directory, fig_average_first_derivative_filename_raw)
                    fig_average_first_derivative.write_html(fig_average_first_derivative_filepath_raw)
                    logging.info(f"Saved the average_first_derivative heartbeat segment plot as an HTML file.")

                    #%% Plotting the derivative analysis
                    
                    # Create a figure with subplots
                    logging.info("Creating a figure with subplots for PPG waveform and derivative analysis.")
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=(
                        'Mean Heartbeat Waveform', 'Mean Heartbeat Derivative',
                        'Median Heartbeat Waveform', 'Median Heartbeat Derivative'
                    ))

                    # Add mean and median heartbeat waveforms and derivatives
                    fig.add_trace(go.Scatter(x=np.arange(len(mean_heartbeat)), y=mean_heartbeat, mode='lines', name='Mean Heartbeat'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(mean_heartbeat_first_derivative)), y=mean_heartbeat_first_derivative, mode='lines', name='Mean Derivative'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(median_heartbeat)), y=median_heartbeat, mode='lines', name='Median Heartbeat'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(median_heartbeat_first_derivative)), y=median_heartbeat_first_derivative, mode='lines', name='Median Derivative'), row=4, col=1)

                    # Determine the y-axis range for the derivative subplots
                    derivative_y_min = min(np.min(mean_heartbeat_first_derivative), np.min(median_heartbeat_first_derivative))
                    derivative_y_max = max(np.max(mean_heartbeat_first_derivative), np.max(median_heartbeat_first_derivative))

                    # Apply padding to y-axis range for the derivative subplots
                    padding = (derivative_y_max - derivative_y_min) * 0.1
                    derivative_y_min -= padding
                    derivative_y_max += padding

                    # Update y-axis for the derivative subplots with the new range
                    fig.update_yaxes(title_text='PPG Amplitude', row=1, col=1)
                    fig.update_yaxes(title_text='Derivative Amplitude', range=[derivative_y_min, derivative_y_max], row=2, col=1)
                    fig.update_yaxes(title_text='PPG Amplitude', row=3, col=1)
                    fig.update_yaxes(title_text='Derivative Amplitude', range=[derivative_y_min, derivative_y_max], row=4, col=1)

                    # Function to add vertical lines across all subplots
                    def add_global_vertical_line(fig, x, row, col, name, color):
                        # Determine global y range for vertical lines based on waveform and derivative subplots
                        global_y_min_waveform = min(np.min(mean_heartbeat), np.min(median_heartbeat))
                        global_y_max_waveform = max(np.max(mean_heartbeat), np.max(median_heartbeat))
                        global_y_min_derivative = derivative_y_min
                        global_y_max_derivative = derivative_y_max

                        # Add lines across waveform subplots
                        fig.add_trace(go.Scatter(
                            x=[x, x], y=[global_y_min_waveform, global_y_max_waveform],
                            mode='lines', line=dict(color=color, width=2, dash='dot'),
                            showlegend=(row == 1),  # Show legend only for the first occurrence
                            name=name
                        ), row=1, col=col)
                        fig.add_trace(go.Scatter(
                            x=[x, x], y=[global_y_min_waveform, global_y_max_waveform],
                            mode='lines', line=dict(color=color, width=2, dash='dot'),
                            showlegend=False,
                            name=name
                        ), row=3, col=col)

                        # Add lines across derivative subplots
                        fig.add_trace(go.Scatter(
                            x=[x, x], y=[global_y_min_derivative, global_y_max_derivative],
                            mode='lines', line=dict(color=color, width=2, dash='dot'),
                            showlegend=False,
                            name=name
                        ), row=2, col=col)
                        fig.add_trace(go.Scatter(
                            x=[x, x], y=[global_y_min_derivative, global_y_max_derivative],
                            mode='lines', line=dict(color=color, width=2, dash='dot'),
                            showlegend=False,
                            name=name
                        ), row=4, col=col)

                    # Add vertical lines for start and end trim indices (you need to define start_index and end_index accordingly)
                    add_global_vertical_line(fig, combined_start_idx, 1, 1, 'Start Index', 'Green')
                    add_global_vertical_line(fig, combined_end_idx, 1, 1, 'End Index', 'Red')

                    # Update layout for the entire figure
                    fig.update_layout(
                        height=1200,
                        title_text=f"PPG Waveform and Derivative Analysis for Artifact Window {true_start} to {true_end}",
                        showlegend=True
                    )

                    # Save the figure as HTML
                    fig_html_path = os.path.join(save_directory, f"artifact_{start}_{end}_analysis_derivative_heartbeat_trimming.html")
                    fig.write_html(fig_html_path)
                    logging.info(f"Saved the PPG waveform analysis plot as an HTML file: {fig_html_path}")
                    
                    # ! We are testing if the trimming is necessary with the new segmentation method
                    
                    # Trim the mean and median heartbeats using the calculated start and end indices
                    mean_heartbeat_trimmed = mean_heartbeat[combined_start_idx:combined_end_idx]
                    logging.info(f"Trimmed the mean heartbeat from {combined_start_idx} to {combined_end_idx} samples with length: {len(mean_heartbeat)}.")
                    median_heartbeat_trimmed = median_heartbeat[combined_start_idx:combined_end_idx]
                    logging.info(f"Trimmed the median heartbeat from {combined_start_idx} to {combined_end_idx} samples with length: {len(median_heartbeat)}.")

                    # Apply the trimming indices to all segmented heartbeats
                    trimmed_heartbeats = [beat[combined_start_idx:combined_end_idx] for beat in segmented_heartbeats]
                    logging.info(f"Trimmed all heartbeats from {combined_start_idx} to {combined_end_idx} samples.")
                    
                    # After trimming, calculate the minimum length across the trimmed beats for consistent length
                    min_length = min(map(len, trimmed_heartbeats))
                    logging.info(f"Minimum length after trimming: {min_length}")

                    # Process each heartbeat, padding as necessary
                    adjusted_heartbeats = []
                    logging.info("Processing heartbeats for consistent length.")
                    for i, beat in enumerate(trimmed_heartbeats):
                        if len(beat) < min_length:
                            # If the heartbeat is shorter than the minimum length, pad it with the last value
                            beat = np.pad(beat, (0, min_length - len(beat)), 'constant', constant_values=(beat[-1],))
                            logging.info("Heartbeat padded to minimum length.")
                        adjusted_heartbeats.append(beat)
                        logging.info(f"Heartbeat index {i + 1} appended to adjusted list.")
                        
                        # Convert numpy array to DataFrame for saving
                        heartbeat_df = pd.DataFrame(beat, columns=['PPG_Values'])
                        
                        # Save the individual heartbeat as a CSV file
                        heartbeat_filename = f'artifact_{start}_{end}_trimmed_heartbeat_{i + 1}.csv'  # use index + 1 to match the key concept
                        heartbeat_filepath = os.path.join(save_directory, heartbeat_filename)
                        heartbeat_df.to_csv(heartbeat_filepath, index=True, index_label='Sample_Index')
                        logging.info(f"Saved individual trimmed heartbeat to CSV file: {heartbeat_filepath}")
                        
                    # Convert to DataFrame for further processing
                    segmented_heartbeats_df = pd.DataFrame(adjusted_heartbeats)
                    logging.info("Converted trimmed segmented heartbeats to DataFrame for further processing.")
                    
                    # Create figure for plotting
                    logging.info("Creating the figure for trimmed heartbeats with quality control.")
                    heartbeats_fig = go.Figure()
                    
                    # Determine the time axis for the individual beats
                    num_points = segmented_heartbeats_df.shape[1]
                    logging.info(f"Number of points in the trimmed segmented heartbeats: {num_points}")
                    
                    # The time duration for each beat based on the number of samples and the sampling rate
                    time_duration_per_beat = num_points / sampling_rate
                    logging.info(f"Time duration per trimmed heartbeat: {time_duration_per_beat} seconds")

                    # Create the time axis from 0 to the time duration for each beat
                    time_axis = np.linspace(0, time_duration_per_beat, num=num_points, endpoint=False)
                    logging.info(f"Time axis created for the trimmed individual beats")

                    # Add individual heartbeats to the figure
                    logging.info("Adding trimmed individual heartbeats to the figure.")
                    for index, beat in enumerate(segmented_heartbeats_df.itertuples(index=False)):
                        # Exclude NaNs for plotting
                        valid_indices = ~np.isnan(beat)
                        valid_time = time_axis[valid_indices]
                        valid_beat = np.array(beat)[valid_indices]

                        heartbeats_fig.add_trace(
                            go.Scatter(
                                x=valid_time,
                                y=valid_beat,
                                mode='lines',
                                line=dict(color='grey', width=1),
                                opacity=0.5,
                                showlegend=False,
                                name=f'Trimmed Beat {index+1}'
                            )
                        )

                    # Add the mean beat shape to the figure
                    logging.info("Adding the trimmed mean beat shape to the figure.")
                    heartbeats_fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=mean_heartbeat_trimmed,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Trimmed Mean Heartbeat'
                        )
                    )
                    
                    # Add the median beat shape to the figure
                    logging.info("Adding the trimmed median beat shape to the figure.")
                    heartbeats_fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=median_heartbeat_trimmed,
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='Trimmed Median Beat'
                        )
                    )
                    
                    # Update the layout of the figure
                    logging.info("Updating the layout of the figure.")
                    heartbeats_fig.update_layout(
                        title=(f"Individual Trimmed Heartbeats and Average Beat Shape for Artifact Window {true_start} to {true_end}"),
                        xaxis_title='Time (s)',
                        yaxis_title='PPG Amplitude',
                        xaxis=dict(range=[0, time_duration_per_beat])  # set the range for x-axis
                    )

                    # Save the figure as an HTML file
                    heartbeat_plot_filename = f'artifact_{start}_{end}_trimmed_average_heartbeat.html'
                    heartbeats_plot_filepath = os.path.join(save_directory, heartbeat_plot_filename)
                    heartbeats_fig.write_html(heartbeats_plot_filepath)
                    logging.info(f"Saving the trimmed heartbeat figure as an HTML file at {heartbeats_plot_filepath}.")
                    
                    #! Line x Line performed up to this point in correct_artifacts() function
                    #%% NOTE: This section is where we calculate local reference signal statistics to guide the interpolation process
                    
                    # Function to calculate the expected number of beats
                    def calculate_expected_beats(artifact_window_ms, std_local_rr_interval_ms, local_rr_interval_ms):
                        try:
                            # Ensure artifact_window_ms is sufficiently larger than std_local_rr_interval_ms
                            if artifact_window_ms < std_local_rr_interval_ms:
                                logging.warning("Artifact window duration is less than the standard deviation of local R-R intervals.")

                            # Calculate the maximum and minimum number of expected beats within the artifact window
                            min_expected_beats = int(np.floor((artifact_window_ms + std_local_rr_interval_ms) / local_rr_interval_ms))
                            max_expected_beats = max(1, int(np.ceil((artifact_window_ms - std_local_rr_interval_ms) / local_rr_interval_ms)))

                            logging.info(f"Estimated maximum number of expected beats within artifact window: {max_expected_beats}")
                            logging.info(f"Estimated minimum number of expected beats within artifact window: {min_expected_beats}")

                            # Ensure min_expected_beats is not greater than max_expected_beats
                            if min_expected_beats > max_expected_beats:
                                logging.warning(f"Minimum expected beats exceeds maximum expected beats, swapping values.")
                                max_expected_beats, min_expected_beats = min_expected_beats, max_expected_beats
                                logging.info(f"Swapped maximum and minimum expected beats.")

                            return max_expected_beats, min_expected_beats

                        except Exception as e:
                            logging.error(f"Error in calculating expected beats: {e}")
                            raise

                    # Function to insert the beat template into the artifact window using spline interpolation
                    def insert_beat_template_into_artifact(artifact_window_signal, mean_heartbeat_trimmed, insertion_points, local_rr_interval_samples, true_start, true_end, valid_ppg, num_beats_to_insert, scaling_factor):
                        try:
                            logging.info(f"Using scaling factor: {scaling_factor}")
                            logging.info(f"Copying the artifact window signal for beat insertion.")
                            inserted_signal = np.copy(artifact_window_signal)
                            logging.info(f"Copied artifact window signal for insertion.")

                            # Clear all values from the inserted signal to start with a clean slate
                            inserted_signal[:] = 0
                            logging.info(f"Cleared all values from the inserted signal.")

                            # Adjust the insertion points based on the scaling factor
                            adjusted_insertion_points = insertion_points / scaling_factor
                            logging.info(f"Adjusted insertion points based on the scaling factor: {adjusted_insertion_points}")

                            # Calculate the number of x-values and y-values to generate
                            num_x_values = int(num_beats_to_insert * len(mean_heartbeat_trimmed) / scaling_factor)
                            logging.info(f"Number of x-values to generate: {num_x_values}")

                            # Create x values for the insertion points and the corresponding y values (amplitude)
                            x_values = np.linspace(true_start, true_end, num=num_x_values)
                            logging.info(f"Created x values for the insertion points: {len(x_values)} samples")
                            y_values = np.tile(mean_heartbeat_trimmed, int(np.ceil(num_x_values / len(mean_heartbeat_trimmed))))[:num_x_values]
                            logging.info(f"Created y values for the insertion points: {len(y_values)} samples")

                            # Calculate the offsets for the first and last beats to match the y-values at true_start and true_end
                            first_beat_offset = valid_ppg[true_start] - mean_heartbeat_trimmed[0]
                            logging.info(f"Calculated the offset for the first beat: {first_beat_offset}")
                            last_beat_offset = valid_ppg[true_end] - mean_heartbeat_trimmed[-1]
                            logging.info(f"Calculated the offset for the last beat: {last_beat_offset}")

                            # Adjust the first and last beats
                            y_values[:len(mean_heartbeat_trimmed)] += first_beat_offset
                            logging.info(f"Adjusted the first beat offset.")
                            y_values[-len(mean_heartbeat_trimmed):] += last_beat_offset
                            logging.info(f"Adjusted the last beat offset.")

                            logging.info(f"x_values length: {len(x_values)}, y_values length: {len(y_values)}")

                            # Fit a cubic spline to the insertion points
                            spline = CubicSpline(x_values, y_values)
                            logging.info(f"Fitted a cubic spline to the insertion points.")

                            # Interpolate the signal using the spline
                            interpolated_signal = spline(np.arange(true_start, true_end + 1))
                            logging.info(f"Interpolated the signal using the spline.")

                            # Plot the results
                            spline_plot = go.Figure()
                            spline_plot.add_trace(go.Scatter(x=np.arange(true_start, true_end + 1), y=artifact_window_signal, mode='lines', line=dict(color='green'), name='Original Signal'))
                            spline_plot.add_trace(go.Scatter(x=np.arange(true_start, true_end + 1), y=interpolated_signal, mode='lines', line=dict(color='blue'), name='Interpolated Signal'))
                            spline_plot.update_layout(title='Spline Interpolation of Inserted Beats', xaxis_title='Samples', yaxis_title='Amplitude')
                            spline_plot_filename = f'artifact_{start}_{end}_spline_interpolation.html'
                            spline_plot_filepath = os.path.join(save_directory, spline_plot_filename)
                            spline_plot.write_html(spline_plot_filepath)
                            logging.info(f"Saved spline interpolation plot as HTML file at {spline_plot_filepath}.")

                            return interpolated_signal

                        except Exception as e:
                            logging.error(f"Error in inserting beats with spline: {e}")
                            raise

                    # Optimization function
                    # ! Not presently optimizing...
                    def optimize_insertion_points(artifact_window_signal, mean_heartbeat_trimmed, valid_ppg, true_start, true_end, local_rr_intervals, sampling_rate, initial_num_beats, max_expected_beats, min_expected_beats):
                        logging.info(f"Starting optimization for insertion points and scaling factor.")

                        def objective_function(params):
                            try:
                                num_beats_to_insert = int(params[0])
                                logging.info(f"Number of beats to insert: {num_beats_to_insert}")
                                scaling_factor = params[1]
                                logging.info(f"Beat scaling factor: {scaling_factor}")

                                insertion_points = np.linspace(0, len(artifact_window_signal) - (num_beats_to_insert * len(mean_heartbeat_trimmed)), num=num_beats_to_insert, endpoint=True).astype(int)
                                logging.info(f"Calculated new insertion points: {insertion_points}")

                                interpolated_signal = insert_beat_template_into_artifact(
                                    artifact_window_signal, mean_heartbeat_trimmed, insertion_points, local_rr_interval_samples, true_start, true_end, valid_ppg, num_beats_to_insert, scaling_factor
                                )
                                logging.info(f"Interpolated signal length: {len(interpolated_signal)}")

                                smoothed_signal = gaussian_filter1d(interpolated_signal, sigma=3)
                                logging.info(f"Smoothed signal length: {len(smoothed_signal)}")

                                peaks, properties = find_peaks(smoothed_signal, prominence=min_prominence, height=min_height, width=min_width)
                                logging.info(f"Detected peaks in the smoothed signal: {peaks}")

                                peaks += true_start
                                logging.info(f"Scaled peaks index to the full range: {peaks}")

                                peaks = np.concatenate(([true_start], peaks, [true_end]))
                                logging.info(f"Added start and end peaks: {peaks}")

                                peaks = np.sort(peaks)
                                logging.info(f"Sorted peaks: {peaks}")

                                rr_intervals = np.diff(peaks) / sampling_rate * 1000
                                logging.info(f"Calculated R-R intervals: {rr_intervals}")

                                mean_rr_interval = np.mean(rr_intervals)
                                logging.info(f"Mean R-R interval: {mean_rr_interval}")
                                std_rr_interval = np.std(rr_intervals)
                                logging.info(f"Standard deviation of R-R intervals: {std_rr_interval}")

                                mean_diff = np.abs(mean_rr_interval - np.mean(local_rr_intervals))
                                logging.info(f"Mean difference: {mean_diff}")
                                std_diff = np.abs(std_rr_interval - np.std(local_rr_intervals))
                                logging.info(f"Standard deviation difference: {std_diff}")

                                return mean_diff + std_diff
                            except Exception as e:
                                logging.error(f"Error in objective function: {e}")
                                return np.inf

                        initial_guess = [initial_num_beats, 1.0]
                        logging.info(f"Initial guess for optimization: {initial_guess}")
                        bounds = [(min_expected_beats, max_expected_beats), (0.5, 2.0)]
                        logging.info(f"Bounds for optimization: {bounds}")

                        try:
                            result = minimize(objective_function, initial_guess, bounds=bounds)
                            logging.info(f"Optimization result: {result}")
                            optimized_num_beats = int(result.x[0])
                            optimized_scaling_factor = result.x[1]
                        except Exception as e:
                            logging.error(f"Error during optimization: {e}")
                            optimized_num_beats, optimized_scaling_factor = initial_guess

                        logging.info(f"Optimized number of beats to insert: {optimized_num_beats}")
                        logging.info(f"Optimized beat scaling factor: {optimized_scaling_factor}")

                        # // optimized_scaling_factor = 0.75 # Hardcoded for now  
                        logging.info(f"Hardcoded optimized beat scaling factor: {optimized_scaling_factor}")
                        return optimized_num_beats, optimized_scaling_factor

                    # Main execution block
                    try:
                        artifact_window_samples = len(valid_ppg[true_start:true_end + 1])
                        logging.info(f"Artifact window duration in samples: {artifact_window_samples} samples")

                        artifact_window_ms = artifact_window_samples / sampling_rate * 1000  # Convert to milliseconds
                        logging.info(f"Artifact window duration: {artifact_window_ms} milliseconds")

                        # Calculate R-R intervals in milliseconds for pre artifact peaks
                        pre_artifact_intervals = np.diff(pre_artifact_peaks) / sampling_rate * 1000
                        logging.info(f"Pre-artifact R-R intervals: {pre_artifact_intervals} milliseconds")

                        # Calculate average R-R interval for pre artifact peaks
                        pre_artifact_interval_mean = np.mean(pre_artifact_intervals) if pre_artifact_intervals.size > 0 else np.nan
                        logging.info(f"Calculated average R-R interval from pre artifact peaks: {pre_artifact_interval_mean} milliseconds")

                        # Calculate standard deviation of R-R intervals for pre artifact peaks
                        pre_artifact_interval_std = np.std(pre_artifact_intervals) if pre_artifact_intervals.size > 0 else np.nan
                        logging.info(f"Standard deviation of pre artifact R-R intervals: {pre_artifact_interval_std} milliseconds")

                        # Calculate R-R intervals in milliseconds for post artifact peaks
                        post_artifact_intervals = np.diff(post_artifact_peaks) / sampling_rate * 1000
                        logging.info(f"Post-artifact R-R intervals: {post_artifact_intervals} milliseconds")

                        # Calculate average R-R interval for post artifact peaks
                        post_artifact_interval_mean = np.mean(post_artifact_intervals) if post_artifact_intervals.size > 0 else np.nan
                        logging.info(f"Calculated average R-R interval from post artifact peaks: {post_artifact_interval_mean} milliseconds")

                        # Calculate standard deviation of R-R intervals for post artifact peaks
                        post_artifact_interval_std = np.std(post_artifact_intervals) if post_artifact_intervals.size > 0 else np.nan
                        logging.info(f"Standard deviation of post artifact R-R intervals: {post_artifact_interval_std} milliseconds")

                        # Concatenate pre- and post-artifact peaks to get the clean peaks around the artifact
                        local_rr_intervals = np.concatenate([pre_artifact_intervals, post_artifact_intervals])
                        logging.info(f"Combined Local R-R Intervals: {local_rr_intervals}")

                        # Calculate the average R-R interval from the clean peaks surrounding the artifact
                        local_rr_interval = np.mean(local_rr_intervals) if local_rr_intervals.size > 0 else np.nan
                        logging.info(f"Calculated average R-R interval from clean peaks surrounding the artifact: {local_rr_interval} milliseconds")

                        # Calculate the standard deviation of the R-R intervals from the clean peaks surrounding the artifact
                        std_local_rr_interval = np.std(local_rr_intervals) if local_rr_intervals.size > 0 else np.nan
                        logging.info(f"Standard deviation of local R-R intervals: {std_local_rr_interval} milliseconds")

                        # Convert the average R-R interval from milliseconds to samples
                        local_rr_interval_samples = int(np.round(local_rr_interval / 1000 * sampling_rate))
                        logging.info(f"Converted average R-R interval from milliseconds to samples: {local_rr_interval_samples} samples")

                        # Convert the average R-R interval from milliseconds to seconds
                        local_rr_interval_seconds = local_rr_interval / 1000 if not np.isnan(local_rr_interval) else np.nan
                        logging.info(f"Converted average R-R interval from milliseconds to seconds: {local_rr_interval_seconds} seconds")

                        # Estimate the maximum and minimum number of expected beats within the artifact window
                        max_expected_beats, min_expected_beats = calculate_expected_beats(artifact_window_ms, std_local_rr_interval, local_rr_interval)
                        logging.info(f"Running function to estimate the maximum and minimum number of expected beats within the artifact window.")

                        num_beats_to_insert = int((max_expected_beats + min_expected_beats) / 2)
                        logging.info(f"Number of beats to insert: {num_beats_to_insert}")

                        artifact_window_signal = valid_ppg[true_start:true_end + 1].copy()
                        logging.info(f"Copied artifact window signal for interpolation.")

                        average_beat_length = len(mean_heartbeat_trimmed)
                        logging.info(f"Average beat length: {average_beat_length} samples")

                        # Calculate the number of beats to insert based on the average beat length
                        logging.info(f"Calculating exact insertion points to insert {num_beats_to_insert} beats.")
                        insertion_points = np.linspace(0, artifact_window_samples - local_rr_interval_samples, num=num_beats_to_insert, endpoint=True).astype(int)
                        logging.info(f"Calculated insertion points: {insertion_points}")

                        # Insert the average beat template into the artifact window using spline interpolation
                        interpolated_signal = insert_beat_template_into_artifact(
                            artifact_window_signal, mean_heartbeat_trimmed, insertion_points, local_rr_interval_samples, true_start, true_end, valid_ppg, num_beats_to_insert, scaling_factor=1.0
                        )
                        logging.info(f"Inserted average beat template into artifact window with spline interpolation.")

                        # Apply Gaussian smoothing to the corrected signal
                        sigma = 3  # Standard deviation for Gaussian kernel
                        smoothed_signal = gaussian_filter1d(interpolated_signal, sigma)
                        logging.info(f"Applied Gaussian smoothing with sigma {sigma} to the corrected signal.")

                        # Create plot for the corrected and smoothed signals
                        corrected_smoothed_plot = go.Figure()
                        corrected_smoothed_plot.add_trace(go.Scatter(x=np.arange(len(interpolated_signal)), y=interpolated_signal, mode='lines', line=dict(color='blue'), name='Interpolated Signal'))
                        corrected_smoothed_plot.add_trace(go.Scatter(x=np.arange(len(smoothed_signal)), y=smoothed_signal, mode='lines', name='Smoothed Signal', line=dict(dash='dash')))
                        corrected_smoothed_plot.update_layout(title='Corrected and Smoothed Signals', xaxis_title='Samples', yaxis_title='Amplitude')
                        corrected_smoothed_plot_filename = f'artifact_{start}_{end}_corrected_smoothed.html'
                        corrected_smoothed_plot_filepath = os.path.join(save_directory, corrected_smoothed_plot_filename)
                        corrected_smoothed_plot.write_html(corrected_smoothed_plot_filepath)
                        logging.info(f"Saved plot for corrected and smoothed signals as HTML file at {corrected_smoothed_plot_filepath}.")

                        # Parameters (these need to be tuned based on your specific signal characteristics)
                        # These help prevent the identification of diastolic peaks as P peaks of interest
                        min_height = np.percentile(smoothed_signal, 75)  # Only consider peaks above the 75th percentile height
                        min_prominence = np.std(smoothed_signal) * 0.5  # Set prominence to be at least half the standard deviation of the signal
                        min_width = 5  # Minimum number of samples over which the peak is wider than at half its prominence

                        # Detect peaks within this segment using scipy's find_peaks for better accuracy
                        peaks, properties = find_peaks(smoothed_signal, prominence=min_prominence, height=min_height, width=min_width)
                        logging.info(f"Found peaks in the smoothed signal: {peaks}")

                        # Scale peaks index to full range
                        peaks += true_start
                        logging.info(f"Scaled peaks index to full range: {peaks}")

                        # Add start and end peaks
                        peaks = np.concatenate(([start], peaks, [end]))
                        logging.info(f"Concatenated start and end peaks to the peak array: {peaks}")

                        # Sort the peaks
                        peaks = np.sort(peaks)
                        logging.info(f"Sorted the peak array: {peaks}")

                        # Calculate R-R intervals from the peaks
                        rr_intervals = np.diff(peaks) / sampling_rate * 1000  # Convert to milliseconds
                        logging.info(f"Calculated R-R intervals from peaks: {rr_intervals} milliseconds")

                        # Calculate mean and standard deviation of R-R intervals
                        mean_rr_interval = np.mean(rr_intervals)
                        std_rr_interval = np.std(rr_intervals)
                        logging.info(f"Mean R-R interval from peaks: {mean_rr_interval} milliseconds")
                        logging.info(f"Standard deviation of R-R intervals from peaks: {std_rr_interval} milliseconds")

                        # Compare with surrounding reference signal
                        mean_diff = np.abs(mean_rr_interval - local_rr_interval)
                        std_diff = np.abs(std_rr_interval - std_local_rr_interval)
                        logging.info(f"Difference between mean R-R interval and local mean: {mean_diff} milliseconds")
                        logging.info(f"Difference between standard deviation of R-R intervals and local standard deviation: {std_diff} milliseconds")

                        # Perform t-test to compare the means
                        t_stat_mean, p_value_mean = ttest_ind(local_rr_intervals, rr_intervals, equal_var=False)
                        logging.info(f"T-statistic for mean comparison: {t_stat_mean}, p-value: {p_value_mean}")

                        # Perform F-test to compare the variances
                        f_stat_std, p_value_std = f_oneway(local_rr_intervals, rr_intervals)
                        logging.info(f"F-statistic for standard deviation comparison: {f_stat_std}, p-value: {p_value_std}")

                        # Check if the mean and standard deviation differences are statistically significant
                        alpha = 0.05  # Significance level
                        if p_value_mean < alpha or p_value_std < alpha:
                            logging.warning(f"R-R interval mean or standard deviation is significantly different from the surrounding reference signal. Optimization may be required.")

                            optimized_num_beats, optimized_scaling_factor = optimize_insertion_points(
                                artifact_window_signal, mean_heartbeat_trimmed, valid_ppg, true_start, true_end, local_rr_intervals, sampling_rate, num_beats_to_insert, max_expected_beats, min_expected_beats
                            )
                            logging.info(f"Optimized number of beats to insert: {optimized_num_beats}")
                            logging.info(f"Optimized beat scaling factor: {optimized_scaling_factor}")

                            insertion_points = np.linspace(0, artifact_window_samples - (optimized_num_beats * local_rr_interval_samples / optimized_scaling_factor), num=optimized_num_beats, endpoint=True).astype(int)
                            logging.info(f"Recalculated insertion points with optimized parameters: {insertion_points}")

                            optimized_interpolated_signal = insert_beat_template_into_artifact(
                                artifact_window_signal, mean_heartbeat_trimmed, insertion_points, local_rr_interval_samples, true_start, true_end, valid_ppg, optimized_num_beats, optimized_scaling_factor
                            )
                            logging.info(f"Re-inserted average beat template into artifact window with optimized parameters.")

                            smoothed_signal = gaussian_filter1d(optimized_interpolated_signal, sigma)
                            logging.info(f"Applied Gaussian smoothing with sigma {sigma} to the optimized corrected signal.")

                            peaks, properties = find_peaks(smoothed_signal, prominence=min_prominence, height=min_height, width=min_width)
                            logging.info(f"Found peaks in the optimized smoothed signal: {peaks}")

                            peaks += true_start
                            logging.info(f"Scaled peaks index to full range: {peaks}")

                            peaks = np.concatenate(([true_start], peaks, [true_end]))
                            logging.info(f"Concatenated start and end peaks to the peak array: {peaks}")

                            peaks = np.sort(peaks)
                            logging.info(f"Sorted the peak array: {peaks}")

                            rr_intervals = np.diff(peaks) / sampling_rate * 1000
                            logging.info(f"Calculated R-R intervals from peaks: {rr_intervals} milliseconds")

                            mean_rr_interval = np.mean(rr_intervals)
                            std_rr_interval = np.std(rr_intervals)
                            logging.info(f"Mean R-R interval from peaks: {mean_rr_interval} milliseconds")
                            logging.info(f"Standard deviation of R-R intervals from peaks: {std_rr_interval} milliseconds")

                            mean_diff = np.abs(mean_rr_interval - local_rr_interval)
                            std_diff = np.abs(std_rr_interval - std_local_rr_interval)
                            logging.info(f"Difference between mean R-R interval and local mean: {mean_diff} milliseconds")
                            logging.info(f"Difference between standard deviation of R-R intervals and local standard deviation: {std_diff} milliseconds")

                            t_stat_mean, p_value_mean = ttest_ind(local_rr_intervals, rr_intervals, equal_var=False)
                            logging.info(f"T-statistic for mean comparison: {t_stat_mean}, p-value: {p_value_mean}")

                            f_stat_std, p_value_std = f_oneway(local_rr_intervals, rr_intervals)
                            logging.info(f"F-statistic for standard deviation comparison: {f_stat_std}, p-value: {p_value_std}")

                            if p_value_mean < alpha or p_value_std < alpha:
                                logging.warning(f"R-R interval mean or standard deviation is still significantly different from the surrounding reference signal after optimization. Further optimization may be required.")

                        # Replace the artifact window in the original signal with the smoothed signal
                        valid_ppg[true_start:true_end + 1] = smoothed_signal
                        logging.info(f"Replaced the artifact window in the original signal with the smoothed signal.")

                        # Create plot for the final corrected and smoothed signal
                        final_signal_plot = go.Figure()
                        final_signal_plot.add_trace(go.Scatter(x=np.arange(true_start, true_end + 1), y=valid_ppg[true_start:true_end + 1], mode='lines', name='Final Signal'))
                        final_signal_plot.add_trace(go.Scatter(x=np.arange(true_start, true_end + 1), y=interpolated_signal, mode='lines', name='Interpolated Signal', line=dict(dash='dash')))
                        final_signal_plot.add_trace(go.Scatter(x=np.arange(true_start, true_end + 1), y=smoothed_signal, mode='lines', name='Smoothed Signal', line=dict(dash='dash')))
                        final_signal_plot.update_layout(title='Final Corrected and Smoothed Signal', xaxis_title='Samples', yaxis_title='Amplitude')
                        final_signal_plot_filename = f'artifact_{start}_{end}_final_corrected_smoothed.html'
                        final_signal_plot_filepath = os.path.join(save_directory, final_signal_plot_filename)
                        final_signal_plot.write_html(final_signal_plot_filepath)
                        logging.info(f"Saved plot for final corrected and smoothed signal as HTML file at {final_signal_plot_filepath}.")

                    except Exception as e:
                        logging.error(f"Error in processing signal: {e}")
                        raise
                    
                    # Sanity check
                    logging.info(f'True start index = {true_start}, True end index = {true_end}')
                        
                    # Ensure you're passing the correctly updated valid_ppg to create_figure
                    fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
                    # Maintain the existing layout
                    current_layout = fig['layout'] if fig else None
                    if current_layout:
                        fig.update_layout(
                            xaxis=current_layout['xaxis'],
                            yaxis=current_layout['yaxis']
                        ) 
                    
                    # Add shaded rectangle marking the artifact window
                    fig.add_shape(
                            type="rect",
                            x0=true_start,
                            x1=true_end,
                            y0=0,  # Start at the bottom of the figure
                            y1=1,  # Extend to the top of the figure
                            line=dict(color="Red"),
                            fillcolor="LightSalmon",
                            opacity=0.5,
                            layer='below',
                            yref='paper'  # Reference to the entire figure's y-axis
                        )

            # Update artifact_windows[-1] with the corrected 'start' and 'end' values
            artifact_windows[-1]['start'] = true_start
            artifact_windows[-1]['end'] = true_end
            logging.info(f"Updated artifact window {artifact_windows[-1]} with corrected start and end indices.")
                                
        return fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows, artifact_windows
    
    except Exception as e:
        logging.error(f"Error in correct_artifacts: {e}")

@app.callback(
    Output('save-status', 'children'), # Update the save status message
    [Input('save-button', 'n_clicks')], # Listen to the save button click
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'), # Keep the data-store state
     State('peaks-store', 'data'), # Keep the peaks-store state 
     State('peak-change-store', 'data'), # Keep the peak-change-store state
     State('ppg-plot', 'figure'), # Access the current figure state
     State('rr-store', 'data'),  # Include rr-store data for saving corrected R-R intervals
     State('artifact-windows-store', 'data')] # Include artifact-windows-store data for saving corrected artifact windows
)

# BUG: Save button able to be clicked twice and double save the data - not urgent to fix but annoying for the log file. 

# Main callback function to save the corrected data to file
# ! need to pass artifact_windows to the function
def save_corrected_data(n_clicks, filename, data_json, valid_peaks, peak_changes, fig, rr_data, artifact_windows):
    """
    Saves the corrected PPG data and updates peak information based on user interactions.

    Triggered by a user clicking the 'Save Corrected Data' button, this function saves the
    corrected data to a new file and updates peak count information. It handles generating
    new filenames for the corrected data and peak count, calculates corrected R-R intervals,
    and saves the updated data back to the filesystem.

    Parameters:
    - n_clicks (int): The number of times the save button has been clicked.
    - filename (str): The name of the uploaded file.
    - data_json (str): JSON string representation of the PPG data DataFrame.
    - valid_peaks (list of int): List of indices representing valid peaks.
    - peak_changes (dict): A dictionary tracking changes to the peaks, including added, deleted, and original counts, and number of samples corrected.
    - fig (dict): The current figure object representing the PPG data and peaks.

    Returns:
    - str: A message indicating the outcome of the save operation, whether success or failure.

    Raises:
    - dash.exceptions.PreventUpdate: Prevents updating the callback output if the save button has not been clicked or if necessary data is missing.
    """ 
   
    if n_clicks is None or not data_json or not valid_peaks:
        logging.info("Save button not clicked, preventing update.")
        raise dash.exceptions.PreventUpdate

    global save_directory # Use the global save directory path from the argument parser

    logging.info(f"Test of artifact_windows passing: {artifact_windows}")

    try:
        logging.info(f"Attempting to save corrected data...")
        
        # Extract subject_id and run_id
        parts = filename.split('_')
        if len(parts) < 4:
            logging.error("Filename does not contain expected parts.")
            return "Error: Filename structure incorrect."

        # Construct the new filename by appending '_corrected' before the file extension
        if filename and not filename.endswith('.tsv.gz'):
            logging.error("Invalid filename format.")
            return "Error: Invalid filename format."
        
        parts = filename.rsplit('.', 2)
        if len(parts) == 3:
            base_name, ext1, ext2 = parts
            ext = f"{ext1}.{ext2}"  # Reassemble the extension
        else:
            # Handle the case where the filename does not have a double extension
            base_name, ext = parts
        
        # Assuming df is already loaded from the JSON
        df = pd.read_json(data_json, orient='split')

        # Add a new column to the DataFrame to store the corrected peaks
        df['PPG_Peaks_elgendi_corrected'] = 0
        df.loc[valid_peaks, 'PPG_Peaks_elgendi_corrected'] = 1

        # Use the rr_data from the store to update the DataFrame with corrected R-R and PPG Data
        df['Regular_Time_Axis'] = rr_data['regular_time_axis']
        logging.info(f"Regular time axis length: {len(rr_data['regular_time_axis'])}")
        df['PPG_Values_Corrected'] = rr_data['valid_ppg']
        logging.info(f"Valid PPG length: {len(rr_data['valid_ppg'])}")
        df['RR_interval_interpolated_corrected'] = rr_data['interpolated_rr']
        logging.info(f"Interpolated R-R intervals length: {len(rr_data['interpolated_rr'])}")

        # Helper function to create censored arrays for PPG and peaks based on artifact windows
        def create_censored_arrays(valid_ppg_corrected, valid_peaks_corrected, tachogram_corrected, artifact_windows):
            valid_ppg_corrected_censored = valid_ppg_corrected.copy()
            valid_peaks_corrected_censored = valid_peaks_corrected.copy()
            tachogram_corrected_censored = np.array(tachogram_corrected, dtype=np.float64)

            for window in artifact_windows:
                start_idx = window['start']
                end_idx = window['end']

                if start_idx >= 0 and end_idx <= len(valid_ppg_corrected_censored):
                    # Mask out the regions in valid_ppg
                    valid_ppg_corrected_censored[start_idx:end_idx] = np.nan
                    logging.info(f"Masked out the region in valid PPG corrected data from {start_idx} to {end_idx}.")
                
                # Mask out the regions in tachogram_corrected
                tachogram_corrected_censored[start_idx:end_idx] = np.nan
                logging.info(f"Masked out the region in tachogram corrected data from {start_idx} to {end_idx}.")

                # Mask out the regions in valid_peaks
                valid_peaks_corrected_censored = [peak for peak in valid_peaks_corrected_censored if peak < start_idx or peak > end_idx]
                logging.info(f"Masked out the region in valid peaks corrected data from {start_idx} to {end_idx}.")

            return valid_ppg_corrected_censored, valid_peaks_corrected_censored, tachogram_corrected_censored

        valid_ppg_corrected = df['PPG_Values_Corrected'].values
        logging.info(f"Valid PPG corrected length: {len(valid_ppg_corrected)}")
        valid_peaks_corrected = df[df['PPG_Peaks_elgendi_corrected'] == 1].index.values
        logging.info(f"Valid peaks corrected length: {len(valid_peaks_corrected)}")
        tachogram_corrected = df['RR_interval_interpolated_corrected'].values
        logging.info(f"Interpolated R-R intervals corrected length: {len(tachogram_corrected)}")

        # Create censored arrays
        logging.info(f"Creating censored arrays for corrected PPG data and peaks based on artifact windows.")
        valid_ppg_corrected_censored, valid_peaks_corrected_censored, tachogram_corrected_censored = create_censored_arrays(valid_ppg_corrected, valid_peaks_corrected, tachogram_corrected, artifact_windows)     
        logging.info(f"Successfully created censored arrays for corrected PPG data and peaks based on artifact windows.")

        # Verify lengths of censored arrays
        logging.info(f"Censored PPG corrected length: {len(valid_ppg_corrected_censored)}")
        logging.info(f"Censored peaks corrected length: {len(valid_peaks_corrected_censored)}")
        logging.info(f"Censored R-R intervals corrected length: {len(tachogram_corrected_censored)}")

        # Plot the original and censored tachogram data using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df['RR_interval_interpolated_corrected'], mode='lines', name='RR_interval_interpolated_corrected'))
        fig.add_trace(go.Scatter(x=df.index[:len(tachogram_corrected_censored)], y=tachogram_corrected_censored, mode='lines', name='RR_interval_interpolated_corrected_censored'))

        fig.update_layout(title='Censored and Uncensored RR Intervals',
                        xaxis_title='Index',
                        yaxis_title='RR Interval',
                        legend_title='Legend')

        # Save the plot as an HTML file

        plot_filename = os.path.join(save_directory, f"_filtered_cleaned_ppg_censored_check.html")
        plotly.offline.plot(fig, filename=plot_filename, auto_open=False)

        # Ensure the length of the censored data matches the original
        df['PPG_Values_Corrected_Censored'] = pd.Series(valid_ppg_corrected_censored, index=df.index[:len(valid_ppg_corrected_censored)])
        logging.info(f"Updating DataFrame with censored corrected PPG data.")

        # Initialize the censored peaks column with zeros
        df['PPG_Peaks_elgendi_corrected_censored'] = 0
        logging.info(f"Updating DataFrame with censored corrected peaks.")

        # Update the peaks in the censored peaks column
        for peak in valid_peaks_corrected_censored:
            if peak < len(df):
                df.at[peak, 'PPG_Peaks_elgendi_corrected_censored'] = 1
        logging.info(f"Successfully updated DataFrame with censored corrected peaks.")

        # Ensure the length of the censored tachogram data matches the original
        df['RR_interval_interpolated_corrected_censored'] = pd.Series(tachogram_corrected_censored, index=df.index[:len(tachogram_corrected_censored)])
        logging.info(f"Updating DataFrame with censored corrected R-R intervals.")

        # Take from peak_changes store
        original_total_peak_count = peak_changes['original_total_peak_count']
        logging.info(f"Original total peak count: {original_total_peak_count}")
        dash_added_peaks_count = peak_changes['dash_added_peaks_count']
        logging.info(f"Dash added peaks count: {dash_added_peaks_count}")
        dash_deleted_peaks_count = peak_changes['dash_deleted_peaks_count']
        logging.info(f"Dash deleted peaks count: {dash_deleted_peaks_count}")
        dash_total_corrected_peak_count = len(valid_peaks)
        logging.info(f"Dash total corrected peak count: {dash_total_corrected_peak_count}")
        kubios_added_peak_count = peak_changes['kubios_added_peak_count']
        logging.info(f"Kubios added peak count: {kubios_added_peak_count}")
        kubios_deleted_peak_count = peak_changes['kubios_deleted_peak_count']
        logging.info(f"Kubios deleted peak count: {kubios_deleted_peak_count}")
        kubios_total_corrected_peak_count = peak_changes['kubios_total_corrected_peak_count']
        logging.info(f"Kubios total corrected peak count: {kubios_total_corrected_peak_count}")
        kubios_total_samples_corrected = peak_changes['kubios_total_samples_corrected']
        logging.info(f"Kubios total samples corrected: {kubios_total_samples_corrected}")
        
        # Helper function to compute artifact window statistics
        def compute_artifact_window_stats(artifact_windows):
            window_sizes = [window['end'] - window['start'] for window in artifact_windows]

            artifact_windows_total_count_number = len(artifact_windows)
            logging.info(f"Total number of artifact windows: {artifact_windows_total_count_number}")
            artifact_windows_total_count_samples = sum(window_sizes)
            logging.info(f"Total number of samples in artifact windows: {artifact_windows_total_count_samples}")
            mean_artifact_window_size_samples = np.mean(window_sizes) if window_sizes else 0
            logging.info(f"Mean size of artifact windows in samples: {mean_artifact_window_size_samples}")
            std_artifact_window_size_samples = np.std(window_sizes) if window_sizes else 0
            logging.info(f"Standard deviation of artifact window sizes in samples: {std_artifact_window_size_samples}")
            min_artifact_window_size_samples = np.min(window_sizes) if window_sizes else 0
            logging.info(f"Minimum size of artifact windows in samples: {min_artifact_window_size_samples}")
            max_artifact_window_size_samples = np.max(window_sizes) if window_sizes else 0
            logging.info(f"Maximum size of artifact windows in samples: {max_artifact_window_size_samples}")

            return {
                'artifact_windows_total_count_number': artifact_windows_total_count_number,
                'artifact_windows_total_count_samples': artifact_windows_total_count_samples,
                'mean_artifact_window_size_samples': mean_artifact_window_size_samples,
                'std_artifact_window_size_samples': std_artifact_window_size_samples,
                'min_artifact_window_size_samples': min_artifact_window_size_samples,
                'max_artifact_window_size_samples': max_artifact_window_size_samples
            }
        
        # Compute statistics for artifact windows
        logging.info(f"Computing statistics for artifact windows.")
        artifact_window_stats = compute_artifact_window_stats(artifact_windows)
        logging.info(f"Successfully computed statistics for artifact windows.")
        peak_changes.update(artifact_window_stats)
        logging.info(f"Updated peak changes with artifact window statistics.")
        
        # Compute artifact window statistics
        artifact_window_censored_total_corrected_peak_count = len(valid_peaks_corrected_censored)
        logging.info(f"Artifact window censored total corrected peak count: {artifact_window_censored_total_corrected_peak_count}")
        artifact_windows_total_count_number = peak_changes['artifact_windows_total_count_number']
        logging.info(f"Artifact windows total count number: {artifact_windows_total_count_number}")
        artifact_windows_total_count_samples = peak_changes['artifact_windows_total_count_samples']
        logging.info(f"Artifact windows total count samples: {artifact_windows_total_count_samples}")
        mean_artifact_window_size_samples = peak_changes['mean_artifact_window_size_samples']
        logging.info(f"Mean artifact window size in samples: {mean_artifact_window_size_samples}")
        std_artifact_window_size_samples = peak_changes['std_artifact_window_size_samples']
        logging.info(f"Standard deviation of artifact window sizes in samples: {std_artifact_window_size_samples}")
        min_artifact_window_size_samples = peak_changes['min_artifact_window_size_samples']
        logging.info(f"Minimum artifact window size in samples: {min_artifact_window_size_samples}")
        max_artifact_window_size_samples = peak_changes['max_artifact_window_size_samples']
        logging.info(f"Maximum artifact window size in samples: {max_artifact_window_size_samples}")
        kubios_ectopic_beats = peak_changes['kubios_ectopic_beats']
        logging.info(f"Kubios ectopic beats: {kubios_ectopic_beats}")	
        kubios_missed_beats = peak_changes['kubios_missed_beats']
        logging.info(f"Kubios missed beats: {kubios_missed_beats}")
        kubios_extra_beats = peak_changes['kubios_extra_beats']
        logging.info(f"Kubios extra beats: {kubios_extra_beats}")	
        kubios_longshort_beats = peak_changes['kubios_longshort_beats']
        logging.info(f"Kubios long or short beats: {kubios_longshort_beats}")

        # Finalize peak count data        
        peak_count_data = {
            'original_total_peak_count': original_total_peak_count, 
            'dash_added_peaks_count': dash_added_peaks_count, # handled by plot clicks data
            'dash_deleted_peaks_count': dash_deleted_peaks_count, # handled by plot clicks data
            'dash_total_corrected_peak_count': dash_total_corrected_peak_count, # handled now with len(valid_peaks) final before save out
            'kubios_added_peak_count': kubios_added_peak_count, 
            'kubios_deleted_peak_count': kubios_deleted_peak_count, 
            'kubios_total_corrected_peak_count': kubios_total_corrected_peak_count, 
            'kubios_total_samples_corrected': kubios_total_samples_corrected, 
            'artifact_window_censored_total_corrected_peak_count': artifact_window_censored_total_corrected_peak_count, # handled now before save out by counting the number of peaks outside the artifact windows 
            'artifact_windows_total_count_number': artifact_windows_total_count_number, # handled now before save out by counting the number of artifact windows
            'artifact_windows_total_count_samples': artifact_windows_total_count_samples, # handled later before save out by counting the total number of samples in the artifact windows
            'mean_artifact_window_size_samples': mean_artifact_window_size_samples, # handled now before save out by calculating the mean size of the artifact windows in samples 
            'std_artifact_window_size_samples': std_artifact_window_size_samples, # handled now before save out by calculating the standard deviation of the artifact window sizes in samples
            'min_artifact_window_size_samples': min_artifact_window_size_samples, # handled now before save out by calculating the minimum size of the artifact windows in samples
            'max_artifact_window_size_samples': max_artifact_window_size_samples, # handled now before save out by calculating the maximum size of the artifact windows in samples
            'kubios_ectopic_beats': kubios_ectopic_beats, # handled now before save out by counting the number of ectopic beats in the Kubios corrected data
            'kubios_missed_beats': kubios_missed_beats, # handled now before save out by counting the number of missed beats in the Kubios corrected data
            'kubios_extra_beats': kubios_extra_beats, # handled now before save out by counting the number of extra beats in the Kubios corrected data
            'kubios_longshort_beats': kubios_longshort_beats # handled now before save out by counting the number of long or short beats in the Kubios corrected data
        }
            
        df_peak_count = pd.DataFrame([peak_count_data])
        logging.info(f"Added peak count data and stats to DataFrame")

        # Define filename for the dash corrected figure
        figure_filename = f"{base_name}_corrected_subplots.html"
        figure_filepath = os.path.join(save_directory, figure_filename)
        
        # Write corrected figure to html file
        pio.write_html(fig, figure_filepath)
        logging.info(f"Saved corrected figure to {figure_filepath}")

        # Define the new filename for the various corrected data output
        new_filename = f"{base_name}_corrected.{ext}"
        full_new_path = os.path.join(save_directory, new_filename)
        logging.info(f"Dash Corrected full path: {full_new_path}")

        # Save the DataFrame to the new file
        df.to_csv(full_new_path, sep='\t', compression='gzip', index=False)
        logging.info(f"Saved corrected data to {full_new_path}")
        
        # Save peak count data
        peak_count_filename = f"{base_name}_corrected_peakCountDataStatistics.tsv"
        peak_count_full_path = os.path.join(save_directory, peak_count_filename)
        
        # Transpose the DataFrame so headers are in the first column and values in the second column
        df_peak_count_transposed = df_peak_count.transpose().reset_index()
        df_peak_count_transposed.columns = ['Statistic', 'Value']

        # Save the transposed DataFrame to a TSV file
        df_peak_count_transposed.to_csv(peak_count_full_path, sep='\t', index=False)
        logging.info(f"Saved corrected peak count data to {peak_count_full_path}")

        # Add conditional logic for different save_suffix cases to run the HRV stats for each case
        for save_suffix in ['dash_corrected', 'kubios_corrected', 'artifact_censored', 'original_uncorrected']:
        
            # Call to compute HRV stats
            compute_hrv_stats(df, valid_peaks, filename, save_directory, save_suffix, artifact_windows) # add save_suffix here to run for each recalculation instance?

        return f"Data and corrected peak counts saved to {full_new_path} and {peak_count_full_path}"

    except Exception as e:
        logging.error(f"Error in save_corrected_data: {e}")
        return "An error occurred while saving data."
    
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

# Function to re-compute HRV statistics after peak and artifact correction
def compute_hrv_stats(df, valid_peaks, filename, save_directory, save_suffix, artifact_windows):
    
    """
    draft code for HRV statistical recomputation
    """
    
    #%% #! SECTION 1: Variables setup
    
    logging.info(f"Recomputing HRV statistics for {save_suffix} corrected PPG data.")
    
    sampling_rate = 100   # downsampled sampling rate
    logging.info(f"Downsampled PPG sampling rate: {sampling_rate} Hz")

    # Current handling in save_corrected_data()
    parts = filename.rsplit('.', 2)
    if len(parts) == 3:
        base_name, ext1, ext2 = parts
        ext = f"{ext1}.{ext2}"  # Reassemble the extension
    else:
        # Handle the case where the filename does not have a double extension
        base_name, ext = parts

    # New handling: Extract base_filename
    base_filename = '_'.join(base_name.split('_')[:5])  # Adjust the index according to your naming convention
    logging.info(f"Base filename for HRV stats files: {base_filename}")
    
    def remove_nan_and_corresponding_rows(ppg_signal, ppg_peaks, tachogram):
        # Create a mask for non-NaN values in ppg_signal
        nan_mask = ~np.isnan(ppg_signal)
        logging.info(f"Created mask for non-NaN values in PPG signal.")
        ppg_signal_cleaned = ppg_signal[nan_mask]
        logging.info(f"Removed NaN values from PPG signal.")

        # Ensure ppg_peaks is a list of peak indices if it's a binary series
        if isinstance(ppg_peaks, pd.Series) or isinstance(ppg_peaks, np.ndarray):
            ppg_peaks = np.where(ppg_peaks == 1)[0].tolist()  # Convert to list of indices
            logging.info(f"Converted PPG peaks to list of indices.")

        # Update ppg_peaks to remove peaks that fall within NaN regions
        ppg_peaks_cleaned = [peak for peak in ppg_peaks if peak < len(nan_mask) and nan_mask[peak]]
        logging.info(f"Removed peaks that fall within NaN regions from PPG peaks.")

        # Update tachogram to match the length of ppg_signal_cleaned
        valid_intervals = [interval for interval, valid in zip(tachogram, nan_mask) if valid]
        logging.info(f"Removed intervals corresponding to NaN regions from tachogram.")

        logging.info(f"Length of ppg_signal_cleaned: {len(ppg_signal_cleaned)}")
        logging.info(f"Length of ppg_peaks_cleaned: {len(ppg_peaks_cleaned)}")
        logging.info(f"Length of tachogram_cleaned: {len(valid_intervals)}")

        return ppg_signal_cleaned, ppg_peaks_cleaned, valid_intervals

    # Add conditional logic for different save_suffix cases
    if save_suffix == 'dash_corrected':
        ppg_signal = df['PPG_Values_Corrected']
        ppg_peaks = df['PPG_Peaks_elgendi_corrected']
        tachogram = df['RR_interval_interpolated_corrected']

    elif save_suffix == 'kubios_corrected':
        ppg_signal = df['PPG_Clean']  # kubios method fixes peaks only and doesn't adjust the ppg signal
        ppg_peaks = df['Peaks_Kubios']
        tachogram = df['RR_interval_interpolated_Kubios']

    elif save_suffix == 'artifact_censored':
        ppg_signal = df['PPG_Values_Corrected_Censored']
        ppg_peaks = df['PPG_Peaks_elgendi_corrected_censored']
        tachogram = df['RR_interval_interpolated_corrected_censored']

    elif save_suffix == 'original_uncorrected':
        ppg_signal = df['PPG_Clean']
        ppg_peaks = df['PPG_Peaks_elgendi']
        tachogram = df['RR_interval_interpolated']

    # Remove NaN values and corresponding rows
    logging.info(f"Removing NaN values from PPG signal and corresponding rows from PPG peaks and tachogram.")
    ppg_signal_cleaned, ppg_peaks_cleaned, tachogram_cleaned = remove_nan_and_corresponding_rows(ppg_signal, ppg_peaks, tachogram)
    logging.info(f"Successfully removed NaN values from PPG signal and corresponding rows from PPG peaks and tachogram.")

    ppg_signal = ppg_signal_cleaned
    ppg_peaks = ppg_peaks_cleaned
    tachogram = tachogram_cleaned
    
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
    
    #%% #! SECTION 2: PSD Plotly Plots (0 - 8 Hz full range)  
    #! Here we need to load the corrected ppg timeseries into a pandas dataframe for neurokit2 functions
    
    # Compute Power Spectral Density 0 - 8 Hz for PPG Tachogram
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG Tachogram using multitapers hann windowing, {save_suffix} method.")
    ppg_filtered_psd = nk.signal_psd(tachogram, sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                        min_frequency=0, max_frequency=8.0, window=None, window_type='hann',
                        silent=False, t=None)

    # Plotting Power Spectral Density
    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 8 Hz for filtered cleaned PPG Tachogram using multitapers hann windowing, {save_suffix} method.")

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
        title=f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for Filtered Cleaned PPG {save_suffix} method.',
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
    plot_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_psd_{save_suffix}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 3: PSD Plotly Plots (0 - 1 Hz HRV range) 
    # Compute Power Spectral Density 0 - 1 Hz for PPG
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG Tachogram HRV range using multitapers hann windowing, {save_suffix} method.")
    ppg_filtered_psd_hrv = nk.signal_psd(tachogram, sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                        silent=False, t=None)

    # Plotting Power Spectral Density
    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned PPG Tachogram HRV using multitapers hann windowing, {save_suffix} method.")

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
        title=f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for R-R Interval Tachogram, {save_suffix} method',
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
    plot_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_psd_hrv_{save_suffix}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 4: Re-calculate HRV Stats
    
    def extract_and_concatenate_non_censored_peaks(peaks, artifact_windows):
        peaks = np.array(peaks)  # Ensure peaks is a NumPy array for proper indexing
        logging.info(f"Converted peaks to numpy array: {peaks[:10]}")

        artifact_windows_adjusted = artifact_windows.copy()  # Copy of artifact windows to adjust  
        peaks_adjusted = peaks.copy()  # Copy of peaks to adjust
        shift_amount = 0  # Initialize total shift amount
        # this needs to reset after each successful shift

        for i, window in enumerate(artifact_windows):
            start_idx = window['start']
            end_idx = window['end']
            logging.info(f"Processing artifact window: {start_idx} to {end_idx}")

            # Find the peaks immediately before and after the artifact window
            peaks_before_window = peaks_adjusted[peaks_adjusted < start_idx]
            logging.info(f"Peaks before the artifact window: {peaks_before_window}")
            peaks_after_window = peaks_adjusted[peaks_adjusted > end_idx]
            logging.info(f"Peaks after the artifact window: {peaks_after_window[:10]}")

            if len(peaks_before_window) > 0 and len(peaks_after_window) > 0:
                peak_before = peaks_before_window[-1]
                peak_after = peaks_after_window[0]
                logging.info(f"Found peak_before: {peak_before}, peak_after: {peak_after}")
                # This is working correctly for at least the first window
                
                # Calculate the shift for this window
                shift_amount = peak_after - peak_before
                logging.info(f"Calculated shift_amount: {shift_amount}")

                # Apply the shift to peaks after the artifact window
                adjusted_peaks_after = peaks_after_window - shift_amount
                # list the 20 peaks after for debugging
                logging.info(f"Adjusted peaks after the artifact window: {adjusted_peaks_after[:10]}")

                # Concatenate the adjusted peaks after the window
                peaks_adjusted = np.concatenate((peaks_before_window, adjusted_peaks_after))
                logging.info(f"Concatenated adjusted peaks: {peaks_adjusted[:10]}")                

                # Remove duplicate peaks from the adjusted array
                peaks_adjusted = np.delete(peaks_adjusted, np.where(peaks_adjusted == peak_before)[0][0])
                logging.info(f"Removed duplicate peak_after from peaks_adjusted: {peaks_adjusted[:10]}")

                # Adjust the subsequent artifact windows
                for adjusted_window in artifact_windows[i+1:]:
                    adjusted_window['start'] -= shift_amount
                    adjusted_window['end'] -= shift_amount
                    logging.info(f"Adjusted artifact window: {adjusted_window}")
                
                # reset the shift amount
                shift_amount = 0
                
            else:
                logging.warning("No peaks found before or after the window. Skipping window adjustment.")

        logging.info(f"Final adjusted peaks: {peaks_adjusted[:20]}")
        
        return peaks_adjusted

    def calculate_hrv_stats(peaks, sampling_rate):
        if len(peaks) > 1:  # Proceed only if there are enough peaks
            try:
                logging.info(f"Calculating HRV for non-censored peaks.")
                hrv_stats = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
                return hrv_stats
            except Exception as e:
                logging.error(f"Error calculating HRV: {e}")
                return pd.DataFrame()
        else:
            logging.warning("Insufficient peaks for HRV calculation.")
            return pd.DataFrame()

    if save_suffix == 'artifact_censored':
        total_length = len(ppg_signal)  # Total length of the PPG signal
        logging.info(f"Total length of PPG signal: {total_length}")

        # Sorting and Logging artifact windows
        artifact_windows = sorted(artifact_windows, key=lambda x: x['start'])
        logging.info(f"Sorted artifact windows: {artifact_windows[:10]}")   

        # Extract and concatenate non-censored peaks with adjusted indices
        peaks_adjusted = extract_and_concatenate_non_censored_peaks(ppg_peaks, artifact_windows)

        # Calculate HRV statistics across the non-censored peaks
        hrv_indices = calculate_hrv_stats(peaks_adjusted, sampling_rate)

        # Logging the HRV statistics
        logging.info("HRV Statistics:")
        logging.info(hrv_indices)

        # Plot the PPG signal and peaks, and the R-R intervals midpoints and interpolated time series using Plotly
        rr_intervals = np.diff(peaks_adjusted)
        logging.info(f"R-R Intervals: {rr_intervals[:10]}")
        rr_intervals_ms = rr_intervals / sampling_rate * 1000  # Convert to milliseconds if sampling rate is in Hz
        rr_midpoints = peaks_adjusted[:-1] + rr_intervals / 2
        logging.info(f"R-R Intervals Midpoints: {rr_midpoints[:10]}")

        # Create cubic spline interpolation of R-R intervals
        cs = CubicSpline(rr_midpoints, rr_intervals_ms)
        logging.info("Created cubic spline interpolation for R-R intervals.")

        # Generate a finer set of points for the spline interpolation
        fine_rr_midpoints = np.linspace(rr_midpoints.min(), rr_midpoints.max(), num=1000)
        fine_rr_intervals_ms = cs(fine_rr_midpoints)
        logging.info(f"Generated fine interpolation points for R-R intervals: {fine_rr_midpoints[:10]}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("PPG Signal with Peaks", "R-R Intervals Midpoints and Interpolated Time Series"))
        logging.info("Created subplots for PPG signal and R-R intervals.")

        # Plot only the peaks_adjusted
        fig.add_trace(go.Scatter(x=peaks_adjusted, y=np.zeros_like(peaks_adjusted), mode='markers', name='Peaks'), row=1, col=1)
        logging.info("Added adjusted peaks to the plot.")

        # Plot the R-R intervals midpoints and interpolated time series
        fig.add_trace(go.Scatter(x=rr_midpoints, y=rr_intervals_ms, mode='markers', name='R-R Intervals Midpoints'), row=2, col=1)
        logging.info("Added R-R Intervals Midpoints to the plot.")
        fig.add_trace(go.Scatter(x=fine_rr_midpoints, y=fine_rr_intervals_ms, mode='lines', name='Interpolated Time Series'), row=2, col=1)
        logging.info("Added Interpolated Time Series to the plot.")

        fig.update_layout(title='PPG Signal and R-R Intervals',
                        xaxis_title='Time',
                        yaxis2_title='R-R Interval (ms)',
                        legend_title='Legend')

        # Save the plot as an HTML file
        plot_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_uncensored_concatenated_{save_suffix}.html")
        plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
        logging.info(f"Plot saved to {plot_filename}")

    else:
        logging.info(f"Calculating HRV Indices via Neurokit for case: {save_suffix}...")
        hrv_indices = nk.hrv(ppg_peaks, sampling_rate=sampling_rate, show=True)

    # Pull FD data from the DataFrame
    fd_upsampled_ppg = df['FD_Upsampled']
    
    # Voxel threshold for FD values
    voxel_threshold = 0.5
    
    # Create a mask for FD values less than or equal to 0.5 voxel threshold (mm)
    mask_ppg = fd_upsampled_ppg <= voxel_threshold

    # Apply the mask to both FD and PPG data
    filtered_fd_ppg = fd_upsampled_ppg[mask_ppg]
    filtered_ppg = ppg_signal[mask_ppg]

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
    if np.any(fd_upsampled_ppg > voxel_threshold):
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

            plot_filename = f"{base_filename}_fd_ppg_correlation_filtered_{save_suffix}.png"
            plot_filepath = os.path.join(save_directory, plot_filename)
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

    num_peaks = len(ppg_peaks)
    logging.info(f"Number of peaks: {num_peaks}")

    total_time_minutes = len(ppg_signal) / (sampling_rate * 60)
    average_ppg_frequency = num_peaks / total_time_minutes

    inter_ppg_intervals = np.diff(ppg_peaks) / sampling_rate
    average_inter_ppg_interval = inter_ppg_intervals.mean() if len(inter_ppg_intervals) > 0 else np.nan
    std_inter_ppg_interval = inter_ppg_intervals.std() if len(inter_ppg_intervals) > 0 else np.nan
    max_inter_ppg_interval = inter_ppg_intervals.max() if len(inter_ppg_intervals) > 0 else np.nan
    min_inter_ppg_interval = inter_ppg_intervals.min() if len(inter_ppg_intervals) > 0 else np.nan

    # Update the ppg_stats dictionary
    ppg_stats = {
        'R-Peak Count (# peaks)': num_peaks,
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
    summary_stats_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_elgendi_summary_statistics_{save_suffix}.tsv")
    ppg_summary_stats_df.to_csv(summary_stats_filename, sep='\t', header=True, index=False)
    logging.info(f"Saving summary statistics to TSV file: {summary_stats_filename}")
                                
# Define the function to create the Plotly figure during initial rendering and re-rendering    
def create_figure(df, valid_peaks, valid_ppg=[], artifact_windows=[], interpolation_windows=[]):
    """
    Generates a Plotly figure with subplots for PPG signal visualization, R-R intervals, and framewise displacement.

    This function creates a multi-panel figure to display the cleaned PPG signal with identified R peaks, 
    the R-R intervals calculated from these peaks, and the framewise displacement of the signal. This visualization 
    aids in the manual correction of R peak detection and the assessment of signal quality.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the PPG signal and framewise displacement data.
    - valid_peaks (list of int): Indices of the valid peaks within the PPG signal.

    Returns:
    - plotly.graph_objs._figure.Figure: The Plotly figure object with the PPG signal, R-R intervals,
      and framewise displacement plotted in separate subplots.
    """
    
    # Create a Plotly figure with the PPG data and peaks
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=('PPG with R Peaks', 'R-R Intervals Tachogram',
                                        'Framewise Displacement'),
                        vertical_spacing=0.065)
   
    # Define constants for signal processing
    sampling_rate = 100  # Hz, down-sampled rate of the PPG signal
    tr = 2  # seconds, repetition time for volume calculations
    
    # Convert the list back to a pandas Series
    #* If you have a specific index you'd like to use, replace `range(len(valid_ppg_list))` accordingly
    
    valid_ppg = pd.Series(valid_ppg, index=range(len(valid_ppg)))
    
    # Add the cleaned and corrected PPG signal to the first subplot
    fig.add_trace(go.Scatter(y=valid_ppg, mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
                  row=1, col=1)
    
    # Add markers for R Peaks on the PPG signal plot
    y_values = valid_ppg.iloc[valid_peaks].tolist()
    fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks',
                             marker=dict(color='red')), row=1, col=1)
    
    # Calculate RR intervals from the valid peaks
    if len(valid_peaks) > 1:
        rr_intervals = np.diff(valid_peaks) / sampling_rate * 1000  # Convert to milliseconds
        midpoint_samples = [(valid_peaks[i] + valid_peaks[i + 1]) // 2 for i in range(len(valid_peaks) - 1)]

        # Cubic spline interpolation with extrapolation
        cs = CubicSpline(midpoint_samples, rr_intervals, extrapolate=True)
        regular_time_axis = np.arange(len(valid_ppg))
        interpolated_rr = cs(regular_time_axis)

        # Initialize the RR interval column with NaNs
        rr_intervals_full = np.full(len(valid_ppg), np.nan)
        
        # Assign interpolated RR intervals to the corresponding midpoints
        for i, midpoint in enumerate(midpoint_samples):
            rr_intervals_full[midpoint] = rr_intervals[i]

        # Fill remaining NaN values by interpolation
        rr_intervals_full = pd.Series(rr_intervals_full).interpolate(method='cubic').to_numpy()

        # Calculate mean value of the nearest 5 R-R intervals for padding
        mean_rr_beginning = np.mean(rr_intervals[:5])
        mean_rr_end = np.mean(rr_intervals[-5:])

        # Extend interpolation to the beginning of the timeseries with mean value padding
        rr_intervals_full[:midpoint_samples[0]] = mean_rr_beginning

        # Extend interpolation to the end of the timeseries with mean value padding
        rr_intervals_full[midpoint_samples[-1]+1:] = mean_rr_end

        # Plotting the R-R intervals and the interpolated line
        fig.add_trace(
            go.Scatter(x=midpoint_samples, y=rr_intervals, mode='markers', name='R-R Midpoints', marker=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=regular_time_axis, y=rr_intervals_full, mode='lines', name='Interpolated R-R', line=dict(color='blue')),
            row=2, col=1
        )
    else:
        rr_intervals_full = np.full(len(valid_ppg), np.nan)
        
    # Set the threshold for framewise displacement outlier identification
    voxel_threshold = 0.5
    
    # Plot the framewise displacement in the third subplot with a horizontal line indicating the threshold
    fig.add_trace(go.Scatter(y=df['FD_Upsampled'], mode='lines', name='Framewise Displacement', line=dict(color='blue')), row=3, col=1)
    fig.add_hline(y=voxel_threshold, line=dict(color='red', dash='dash'), row=3, col=1)

    # Update layout and size
    fig.update_layout(height=1200, width=1800, title_text=f'PPG R Peak Correction Interface')

    # Update y-axis labels for each subplot
    fig.update_yaxes(title_text='Amplitude (Volts)', row=1, col=1)
    fig.update_yaxes(title_text='R-R Interval (ms)', row=2, col=1)
    fig.update_yaxes(title_text='FD (mm)', row=3, col=1)

    # Calculate the number of volumes (assuming 2 sec TR and given sampling rate)
    num_volumes = len(df['FD_Upsampled']) / (sampling_rate * tr)

    # Generate the volume numbers for the x-axis
    volume_numbers = np.arange(0, num_volumes)
    
    # Calculate the tick positions for the fourth subplot
    tick_interval_fd = 5  # Adjust this value as needed
    tick_positions_fd = np.arange(0, len(df['FD_Upsampled']), tick_interval_fd * sampling_rate * tr)
    tick_labels_fd = [f"{int(vol)}" for vol in volume_numbers[::tick_interval_fd]]
    
    # Update x-axis labels for each subplot
    fig.update_xaxes(title_text='Samples', row=1, col=1, matches='x')
    fig.update_xaxes(title_text='Samples', row=2, col=1, matches='x')
    fig.update_xaxes(title_text='Volume Number (2 sec TR)', tickvals=tick_positions_fd, ticktext=tick_labels_fd, row=3, col=1, matches='x')

    # Disable y-axis zooming for all subplots
    fig.update_yaxes(fixedrange=True)
    
    for window in artifact_windows:
        fig.add_shape(
            type="rect",
            x0=window['start'],
            x1=window['end'],
            y0=0,
            y1=1,
            line=dict(color="Red"),
            fillcolor="LightSalmon",
            opacity=0.5,
            layer='below',
            yref='paper'
        )
         
    # Assuming fig is your Plotly figure object
    for window in interpolation_windows:
        pre_artifact = window['pre_artifact']
        post_artifact = window['post_artifact']
        
        # Add shaded area for pre_artifact range
        fig.add_shape(type="rect",
                    x0=pre_artifact[0], x1=pre_artifact[1],
                    y0=0, y1=1,
                    fillcolor="LightGreen", opacity=0.5, layer="below", line_width=0,
                    yref='paper')  # Ensure this matches with your subplot reference if needed

        # Add shaded area for post_artifact range
        fig.add_shape(type="rect",
                    x0=post_artifact[0], x1=post_artifact[1],
                    y0=0, y1=1,
                    fillcolor="LightGreen", opacity=0.5, layer="below", line_width=0,
                    yref='paper')  # Ensure this matches with your subplot reference if needed

    # Package necessary data for reuse
    rr_data = {
        'valid_peaks': valid_peaks,
        'valid_ppg': valid_ppg,
        'interpolated_rr': rr_intervals_full,
        'rr_intervals': rr_intervals,
        'midpoint_samples': midpoint_samples,
        'regular_time_axis': regular_time_axis
    }

    # Return the figure
    return fig, rr_data

# Function to open Dash app in web browser
def open_browser():
    """
    Opens the default web browser to the Dash application's URL.

    This function is designed to be called after the Dash server starts, automatically
    launching the application in the user's default web browser. It targets the local
    server address where the Dash app is hosted.

    Exceptions are caught and logged to provide feedback in case the browser cannot be
    opened automatically, which might occur due to issues with web browser access or
    system permissions.

    Raises:
    - Exception: Catches and logs any exception that occurs while trying to open the web browser,
      ensuring that the application continues running and remains accessible via the URL.
    """
    # Adjust the logging level of Werkzeug
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Attempt to open a new web browser tab to the Dash application's local server address
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception as e:
        print(f"Error opening browser: {e}")

# Main entry point of the Dash application.
if __name__ == '__main__':
    
    # Set up a Timer to open the web browser to the application URL after a 1-second delay.
    Timer(1, open_browser).start()

    try:
        # Attempt to run the Dash server on the specified port.
        # Debug mode is set to False for production use, but can be enabled for development to provide additional debugging information.
        app.run_server(debug=False, port=8050)
    except Exception as e:
        # If an error occurs while attempting to run the server, log the error.
        # This could be due to issues such as the port being already in use or insufficient privileges.
        print(f"Error running the app: {e}")
        