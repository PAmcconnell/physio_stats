""" 

Overview:

This script implements a Dash-based web application for correcting peaks in photoplethysmogram (PPG) data. 
It provides an interactive interface for uploading PPG data files, visualizing them, manually correcting the identified peaks, and saving the corrected data. 
The application integrates various Python packages including Pandas and Numpy for data manipulation, Plotly for visualization, and SciPy for signal processing. 

Usage:

- Activate the appropriate Python environment that has the required packages installed.
  For Conda environments: `conda activate nipype`
- Run the script from the command line: python correct_peaks_artifacts_rev1.0.py --save_dir "/path/to/your/save/directory"
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
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
import os
import argparse
import datetime
import sys
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend for rendering
import matplotlib.pyplot as plt
import neurokit2 as nk
import bisect

# ! This is a functional peak correction interface for PPG data with artifact selection and correction (rev4.0) [- 2021-09-30] 
# ! Working to finalize save and output

# NOTE: conda activate nipype

# // TODO: track number of corrections made and save to a file
# // TODO - Fix the save function to handle different directory structures
# // TODO - Add plotly html output here or elsewhere?
# // TODO: implement subject- and run-specific archived logging

# // TODO: implement artifact selection and correction
# // TODO: Handle edge cases for artifact selection and correction

# TODO: add pre and post average heartbeat plotting for QC
# TODO: implement recalculation and saving of PPG and HRV statistics

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
    dcc.Store(id='peak-change-store', data={'added': 0, 'deleted': 0, 'original': 0, 'samples_corrected': 0}),  # To store the number of peak changes and samples corrected for logging
    dcc.Store(id='artifact-windows-store', data=[]),  # Store for tracking indices of corrected artifact windows
    dcc.Store(id='artifact-interpolation-store', data=[]),  # Store for tracking surrounding ppg timeseries used for artifact interpolation
    
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
     Output('artifact-interpolation-store', 'data')],  # Store for tracking surrounding ppg timeseries used for artifact interpolation
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
            df = parse_contents(contents)
            
            #FIXME: correct loading of valid_ppg for interpolation and saving out corrected data
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            valid_ppg = df['PPG_Clean']
            
            # Render initial Plotly figure with the PPG data and peaks
            fig = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            # Initialize peak changes data
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks), 'samples_corrected': 0}
            
            # TODO - Can we add other columns to peak_changes for interpolated peaks and artifact windows?
            # TODO - OR, can we merge interpolated peaks into valid_peaks and track samples corrected?
 
            # BUG: Double peak correction when clicking on plot (R-R interval goes to 0 ms)

            return fig, df.to_json(date_format='iso', orient='split'), valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update
            # NOTE: Update everything except artifact variables on first file upload (4 outputs updated, 5 unchanged = 9 total outputs)
        
        # Handling peak correction via plot clicks
        if triggered_id == 'ppg-plot' and clickData:
            clicked_x = clickData['points'][0]['x']
            df = pd.read_json(data_json, orient='split')

            # Handle peak addition
            if clicked_x in valid_peaks:
                logging.info(f"Deleting a peak at sample index: {clicked_x}")
                valid_peaks.remove(clicked_x)
                peak_changes['deleted'] += 1
                
            # Handle peak deletion
            else:
                logging.info(f"Adding a new peak at sample index: {clicked_x}")
                peak_indices, _ = find_peaks(valid_ppg)
                nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
                valid_peaks.append(nearest_peak)
                peak_changes['added'] += 1
                valid_peaks.sort()

            # Update the figure with the corrected peaks after each click correction
            fig = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            """             
            # Load existing figure
            if existing_figure:
                fig = go.Figure(existing_figure)
            """
                
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                ) 
            
            return fig, dash.no_update, valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update
            # NOTE: We are not updating the data-store (df), we are only updating the figure, valid_peaks, and peak_changes record for tracking 

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
            fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows = correct_artifacts(df, fig, valid_peaks, valid_ppg, peak_changes, artifact_windows, interpolation_windows)
            
            # Ensure the updated valid_ppg is passed to create_figure
            fig = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                ) 
            
            # Return updated stores and output & Reset start and end inputs after confirmation
            return fig, dash.no_update, dash.no_update, valid_ppg, dash.no_update, artifact_windows, artifact_window_output, None, None, show_artifact_selection, interpolation_windows
        
    # FIXME: More precise handling of errors and error messages. 
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, None, dash.no_update, dash.no_update]

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
                logging.info(f"Proccessing Artifact Window from Boundary Peaks: Start - {start}, End - {end}")
                  
                if start < end:
                                    
                    # Identify indices of surrounding valid peaks
                    num_local_peaks = 5 # Number of peaks to include on either side of the artifact window
                    logging.info(f"Number of local peaks to search: {num_local_peaks}")

                    # Define search ranges, limited to 50 samples from the start and end points
                    search_range_start = start + 1
                    search_range_end = end - 1
                    search_limit_start = 75  # Limit the search to 75 samples
                    search_limit_end = 50  # Limit the search to 50 samples

                    # Look for the nadir after the start peak, limited by search range
                    start_search_end = min(search_range_start + search_limit_start, search_range_end)
                    start_nadir = valid_ppg[search_range_start:start_search_end].idxmin()
                    
                    # Look for the nadir before the end peak, limited by search range
                    end_search_start = max(search_range_end - search_limit_end, search_range_start)
                    end_nadir = valid_ppg[end_search_start:search_range_end].idxmin()

                    # Check if start and end nadirs are valid
                    if start_nadir is None or end_nadir is None or start_nadir >= end_nadir:
                        logging.error("Nadir detection failed: start nadir and end nadir are not valid")
                        return fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows
                    else:
                        logging.info(f"True Interpolation Window Range: Start Nadir - {start_nadir}, End Nadir - {end_nadir}")
                        
                        # Adjust start and end to the nadirs for true interpolation window
                        true_start = start_nadir
                        true_end = end_nadir
                        
                    # Adjust start and end to the nadirs for true interpolation window
                    latest_artifact['start'] = true_start
                    latest_artifact['end'] = true_end
                        
                    # Calculate the expected length of the artifact window
                    interpolated_length = true_end - true_start + 1
                    logging.info(f"Expected interpolated length: {interpolated_length} samples")
                    
                    # Calculate the duration of the artifact window in seconds
                    artifact_duration = (true_end - true_start) / sampling_rate
                    logging.info(f"Artifact window duration: {artifact_duration} seconds")
                    
                    # Find the index of the peak immediately preceding the 'start' of the artifact window in 'valid_peaks'
                    start_peak_idx = bisect.bisect_left(valid_peaks, start)
                    
                    # Find the index of the peak immediately following the 'end' of the artifact window in 'valid_peaks'
                    # Subtract 1 to get the last peak that's within the end of the artifact window
                    end_peak_idx = bisect.bisect_right(valid_peaks, end) - 1  
                    
                    # Ensure that the start_peak_idx is not less than 0 (start of the data)
                    start_peak_idx = max(start_peak_idx, 0)
                    
                    # Ensure that the end_peak_idx is not beyond the last index of valid_peaks (end of the data)
                    end_peak_idx = min(end_peak_idx, len(valid_peaks) - 1)
                    
                    # Calculate the index for the start of the pre-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact start peak
                    # We add 1 because bisect_left gives us the index of the artifact start peak itself
                    pre_artifact_start_idx = max(0, start_peak_idx - num_local_peaks + 1)
                    
                    # Calculate the index for the end of the post-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact end peak
                    # We subtract 1 because bisect_right gives us the index of the peak after the artifact end peak
                    post_artifact_end_idx = min(end_peak_idx + num_local_peaks - 1, len(valid_peaks) - 1)
                    
                    # Determine the actual sample number for the start of the pre-artifact window based on pre_artifact_start_idx
                    pre_artifact_start = valid_peaks[pre_artifact_start_idx] if pre_artifact_start_idx < len(valid_peaks) else 0
                    
                    # The end of the pre-artifact window is the same as the start of the artifact window
                    pre_artifact_end = true_start 
                    
                    # Find the nadir (lowest point) before the pre-artifact window to include the complete waveform
                    pre_artifact_nadir = valid_ppg[pre_artifact_start - 75 : pre_artifact_start].idxmin()
                    
                    logging.info(f"Pre artifact nadir: {pre_artifact_nadir} - Pre artifact start: {pre_artifact_start} - Pre artifact end: {pre_artifact_end}")
                    
                    # The start of the post-artifact window is the same as the end of the artifact window
                    post_artifact_start = true_end  
                    
                    # Determine the actual sample number for the end of the post-artifact window based on post_artifact_end_idx
                    post_artifact_end = valid_peaks[post_artifact_end_idx] if post_artifact_end_idx >= 0 else len(valid_ppg)
                    
                    # Find the nadir (lowest point) after the post-artifact window to include the complete waveform
                    post_artifact_nadir = valid_ppg[post_artifact_end : post_artifact_end + 75].idxmin()

                    logging.info(f"Post artifact start: {post_artifact_start} - Post artifact end: {post_artifact_end} - Post artifact nadir: {post_artifact_nadir}")
                    
                    # Adjust the start of the pre-artifact window to the nadir to include the full waveform
                    pre_artifact_start = pre_artifact_nadir
                    logging.info(f"Extended pre_artifact window: Start = {pre_artifact_start}, End = {pre_artifact_end}")
                    
                    # Adjust the end of the post-artifact window to the nadir to include the full waveform
                    post_artifact_end = post_artifact_nadir
                    logging.info(f"Extended post_artifact window: Start = {post_artifact_start}, End = {post_artifact_end}")

                    # Update interpolation_windows with pre and post artifact ranges
                    interpolation_windows.append({'pre_artifact': (pre_artifact_start, pre_artifact_end),
                                                'post_artifact': (post_artifact_start, post_artifact_end)})
                    logging.info(f"Interpolation windows successfully appended: {interpolation_windows}")
                    
                    # Ensure valid_peaks is a NumPy array
                    valid_peaks = np.array(valid_peaks)

                    # Ensure pre_artifact_start and post_artifact_end are single integer values
                    pre_artifact_start = int(pre_artifact_start)
                    post_artifact_end = int(post_artifact_end)

                    # Calculate the average R-R interval from the clean peaks surrounding the artifact
                    # Adjust the indexing to correctly identify pre and post artifact peaks based on extended windows including nadirs
                    pre_artifact_peaks_indices = np.where((valid_peaks >= pre_artifact_start) & (valid_peaks < pre_artifact_end))[0]
                    post_artifact_peaks_indices = np.where((valid_peaks > post_artifact_start) & (valid_peaks <= post_artifact_end))[0]

                    # Extract the pre and post artifact peaks using the calculated indices
                    pre_artifact_peaks = valid_peaks[pre_artifact_peaks_indices]
                    post_artifact_peaks = valid_peaks[post_artifact_peaks_indices]

                    # Ensure 'valid_peaks' is an array of integers
                    valid_peaks = np.array(valid_peaks, dtype=int)

                    # Concatenate pre- and post-artifact peaks to get the clean peaks around the artifact
                    clean_peaks = np.concatenate((pre_artifact_peaks, post_artifact_peaks))

                    # Segment the PPG signal into heartbeats using the clean peaks
                    heartbeats = nk.ppg_segment(ppg_cleaned=valid_ppg, 
                                                peaks=clean_peaks, 
                                                sampling_rate=sampling_rate, 
                                                show=False)
                    
                    # Ensure clean_peaks are sorted; they correspond one-to-one with the heartbeats in the dictionary
                    clean_peaks.sort()
                    logging.info(f"Sorted clean_peaks: {clean_peaks}")

                    # Initialize a list to hold all segmented heartbeats
                    segmented_heartbeats = []

                    # Iterate over the clean peaks and corresponding heartbeats
                    logging.info("Iterating over the clean peaks to align the heartbeats.")
                    for i, peak_index in enumerate(clean_peaks):
                        
                        # The key for accessing heartbeats is 1-based
                        key = str(i + 1)
                        logging.info(f"Accessing the heartbeat using the key: {key}")
                        logging.info(f"Processing peak index: {peak_index}")

                        if key in heartbeats:
                            heartbeat = heartbeats[key].copy()

                            # Rename 'Signal' to 'PPG_Values'
                            heartbeat.rename(columns={'Signal': 'PPG_Values'}, inplace=True)

                            # Set 'Index' as the DataFrame index
                            heartbeat.set_index('Index', inplace=True)

                            # Drop the 'Label' column as it is not needed
                            heartbeat.drop(columns=['Label'], inplace=True)

                            # Calculate the derivative (slope) of the heartbeat signal
                            heartbeat['Slope'] = heartbeat['PPG_Values'].diff() / np.diff(heartbeat.index.to_numpy(), prepend=heartbeat.index[0])
                            logging.info(f"Calculated the derivative (slope) of the heartbeat signal.")

                            # Check the slope of the last few points and find where it starts to rise
                            end_slope_window = heartbeat['Slope'][-10:]  # last 10 points as an example
                            rise_start_index = end_slope_window[end_slope_window > 0].first_valid_index()
                            logging.info(f"Rise start index identified...trimming segmentation...: {rise_start_index}")   

                            # If a rise is detected, trim the heartbeat before the rise starts
                            if rise_start_index is not None:
                                heartbeat = heartbeat.loc[:rise_start_index - 1]
                                logging.info(f"Trimmed the heartbeat before the rise starts.")

                            # Log the start index, peak index, end index, and total length in samples for the heartbeat
                            start_index = heartbeat.index[0]
                            end_index = heartbeat.index[-1]
                            total_length = len(heartbeat)
                            logging.info(f"Heartbeat start index: {start_index}, peak index: {peak_index}, end index: {end_index}, total length in samples: {total_length}")

                            # Append the original heartbeat segment to the list
                            segmented_heartbeats.append(heartbeat['PPG_Values'].values)
                            logging.info(f"Appended heartbeat segment from key {key}.")
                                
                            # Save the individual heartbeat as a CSV file
                            heartbeat_filename = f'heartbeat_{true_start}_{true_end}_{key}.csv'
                            heartbeat_filepath = os.path.join(save_directory, heartbeat_filename)
                            heartbeat.to_csv(heartbeat_filepath, index=True, index_label='Sample_Indices')
                    
                    # Determine the minimum length to truncate or pad heartbeats
                    logging.info("Performing quality control on segmented heartbeats.")
                    min_length = min(len(beat) for beat in segmented_heartbeats)
                    max_length = max(len(beat) for beat in segmented_heartbeats)
                    logging.info(f"Minimum heartbeat length: {min_length}, Maximum heartbeat length: {max_length}")
                    
                    # Truncate or pad all heartbeats to the minimum length
                    adjusted_heartbeats = []
                    for beat in segmented_heartbeats:
                        if len(beat) > min_length:
                            # Truncate the beat if it is longer than the minimum length
                            adjusted_beat = beat[:min_length]
                            logging.info(f"Truncated heartbeat to minimum length: {min_length}")
                        elif len(beat) < min_length:
                            # Pad the beat if it is shorter than the minimum length
                            padding = np.full(min_length - len(beat), beat[-1])  # pad with the last value of the beat
                            adjusted_beat = np.concatenate((beat, padding))
                            logging.info(f"Padded heartbeat to minimum length: {min_length}")
                        else:
                            adjusted_beat = beat  # no adjustment needed
                            logging.info(f"No adjustment needed for heartbeat")
                        adjusted_heartbeats.append(adjusted_beat)
                        logging.info(f"Heartbeat appended to adjusted_heartbeats.")
                    
                    # Convert the list of Series to a DataFrame for easier processing
                    segmented_heartbeats_df = pd.DataFrame(adjusted_heartbeats).fillna(0)  # Fill NaNs with 0 for a consistent length
                    
                    # Calculate the average beat shape, ignoring NaNs
                    logging.info("Calculating the average beat shape.")
                    average_beat = segmented_heartbeats_df.mean(axis=0, skipna=True)

                    # Create the figure for heartbeats, ensuring to handle NaNs in the plot
                    logging.info("Creating the figure for heartbeats with quality control.")
                    heartbeats_fig = go.Figure()

                    # Determine the time axis for the individual beats
                    num_points = segmented_heartbeats_df.shape[1]
                    logging.info(f"Number of points in the segmented heartbeats: {num_points}")
                    
                    # The time duration for each beat based on the number of samples and the sampling rate
                    time_duration_per_beat = num_points / sampling_rate

                    # Create the time axis from 0 to the time duration for each beat
                    time_axis = np.linspace(0, time_duration_per_beat, num=num_points, endpoint=False)
                    logging.info(f"Time axis created for the individual beats")

                    # Add individual heartbeats to the figure
                    logging.info("Adding individual heartbeats to the figure.")
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
                                name=f'Beat {index+1}'
                            )
                        )

                    # Add the average beat shape to the figure
                    logging.info("Adding the average beat shape to the figure.")
                    heartbeats_fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=average_beat,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Average Beat'
                        )
                    )

                    # Update the layout of the figure
                    logging.info("Updating the layout of the figure.")
                    heartbeats_fig.update_layout(
                        title=(f"Individual Heart Beats and Average Beat Shape for Artifact Window {true_start} to {true_end}"),
                        xaxis_title='Time (s)',
                        yaxis_title='PPG Amplitude',
                        xaxis=dict(range=[0, time_duration_per_beat])  # set the range for x-axis
                    )

                    # Save the figure as an HTML file
                    heartbeat_plot_filename = f'average_heartbeats_artifact_{true_start}_{true_end}.html'
                    heartbeats_plot_filepath = os.path.join(save_directory, heartbeat_plot_filename)
                    heartbeats_fig.write_html(heartbeats_plot_filepath)
                    logging.info(f"Saving the heartbeats figure as an HTML file at {heartbeats_plot_filepath}.")
                    
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
                    
                    # Calculate the number of samples in the artifact window
                    artifact_window_samples = int(np.round(artifact_duration * sampling_rate))
                    logging.info(f"Artifact window duration in samples: {artifact_window_samples} samples")

                    # Calculate expected slice length
                    expected_slice_length = true_end - true_start + 1
                    logging.info(f"Expected slice length: {expected_slice_length} samples")
                    
                    # Estimate the number of beats within the artifact window using the local average R-R interval
                    estimated_beats_artifact_window = int(np.round(artifact_duration / local_rr_interval_seconds))
                    logging.info(f"Estimated number of beats in artifact window: {estimated_beats_artifact_window}")

                    # If the estimated number of beats is not fitting well, adjust by comparing with the artifact duration
                    actual_beats_artifact_window = estimated_beats_artifact_window if (estimated_beats_artifact_window * local_rr_interval_samples <= artifact_window_samples) else estimated_beats_artifact_window - 1
                    logging.info(f"Adjusted estimated number of beats in artifact window: {actual_beats_artifact_window}")

                    # Convert the average beat from DataFrame to a NumPy array for easier manipulation
                    average_beat_array = average_beat.values.flatten()

                    # Repeat the average beat shape to fill the adjusted estimated missing beats
                    replicated_beats = np.tile(average_beat_array, actual_beats_artifact_window)
                    logging.info(f"Replicated beats length: {len(replicated_beats)} samples")

                    # Adjust the length of replicated beats to match the artifact window duration by interpolation
                    x_old = np.linspace(0, 1, len(replicated_beats))
                    logging.info(f"Old length of replicated beats: {len(replicated_beats)} samples")
                    x_new = np.linspace(0, 1, expected_slice_length) 
                    logging.info(f"New length of replicated beats: {expected_slice_length} samples")
                    replicated_beats = np.interp(x_new, x_old, replicated_beats)  # Adjust the length of replicated beats
                    logging.info(f"Replicated beats length (interpolated): {len(replicated_beats)} samples")
                    x_old = x_new # Update x_old to match the new length
                    logging.info(f"Updated x_old to match the new length: {len(x_old)} samples")
                    
                    # To create a smooth transition, generate a tapered window for cross-fading
                    fade_length = int(sampling_rate * 0.05)  # 5% of a second, adjust as needed
                    logging.info(f"Fade length for cross-fading: {fade_length} samples")
                    taper_window = np.linspace(0, 1, fade_length)
                    logging.info("Created taper window for cross-fading.")

                    # Apply cross-fade at the beginning of the artifact window
                    start_faded = (1 - taper_window) * valid_ppg[true_start:true_start + fade_length] + taper_window * replicated_beats[:fade_length]

                    # Apply cross-fade at the end of the artifact window
                    end_faded = (1 - taper_window) * replicated_beats[-fade_length:] + taper_window * valid_ppg[true_start + artifact_window_samples - fade_length:true_start + artifact_window_samples]
                
                    # Prepare for replacement, Calculate the total length of the section to be replaced in valid_ppg
                    total_replacement_length = artifact_window_samples 
                    logging.info(f"Total length of the section to be replaced: {total_replacement_length} samples")

                    # The middle segment is adjusted to fit exactly after considering start and end fades
                    middle_segment = replicated_beats[fade_length:-fade_length]
                    logging.info(f"Middle segment length: {len(middle_segment)} samples")
                    
                    # Generate the concatenated array for replacement correctly within the artifact window
                    concatenated_beats = np.concatenate((start_faded, middle_segment, end_faded))
                    concatenated_beats_length = len(concatenated_beats)
                    logging.info(f"Concatenated beats length: {concatenated_beats_length} samples")

                    # Get the y values (amplitudes) from concatenated_beats and their corresponding x indices
                    y_values = concatenated_beats
                    x_indices = np.arange(len(y_values))

                    # Find the indices of the top actual_beats_artifact_window (N)# of maxima
                    # argsort sorts in ascending order, so we take the last N indices for the highest values
                    top_maxima_indices = np.argsort(y_values)[-actual_beats_artifact_window:]

                    # Since we're interested in the exact x (sample indices) of these maxima
                    peaks_indices = x_indices[top_maxima_indices]

                    # Sort the x indices to maintain temporal order
                    peaks_indices = np.sort(peaks_indices)
                    
                    # Perform peak detection on the concatenated_beats within the boundaries
                    boundary_indices = [start, end]  # Use the correct indices from your context
                    logging.info(f"Boundary indices for peak detection: {boundary_indices}")
                    
                    nadir_indices = [start_nadir, end_nadir]  # Use the correct indices from your context
                    logging.info(f"Nadir indices for peak detection: {nadir_indices}")
                    
                    peaks_indices = [index + nadir_indices[0] for index in peaks_indices]
    
                    # Convert peaks_indices to a NumPy array if not already
                    peaks_indices = np.array(peaks_indices)

                    # Explicitly include boundary peaks if not already detected
                    if start not in peaks_indices:
                        peaks_indices = np.append(peaks_indices, start)
                        logging.info(f"Including start boundary peak: {start}")
                    if end not in peaks_indices:
                        peaks_indices = np.append(peaks_indices, end)
                        logging.info(f"Including end boundary peak: {end}")

                    # Ensure the peaks_indices are sorted since we might have appended boundary indices
                    peaks_indices = np.sort(peaks_indices)
                    logging.info(f"Including boundary peaks, adjusted peaks indices sorted: {peaks_indices}")

                    # Calculate the R-R intervals from the detected peaks
                    concatenated_rr_intervals = np.diff(peaks_indices) / sampling_rate * 1000
                    logging.info(f"R-R intervals from the detected peaks: {concatenated_rr_intervals}")

                    # Compute the mean and standard deviation of these intervals
                    concatenated_rr_mean = np.mean(concatenated_rr_intervals)
                    logging.info(f"Mean R-R interval within concatenated beats: {concatenated_rr_mean} milliseconds")
                    logging.info(f"Average Local R-R Interval: {local_rr_interval} milliseconds")
                    concatenated_rr_std = np.std(concatenated_rr_intervals)
                    logging.info(f"Standard deviation of R-R intervals within concatenated beats: {concatenated_rr_std}")
                    logging.info(f"Standard deviation of local R-R intervals: {std_local_rr_interval} milliseconds")

                    # Calculate the difference in mean R-R intervals and decide if adjustment is needed
                    mean_rr_difference = abs(local_rr_interval - concatenated_rr_mean)
                    logging.info(f"Difference in mean R-R intervals: {mean_rr_difference} milliseconds")
                    
                    if mean_rr_difference > 25:
                        logging.info(f"Significant mean R-R interval difference detected: {mean_rr_difference} milliseconds")
                        logging.info("Further adjustment of artifact correction needed.")

                        # Determine stretch or compression factor based on whether the mean R-R interval of concatenated beats is longer or shorter than desired
                        if concatenated_rr_mean > local_rr_interval:
                            logging.info("Concatenated beats have longer intervals, compressing them slightly.")
                            # If concatenated beats have longer intervals, compress them slightly
                            stretch_factor = local_rr_interval / concatenated_rr_mean
                            logging.info(f"Stretch factor: {stretch_factor}")
                        else:
                            logging.info("Concatenated beats have shorter intervals, stretching them slightly.")
                            # If concatenated beats have shorter intervals, stretch them slightly
                            stretch_factor = concatenated_rr_mean / local_rr_interval
                            logging.info(f"Stretch factor: {stretch_factor}")

                        # Adjust 'x_new' for stretching or compressing the average beat shape
                        x_old = np.linspace(0, 1, len(average_beat_array))
                        logging.info(f"Old length of average beat: {len(x_old)} samples")
                        x_new_adjusted = np.linspace(0, 1, int(len(average_beat_array) * stretch_factor))
                        logging.info(f"New length of average beat after adjustment: {len(x_new_adjusted)} samples")

                        # Interpolate the average beat to adjust its length
                        adjusted_average_beat = np.interp(x_new_adjusted, x_old, average_beat_array)
                        logging.info(f"Adjusted average beat length: {len(adjusted_average_beat)} samples")

                        # Calculate the total duration in seconds that the artifact window should cover
                        artifact_window_duration_seconds = expected_slice_length / sampling_rate
                        logging.info(f"Artifact window duration in seconds: {artifact_window_duration_seconds} seconds")

                        # Calculate the mean R-R interval for the estimated and actual number of beats
                        mean_rr_estimated = artifact_window_duration_seconds / estimated_beats_artifact_window * 1000  # Convert to milliseconds
                        logging.info(f"Mean R-R interval for first estimated number of beats: {mean_rr_estimated} milliseconds")
                        mean_rr_actual = artifact_window_duration_seconds / actual_beats_artifact_window * 1000  # Convert to milliseconds
                        logging.info(f"Mean R-R interval for adjusted number of beats (actual_beats_artifact_window): {mean_rr_actual} milliseconds")
                        
                        # Determine the deviation from the local_rr_interval for each
                        deviation_estimated = abs(local_rr_interval - mean_rr_estimated)
                        logging.info(f"Deviation from local R-R interval for estimated number of beats: {deviation_estimated} milliseconds")
                        deviation_actual = abs(local_rr_interval - mean_rr_actual)
                        logging.info(f"Deviation from local R-R interval for adjusted number of beats: {deviation_actual} milliseconds")

                        # Choose the option with the smallest deviation
                        if deviation_estimated <= deviation_actual:
                            logging.info("Estimated number of beats has smaller deviation from local R-R interval.")
                            chosen_beats_artifact_window = estimated_beats_artifact_window
                            logging.info("Using estimated_beats_artifact_window for replication.")
                        else:
                            logging.info("Adjusted number of beats has smaller deviation from local R-R interval.")
                            chosen_beats_artifact_window = actual_beats_artifact_window
                            logging.info("Using actual_beats_artifact_window for replication.")

                        # Now replicate the adjusted average beat according to the chosen option
                        replicated_adjusted_beats = np.tile(adjusted_average_beat, chosen_beats_artifact_window)
                        logging.info(f"Replicated adjusted beats length: {len(replicated_adjusted_beats)} samples")
                        
                        # Adjust the length of replicated beats to exactly match the artifact window's duration
                        # Use interpolation to fit the replicated beats into the expected_slice_length
                        x_old_replicated = np.linspace(0, 1, len(replicated_adjusted_beats))
                        logging.info(f"Old length of replicated adjusted beats: {len(x_old_replicated)} samples")
                        x_new_replicated = np.linspace(0, 1, expected_slice_length)
                        logging.info(f"New length of replicated adjusted beats: {len(x_new_replicated)} samples")
                        adjusted_replicated_beats = np.interp(x_new_replicated, x_old_replicated, replicated_adjusted_beats)
                        logging.info(f"Adjusted replicated beats length: {len(adjusted_replicated_beats)} samples")

                        # Prepare adjusted beats for insertion
                        # Apply cross-fade at the beginning and end of the artifact window for a smooth transition
                        start_faded = (1 - taper_window) * valid_ppg[true_start:true_start + fade_length] + taper_window * adjusted_replicated_beats[:fade_length]
                        logging.info(f"Start faded: {len(start_faded)} samples")
                        end_faded = (1 - taper_window) * adjusted_replicated_beats[-fade_length:] + taper_window * valid_ppg[true_start + artifact_window_samples - fade_length:true_start + artifact_window_samples]
                        logging.info(f"End faded: {len(end_faded)} samples")
                        middle_segment_adjusted = adjusted_replicated_beats[fade_length:-fade_length]
                        logging.info(f"Middle segment adjusted: {len(middle_segment_adjusted)} samples")
                        corrected_signal = np.concatenate((start_faded, middle_segment_adjusted, end_faded))
                        logging.info(f"Corrected signal length: {len(corrected_signal)} samples")

                        valid_ppg[true_start:true_end + 1] = corrected_signal
                        logging.info("Corrected signal with adjusted R-R intervals successfully updated in valid_ppg.")
                    else:
                        # If mean R-R interval difference is not significant, no adjustment needed
                        logging.info(f"No significant mean R-R interval difference detected: {mean_rr_difference} milliseconds")    
                        logging.info("No further adjusted artifact correction needed.")  

                        # Insert the first calculation of concatenated beats into the valid_ppg artifact window
                        valid_ppg[true_start:true_end + 1] = concatenated_beats
                        logging.info(f"Concatenated beats successfully assigned to valid_ppg.")

                        
                    # Ensure you're passing the correctly updated valid_ppg to create_figure
                    fig = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
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
                    
        return fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows
    
    except Exception as e:
        logging.error(f"Error in correct_artifacts: {e}")

@app.callback(
    Output('save-status', 'children'), # Update the save status message
    [Input('save-button', 'n_clicks')], # Listen to the save button click
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'), # Keep the data-store state
     State('peaks-store', 'data'), # Keep the peaks-store state
     State('peak-change-store', 'data'), # Keep the peak-change-store state
     State('ppg-plot', 'figure')] # Access the current figure state
)

# BUG: Save button able to be clicked twice and double save the data - not urgent to fix but annoying for the log file. 

# Main callback function to save the corrected data to file
def save_corrected_data(n_clicks, filename, data_json, valid_peaks, peak_changes, fig):
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
        
        # Define the new filename for the corrected data
        new_filename = f"{base_name}_corrected.{ext}"
        figure_filename = f"{base_name}_corrected_subplots.html"
        logging.info(f"Corrected filename: {new_filename}")
        logging.info(f"Figure filename: {figure_filename}")

        # Construct the full path for the new file
        full_new_path = os.path.join(save_directory, new_filename)
        figure_filepath = os.path.join(save_directory, figure_filename)
        
        # Write corrected figure to html file
        pio.write_html(fig, figure_filepath)

        # Load the data from JSON
        df = pd.read_json(data_json, orient='split')
        
        # Add a new column to the DataFrame to store the corrected peaks
        df['PPG_Peaks_elgendi_corrected'] = 0
        df.loc[valid_peaks, 'PPG_Peaks_elgendi_corrected'] = 1

        # Define sampling rate (down-sampled rate)
        sampling_rate = 100  # Hz

        # Calculate Corrected R-R intervals in milliseconds
        corrected_rr_intervals = np.diff(valid_peaks) / sampling_rate * 1000  # in ms

        # Calculate midpoints between R peaks in terms of the sample indices
        corrected_midpoint_samples = [(valid_peaks[i] + valid_peaks[i + 1]) // 2 for i in range(len(valid_peaks) - 1)]

        # Convert midpoint_samples to a NumPy array if needed
        corrected_midpoint_samples = np.array(corrected_midpoint_samples)
        
        # Generate a regular time axis for interpolation
        corrected_regular_time_axis = np.linspace(corrected_midpoint_samples.min(), corrected_midpoint_samples.max(), num=len(df))

        # Create a cubic spline interpolator
        cs = CubicSpline(corrected_midpoint_samples, corrected_rr_intervals)

        # Interpolate over the regular time axis
        corrected_interpolated_rr = cs(corrected_regular_time_axis)

        # Assign interpolated corrected R-R intervals to the DataFrame
        df['RR_Intervals_Corrected'] = pd.Series(corrected_interpolated_rr, index=df.index)
        
        # Save the DataFrame to the new file
        df.to_csv(full_new_path, sep='\t', compression='gzip', index=False)
    
        # Calculate corrected number of peaks
        corrected_peaks = len(valid_peaks)
        original_peaks = peak_changes['original']
        peaks_added = peak_changes['added']
        peaks_deleted = peak_changes['deleted']
        samples_corrected = peak_changes['samples_corrected']

        # Prepare data for saving
        peak_count_data = {
            'original_peaks': original_peaks,
            'peaks_deleted': peaks_deleted,
            'peaks_added': peaks_added,
            'corrected_peaks': corrected_peaks,
            'samples_corrected': samples_corrected
        }
        df_peak_count = pd.DataFrame([peak_count_data])

        # Save peak count data
        count_filename = f"{base_name}_corrected_peakCount.{ext}"
        logging.info(f"Corrected peak count filename: {count_filename}")
        count_full_path = os.path.join(save_directory, count_filename)
        df_peak_count.to_csv(count_full_path, sep='\t', compression='gzip', index=False)

        return f"Data and corrected peak counts saved to {full_new_path} and {count_full_path}"

    except Exception as e:
        logging.error(f"Error in save_corrected_data: {e}")
        return "An error occurred while saving data."

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
    
    # // # Add the cleaned PPG signal to the first subplot
    # // fig.add_trace(go.Scatter(y=df['PPG_Clean'], mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
    # //              row=1, col=1)
    
    # Add the cleaned and corrected PPG signal to the first subplot
    fig.add_trace(go.Scatter(y=valid_ppg, mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
                  row=1, col=1)
    
    # // # Add markers for R Peaks on the PPG signal plot
    # // y_values = df.loc[valid_peaks, 'PPG_Clean'].tolist()
    # //fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks',
    # //                         marker=dict(color='red')), row=1, col=1)
    
    # Add markers for R Peaks on the PPG signal plot
    #//y_values = df.loc[valid_peaks, valid_ppg].tolist()
    #//y_values = valid_ppg[valid_peaks].tolist()
    y_values = valid_ppg.iloc[valid_peaks].tolist()
    fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks',
                             marker=dict(color='red')), row=1, col=1)
    
    # Calculate R-R intervals from the valid peaks and plot them in the second subplot
    rr_intervals = np.diff(valid_peaks) / sampling_rate * 1000  # in ms

    # Calculate midpoints between R peaks in terms of the sample indices
    midpoint_samples = [(valid_peaks[i] + valid_peaks[i + 1]) // 2 for i in range(len(valid_peaks) - 1)]

    # Convert midpoint_samples to a NumPy array if needed
    midpoint_samples = np.array(midpoint_samples)
    
    # Generate a regular time axis for interpolation
    regular_time_axis = np.linspace(midpoint_samples.min(), midpoint_samples.max(), num=len(df))

    # Create a cubic spline interpolator
    cs = CubicSpline(midpoint_samples, rr_intervals)

    # Interpolate over the regular time axis
    interpolated_rr = cs(regular_time_axis)

    # Third Subplot: R-R Intervals Midpoints
    fig.add_trace(go.Scatter(x=midpoint_samples, y=rr_intervals, mode='markers', name='R-R Midpoints', marker=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=regular_time_axis, y=interpolated_rr, mode='lines', name='Interpolated R-R Intervals', line=dict(color='blue')), row=2, col=1)
    
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

    # Return the figure
    return fig

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
        