""" 

Overview:

This script implements a Dash-based web application for correcting peaks in photoplethysmogram (PPG) data. 
It provides an interactive interface for uploading PPG data files, visualizing them, manually correcting the identified peaks, and saving the corrected data. 
The application integrates various Python packages including Pandas and Numpy for data manipulation, Plotly for visualization, and SciPy for signal processing. 

Usage:

- Activate the appropriate Python environment that has the required packages installed.
  For Conda environments: `conda activate nipype`
- Run the script from the command line: python correct_peaks_artifacts_revX.x.py --save_dir "/path/to/your/save/directory"
- The script automatically opens a web interface in the default browser where users can upload, correct, and save PPG data.
- Use control-c to stop the server and exit the application when finished with corrections and file saving. 

"""
from scipy.stats import linregress
from scipy.stats import pearsonr
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
from scipy.stats import t

# ! This is a functional peak correction interface for PPG data with artifact selection and (buggy) correction (rev9.0) [- 2024-04-25] 
# ! Working to finalize save and output (rev 9.x) etc
# ! Working to finalize artifact correction (rev 9.x) etc
# ! Working to integrate nk fixpeaks testing

# NOTE: conda activate nipype
# python correct_peaks_artifacts_rev5.0.py

# // TODO: track number of corrections made and save to a file
# // TODO - Fix the save function to handle different directory structures
# // TODO - Add plotly html output here or elsewhere?
# // TODO: implement subject- and run-specific archived logging
# // TODO: implement artifact selection and correction
# // TODO: Handle edge cases for artifact selection and correction
# // TODO: fix the trimming of heartbeats to remove the rise

# TODO: smooth the transitions between heartbeats at window boundaries
#? Much better now but still a little jagged at the edges
# TODO: implement amplitude scaling for inserted heartbeats
# Note: The amplitude scaling issue seems resolved with the revised beat template creation
# TODO: add pre and post average heartbeat plotting for QC
#? Is this worth implementing? - It could be useful for visualizing the interpolation process
# TODO: Median and standard deviation for heartbeat shape plots? Pre- and Post- correction?
#? This could be useful for visualizing the variability in the heartbeats before and after correction
# REVIEW: implement recalculation and saving of PPG and HRV statistics
# TODO: Fix sample tracking for artifact windows
# TODO: Test signal_fixpeaks: https://neuropsychology.github.io/NeuroKit/functions/signal.html#signal-fixpeaks
# TODO: Implement artifact-corrected and artifact-uncorrected HRV statistics and timeseries save out
# TODO: Implement statistical comparisons of artifact-corrected and artifact-free HRV stats - does interpolation bias findings?
# TODO: Need to add padding of some kind for the edges of the timeseries so that we have full sample (e.g., 35k samples) for ppg and r-r interval timeseries - during final save?
# TODO: Fix BUG where points can be click added/removed to r-r interval plot

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

        # REVIEW: Returns? ... return subject_id, session_id, taskName, run_id

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
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            # imported as list for dcc.Store since not accessed here directely
            valid_ppg = df['PPG_Clean']
            
            #%% Integration of NeuroKit2 fixpeaks function
            
            # NOTE: At present (v9.0) this test is redundant given the call to nk.ppg_peaks() with artifact correction in the preprocessing script. 
            
            initial_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.to_numpy()
            logging.info(f"Initial peaks imported from preprocessed dataframe for automated peak correction") 

            # Methods to apply
            methods = ['Kubios', 'neurokit']
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

                            # Initialize the artifact column to 0
                            df[artifact_type] = 0
                            
                            # Set 1 at the corresponding sample indices to indicate artifacts
                            df.loc[artifact_sample_indices, artifact_type] = 1
                    
                    # Process subspaces data
                    for subspace in subspaces:
                        if subspace in artifacts:
                            # Interpolate subspace data
                            subspace_data = artifacts[subspace]
                            if len(valid_peaks) > 1 and len(subspace_data) >= len(valid_peaks):
                                cs = CubicSpline(valid_peaks, subspace_data[:len(valid_peaks)])  # Ensure data length matches
                                interpolated_values = cs(np.arange(len(df)))
                                df[subspace] = interpolated_values
                            else:
                                logging.warning(f"Not enough data points to interpolate for subspace: {subspace}")
                else:
                    logging.info("No artifacts to process.")

            # Create subplots with shared x-axes
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=('Initial Peaks', 'NeuroKit Corrected Peaks', 'Kubios Corrected Peaks'))

            # Plot valid PPG with valid peaks
            valid_peak_vals = valid_ppg.iloc[valid_peaks].tolist()
            fig.add_trace(
                go.Scatter(x=valid_peaks, y=valid_peak_vals, mode='markers', name='Valid Peaks', marker=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=valid_ppg, mode='lines', name='Valid PPG', line=dict(color='green')),
                row=1, col=1
            )

            # Plot valid PPG with NeuroKit corrected peaks
            nk_corrected_peaks = df[df['Peaks_neurokit'] == 1].index.tolist()
            nk_corrected_vals = valid_ppg.iloc[nk_corrected_peaks].tolist()
            fig.add_trace(
                go.Scatter(x=nk_corrected_peaks, y=nk_corrected_vals, mode='markers', name='NeuroKit Peaks', marker=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=valid_ppg, mode='lines', name='Valid PPG', line=dict(color='green'), showlegend=False),
                row=2, col=1
            )

            # Plot valid PPG with Kubios corrected peaks and artifacts
            kubios_corrected_peaks = df[df['Peaks_Kubios'] == 1].index.tolist()
            kubios_corrected_vals = valid_ppg.iloc[kubios_corrected_peaks].tolist()
            fig.add_trace(
                go.Scatter(x=kubios_corrected_peaks, y=kubios_corrected_vals, mode='markers', name='Kubios Peaks', marker=dict(color='purple')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=valid_ppg, mode='lines', name='Valid PPG', line=dict(color='green'), showlegend=False),
                row=3, col=1
            )

            # If artifacts exist for the Kubios method, plot them
            if 'Kubios' in methods and artifacts:
                artifact_symbols = {
                    'ectopic': 'triangle-up', 
                    'missed': 'triangle-down', 
                    'extra': 'triangle-left', 
                    'longshort': 'triangle-right'
                }
                
                # We will map the peak indices (as they are numbered in sequence) to their corresponding sample indices.
                peak_to_sample_index = dict(zip(range(1, len(initial_peaks) + 1), initial_peaks))
                
                for artifact_type in artifact_types:
                    if artifact_type in artifacts:
                        # Map peak indices to sample indices
                        artifact_sample_indices = [peak_to_sample_index[peak_index] for peak_index in artifacts[artifact_type] if peak_index in peak_to_sample_index]
                        
                        artifact_vals = valid_ppg.loc[artifact_sample_indices].tolist() if artifact_sample_indices else []
                        
                        fig.add_trace(
                            go.Scatter(x=artifact_sample_indices, y=artifact_vals, mode='markers', 
                                    name=f'{artifact_type}', 
                                    marker=dict(symbol=artifact_symbols[artifact_type], color='orange', size=10)),
                            row=3, col=1
                        )
            else:
                logging.info("No artifacts to plot for Kubios.")

            # Calculate differences for NeuroKit and Kubios peaks compared to valid peaks
            neurokit_diff_added = set(df[df['Peaks_neurokit'] == 1].index.tolist()) - set(valid_peaks)
            neurokit_diff_removed = set(valid_peaks) - set(df[df['Peaks_neurokit'] == 1].index.tolist())
            logging.info(f"NeuroKit peaks added: {neurokit_diff_added}")
            logging.info(f"NeuroKit peaks removed: {neurokit_diff_removed}")

            kubios_diff_added = set(df[df['Peaks_Kubios'] == 1].index.tolist()) - set(valid_peaks)
            kubios_diff_removed = set(valid_peaks) - set(df[df['Peaks_Kubios'] == 1].index.tolist())
            logging.info(f"Kubios peaks added: {kubios_diff_added}")
            logging.info(f"Kubios peaks removed: {kubios_diff_removed}")
            
            # Add vertical lines for differing peaks
            # For NeuroKit
            for diff in neurokit_diff_added.union(neurokit_diff_removed):
                fig.add_shape(type="line",
                            x0=diff, y0=min(valid_ppg), x1=diff, y1=max(valid_ppg),
                            line=dict(color="RoyalBlue", width=2, dash="dash"),
                            row='all', col=1)
            # For Kubios
            for diff in kubios_diff_added.union(kubios_diff_removed):
                fig.add_shape(type="line",
                            x0=diff, y0=min(valid_ppg), x1=diff, y1=max(valid_ppg),
                            line=dict(color="Purple", width=2, dash="dash"),
                            row='all', col=1)

            # Update layout to show titles and adjust axes
            fig.update_layout(
                title='Evaluation of NeuroKit2 Automated Peak Correction Methods',
                #xaxis_title='Evaluation of NeuroKit2 Automated Peak Correction Methods',
                yaxis_title='PPG Signal',
                height=900
            )

            # Disable y-axis zooming for all subplots
            fig.update_yaxes(fixedrange=True)
            
            # Update x-axis labels for each subplot
            fig.update_xaxes(title_text='Samples', row=1, col=1, matches='x')
            fig.update_xaxes(title_text='Samples', row=2, col=1, matches='x')
            fig.update_xaxes(title_text='Samples', row=3, col=1, matches='x')
    
            # Update y-axis labels
            fig.update_yaxes(title_text="PPG Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="PPG Amplitude", row=2, col=1)
            fig.update_yaxes(title_text="PPG Amplitude", row=3, col=1)

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

            df.to_csv(full_new_path, sep='\t', compression='gzip', index=False)
            pio.write_html(fig, figure_filepath)
            logging.info(f"Saved corrected data to: {full_new_path}")
                        
            # Render initial Plotly figure with the PPG data and peaks
            fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
            # Initialize peak changes data
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks), 'samples_corrected': 0}
            
            # TODO - Can we add other columns to peak_changes for interpolated peaks and artifact windows?
            # TODO - OR, can we merge interpolated peaks into valid_peaks and track samples corrected?
 
            # BUG: Double peak correction when clicking on plot (R-R interval goes to 0 ms)

            return fig, df.to_json(date_format='iso', orient='split'), valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update, rr_data
            # NOTE: Update everything except artifact variables on first file upload (4 outputs updated, 5 unchanged = 9 total outputs)
            # REVIEW corrected output counts (12?)
        
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
            fig, rr_data = create_figure(df, valid_peaks, valid_ppg, artifact_windows, interpolation_windows)
            
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
            
            return fig, dash.no_update, valid_peaks, valid_ppg, peak_changes, dash.no_update, dash.no_update, None, None, show_artifact_selection, dash.no_update, rr_data
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
                    """
         
                    # Identify indices of surrounding valid peaks
                    num_local_peaks = 5 # Hardcoded number of peaks to include on either side of the artifact window
                    logging.info(f"Number of local peaks to search: {num_local_peaks}")

                    # Define search ranges, limited to 75 samples from the start and 50 samples from the end points
                    """Defines search_range_start and search_range_end which are adjusted indices to avoid including the boundary peaks directly in the search for nadirs, 
                    providing a buffer to ensure nadir detection is within valid data ranges."""
                    search_range_start = start + 1
                    logging.info(f"Search range start: {search_range_start}")
                    search_range_end = end - 1
                    logging.info(f"Search range end: {search_range_end}")
                    
                    # Note: These ranges were selected based on NeuroKit2 code for heartbeat segmentation and ppg waveform characteristics
                    # FIXME: Dynamic searching based on nearest 2 valid peaks instead of hard coding 
                    search_limit_start = 75  # Limit the search to 75 samples before the start peak to locate the pre-peak nadir
                    logging.info(f"Search limit start: {search_limit_start} samples after the manually selected start peak")
                    search_limit_end = 50  # Limit the search to 50 samples after the end peak to locate the post-peak nadir
                    logging.info(f"Search limit end: {search_limit_end} samples before the manually selected end peak")

                    # Using idxmin, look for the nadir after the artifact window start peak, limited by search range
                    start_search_end = min(search_range_start + search_limit_start, search_range_end)
                    logging.info(f"Start search end: {start_search_end}")
                    start_nadir = valid_ppg[search_range_start:start_search_end].idxmin()
                    logging.info(f"Start nadir: {start_nadir}")
                    
                    # Look for the nadir before the end peak, limited by search range
                    end_search_start = max(search_range_end - search_limit_end, search_range_start)
                    logging.info(f"End search start: {end_search_start}")
                    end_nadir = valid_ppg[end_search_start:search_range_end].idxmin()
                    logging.info(f"End nadir: {end_nadir}")

                    # Check if start and end nadirs are valid
                    if start_nadir is None or end_nadir is None or start_nadir >= end_nadir:
                        logging.error("Nadir detection failed: start nadir and end nadir are not valid")
                        return fig, valid_peaks, valid_ppg, peak_changes, interpolation_windows
                    else:
                        logging.info(f"True Interpolation Window Range: Start Nadir - {start_nadir}, End Nadir - {end_nadir}")
                        
                        # Adjust start and end to the nadirs for true interpolation window
                        true_start = start_nadir
                        logging.info(f"Start nadir {start_nadir} defined as True start of artifact window: {true_start}")
                        true_end = end_nadir
                        logging.info(f"End nadir {end_nadir} defined as True end of artifact window: {true_end}")
                          
                    # Adjust start and end to the nadirs for true interpolation window
                    """updates the latest_artifact dictionary to record these values for samples corrected logging."""
                    latest_artifact['start'] = true_start
                    latest_artifact['end'] = true_end
                    
                    #! BUG: the corrected samples is not working correctly
                     
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
                    
                    # Calculate the index for the start of the pre-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact start peak
                    # We add 1 because bisect_left gives us the index of the artifact start peak itself
                    pre_artifact_start_idx = max(0, start_peak_idx - num_local_peaks + 1)
                    logging.info(f"Pre artifact start index (peak index): {pre_artifact_start_idx}")
                    
                    # Calculate the index for the end of the post-artifact window
                    # This is the index of the peak num_local_peaks away from the artifact end peak
                    # We subtract 1 because bisect_right gives us the index of the peak after the artifact end peak
                    post_artifact_end_idx = min(end_peak_idx + num_local_peaks - 1, len(valid_peaks) - 1)
                    logging.info(f"Post artifact end index (peak index): {post_artifact_end_idx}")
                    
                    # Determine the actual sample number for the start of the pre-artifact window based on pre_artifact_start_idx
                    pre_artifact_start = valid_peaks[pre_artifact_start_idx] if pre_artifact_start_idx < len(valid_peaks) else 0
                    logging.info(f"Pre artifact start peak (sample index): {pre_artifact_start}")
                    
                    # The end of the pre-artifact window is the same as the start of the artifact window
                    pre_artifact_end = true_start
                    logging.info(f"Pre artifact end peak (sample index) {pre_artifact_end} marked as the start of the artifact window") 
                    
                    # Find the nadir (lowest point) before the pre-artifact window to include the complete waveform using hardcoded search_limit_start (here, 75 samples)
                    # NOTE: Here idxmin() returns the index of the first occurrence of the minimum value as the start of the ppg waveform pre-nadir
                    pre_artifact_nadir = valid_ppg[pre_artifact_start - search_limit_start: pre_artifact_start].idxmin()
                    logging.info(f"Pre artifact nadir (sample index): {pre_artifact_nadir} - Pre artifact start (sample index): {pre_artifact_start} - Pre artifact end (sample index): {pre_artifact_end}")
                    
                    # The start of the post-artifact window is the same as the end of the artifact window
                    post_artifact_start = true_end
                    logging.info(f"Post artifact start (sample index) {post_artifact_start} marked as the end of the artifact window {true_end}")  
                    
                    # Determine the actual sample number for the end of the post-artifact window based on post_artifact_end_idx
                    post_artifact_end = valid_peaks[post_artifact_end_idx] if post_artifact_end_idx >= 0 else len(valid_ppg)
                    logging.info(f"Post artifact end (sample index): {post_artifact_end}")
                    
                    # Find the nadir (lowest point) after the post-artifact window to include the complete waveform
                    # NOTE: Here idxmin() returns the index of the first occurrence of the minimum value as the end of the ppg waveform post-nadir
                    #! Should this be using search_limit_end (i.e., 50 samples) or search_limit_start (i.e., 75 samples)?
                    post_artifact_nadir = valid_ppg[post_artifact_end : post_artifact_end + search_limit_start].idxmin()
                    logging.info(f"Post artifact start (sample index): {post_artifact_start} - Post artifact end (sample index): {post_artifact_end} - Post artifact nadir (sample index): {post_artifact_nadir}")
                    
                    # Adjust the start of the pre-artifact window to the nadir to include the full waveform
                    # Note: Here we are shifting the start index from the peak to the nadir to include the full waveform
                    pre_artifact_start = pre_artifact_nadir
                    logging.info(f"Extended pre_artifact window: Start nadir (sample index) = {pre_artifact_start}, End nadir (i.e., interpolation start point) (sample index) = {pre_artifact_end}")
                    
                    # Adjust the end of the post-artifact window to the nadir to include the full waveform
                    # Note: Here we are shifting the end index from the peak to the nadir to include the full waveform
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
                    
                    """
                    NOTE: The last beat problem likely involves this portion of nk.ppg_segment code:
                    # pad last heartbeat with nan so that segments are equal length
                    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
                    after_last_index = heartbeats[last_heartbeat_key]["Index"] < len(ppg_cleaned)
                    heartbeats[last_heartbeat_key].loc[after_last_index, "Signal"] = np.nan
                    """
                    
                    # HACK: There is bug with last segmented heartbeat, so we add an extra valid peak index to the post_artifact_peaks array
                    # Add an extra valid peak index to the post_artifact_peaks array
                    # Verify that post_artifact_peaks_indices is not empty first (e.g., in case of edge artifact)
                    if post_artifact_peaks_indices.size > 0:
                        # Proceed with the existing logic
                        if len(valid_peaks) > post_artifact_peaks_indices[-1] + 1:
                            next_valid_peak_index = valid_peaks[post_artifact_peaks_indices[-1] + 1]
                            post_artifact_peaks = np.append(post_artifact_peaks, next_valid_peak_index)
                            logging.info(f"Post artifact peaks sample indices (including next valid peak): {post_artifact_peaks}")
                        else:
                            logging.warning("No valid peak found after the last post-artifact peak.")
                    else:
                        logging.warning("Post artifact peaks indices array is empty.")

                    # Concatenate pre- and post-artifact peaks to get the clean peaks around the artifact
                    clean_peaks = np.concatenate((pre_artifact_peaks, post_artifact_peaks))
                    logging.info(f"Clean peaks around the artifact window: {clean_peaks}")
                    
                    # Ensure clean_peaks are sorted; they correspond one-to-one with the heartbeats in the dictionary
                    clean_peaks.sort()
                    logging.info(f"Sorted clean_peaks: {clean_peaks}")

                    """Using neurokit2 to segment the PPG signal into heartbeats using the clean peaks, 
                    see https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ppg/ppg_segment.html#ppg_segment for more information."""
                    
                    # Segment the PPG signal into heartbeats using the clean peaks, Returns a dictionary containing DataFrames for all segmented heartbeats.
                    heartbeats = nk.ppg_segment(ppg_cleaned=valid_ppg, 
                                                peaks=clean_peaks, 
                                                sampling_rate=sampling_rate, 
                                                show=False)
                    logging.info(f"Segmented heartbeats within clean peaks using nk.ppg_segment function.")

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
                            
                            logging.info(f"Heartbeat DataFrame head for key {key} pre-reformatting:")
                            logging.info(heartbeat.head())
                            
                            # Rename 'Signal' to 'PPG_Values'
                            heartbeat.rename(columns={'Signal': 'PPG_Values'}, inplace=True)
                            #//logging.info(f"Renamed 'Signal' column to 'PPG_Values'.")

                            # Set 'Index' as the DataFrame index
                            heartbeat.set_index('Index', inplace=True)
                            #//logging.info(f"Set 'Index' as the DataFrame index.")

                            # Drop the 'Label' column as it is not needed
                            heartbeat.drop(columns=['Label'], inplace=True)
                            #//logging.info(f"Dropped the 'Label' column.")

                            logging.info(f"Heartbeat DataFrame head for key {key} post-reformatting:")
                            logging.info(heartbeat.head())
                            
                            # Check for NaNs in 'PPG_Values' and only proceed if the segment is clean
                            if not heartbeat['PPG_Values'].isna().any():
                                valid_keys.append(key)
                                logging.info(f"Valid heartbeat segment copied.")
                                
                                # Expected maximum length of a heartbeat in samples
                                max_heartbeat_length = int(sampling_rate * 1.3)  # 1300 ms or 1.3 seconds is beats per minute (bpm) of 46
                                logging.info(f"Expected maximum length of a heartbeat in samples: {max_heartbeat_length}")
                                
                                # Trim the heartbeat to the max_heartbeat_length
                                # HACK: This is a quick solution but may not be the best approach for all cases where mean resting bpm may be low
                                if len(heartbeat) > max_heartbeat_length:
                                    heartbeat = heartbeat.iloc[:max_heartbeat_length]
                                    logging.info(f"Heartbeat trimmed to the maximum expected length.")
                                
                                segmented_heartbeats.append(heartbeat['PPG_Values'].values)
                                logging.info(f"Appended the segmented heartbeat from key {key} to the segmented heartbeat list.")
                                
                                # Save the individual heartbeat as a raw CSV file
                                heartbeat_filename_raw = f'heartbeat_{true_start}_{true_end}_{key}_raw.csv'
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
                        heartbeat_plot_filename_raw = f'heartbeat_{true_start}_{true_end}_{key}_raw.html'
                        heartbeat_plot_filepath_raw = os.path.join(save_directory, heartbeat_plot_filename_raw)
                        fig.write_html(heartbeat_plot_filepath_raw)
                        logging.info(f"Saved the individual heartbeat segment plot as an HTML file.")
                        
                    # Find the maximum length of all heartbeats
                    max_length = max(len(heartbeat) for heartbeat in segmented_heartbeats)
                    logging.info(f"Maximum length found among segmented heartbeats: {max_length}")

                    # Pad heartbeats to the maximum length to enable mean and median waveform calculation
                    padded_heartbeats = []
                    for heartbeat in segmented_heartbeats:
                        if len(heartbeat) < max_length:
                            # Pad with the last value if shorter than max_length
                            padding = np.full(max_length - len(heartbeat), heartbeat[-1])
                            heartbeat = np.concatenate((heartbeat, padding))
                        padded_heartbeats.append(heartbeat)
                        logging.info(f"Heartbeat padded to maximum length: {len(heartbeat)}")

                    # Replace original list with padded heartbeats for mean and median calculation
                    segmented_heartbeats = padded_heartbeats

                    # Calculate the mean and median heartbeat waveforms
                    #//logging.info("Calculating mean and median heartbeat waveforms.")
                    mean_heartbeat = np.mean(segmented_heartbeats, axis=0)
                    #//logging.info(f"Mean heartbeat calculated: {mean_heartbeat}")
                    median_heartbeat = np.median(segmented_heartbeats, axis=0)
                    #//logging.info(f"Median heartbeat calculated: {median_heartbeat}")
                    
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
                    fig_mean_filename_raw = f'mean_heartbeat_{true_start}_{true_end}_raw.html'
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
                        title="Median Heartbeat Waveform",
                        xaxis_title='Sample Index',
                        yaxis_title='PPG Amplitude'
                    )
                    # Save the figure as HTML
                    fig_median_filename_raw = f'median_heartbeat_{true_start}_{true_end}_raw.html'
                    fig_median_filepath_raw = os.path.join(save_directory, fig_median_filename_raw)
                    fig_median.write_html(fig_median_filepath_raw)
                    logging.info(f"Saved the median heartbeat segment plot as an HTML file.")
                        
                    """
                    How the Derivative is Calculated:
                    
                    np.diff() Function: 
                    
                    The np.diff() function calculates the difference between consecutive elements in an array. 
                    When applied to the heartbeat signal, it computes the change in signal amplitude from one sample to the next across the entire array.
                    
                    Time Interval: The denominator in the derivative calculation is np.diff(np.arange(len(mean_heartbeat))), 
                    which represents the change in time between consecutive samples. 
                    Since the time between samples is uniform (given a constant sampling rate), 
                    this array is essentially a series of ones (i.e., [1, 1, 1, ..., 1]).
                    
                    Resulting Slope Array: 
                    The resulting mean_heartbeat_slope and median_heartbeat_slope arrays represent the slope of the signal at each sample point. 
                    The slope is defined as the change in signal amplitude over the change in time, which, in this case, is the sampling interval.
                    
                    Why the Derivative is Performed:
                    
                    The derivative of a PPG signal is used to locate the systolic peaks (positive slopes) and diastolic troughs (negative slopes) within a heartbeat. 
                    When the derivative changes sign from positive to negative, it indicates a peak, and vice versa for a trough.         
                    
                    """
                                        
                    # Calculate the derivative (slope) of the mean and median heartbeat signal
                    mean_heartbeat_slope = np.diff(mean_heartbeat) / np.diff(np.arange(len(mean_heartbeat)))
                    #//logging.info(f"Calculated the mean heartbeat slope {mean_heartbeat_slope}.")
                    median_heartbeat_slope = np.diff(median_heartbeat) / np.diff(np.arange(len(median_heartbeat)))
                    #//logging.info(f"Calculated the median heartbeat slope {median_heartbeat_slope}.")

                    # Plot mean heartbeat derivative waveform
                    fig_mean_derivative = go.Figure()
                    fig_mean_derivative.add_trace(go.Scatter(
                        x=np.arange(len(mean_heartbeat_slope)),
                        y=mean_heartbeat_slope,
                        mode='lines',
                        name='Mean Heartbeat Derivative'
                    ))
                    fig_mean_derivative.update_layout(
                        title="Mean Heartbeat Derivative",
                        xaxis_title='Sample Index',
                        yaxis_title='Rate of PPG Amplitude Change per Sample'
                    )
                    # Save the figure as HTML
                    fig_mean_derivative_filename_raw = f'mean_derivative_heartbeat_{true_start}_{true_end}_raw.html'
                    fig_mean_derivative_filepath_raw = os.path.join(save_directory, fig_mean_derivative_filename_raw)
                    fig_mean_derivative.write_html(fig_mean_derivative_filepath_raw)
                    logging.info(f"Saved the mean derivative heartbeat segment plot as an HTML file.")
                    
                    # Plot median heartbeat derivative waveform
                    fig_median_derivative = go.Figure()
                    fig_median_derivative.add_trace(go.Scatter(
                        x=np.arange(len(median_heartbeat_slope)),
                        y=median_heartbeat_slope,
                        mode='lines',
                        name='Median Heartbeat Derivative'
                    ))
                    fig_median_derivative.update_layout(
                        title="Median Heartbeat Derivative",
                        xaxis_title='Sample Index',
                        yaxis_title='Rate of PPG Amplitude Change per Sample'
                    )
                    # Save the figure as HTML
                    fig_median_derivative_filename_raw = f'median_derivative_heartbeat_{true_start}_{true_end}_raw.html'
                    fig_median_derivative_filepath_raw = os.path.join(save_directory, fig_median_derivative_filename_raw)
                    fig_median_derivative.write_html(fig_median_derivative_filepath_raw)
                    logging.info(f"Saved the median derivative heartbeat segment plot as an HTML file.")

                    #%% Next implement trimming of heartbeats 
                    
                    """
                    np.diff() calculates the difference between consecutive elements of an array. 
                    When you use it to find the zero crossings in a signal derivative, it effectively reduces the length of the array by 1 because it computes the difference between n and n+1 elements.
                    After using np.diff() to identify where the sign changes (which would indicate a zero crossing), the indices in the resulting array correspond to the position in the np.diff() array, 
                    not the original array. Therefore, when you locate a zero crossing using the indices from the np.diff() result, you need to add 1 to align it with the correct position in the original array. 
                    If you don't account for this shift, the identified zero crossings would be one position earlier than they actually are in the original signal.
                    """
                    
                    # Find zero crossings for the mean derivative
                    mean_crossings = np.where(np.diff(np.sign(mean_heartbeat_slope)))[0]
                    logging.info(f"Mean derivative zero crossings: {mean_crossings}")
                    mean_negative_to_positive_crossings = mean_crossings[mean_heartbeat_slope[mean_crossings] < 0]
                    logging.info(f"Mean negative-to-positive crossings: {mean_negative_to_positive_crossings}")
                    mean_positive_to_negative_crossings = mean_crossings[mean_heartbeat_slope[mean_crossings] > 0]
                    logging.info(f"Mean positive-to-negative crossings: {mean_positive_to_negative_crossings}")

                    # Find zero crossings for the median derivative
                    median_crossings = np.where(np.diff(np.sign(median_heartbeat_slope)))[0]
                    logging.info(f"Median derivative zero crossings: {median_crossings}")
                    median_negative_to_positive_crossings = median_crossings[median_heartbeat_slope[median_crossings] < 0]
                    logging.info(f"Median negative-to-positive crossings: {median_negative_to_positive_crossings}")
                    median_positive_to_negative_crossings = median_crossings[median_heartbeat_slope[median_crossings] > 0]
                    logging.info(f"Median positive-to-negative crossings: {median_positive_to_negative_crossings}")

                    # Define tolerance for matching zero crossings (in samples)
                    tolerance = 3

                    # Adjust existing function to account for tolerance
                    def find_common_crossings(mean_crossings, median_crossings, tolerance):
                        common_crossings = []
                        for mc in mean_crossings:
                            # Check if there is a crossing in median_crossings within the tolerance range
                            if any(abs(mc - median_crossings) <= tolerance):
                                common_crossings.append(mc)
                        return np.array(common_crossings)

                    # Calculate the common zero crossings
                    common_negative_to_positive_crossings = find_common_crossings(mean_negative_to_positive_crossings, median_negative_to_positive_crossings, tolerance)
                    logging.info(f"Common negative-to-positive crossings: {common_negative_to_positive_crossings}")
                    common_positive_to_negative_crossings = find_common_crossings(mean_positive_to_negative_crossings, median_positive_to_negative_crossings, tolerance)
                    logging.info(f"Common positive-to-negative crossings: {common_positive_to_negative_crossings}")

                    # Function to select index based on joint mean and median information
                    def select_joint_index(negative_to_positive_crossings, positive_to_negative_crossings, default_index, tolerance):
                        # Ensure that there is at least one crossing of each type
                        if negative_to_positive_crossings.size > 0:
                            # Select the first negative-to-positive crossing as the start
                            start_index = negative_to_positive_crossings[0] + 1  # Account for diff shift

                            # Find the closest subsequent negative-to-positive crossing after the peak
                            # This represents the nadir after the systolic peak
                            subsequent_np_crossings = negative_to_positive_crossings[negative_to_positive_crossings > start_index]
                            if subsequent_np_crossings.size > 0:
                                # Find the closest subsequent np crossing within a tolerance range
                                closest_subsequent_np = subsequent_np_crossings[0]
                                for crossing in subsequent_np_crossings:
                                    if abs(crossing - positive_to_negative_crossings[0]) <= tolerance:
                                        closest_subsequent_np = crossing
                                        break
                                end_index = closest_subsequent_np + 1
                            else:
                                end_index = default_index
                        else:
                            start_index, end_index = default_index, default_index
                        return start_index, end_index

                    # Select the indices based on the joint zero crossings with tolerance
                    start_index, end_index = select_joint_index(common_negative_to_positive_crossings, common_positive_to_negative_crossings, 0, tolerance)
                    logging.info(f"Selecting start and end indices based off of mean and median joint information.")
                    
                    # Ensure that start_index is before end_index and they are not the same
                    if start_index >= end_index or start_index == 0:
                        logging.error("Invalid indices for trimming. Adjusting to default.")
                        start_index = 0
                        end_index = len(mean_heartbeat_slope) - 1
                        # OPTIMIZE: Consider changing default indices to something more meaningful

                    logging.info(f"Refined Start trim index: {start_index}")
                    logging.info(f"Refined End trim index: {end_index}")

                    #%% Plotting the derivative analysis for heartbeat trimming
                    
                    # Create a figure with subplots
                    logging.info("Creating a figure with subplots for PPG waveform and derivative analysis.")
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=(
                        'Mean Heartbeat Waveform', 'Mean Heartbeat Derivative',
                        'Median Heartbeat Waveform', 'Median Heartbeat Derivative'
                    ))

                    # Add mean and median heartbeat waveforms and derivatives
                    fig.add_trace(go.Scatter(x=np.arange(len(mean_heartbeat)), y=mean_heartbeat, mode='lines', name='Mean Heartbeat'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(mean_heartbeat_slope)), y=mean_heartbeat_slope, mode='lines', name='Mean Derivative'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(median_heartbeat)), y=median_heartbeat, mode='lines', name='Median Heartbeat'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(median_heartbeat_slope)), y=median_heartbeat_slope, mode='lines', name='Median Derivative'), row=4, col=1)

                    # Determine the y-axis range for the derivative subplots
                    derivative_y_min = min(np.min(mean_heartbeat_slope), np.min(median_heartbeat_slope))
                    derivative_y_max = max(np.max(mean_heartbeat_slope), np.max(median_heartbeat_slope))

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
                    add_global_vertical_line(fig, start_index, 1, 1, 'Start Index', 'Green')
                    add_global_vertical_line(fig, end_index, 1, 1, 'End Index', 'Red')

                    # Update layout for the entire figure
                    fig.update_layout(
                        height=1200,
                        title_text=f"PPG Waveform and Derivative Analysis for Artifact Window {true_start} to {true_end}",
                        showlegend=True
                    )

                    # Save the figure as HTML
                    fig_html_path = os.path.join(save_directory, f"analysis_derivative_heartbeat_trimming_{true_start}_{true_end}.html")
                    fig.write_html(fig_html_path)
                    logging.info(f"Saved the PPG waveform analysis plot as an HTML file: {fig_html_path}")

                    # Trim the mean and median heartbeats using the calculated start and end indices
                    mean_heartbeat_trimmed = mean_heartbeat[start_index:end_index]
                    logging.info(f"Trimmed the mean heartbeat from {start_index} to {end_index} samples.")
                    median_heartbeat_trimmed = median_heartbeat[start_index:end_index]
                    logging.info(f"Trimmed the median heartbeat from {start_index} to {end_index} samples.")

                    # Apply the trimming indices to all segmented heartbeats
                    trimmed_heartbeats = [beat[start_index:end_index] for beat in segmented_heartbeats]
                    logging.info(f"Trimmed all heartbeats from {start_index} to {end_index} samples.")
                    
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
                        heartbeat_filename = f'trimmed_heartbeat_{true_start}_{true_end}_{i + 1}.csv'  # use index + 1 to match the key concept
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
                    heartbeat_plot_filename = f'trimmed_average_heartbeats_artifact_{true_start}_{true_end}.html'
                    heartbeats_plot_filepath = os.path.join(save_directory, heartbeat_plot_filename)
                    heartbeats_fig.write_html(heartbeats_plot_filepath)
                    logging.info(f"Saving the trimmed heartbeats figure as an HTML file at {heartbeats_plot_filepath}.")
                    
                    #! Line x Line performed up to this point in correct_artifacts() function
                    #%% NOTE: This section is where we calculate local reference signal statistics to guide the interpolation process
                    
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

                    # FIXME: Not optimal fitting choosing between estimated and actual beats
                    # If the estimated number of beats is not fitting well, adjust by comparing with the artifact duration
                    actual_beats_artifact_window = estimated_beats_artifact_window if (estimated_beats_artifact_window * local_rr_interval_samples <= artifact_window_samples) else estimated_beats_artifact_window - 1
                    logging.info(f"Adjusted estimated number of beats in artifact window: {actual_beats_artifact_window}")

                    #// Convert the average beat from DataFrame to a NumPy array for easier manipulation
                    #//mean_heartbeat_array = mean_heartbeat.values.flatten()

                    # Ensure mean_heartbeat is a flat NumPy array
                    if isinstance(mean_heartbeat, np.ndarray):
                        mean_heartbeat_array = mean_heartbeat_trimmed.flatten()
                    else:
                        mean_heartbeat_array = mean_heartbeat_trimmed.values.flatten()

                    # Repeat the average beat shape to fill the adjusted estimated missing beats
                    replicated_beats = np.tile(mean_heartbeat_array, actual_beats_artifact_window)
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
                    
                    # Calculate deviations of each R-R interval from the local mean
                    rr_deviations = np.abs(concatenated_rr_intervals - local_rr_interval)
                    logging.info(f"Individual R-R interval deviations: {rr_deviations}")

                    # Define a threshold for significant deviation
                    deviation_threshold = 25  # milliseconds

                    # Check if any individual R-R intervals deviate significantly from the local mean
                    significant_deviation = np.any(rr_deviations > deviation_threshold)
                    if significant_deviation:
                        logging.info("Significant individual R-R interval deviation detected, requiring adjustment.")
                        
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
                        x_old = np.linspace(0, 1, len(mean_heartbeat_array))
                        logging.info(f"Old length of average beat: {len(x_old)} samples")
                        x_new_adjusted = np.linspace(0, 1, int(len(mean_heartbeat_array) * stretch_factor))
                        logging.info(f"New length of average beat after adjustment: {len(x_new_adjusted)} samples")

                        # Interpolate the average beat to adjust its length
                        adjusted_mean_heartbeat = np.interp(x_new_adjusted, x_old, mean_heartbeat_array)
                        logging.info(f"Adjusted average beat length: {len(adjusted_mean_heartbeat)} samples")

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
                        replicated_adjusted_beats = np.tile(adjusted_mean_heartbeat, chosen_beats_artifact_window)
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
                        ## Apply cross-fade at the beginning and end of the artifact window for a smooth transition
                        start_faded = (1 - taper_window) * valid_ppg[true_start:true_start + fade_length] + taper_window * adjusted_replicated_beats[:fade_length]
                        logging.info(f"Start faded: {len(start_faded)} samples")
                        end_faded = (1 - taper_window) * adjusted_replicated_beats[-fade_length:] + taper_window * valid_ppg[true_start + artifact_window_samples - fade_length:true_start + artifact_window_samples]
                        logging.info(f"End faded: {len(end_faded)} samples")
                        middle_segment_adjusted = adjusted_replicated_beats[fade_length:-fade_length]
                        logging.info(f"Middle segment adjusted: {len(middle_segment_adjusted)} samples")
                        corrected_signal = np.concatenate((start_faded, middle_segment_adjusted, end_faded))
                        logging.info(f"Corrected signal length: {len(corrected_signal)} samples")
                        
                        #! Insert peak detection again on the corrected signal including boundary peaks to get the asymmetrical first and last r-r intervals
                        
                        # Get the y values (amplitudes) from concatenated_beats and their corresponding x indices
                        y_values = corrected_signal
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
                        concatenated_corrected_rr_intervals = np.diff(peaks_indices) / sampling_rate * 1000
                        logging.info(f"R-R intervals from the detected peaks: {concatenated_corrected_rr_intervals}")
                        
                        # Calculate the deviation of the first and last R-R interval from the local mean
                        first_rr_deviation = abs(concatenated_corrected_rr_intervals[0] - local_rr_interval)
                        logging.info(f"First R-R interval deviation: {first_rr_deviation} milliseconds")
                        first_rr_interval = concatenated_corrected_rr_intervals[0]
                        logging.info(f"First R-R interval: {first_rr_interval} milliseconds")
                        last_rr_deviation = abs(concatenated_corrected_rr_intervals[-1] - local_rr_interval)
                        logging.info(f"Last R-R interval deviation: {last_rr_deviation} milliseconds")
                        last_rr_interval = concatenated_corrected_rr_intervals[-1]
                        logging.info(f"Last R-R interval: {last_rr_interval} milliseconds")
                        
                        # Calculate the midpoint index of the first R-R interval
                        midpoint_first_rr = (peaks_indices[1] + peaks_indices[0]) // 2
                        logging.info(f"Midpoint of the first R-R interval: {midpoint_first_rr} samples")
                        
                        # Calculate the target R-R interval (average of first and last)
                        target_rr_interval = (first_rr_interval + last_rr_interval) / 2
                        logging.info(f"Target R-R interval: {target_rr_interval} milliseconds")

                        # Calculate how many samples each interval needs to change to meet the target interval
                        first_adjustment_samples = (first_rr_interval - target_rr_interval) / 1000 * sampling_rate
                        last_adjustment_samples = (last_rr_interval - target_rr_interval) / 1000 * sampling_rate
                        logging.info(f"First interval adjustment in samples: {first_adjustment_samples}")
                        logging.info(f"Last interval adjustment in samples: {last_adjustment_samples}")

                        """
                        #! Temporary fix to avoid the phase shift calculation
                        
                        # Calculate the shift required
                        # If first_adjustment_samples is positive, we need to shift left to decrease the first interval
                        # If last_adjustment_samples is negative, we need to shift right to increase the last interval
                        # Adjust by the average of these two values to try and center the adjustment
                        shift_samples = int(round((first_adjustment_samples - last_adjustment_samples) / 2))
                        shift_direction = 'left' if shift_samples > 0 else 'right'
                        shift_samples = abs(shift_samples)
                        logging.info(f"Calculated shift of {shift_samples} samples to the {shift_direction}")

                        # Apply the calculated shift to the signal
                        if shift_direction == 'left':
                            shifted_signal = np.roll(corrected_signal, -shift_samples)
                            shifted_signal[-shift_samples:] = corrected_signal[-1]  # Fill the end with the last value
                        else:
                            shifted_signal = np.roll(corrected_signal, shift_samples)
                            shifted_signal[:shift_samples] = corrected_signal[0]  # Fill the beginning with the first value

                        # Ensure the shifted signal has the correct length to fit into the artifact window
                        shifted_signal = shifted_signal[:len(corrected_signal)]  # Adjust length after the shift
                        valid_ppg[true_start:true_end + 1] = shifted_signal  # Insert the shifted signal into the valid_ppg array
                        logging.info(f"Shifted and adjusted signal inserted with length {len(shifted_signal)}")
                        logging.info(f"Applied a phase shift of {shift_samples} samples to the {shift_direction} to balance the first and last R-R intervals.")
                    else:
                        # If mean R-R interval difference is not significant, no adjustment needed
                        logging.info(f"No significant mean R-R interval difference detected: {mean_rr_difference} milliseconds")    
                        logging.info("No further adjusted artifact correction needed.")  
                        """
                        # Insert the first calculation of concatenated beats into the valid_ppg artifact window
                        valid_ppg[true_start:true_end + 1] = concatenated_beats
                        logging.info(f"Concatenated beats successfully assigned to valid_ppg.")

                        
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
     State('ppg-plot', 'figure'), # Access the current figure state
     State('rr-store', 'data')]  # Include rr-store data for saving corrected R-R intervals
)

# BUG: Save button able to be clicked twice and double save the data - not urgent to fix but annoying for the log file. 

# Main callback function to save the corrected data to file
def save_corrected_data(n_clicks, filename, data_json, valid_peaks, peak_changes, fig, rr_data):
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
        
        # Use the rr_data from the store to update the DataFrame with corrected R-R and PPG Data
        df['Regular_Time_Axis'] = pd.Series(rr_data['regular_time_axis'], index=df.index)
        logging.info(f"Regular time axis length: {len(rr_data['regular_time_axis'])}")
        df['PPG_Values_Corrected'] = pd.Series(rr_data['valid_ppg'], index=df.index)
        logging.info(f"Valid PPG length: {len(rr_data['valid_ppg'])}")
        #// df['PPG_Peaks_Corrected'] = pd.Series(rr_data['valid_peaks'], index=df.index)
        #// logging.info(f"Valid peaks length: {len(rr_data['valid_peaks'])}")
        #//df['Midpoint_Samples'] = pd.Series(rr_data['midpoint_samples'], index=df.index)
        #//logging.info(f"Midpoint samples length: {len(rr_data['midpoint_samples'])}")
        df['Interpolated_RR'] = pd.Series(rr_data['interpolated_rr'], index=df.index)
        logging.info(f"Interpolated R-R intervals length: {len(rr_data['interpolated_rr'])}")
        #//df['RR_Intervals_Corrected'] = pd.Series(rr_data['rr_intervals'], index=df.index)
        #//logging.info(f"R-R intervals length: {len(rr_data['rr_intervals'])}")
        
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

        # Call to compute HRV stats
        compute_hrv_stats(df, valid_peaks, filename, save_directory)

        return f"Data and corrected peak counts saved to {full_new_path} and {count_full_path}"

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
def compute_hrv_stats(df, valid_peaks, filename, save_directory):
    
    """
    draft code for HRV statistical recomputation
    """
    
    #%% #! SECTION 1: Variables setup
    
    sampling_rate = 100   # downsampled sampling rate
    logging.info(f"Downsampled PPG sampling rate: {sampling_rate} Hz")

    # FIXME: base_name and path handling
    
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
    
    # Compute Power Spectral Density 0 - 8 Hz for PPG
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG using multitapers hann windowing.")
    ppg_filtered_psd = nk.signal_psd(df['PPG_Values_Corrected'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
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
    plot_append = (f"corrected")
    plot_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_psd_{plot_append}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 3: PSD Plotly Plots (0 - 1 Hz HRV range) 
    # Compute Power Spectral Density 0 - 1 Hz for PPG
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG HRV using multitapers hann windowing.")
    ppg_filtered_psd_hrv = nk.signal_psd(df['Interpolated_RR'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
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
    plot_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_psd_hrv_{plot_append}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 4: Re-calculate HRV Stats
    
    # Pull FD data from the DataFrame
    fd_upsampled_ppg = df['FD_Upsampled']
    
    # Voxel threshold for FD values
    voxel_threshold = 0.5
    
    # Create a mask for FD values less than or equal to 0.5 voxel threshold (mm)
    mask_ppg = fd_upsampled_ppg <= voxel_threshold

    # Pull PPG data from the DataFrame
    ppg_corrected = df['PPG_Values_Corrected']

    # Apply the mask to both FD and PPG data
    filtered_fd_ppg = fd_upsampled_ppg[mask_ppg]
    filtered_ppg = ppg_corrected[mask_ppg]

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

            plot_filename = f"{base_filename}_fd_ppg_correlation_filtered_{plot_append}.png"
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

    # Average ppg Frequency (counts/min)
    total_time_minutes = len(ppg_corrected) / (sampling_rate * 60)
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
    summary_stats_filename = os.path.join(save_directory, f"{base_filename}_filtered_cleaned_ppg_elgendi_summary_statistics_{plot_append}.tsv")
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

    # Package necessary data for reuse
    rr_data = {
        'valid_peaks': valid_peaks,
        'valid_ppg': valid_ppg,
        'interpolated_rr': interpolated_rr,
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
        