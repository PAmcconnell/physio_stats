""" 

Overview:

This script implements a Dash-based web application for correcting peaks in photoplethysmogram (PPG) data. 
It provides an interactive interface for uploading PPG data files, visualizing them, manually correcting the identified peaks, and saving the corrected data. 
The application integrates various Python packages including Pandas and Numpy for data manipulation, Plotly for visualization, and SciPy for signal processing. 

Usage:

- Activate the appropriate Python environment that has the required packages installed.
  For Conda environments: `conda activate nipype`
- Run the script from the command line: python correct_peaks_rev2.1.py --save_dir "/path/to/your/save/directory"
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
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import os
import argparse 

# ! This is a functional peak correction interface for PPG data without artifact selection and correction built in yet. 

# NOTE: conda activate nipype

# // TODO: track number of corrections made and save to a file
# TODO: implement artifact selection and correction
# TODO: implement recalculation and saving of PPG and HRV statistics
# TODO: implement subject and run specific archived logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

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

# Define the layout of the Dash application
app.layout = html.Div([
    
    # Store components for holding data in the browser's memory without displaying it
    dcc.Store(id='data-store'),  # To store the main DataFrame after processing
    dcc.Store(id='peaks-store'),  # To store indices of valid peaks identified in the PPG data
    dcc.Store(id='filename-store'),  # To store the name of the original file uploaded by the user
    dcc.Store(id='peak-change-store', data={'added': 0, 'deleted': 0, 'original': 0}), # To store the number of peak changes for logging
    
    # Upload component to allow users to select and upload PPG data files for correction
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select _processed.tsv.gz File to Correct Peaks'),
        style={},  # OPTIMIZE: Style properties can be added here for customization
        multiple=False  # Restrict to uploading a single file at a time
    ),
    
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

# Define content upload function
def upload_data(contents):
    """
    Processes the uploaded file's content, extracts valid peaks from the PPG data,
    and returns the data as a JSON string along with a list of valid peak indices.

    This function is intended to be used as a callback within the Dash application. It parses the
    uploaded file's contents, identifies valid PPG peaks based on a specific column criterion,
    and prepares the data for further processing or visualization within the app.

    Parameters:
    - contents (str): The base64 encoded contents of the uploaded file.

    Returns:
    - tuple: A tuple containing two elements:
        - The first element is a JSON string representation of the DataFrame generated from the uploaded file,
          formatted according to ISO standards and split orientation.
        - The second element is a list of indices representing valid PPG peaks identified within the data.

    Raises:
    - dash.exceptions.PreventUpdate: This exception is raised to prevent the Dash callback from
      updating its output if there are no contents in the uploaded file or if an error occurs
      during file processing, ensuring the application remains in a consistent state.
    """
    
    if contents:
        try:
            logging.info("Attempting to process uploaded file")
            
            # Parse the uploaded file content to a DataFrame
            df = parse_contents(contents)
            
            # Extract indices of rows where PPG peaks are marked as valid (e.g., marked with a 1)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            logging.info("File processed successfully")
            
            # Return the DataFrame as a JSON string and the list of valid peaks
            return df.to_json(date_format='iso', orient='split'), valid_peaks
        
        except Exception as e:
            # Log and raise an exception to prevent Dash callback update on error
            logging.error(f"Error processing uploaded file: {e}")
            raise dash.exceptions.PreventUpdate
    else:
        # Log and raise an exception to prevent Dash callback update if no file content is received
        logging.info("No file content received")
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('ppg-plot', 'figure'), # Update the figure with the PPG data and peaks
     Output('data-store', 'data'), # Update the data-store with the DataFrame
     Output('peaks-store', 'data'), # Update the peaks-store with the valid peaks
     Output('peak-change-store', 'data')],  # Add output for peak change tracking
    [Input('upload-data', 'contents'), # Listen to the contents of the uploaded file
     Input('ppg-plot', 'clickData')], # Listen to clicks on the PPG plot
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'), # Keep the data-store state
     State('peaks-store', 'data'), # Keep the peaks-store state
     State('ppg-plot', 'figure'), # Keep the existing figure state
     State('peak-change-store', 'data')] # Keep the peak-change-store state
)

# Main callback function to update the PPG plot and peak data
def update_plot_and_peaks(contents, clickData, filename, data_json, valid_peaks, existing_figure, peak_changes):
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
    - peak_changes (dict): A dictionary tracking the number of peaks added, deleted, or originally present.

    Returns:
    - A tuple containing the updated figure, data store JSON, valid peaks list, and peak changes dictionary.

    Raises:
    - dash.exceptions.PreventUpdate: Prevents updating the outputs if there's no new file uploaded or no click interaction on the plot.
    """
    
    ctx = dash.callback_context

    if not ctx.triggered:
        logging.info("No trigger for the callback")
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        # Handle file uploads and initialize the plot and peak data
        if triggered_id == 'upload-data' and contents:
            logging.info(f"Handling file upload: triggered ID: {triggered_id}")
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            
            # Render initial Plotly figure with the PPG data and peaks
            fig = create_figure(df, valid_peaks)
            
            # Initialize peak changes data
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks)}
            
            # TODO - Can we add other columns to peak_changes for interpolated peaks and artifact windows?
            # TODO - OR, can we merge interpolated peaks into valid_peaks and track samples corrected?
            
            # Log file names for debugging
            logging.info(f"Filename from update_plot_and_peaks: {filename}")
 
            # BUG: Double peak correction when clicking on plot (R-R interval goes to 0 ms)
            
            # TODO: Identify path to source directory for corrected file saving?
            
            return fig, df.to_json(date_format='iso', orient='split'), valid_peaks, peak_changes
            
            # NOTE: Update everything on first file upload (4 outputs)
        
        # Handling peak correction via plot clicks
        if triggered_id == 'ppg-plot' and clickData:
            logging.info(f"Handling peak correction: triggered ID: {triggered_id}")
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
                peak_indices, _ = find_peaks(df['PPG_Clean'])
                nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
                valid_peaks.append(nearest_peak)
                peak_changes['added'] += 1
                valid_peaks.sort()

            # Update the figure with the corrected peaks after each click correction
            fig = create_figure(df, valid_peaks)
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(
                    xaxis=current_layout['xaxis'],
                    yaxis=current_layout['yaxis']
                )
            return fig, dash.no_update, valid_peaks, peak_changes
            # NOTE: We are not updating the data-store (df), we are only updating the figure, valid_peaks, and peak_changes record for tracking 
        
        # FIXME: More precise handling of errors and error messages. 
        
        logging.error("Unexpected trigger in callback")
        raise dash.exceptions.PreventUpdate
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [fig, dash.no_update, dash.no_update, dash.no_update]
        # FIXME: return [dash.no_update] * 4 (necessary to return fig here?)

# TODO - Fix the save function to handle different directory structures

@app.callback(
    Output('save-status', 'children'), # Update the save status message
    [Input('save-button', 'n_clicks')], # Listen to the save button click
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'), # Keep the data-store state
     State('peaks-store', 'data'), # Keep the peaks-store state
     State('peak-change-store', 'data')] # Keep the peak-change-store state
)

# Main callback function to save the corrected data to file
def save_corrected_data(n_clicks, filename, data_json, valid_peaks, peak_changes):
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
    - peak_changes (dict): A dictionary tracking changes to the peaks, including added, deleted, and original counts.

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
        logging.info(f"Attempting to save corrected data for hidden filename: {filename}")
        
        # Extract subject_id and run_id
        parts = filename.split('_')
        if len(parts) < 4:
            logging.error("Filename does not contain expected parts.")
            return "Error: Filename structure incorrect."

        subject_id = parts[0]
        logging.info(f"Subject ID: {subject_id}")
        run_id = parts[3]
        logging.info(f"Run ID: {run_id}")
        
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
        logging.info(f"New filename: {new_filename}")

        # Construct the full path for the new file
        full_new_path = os.path.join(save_directory, new_filename)
        logging.info(f"Full path for new file: {full_new_path}")

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

        # Prepare data for saving
        peak_count_data = {
            'original_peaks': original_peaks,
            'peaks_deleted': peaks_deleted,
            'peaks_added': peaks_added,
            'corrected_peaks': corrected_peaks
        }
        df_peak_count = pd.DataFrame([peak_count_data])

        # Save peak count data
        count_filename = f"{base_name}_corrected_peakCount.{ext}"
        logging.info(f"Corrected peak count filename: {count_filename}")
        count_full_path = os.path.join(save_directory, count_filename)
        logging.info(f"Full path for corrected peak count file: {count_full_path}")
        df_peak_count.to_csv(count_full_path, sep='\t', compression='gzip', index=False)

        return f"Data and corrected peak counts saved to {full_new_path} and {count_full_path}"

    except Exception as e:
        logging.error(f"Error in save_corrected_data: {e}")
        return "An error occurred while saving data."
    
    # TODO - Add plotly html output here or elsewhere?

# Define the function to create the Plotly figure during initial rendering and re-rendering    
def create_figure(df, valid_peaks):
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
    
    # Add the cleaned PPG signal to the first subplot
    fig.add_trace(go.Scatter(y=df['PPG_Clean'], mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
                  row=1, col=1)
    
    # Add markers for R Peaks on the PPG signal plot
    y_values = df.loc[valid_peaks, 'PPG_Clean'].tolist()
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
    
    # Attempt to open a new web browser tab to the Dash application's local server address
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception as e:
        logging.error(f"Error opening browser: {e}")

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
        logging.error(f"Error running the app: {e}")
        