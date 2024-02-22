import base64
import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import Input, Output, State
import webbrowser
from threading import Timer
import logging
import pandas as pd
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import os
import bisect

# conda activate nipype
# TODO: implement artifact selection and correction
# TODO: implement recalculation and saving of PPG and HRV statistics
# TODO: implement subject and run specific archived logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout with Store components
app.layout = html.Div([
    dcc.Store(id='data-store'),  # To store the DataFrame
    dcc.Store(id='peaks-store'),  # To store the valid peaks
    dcc.Store(id='filename-store'),  # To store the original filename
    dcc.Store(id='peak-change-store', data={'added': 0, 'deleted': 0, 'original': 0}),
    html.Div(id='hidden-filename', style={'display': 'none'}),  # Hidden div for filename
    dcc.Upload(
    id='upload-data',
    children=html.Button('Select _processed.tsv.gz File to Correct Peaks'),
    style={},  # Add necessary style properties
    multiple=False),
    html.Div([
        html.Button('Toggle Mode', id='toggle-mode-button'),
        html.Div(id='mode-indicator')
    ]),
    dcc.Graph(id='ppg-plot'),
    html.Button('Save Corrected Data', id='save-button'),
    html.Div(id='save-status'),  # To display the status of the save operation
])

def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.BytesIO(decoded), sep='\t', compression='gzip')
    return df

def upload_data(contents):
    if contents:
        try:
            logging.info("Attempting to process uploaded file")
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            logging.info("File processed successfully")
            return df.to_json(date_format='iso', orient='split'), valid_peaks
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            raise dash.exceptions.PreventUpdate
    else:
        logging.info("No file content received")
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('ppg-plot', 'figure'),
     Output('data-store', 'data'),
     Output('peaks-store', 'data'),
     Output('hidden-filename', 'children'),  # Keep the hidden filename output
     Output('peak-change-store', 'data')],  # Add output for peak change tracking
    [Input('upload-data', 'contents'),
     Input('ppg-plot', 'clickData')],
    [State('upload-data', 'filename'),  # Keep filename state
     State('data-store', 'data'),
     State('peaks-store', 'data'),
     State('ppg-plot', 'figure'),
     State('peak-change-store', 'data'),
     State('hidden-filename', 'children')]  # Keep track of the filename
)
def update_plot_and_peaks(contents, clickData, filename, data_json, valid_peaks, existing_figure, peak_changes, hidden_filename):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logging.info(f"Triggered ID: {triggered_id}")
    
    if not ctx.triggered:
        logging.info("No trigger for the callback")
        raise dash.exceptions.PreventUpdate
    
    # Initialize outputs
    fig = go.Figure(existing_figure) if existing_figure else go.Figure()
    artifact_output = ""
    logging.info(F"Artifact output initialized as empty string: {artifact_output}")
    cancel_button_style = {'display': 'none'}
    confirm_button_style = {'display': 'block'}
    next_button_style = {'display': 'none'}
    confirmation_text = ""
    logging.info(f"Confirmation text initialized as empty string: {confirmation_text}")
    trigger_mode_change = ""
    logging.info(f"Trigger mode change initialized as empty string: {trigger_mode_change}")
    n_clicks_toggle = 0

    try:
        # Handling file upload
        if triggered_id == 'upload-data' and contents:
            logging.info(f"Handling file upload: triggered ID: {triggered_id}")
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            fig = create_figure(df, valid_peaks, updated_valid_peaks, corrected_artifacts, updated_rr_intervals)
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks)}
            new_filename = filename if isinstance(filename, str) else hidden_filename
            
            # Reset all artifact related outputs to default
            return [fig, df.to_json(date_format='iso', orient='split'), valid_peaks, new_filename, peak_changes, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update]
            
        # TODO: Fix double peak correction bug
        # TODO: Fix peak correction after artifact selection figure reset
     
        # Handling peak correction via plot clicks
        if mode_store['mode'] == 'peak_correction' and triggered_id == 'ppg-plot' and clickData:
            logging.info(f"Handling peak correction: triggered ID: {triggered_id}")
            clicked_x = clickData['points'][0]['x']
            df = pd.read_json(data_json, orient='split')
            if clicked_x in valid_peaks:
                logging.info(f"Deleting a peak at sample index: {clicked_x}")
                valid_peaks.remove(clicked_x)
                peak_changes['deleted'] += 1
            else:
                peak_indices, _ = find_peaks(df['PPG_Clean'])
                nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
                valid_peaks.append(nearest_peak)
                logging.info(f"Adding a new peak at sample index: {nearest_peak}")
                peak_changes['added'] += 1
                valid_peaks.sort()
        
            fig = create_figure(df, valid_peaks, updated_valid_peaks, corrected_artifacts, updated_rr_intervals)
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(xaxis=current_layout['xaxis'], yaxis=current_layout['yaxis'])
            return [fig, dash.no_update, valid_peaks, dash.no_update, peak_changes, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update]
        
        # Handle artifact selection
        if mode_store['mode'] == 'artifact_selection':
            logging.info(f"Handling artifact selection: triggered ID: {triggered_id}")
            
            # Define artifact start and end based on input values
            artifact_start = start_input
            artifact_end = end_input
            artifact_output = ""
            
            # Handle changes in artifact start/end inputs
            if 'artifact-start-input' in triggered_id or 'artifact-end-input' in triggered_id:
                logging.info(f"Handling artifact input change: triggered ID: {triggered_id}")
                
                # Make the confirm button visible again
                cancel_button_style = {'display': 'none'}
                confirm_button_style = {'display': 'block'}
                next_button_style = {'display': 'none'}
                
                # Return the current state with updated confirm button visibility
                return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                        artifact_output, existing_artifact_windows, artifact_start, 
                        artifact_end, cancel_button_style, dash.no_update, dash.no_update,
                        confirm_button_style, dash.no_update, dash.no_update, dash.no_update, 
                        dash.no_update, next_button_style, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update]
                
            # Manual mode toggle logic - Check if the mode toggle button was clicked
            if triggered_id == 'toggle-mode-button.n_clicks' and n_clicks_toggle > 0:
                
                # If the toggle button is clicked in artifact selection mode, switch back to peak correction mode
                logging.info(f"Manually switching back to peak correction mode: triggered id: {triggered_id}.")
                return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                        dash.no_update, existing_artifact_windows, dash.no_update, dash.no_update, 
                        cancel_button_style, dash.no_update, 0, confirm_button_style, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, next_button_style, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update]
                
            # TODO: Verify artifact-store content for multiple artifact windows.
            # TODO: Implement artifact correction.
            # TODO: Fix file save after artifact selection. 
            
            if 'confirm-artifact-button' in triggered_id and n_clicks_confirm > 0:
                try:
                    # Ensure start_input and end_input are strings
                    start_input_str = str(start_input)
                    end_input_str = str(end_input)

                    # Check for valid input values
                    if start_input_str.isdigit() and end_input_str.isdigit():
                        artifact_start = int(start_input_str)
                        artifact_end = int(end_input_str)
                        logging.info(f"Artifact start value: {artifact_start}, Artifact end value: {artifact_end}")

                        # Check for valid input range
                        if artifact_start >= artifact_end:
                            logging.error("Start value must be less than end value.")
                            raise ValueError("Start value must be less than end value.")
                    else:
                        # Invalid input values
                        logging.error(f"Invalid input values: {start_input_str}, {end_input_str}")
                        return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                            f"Error: Please enter valid numeric start and end values.", existing_artifact_windows, 0, 0, 
                            cancel_button_style, dash.no_update, dash.no_update, confirm_button_style, 
                            dash.no_update, 0, dash.no_update, dash.no_update, next_button_style, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update]
                        
                except ValueError as e:
                    logging.error(f"An error occurred: {e}")
                    # Reset the inputs and provide error feedback
                    return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            f"Error: {e}. Please enter valid start and end values.", existing_artifact_windows, 0, 0,
                            cancel_button_style, dash.no_update, dash.no_update, confirm_button_style,
                            dash.no_update, 0, dash.no_update, dash.no_update, next_button_style, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update]
                    
                logging.info(f"Artifact window confirmed: {artifact_start} to {artifact_end}: triggered ID: {triggered_id}")

                # Update the figure
                existing_artifact_windows.append({'start': artifact_start, 'end': artifact_end})
                logging.info(f"Appending artifact windows to existing: {existing_artifact_windows}")
                logging.info(f"Debug: Existing artifact windows type: {type(existing_artifact_windows)}")
                
                # Hide the confirm button as the artifact is now confirmed
                confirm_button_style = {'display': 'none'}

                # Show the cancel button
                cancel_button_style = {'display': 'block'}
                
                # Show the next selection button
                next_button_style = {'display': 'block'}
                
                # Add shaded rectangle marking the artifact window
                fig.add_shape(
                        type="rect",
                        x0=artifact_start,
                        x1=artifact_end,
                        y0=0,  # Start at the bottom of the figure
                        y1=1,  # Extend to the top of the figure
                        line=dict(color="Red"),
                        fillcolor="LightSalmon",
                        opacity=0.5,
                        layer='below',
                        yref='paper'  # Reference to the entire figure's y-axis
                    )

                # Hide the confirm button and show confirmation text
                return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                        "", existing_artifact_windows, 0, 0, cancel_button_style, dash.no_update, 
                        dash.no_update, confirm_button_style, f"Artifact window confirmed at {artifact_start} to {artifact_end}",
                        0, dash.no_update, dash.no_update, next_button_style, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update]
                
            # Cancelling the last confirmed artifact window
            if 'cancel-artifact-button' in triggered_id and n_clicks_cancel > 0:
                logging.info(f"Cancelling last artifact window: triggered ID: {triggered_id}")
           
                if saved_figure_json:
                    fig = go.Figure(saved_figure_json)  # Restore the saved figure state
                    # existing_artifact_windows.clear()  # Clear the artifact windows
                    if existing_artifact_windows:
                        logging.info(f"Removing last artifact window: {existing_artifact_windows[-1]}")
                        existing_artifact_windows.pop()  # Remove the last artifact window
                        logging.info(f"Remaining artifact windows: {existing_artifact_windows}")
                    if corrected_artifacts:
                        logging.info(f"Removing last corrected artifact window: {corrected_artifacts[-1]}")
                        existing_artifact_windows.pop()  # Remove the last artifact window
                        logging.info(f"Remaining corrected artifact windows: {corrected_artifacts}")
                        
                # Hide the cancel button again, show confirm button
                cancel_button_style = {'display': 'none'}
                next_button_style = {'display': 'none'}
                confirm_button_style = {'display': 'block'}
                
                # Reset the artifact window start and end input values to zero
                artifact_output = f"Artifact window cancelled from {artifact_start} to {artifact_end} Resetting values to zero."
                confirmation_text = f"Previous artifact window cancelled."
                artifact_start = artifact_end = 0
                
                # Trigger the client-side callback to switch mode back to peak correction
                trigger_mode_change = 'Switch to Peak Correction'

                return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                        artifact_output, existing_artifact_windows, 0, 0, cancel_button_style, 
                        trigger_mode_change, dash.no_update, confirm_button_style, confirmation_text, 
                        dash.no_update, 0, dash.no_update, next_button_style, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update] # TODO update returns for correction cancelation
            
            # Proceeding with next selection and artifact correction
            if 'next-selection-button' in triggered_id and n_clicks_next > 0:
                logging.info(f"Correcting confirmed artifact window: triggered ID: {triggered_id}")
            
                saved_figure_json = go.Figure(fig)  # Save current figure state
                df = pd.read_json(data_json, orient='split')
                
                # Initialize updated_valid_peaks as a zero-filled array matching the DataFrame's length
                if df is not None:  # Assuming df is your DataFrame
                    if updated_valid_peaks is None or len(updated_valid_peaks) != len(df):
                        updated_valid_peaks = np.zeros(len(df), dtype=int)
                        logging.info(f"Updated valid peaks initialized as zero-filled array with length: {len(df)}")
                        
                # Initalize updated_rr_intervals array
                if df is not None:
                    if updated_rr_intervals is None or len(updated_rr_intervals) != len(df):
                        updated_rr_intervals = np.zeros(len(df), dtype=float)
                        logging.info(f"Updated rr intervals initialized as zero-filled array with length: {len(df)}")
                        
                # # Initialize corrected_artifacts as an empty list (if this is the expected default)
                # if corrected_artifacts is None:
                #     corrected_artifacts = []
                #     logging.info(f"Corrected artifacts initialized as empty list: {corrected_artifacts}")

                # Initialize updated_peak_changes dictionary 
                if updated_peak_changes is None:
                    updated_peak_changes = {'added': 0, 'deleted': 0, 'original': 0, 'interpolated_length': 0, 'interpolated_peaks': 0}
                    logging.info(f"Updated peak changes initialized as empty dictionary: {updated_peak_changes}")
                    
                try: 
                    # Perform artifact correction
                    updated_data_json, updated_valid_peaks, updated_peak_changes, corrected_artifacts, updated_rr_intervals, updated_midpoint_flags = correct_artifacts(existing_artifact_windows, data_json, valid_peaks, corrected_artifacts, updated_valid_peaks, updated_peak_changes, updated_rr_intervals, updated_midpoint_flags)
                except Exception as e:
                    logging.error(f"Error in artifact correction: {e}")
                    raise dash.exceptions.PreventUpdate
                
                try:
                    # Update the figure with corrected data
                    updated_df = pd.read_json(updated_data_json, orient='split')
                    fig = create_figure(updated_df, valid_peaks, updated_valid_peaks, existing_artifact_windows, updated_rr_intervals)
                except Exception as e:
                    logging.error(f"Error in post-artifact correction figure updating: {e}")
                    raise dash.exceptions.PreventUpdate
                
                try:
                    # Apply the saved layout if available
                    if saved_layout:
                        fig.update_layout(saved_layout)
                    
                    # Add shaded rectangle marking the artifact window
                    if existing_artifact_windows:
                        latest_artifact = existing_artifact_windows[-1]
                        logging.info(f"Visualizing new figure: {latest_artifact}")

                    if 'start' in latest_artifact and 'end' in latest_artifact:
                        start, end = latest_artifact['start'], latest_artifact['end']
                        
                    fig.add_shape(
                            type="rect",
                            x0=start,
                            x1=end,
                            y0=0,  # Start at the bottom of the figure
                            y1=1,  # Extend to the top of the figure
                            line=dict(color="Red"),
                            fillcolor="LightSalmon",
                            opacity=0.5,
                            layer='below',
                            yref='paper'  # Reference to the entire figure's y-axis
                        )
                    logging.info(f"New figure created with artifact windows marked")
                
                except Exception as e:
                    logging.error(f"Error in plot layout updating: {e}")
                    raise dash.exceptions.PreventUpdate
                
                # TODO - Enable cancelation of last artifact window correction and reverting to previous state
                
                # TODO - Integration of updated data and peaks into the dataframe
                
                # valid_peaks = updated_valid_peaks  # Update valid peaks
                # peak_changes = updated_peak_changes  # Update peak changes
                
                # Reset the artifact selection process for next selection
                confirm_button_style = {'display': 'block'}
                next_button_style = {'display': 'none'}
                cancel_button_style = {'display': 'none'}
                trigger_mode_change = 'peak_correction'  # Auto-toggle back to peak correction mode

                # Reset start and end inputs
                start_input = end_input = ''
                logging.info(f"Reset start and end inputs: start_input={start_input}, end_input={end_input}")

                return [fig, updated_df.to_json(date_format='iso', orient='split'), valid_peaks, dash.no_update, peak_changes,
                        f"Proceeding with next selection...", existing_artifact_windows, start_input, end_input, 
                        cancel_button_style, trigger_mode_change, dash.no_update, confirm_button_style, 
                        confirmation_text, dash.no_update, dash.no_update, 0, next_button_style, corrected_artifacts,
                        updated_peak_changes, updated_valid_peaks, updated_rr_intervals, updated_midpoint_flags]
        else:
            cancel_button_style = {'display': 'none'} if not existing_artifact_windows else {'display': 'block'}

        # Default return if none of the conditions are met
        logging.info(f"Default return: {triggered_id}, none of the conditions are met")
        return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                f"No valid artifacts to process: triggered id = {triggered_id}.", existing_artifact_windows, 
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update]
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                f"Error in plot updating: {e}", existing_artifact_windows, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update]

def correct_artifacts(artifact_data, data_json, valid_peaks, corrected_artifacts, updated_valid_peaks, updated_peak_changes, updated_rr_intervals, updated_midpoint_flags):
    logging.info(f"Starting artifact correction. Artifact data: {artifact_data}") # Valid Peaks: {valid_peaks}")

    # # Initialize corrected artifacts windows list
    # if corrected_artifacts is None:
    #     corrected_artifacts = []
    #     logging.info("Initializing corrected_artifacts list.")
    
    # Load data from JSON into updated DataFrame
    updated_df = pd.read_json(data_json, orient='split')
    logging.info("Data loaded successfully from JSON, creating updated dataframe.")

    # Initialize PPG_Clean_Corrected as empty series
    if 'PPG_Clean_Corrected' not in updated_df.columns:
        updated_df['PPG_Clean_Corrected'] = pd.Series(index=updated_df.index, dtype='float64')
        logging.info("Initializing PPG_Clean_Corrected column.")
    
    # Initialize PPG_Peaks_elgendi_corrected_interpolated as empty series
    if 'PPG_Peaks_elgendi_corrected_interpolated' not in updated_df.columns:
        updated_df['PPG_Peaks_elgendi_corrected_interpolated'] = 0  # Initialize with zeros
        logging.info("Initializing PPG_Peaks_elgendi_corrected_interpolated column.")
        
    # Initialize PPG_Peaks_elgendi_corrected_interpolated as empty series
    if 'RR_Intervals_Corrected_Interpolated' not in updated_df.columns:
        updated_df['RR_Intervals_Corrected_Interpolated'] = 0  # Initialize with zeros
        logging.info("Initializing RR_Intervals_Corrected_Interpolated column.")
        
    # Initialize updated interpolated valid peaks array (contains interpolated peaks only)   
    if updated_valid_peaks is None:
        updated_valid_peaks = np.zeros(len(updated_df), dtype=int)  # Array to store updated peak information
        logging.info("Initializing updated_valid_peaks list.")
        
    # Ensure updated_valid_peaks is initialized correctly
    if updated_valid_peaks is None or len(updated_valid_peaks) != len(updated_df):
        updated_valid_peaks = np.zeros(len(updated_df), dtype=int)
        logging.warning(f"Reinitialized updated_valid_peak due to a size mismatch.")
        logging.info(f"Updated_valid_peaks length: {len(updated_valid_peaks)}, updated_df length: {len(updated_df)}")
        logging.info(f"Updated_valid_peaks type: {type(updated_valid_peaks)}, updated_df type: {type(updated_df)}")

    # # Initialize updated interpolated r-r interval array (contains interpolated artifact window peaks only)   
    # if updated_rr_intervals is None:
    #     updated_rr_intervals = np.zeros(len(updated_df), dtype=int)  # Array to store updated rr interval information
    #     logging.info("Initializing updated_rr_intervals list.")
     
    # Initialize interpolated peak changes tracking dictionary    
    if updated_peak_changes is None:
        updated_peak_changes = {'added': 0, 'deleted': 0, 'original': 0, 'interpolated_length': 0, 'interpolated_peaks': 0}
        logging.info("Initializing updated_peak_changes dictionary.")

    if not artifact_data:
        logging.warning("No artifact data provided. Exiting function.")
        raise dash.exceptions.PreventUpdate
    
    sampling_rate = 100  # Hz
    
    try:
        # TODO - Do we need to reset the peak changes tracking dictionary for each artifact?
        # Reset peak changes tracking for the current artifact correction session
        # updated_peak_changes = {'added': 0, 'deleted': 0, 'interpolated_length': 0, 'interpolated_peaks': 0}

        if artifact_data:
            latest_artifact = artifact_data[-1]
            logging.info(f"Processing artifact: {latest_artifact}")

            if 'start' in latest_artifact and 'end' in latest_artifact:
                start, end = latest_artifact['start'], latest_artifact['end']
                logging.info(f"Artifact window: Start - {start}, End - {end}")

                # TODO - Fix PPG interpolation timeseries
                
                if start < end:
                    
                    # Identify indices of surrounding valid peaks
                    num_local_peaks = 5 # Number of peaks to include on either side of the artifact window
                    logging.info(f"Number of local peaks to search: {num_local_peaks}")
                    
                    # Find the indices for the boundary peaks of the artifact window
                    start_peak_idx = bisect.bisect_left(valid_peaks, start)
                    logging.info(f"Start peak index: {start_peak_idx}")
                    
                    end_peak_idx = bisect.bisect_right(valid_peaks, end) - 1
                    logging.info(f"End peak index: {end_peak_idx}")

                    # Calculate R-R intervals before the artifact window
                    pre_artifact_intervals = valid_peaks[max(0, start_peak_idx - num_local_peaks):start_peak_idx + 1]
                    logging.info(f"Pre artifact intervals: {pre_artifact_intervals}")
                    
                    pre_rr_intervals = np.diff(pre_artifact_intervals) / sampling_rate * 1000
                    logging.info(f"Pre artifact R-R intervals: {pre_rr_intervals}")
                    
                    # Calculate R-R intervals after the artifact window
                    post_artifact_intervals = valid_peaks[end_peak_idx:min(len(valid_peaks), end_peak_idx + num_local_peaks + 1)]
                    logging.info(f"Post artifact intervals: {post_artifact_intervals}")
                    
                    post_rr_intervals = np.diff(post_artifact_intervals) / sampling_rate * 1000
                    logging.info(f"Post artifact R-R intervals: {post_rr_intervals}")
                    
                    # Combine and calculate local average R-R interval
                    combined_rr_intervals = np.concatenate((pre_rr_intervals, post_rr_intervals))
                    
                    pre_artifact_intervals_avg = np.mean(pre_rr_intervals)
                    logging.info(f"Pre artifact intervals average: {pre_artifact_intervals_avg}")
                    pre_artifact_intervals_std = np.std(pre_rr_intervals)   
                    logging.info(f"Pre artifact intervals standard deviation: {pre_artifact_intervals_std}")
                    
                    post_artifact_intervals_avg = np.mean(post_rr_intervals)
                    logging.info(f"Post artifact intervals average: {post_artifact_intervals_avg}")
                    post_artifact_intervals_std = np.std(post_rr_intervals)
                    logging.info(f"Post artifact intervals standard deviation: {post_artifact_intervals_std}")
                    
                    local_avg_rr_interval = np.mean(combined_rr_intervals)
                    local_std_rr_interval = np.std(combined_rr_intervals)
                    logging.info(f"Local average R-R interval (ms): {local_avg_rr_interval}")
                    logging.info(f"Local standard deviation of R-R interval (ms): {local_std_rr_interval}") 

                    try:
                        # Apply cubic spline interpolation to the PPG signal
                        x_range = np.arange(start, end + 1)
                        logging.info(f"X range for interpolation: {x_range}")
                        
                        interpolated_length = len(x_range)
                        logging.info(f"Interpolated length: {interpolated_length}")
                        
                        # TODO - integrate r-r interval changes into plot
                        # Update peak changes
                        if 'interpolated_length' in updated_peak_changes:
                            updated_peak_changes['interpolated_length'] += interpolated_length
                            logging.info(f"Updated interpolated length: {updated_peak_changes['interpolated_length']}")
                        else:
                            updated_peak_changes['interpolated_length'] = interpolated_length  # Initialize if not present
                            logging.info(f"Initialized interpolated length: {updated_peak_changes['interpolated_length']}")
                        
                        # Select data for interpolation - segments spanning through five valid peaks before and after the artifact window
                        pre_artifact_end = max(0, start - 1)
                        logging.info(f"Pre artifact end: {pre_artifact_end}")
                        
                        pre_artifact_start = max(0, pre_artifact_end - num_local_peaks * interpolated_length)
                        logging.info(f"Pre artifact start: {pre_artifact_start}")
                        
                        post_artifact_start = min(end + 1, len(updated_df))
                        logging.info(f"Post artifact start: {post_artifact_start}")
                        
                        post_artifact_end = min(post_artifact_start + num_local_peaks * interpolated_length, len(updated_df))
                        logging.info(f"Post artifact end: {post_artifact_end}")
                        
                        # Calculate the number of samples for pre and post artifact segments
                        pre_samples = min(num_local_peaks, start - pre_artifact_start)
                        logging.info(f"Pre samples: {pre_samples}")
                        
                        post_samples = min(num_local_peaks, len(updated_df) - post_artifact_start)
                        logging.info(f"Post samples: {post_samples}")

                        # Select pre and post artifact data segments for y axis
                        y_range_pre = updated_df.loc[pre_artifact_start:pre_artifact_end, 'PPG_Clean']
                        logging.info(f"Y range pre: {y_range_pre}")
                        
                        y_range_post = updated_df.loc[post_artifact_start:post_artifact_end, 'PPG_Clean']
                        logging.info(f"Y range post: {y_range_post}")
                        
                        # Modify x_range_combined and y_range_combined to include boundary points
                        boundary_start_y = updated_df.at[start, 'PPG_Clean']
                        logging.info(f"Boundary start y: {boundary_start_y}")
                        
                        boundary_end_y = updated_df.at[end, 'PPG_Clean']
                        logging.info(f"Boundary end y: {boundary_end_y}")
                        
                        # Combine pre, boundary, and post artifact data for fitting the cubic spline
                        y_range_combined = pd.concat([y_range_pre, pd.Series([boundary_start_y, boundary_end_y]), y_range_post])
                        logging.info(f"Y range combined: {y_range_combined}")
                    
                        # Generate x values for combined y range
                        x_range_combined = np.linspace(pre_artifact_start, post_artifact_end, num=len(y_range_combined))
                        logging.info(f"X range combined: {x_range_combined}")

                        # Ensure x_range_combined and y_range_combined are NumPy array for cubic spline
                        x_range_combined = np.array(x_range_combined)
                        y_range_combined = np.array(y_range_combined)
    
                        # Fit cubic spline to combined data
                        cs = CubicSpline(x_range_combined, y_range_combined)
                        logging.info(f"Cubic spline successfully fitted to combined data.")
                        
                        # Generate interpolated PPG signal within the artifact window
                        updated_df.loc[start:end, 'PPG_Clean_Corrected'] = cs(np.arange(start, end + 1))
                        logging.info(f"Interpolated PPG signal generated within the artifact window.")
                        
                        # Define the size of the smoothing window
                        smoothing_window_size = 9  # Approximately 10% of average peak-to-peak span

                        # Apply smoothing to the PPG_Clean_Corrected data within the artifact window
                        updated_df.loc[start:end, 'PPG_Clean_Corrected'] = uniform_filter1d(
                            updated_df.loc[start:end, 'PPG_Clean_Corrected'], 
                            size=smoothing_window_size)

                    except Exception as e:
                        logging.error(f"Error during spline interpolation: {e}")
                        raise dash.exceptions.PreventUpdate

                    # Correct handling of updated_corrected_artifacts
                    if not isinstance(corrected_artifacts, dict):
                        logging.warning("corrected_artifacts is not a dictionary. Reinitializing.")
                        corrected_artifacts = {'corrected_windows': [], 'total_samples_corrected': 0, 'total_interpolated_peaks': 0}

                    try:
                        # Append new artifact window to the 'corrected_windows' list within the dictionary
                        corrected_artifacts['corrected_windows'].append({'start': start, 'end': end})
                        logging.info(f"Updated corrected artifacts: {corrected_artifacts['corrected_windows']}")
                    except KeyError as e:
                        logging.error(f"KeyError when updating corrected artifacts: {e}")
                        corrected_artifacts = {'corrected_windows': [{'start': start, 'end': end}], 'total_samples_corrected': 0, 'total_interpolated_peaks': 0}
                    except Exception as e:
                        logging.error(f"Unexpected error updating corrected artifacts: {e}")
                        corrected_artifacts = {'corrected_windows': [{'start': start, 'end': end}], 'total_samples_corrected': 0, 'total_interpolated_peaks': 0}
                    
                    # Apply peak detection on the interpolated PPG signal within the artifact window
                    interpolated_peaks_window, _ = find_peaks(updated_df.loc[start:end, 'PPG_Clean_Corrected'])
                    
                    # After finding peaks
                    interpolated_peaks = interpolated_peaks_window + start

                    # Update valid peaks array and peak changes dictionary
                    for peak in interpolated_peaks:
                        logging.info(f"Updating peak in updated_valid_peaks dict: {peak}")
                        
                        if 0 <= peak < len(updated_df):
                            
                            logging.info(f"Peak index within bounds: {peak}")
                            try:
                                # Directly update the updated_df dataframe
                                updated_df.at[peak, 'PPG_Peaks_elgendi_corrected_interpolated'] = 1 
                                logging.info(f"Marked interpolated peak in Updated DataFrame at index {peak}")
                                
                                # Update the updated_valid_peaks array
                                updated_valid_peaks[peak] = 1
                                logging.info(f"Updated interpolated peak in updated_valid_peaks array: {peak}")
                                logging.info(f"Type of updated_valid_peaks during peak assignment: {type(updated_valid_peaks)}")
                                
                                # Update the updated_peak_changes dictionary
                                updated_peak_changes['interpolated_peaks'] = updated_peak_changes.get('interpolated_peaks', 0) + 1
                                logging.info(f"Updated interpolated peaks in updated_peak_changes: {updated_peak_changes['interpolated_peaks']}")
        
                            except IndexError as e:
                                logging.error(f"Index out of range error: {e}. Peak: {peak}, Array Length: {len(updated_valid_peaks)}")
                                
                        else:
                            logging.warning(f"Skipped updating peak as it is out of range: {peak}. Array Length: {len(updated_valid_peaks)}")

                    # Assuming 'PPG_Peaks_elgendi_corrected_interpolated' is the column in 'updated_df' DataFrame
                    # Find indices where PPG_Peaks_elgendi_corrected_interpolated is 1
                    interpolated_indices = np.where(updated_df['PPG_Peaks_elgendi_corrected_interpolated'] == 1)[0]
                    logging.info(f"Interpolated indices: {interpolated_indices}")
                    
                    boundary_indices = [start - 1, end + 1]  # Adjust as needed based on actual boundary peak indices
                    logging.info(f"Boundary indices: {boundary_indices}")
                    
                    # Check if there are any 1 values and log the indices
                    if interpolated_indices.size > 0:
                        logging.info(f"Interpolated peaks found at indices: {interpolated_indices}")
                    else:
                        logging.info("No interpolated peaks found during beat estimation operation.")

                    # TODO - Handle edge cases near start or end of timeseries with < num_peaks available for sampling
                    # TODO - Handle artifact windows with multiple interpolated peaks
                    
                    # Calculate R-R intervals and midpoints
                    midpoints = []
                    rr_intervals = []

                    # R-R intervals before the artifact window
                    for i in range(max(0, start_peak_idx - num_local_peaks), start_peak_idx):
                        if i < len(valid_peaks) - 1:
                            midpoints.append((valid_peaks[i] + valid_peaks[i + 1]) // 2)
                            rr_intervals.append((valid_peaks[i + 1] - valid_peaks[i]) * 1000 / sampling_rate)
                    
                    logging.info(f"Start peak index: {start_peak_idx}")
                    logging.info(f"End peak index: {end_peak_idx}")
                    logging.info(f"Midpoints before artifact window: {midpoints}")
                    logging.info(f"R-R intervals before artifact window: {rr_intervals}")
                    
                    # Calculate R-R intervals and midpoints for interpolated peaks within the artifact window
                    for idx, peak in enumerate(interpolated_peaks):
                        # Determine the previous and next valid peaks
                        prev_peak = valid_peaks[start_peak_idx - 1] if idx == 0 else valid_peaks[bisect.bisect_left(valid_peaks, peak) - 1]
                        next_peak = valid_peaks[end_peak_idx + 1] if idx == len(interpolated_peaks) - 1 else valid_peaks[bisect.bisect_right(valid_peaks, peak)]

                        # Calculate midpoints and intervals
                        midpoint_before = (peak + prev_peak) // 2
                        rr_interval_before = (peak - prev_peak) * 1000 / sampling_rate
                        midpoint_after = (peak + next_peak) // 2
                        rr_interval_after = (next_peak - peak) * 1000 / sampling_rate

                        # Append midpoints and intervals if they are within a reasonable range
                        if 0 < rr_interval_before < 1600:
                            midpoints.append(midpoint_before)
                            rr_intervals.append(rr_interval_before)
                        
                        if 0 < rr_interval_after < 1600:
                            midpoints.append(midpoint_after)
                            rr_intervals.append(rr_interval_after)

                        logging.info(f"Interpolated peak: {peak}, Previous peak: {prev_peak}, Next peak: {next_peak}")
                        logging.info(f"Midpoint before: {midpoint_before}, Midpoint after: {midpoint_after}")
                        logging.info(f"R-R interval before: {rr_interval_before}, R-R interval after: {rr_interval_after}")
                    
                    # Append midpoints and intervals for interpolated peaks
                    midpoints.extend([midpoint_before for midpoint_before, _, _ in interpolated_peaks])
                    midpoints.extend([midpoint_after for _, _, midpoint_after in interpolated_peaks])
                    rr_intervals.extend([rr_interval_before for _, rr_interval_before, _ in interpolated_peaks])
                    rr_intervals.extend([rr_interval_after for _, _, rr_interval_after in interpolated_peaks])

                    # Ensure unique and sorted midpoints for interpolation
                    midpoints, unique_indices = np.unique(midpoints, return_index=True)
                    rr_intervals = np.array(rr_intervals)[unique_indices]

                    # Interpolate R-R intervals using cubic spline
                    rr_interp = CubicSpline(midpoints, rr_intervals)
                    interpolated_rr_intervals = rr_interp(np.arange(len(updated_df)))

                    # Update the DataFrame with interpolated R-R intervals
                    updated_df['RR_Intervals_Corrected_Interpolated'] = interpolated_rr_intervals
                    # Ensure unique and sorted midpoints for interpolation
                    midpoints, unique_indices = np.unique(midpoints, return_index=True)
                    rr_intervals = np.array(rr_intervals)[unique_indices]
                    logging.info(f"Unique midpoints: {midpoints}")
                    logging.info(f"Unique R-R intervals: {rr_intervals}")

                    # Interpolate R-R intervals using cubic spline
                    rr_interp = CubicSpline(midpoints, rr_intervals)
                    interpolated_rr_intervals = rr_interp(np.arange(len(updated_df)))
                    logging.info(f"Interpolated R-R intervals: {interpolated_rr_intervals}")

                    # Update the DataFrame with interpolated R-R intervals
                    updated_df['RR_Intervals_Corrected_Interpolated'] = interpolated_rr_intervals
                    logging.info("Updated DataFrame with new interpolated R-R intervals.")

                    # Set flags for midpoints where interpolation occurred
                    updated_df['RR_Intervals_Corrected_Interpolated_Midpoints'] = 0
                    logging.info("Initialized DataFrame with new interpolated R-R intervals and midpoint flags.")
                    updated_df.loc[midpoints, 'RR_Intervals_Corrected_Interpolated_Midpoints'] = 1
                    logging.info("Updated DataFrame with new interpolated R-R intervals and midpoint flags.")

        logging.info("Updated DataFrame with new interpolated R-R intervals and midpoint flags.")
        updated_data_json = updated_df.to_json(date_format='iso', orient='split')
        logging.info("Artifact correction completed successfully.")

        return updated_data_json, updated_valid_peaks, updated_peak_changes, corrected_artifacts, updated_rr_intervals, updated_midpoint_flags

    except Exception as e:
        logging.error(f"An error occurred during artifact correction: {e}")
        raise dash.exceptions.PreventUpdate

# TODO - Fix the save function to handle different directory structures
        
@app.callback(
    Output('save-status', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('data-store', 'data'),
     State('peaks-store', 'data'),
     State('hidden-filename', 'children'),
     State('peak-change-store', 'data'),
     State('mode-store', 'data')]
)
def save_corrected_data(n_clicks, data_json, valid_peaks, hidden_filename, peak_changes, mode_store):
    
    if mode_store['mode'] != 'peak_correction':
        raise dash.exceptions.PreventUpdate
   
    if n_clicks is None or not data_json or not valid_peaks or not hidden_filename:
        logging.info("Save button not clicked, preventing update.")
        raise dash.exceptions.PreventUpdate

    try:
        # Extract subject_id and run_id
        parts = hidden_filename.split('_')
        if len(parts) < 4:
            logging.error("Filename does not contain expected parts.")
            return "Error: Filename structure incorrect."

        subject_id = parts[0]
        logging.info(f"Subject ID: {subject_id}")
        run_id = parts[3]
        logging.info(f"Run ID: {run_id}")
        
        # Construct the new filename by appending '_corrected' before the file extension
        if hidden_filename and not hidden_filename.endswith('.tsv.gz'):
            logging.error("Invalid filename format.")
            return "Error: Invalid filename format."
        
        parts = hidden_filename.rsplit('.', 2)
        if len(parts) == 3:
            base_name, ext1, ext2 = parts
            ext = f"{ext1}.{ext2}"  # Reassemble the extension
        else:
            # Handle the case where the filename does not have a double extension
            base_name, ext = parts
        logging.info(f"Base name: {base_name}")
        logging.info(f"Extension: {ext}")
        new_filename = f"{base_name}_corrected.{ext}"
        logging.info(f"New filename: {new_filename}")

        # Construct the full path for the new file
        derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/ppg/')
        full_new_path = os.path.join(derivatives_dir, subject_id, run_id, new_filename)
        logging.info(f"Full path for new file: {full_new_path}")

        # Load the data from JSON
        df = pd.read_json(data_json, orient='split')
        
        sampling_rate = 100  # Hz
        
        # Add a new column to the DataFrame to store the corrected peaks
        df['PPG_Peaks_elgendi_corrected'] = 0
        df.loc[valid_peaks, 'PPG_Peaks_elgendi_corrected'] = 1
    
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

        # Save corrected peak count data log
        count_filename = f"{base_name}_corrected_peakCount.{ext}"
        logging.info(f"Corrected peak count filename: {count_filename}")
        count_full_path = os.path.join(derivatives_dir, subject_id, run_id, count_filename)
        logging.info(f"Full path for corrected peak count file: {count_full_path}")
        df_peak_count.to_csv(count_full_path, sep='\t', compression='gzip', index=False)

        return f"Data, corrected R-R intervals, and corrected peak counts saved to {full_new_path} and {count_full_path}"

    except Exception as e:
        logging.error(f"Error in save_corrected_data: {e}")
        return "An error occurred while saving data."

def create_figure(df, valid_peaks, updated_valid_peaks, corrected_artifacts, updated_rr_intervals):
    
    # Create a Plotly figure with the PPG data and peaks
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=('PPG with R Peaks', 'R-R Intervals Tachogram',
                                        'Framewise Displacement'),
                        vertical_spacing=0.065)
    
    sampling_rate = 100 # Hz
 
    # Add traces to the first subplot (PPG with R Peaks)
    fig.add_trace(go.Scatter(y=df['PPG_Clean'], mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
                  row=1, col=1)
    
    # Add traces for R Peaks
    y_values = df.loc[valid_peaks, 'PPG_Clean'].tolist()
    fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks',
                             marker=dict(color='red')), row=1, col=1)
    
    # Add segments for corrected PPG_Clean_Corrected data
    if 'PPG_Clean_Corrected' in df.columns and corrected_artifacts:
        for artifact in corrected_artifacts:
            start, end = artifact['start'], artifact['end']
            corrected_segment = df.loc[start:end, 'PPG_Clean_Corrected']
            fig.add_trace(go.Scatter(x=corrected_segment.index, y=corrected_segment, mode='lines', 
                                     name='Corrected PPG Segment', line=dict(color='blue')), row=1, col=1)
        
        # Assuming 'PPG_Peaks_elgendi_corrected_interpolated' is the column in 'updated_df' DataFrame
        # Find indices where PPG_Peaks_elgendi_corrected_interpolated is 1
        interpolated_indices = np.where(df['PPG_Peaks_elgendi_corrected_interpolated'] == 1)[0]

        # Check if there are any 1 values and log the indices
        if interpolated_indices.size > 0:
            logging.info(f"Interpolated peaks found at indices: {interpolated_indices}")
        else:
            logging.info("No interpolated peaks found during create_figure operation.")
            
        # Check if 'PPG_Peaks_elgendi_corrected_interpolated' column exists in the DataFrame
        if 'PPG_Peaks_elgendi_corrected_interpolated' in df.columns:
            # Filter for rows where 'PPG_Peaks_elgendi_corrected_interpolated' is 1 (true for interpolated peaks)
            interpolated_peaks_mask = df['PPG_Peaks_elgendi_corrected_interpolated'].astype(bool)
            logging.info(f"Interpolated peaks mask for plotting: {interpolated_peaks_mask}")
            
            # Use the mask to filter the y-values of the interpolated peaks
            interpolated_y_values = df.loc[interpolated_peaks_mask, 'PPG_Clean_Corrected']
            logging.info(f"Interpolated y values for plotting: {interpolated_y_values}")

            # Plot only the interpolated peaks on the figure
            fig.add_trace(go.Scatter(
                x=interpolated_y_values.index, 
                y=interpolated_y_values, 
                mode='markers',
                name='Interpolated R Peaks', 
                marker=dict(color='purple')
            ), row=1, col=1)
            
            # Find the boundary peak indices
            boundary_start_idx = bisect.bisect_left(valid_peaks, start) - 1 if start > 0 else 0
            logging.info(f"Boundary start index for rr interval plotting: {boundary_start_idx}")
            boundary_end_idx = bisect.bisect_right(valid_peaks, end, 0, len(valid_peaks)) - 1
            logging.info(f"Boundary end index for rr interval plotting: {boundary_end_idx}")

            # Correctly identify the midpoints adjacent to the artifact window
            midpoint_before_artifact = (valid_peaks[boundary_start_idx] + valid_peaks[boundary_start_idx + 1]) // 2
            logging.info(f"Midpoint before artifact for rr interval plotting: {midpoint_before_artifact}")
            midpoint_after_artifact = (valid_peaks[boundary_end_idx] + valid_peaks[boundary_end_idx + 1]) // 2 if boundary_end_idx + 1 < len(valid_peaks) else valid_peaks[-1]
            logging.info(f"Midpoint after artifact for rr interval plotting: {midpoint_after_artifact}")
            
            # Adjust the end of the extended time axis if necessary
            if midpoint_after_artifact >= len(df):
                midpoint_after_artifact = len(df) - 1

            # Define the extended time axis for the interpolated R-R intervals
            extended_time_axis = np.arange(midpoint_before_artifact, midpoint_after_artifact + 1)
            logging.info(f"Extended time axis for rr interval plotting: {extended_time_axis}")

            # Ensure that interpolated_rr_intervals have the correct range
            interpolated_extended_rr = df['RR_Intervals_Corrected_Interpolated'][midpoint_before_artifact:midpoint_after_artifact + 1]
            logging.info(f"Interpolated extended rr intervals before plotting (from dataframe): {interpolated_extended_rr}")
            
            # [Plotting Code]

            # Plotting updated R-R intervals within the artifact window
            # Extract midpoints and interpolated R-R intervals from DataFrame
            updated_midpoints = df.index[df['RR_Intervals_Corrected_Interpolated_Midpoints'] == 1]
            logging.info(f"Updated midpoints for rr interval plotting: {updated_midpoints}")
            updated_rr_values = df['RR_Intervals_Corrected_Interpolated'][updated_midpoints]
            logging.info(f"Updated R-R values for rr interval plotting: {updated_rr_values}")

            # Plotting the updated R-R intervals
            fig.add_trace(go.Scatter(x=updated_midpoints, y=updated_rr_values, mode='markers', name='Updated R-R Midpoints', marker=dict(color='green')), row=2, col=1)

            # Plotting a line for interpolated intervals within the extended range
            # This assumes continuous interpolated R-R intervals within the extended range
            fig.add_trace(go.Scatter(x=extended_time_axis, y=interpolated_extended_rr, mode='lines', name='Extended Interpolated R-R Intervals', line=dict(color='purple')), row=2, col=1)
                
    # Calculate R-R intervals in milliseconds
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
    
    voxel_threshold = 0.5
    
    # Add traces to the fourth subplot (Framewise Displacement)
    fig.add_trace(go.Scatter(y=df['FD_Upsampled'], mode='lines', name='Framewise Displacement', line=dict(color='blue')), row=3, col=1)
    fig.add_hline(y=voxel_threshold, line=dict(color='red', dash='dash'), row=3, col=1)

    # Update layout and size
    fig.update_layout(height=1200, width=1800, title_text=f'PPG R Peak Correction Interface')

    # Update y-axis labels for each subplot
    fig.update_yaxes(title_text='Amplitude (Volts)', row=1, col=1)
    fig.update_yaxes(title_text='Interval (ms)', row=2, col=1)
    fig.update_yaxes(title_text='FD (mm)', row=3, col=1)

    # Calculate the number of volumes (assuming 2 sec TR and given sampling rate)
    num_volumes = len(df['FD_Upsampled']) / (sampling_rate * 2)

    # Generate the volume numbers for the x-axis
    volume_numbers = np.arange(0, num_volumes)
    
    # Calculate the tick positions for the fourth subplot
    tick_interval_fd = 5  # Adjust this value as needed
    tick_positions_fd = np.arange(0, len(df['FD_Upsampled']), tick_interval_fd * sampling_rate * 2)
    tick_labels_fd = [f"{int(vol)}" for vol in volume_numbers[::tick_interval_fd]]
    
    # Update x-axis labels for each subplot
    fig.update_xaxes(title_text='Samples', row=1, col=1, matches='x')
    fig.update_xaxes(title_text='Samples', row=2, col=1, matches='x')
    fig.update_xaxes(title_text='Volume Number (2 sec TR)', tickvals=tick_positions_fd, ticktext=tick_labels_fd, row=3, col=1, matches='x')

    # Disable y-axis zooming for all subplots
    fig.update_yaxes(fixedrange=True)
    
    # Return the figure
    return fig

# Function to open the web browser
def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception as e:
        logging.error(f"Error opening browser: {e}")

# Run the app
if __name__ == '__main__':
    Timer(1, open_browser).start()
    try:
        app.run_server(debug=False, port=8050)
    except Exception as e:
        logging.error(f"Error running the app: {e}")
