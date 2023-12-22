import base64
import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
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

# conda activate nipype
# TODO: implement artifact selection and correction
# TODO: implement recalculation and saving of PPG and HRV statistics
# TODO: implement subject and run specific archived logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

# Add a clientside callback to switch between modes
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='switchMode'
    ),
    Output('client-mode-store', 'data'),
    [Input('trigger-mode-change', 'children')]
)

app.layout = html.Div([
    dcc.Store(id='data-store'),  # To store the DataFrame
    dcc.Store(id='peaks-store'),  # To store the valid peaks
    dcc.Store(id='filename-store'),  # To store the original filename
    dcc.Store(id='peak-change-store', data={'added': 0, 'deleted': 0, 'original': 0}),
    dcc.Store(id='rr-intervals-store'),  # New store for R-R intervals
    dcc.Store(id='mode-store', data={'mode': 'peak_correction'}),  # To store the current mode
    dcc.Store(id='figure-store'),  # To store the figure state
    dcc.Store(id='client-mode-store', data={'mode': None}),  # To handle mode switching on the client side
    dcc.Store(id='artifact-windows-store', data=[]),  # Store for tracking artifact windows
    html.Div(id='hidden-filename', style={'display': 'none'}),  # Hidden div for filename
    html.Div(id='trigger-mode-change', style={'display': 'none'}), # Hidden div for triggering mode change
    dcc.Upload(
    id='upload-data',
    children=html.Button('Select _processed.tsv.gz File to Correct Peaks'),
    style={},  # Add necessary style properties
    multiple=False),
    html.Div([
        html.Button('Toggle Mode', id='toggle-mode-button'),
        html.Div(id='mode-indicator')
    ]),
    # Group artifact selection components
    html.Div([
        html.Div(id='artifact-window-output'),
        html.Div([
            html.Label('Start:'),
            dcc.Input(id='artifact-start-input', type='number', value=0),
            html.Label('End:'),
            dcc.Input(id='artifact-end-input', type='number', value=0)
        ]),
        html.Button('Confirm Artifact Selection', id='confirm-artifact-button', n_clicks=0),
        html.Button('Cancel Last Artifact', id='cancel-artifact-button', n_clicks=0, style={'display': 'none'}),
    ], id='artifact-selection-components', style={'display': 'none'}),  # Initially hidden
    dcc.Graph(id='ppg-plot'),
    html.Button('Save Corrected Data', id='save-button'),
    html.Div(id='save-status'),  # To display the status of the save operation
])

@app.callback(
    [Output('mode-store', 'data'),
     Output('mode-indicator', 'children'),
     Output('artifact-selection-components', 'style'),
     Output('figure-store', 'data')],
    [Input('toggle-mode-button', 'n_clicks'),
     Input('client-mode-store', 'data')],
    [State('mode-store', 'data'),
     State('ppg-plot', 'figure')]
)
def toggle_mode(n_clicks, client_mode_data, mode_data, current_figure):
    # Determine if the callback was triggered by the client-side mode store
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'client-mode-store.data':
        if client_mode_data and client_mode_data['mode'] == 'peak_correction':
            return {'mode': 'peak_correction'}, "Current Mode: Peak Correction", {'display': 'none'}, dash.no_update

    if n_clicks is None:
        return mode_data, "Current Mode: Peak Correction", {'display': 'none'}, dash.no_update

    if mode_data['mode'] == 'peak_correction' or client_mode_data['mode'] == 'peak_correction':
        new_mode = 'artifact_selection'
        mode_text = "Current Mode: Artifact Selection"
        artifact_style = {'display': 'block'}
        saved_figure_json = current_figure  # Save current figure state
    else:
        new_mode = 'peak_correction'
        mode_text = "Current Mode: Peak Correction"
        artifact_style = {'display': 'none'}
        saved_figure_json = dash.no_update

    return {'mode': new_mode}, mode_text, artifact_style, saved_figure_json

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
    [Output('ppg-plot', 'figure'), # Update the figure (plot)
     Output('data-store', 'data'), # Update the data store (DataFrame)
     Output('peaks-store', 'data'), # Update the peaks store (valid peaks)
     Output('hidden-filename', 'children'), # Update the hidden filename 
     Output('peak-change-store', 'data'), # Update the peak changes store (added, deleted, original)
     Output('artifact-window-output', 'children'), # Update the artifact window output 
     Output('artifact-windows-store', 'data'), # Update the artifact windows store
     Output('artifact-start-input', 'value'),  # Reset start input after cancel
     Output('artifact-end-input', 'value'),    # Reset end input after cancel
     Output('cancel-artifact-button', 'style'), # Update cancel button style to hide it
     #Output('client-mode-store', 'data'), # Update client-side mode store
     Output('trigger-mode-change', 'children')],  # New output to trigger mode change
    [Input('upload-data', 'contents'), # Handle file upload via upload button
     Input('ppg-plot', 'clickData'), # Handle peak correction via plot clicks 
     Input('confirm-artifact-button', 'n_clicks'), # Handle artifact selection confirmation 
     Input('cancel-artifact-button', 'n_clicks'), # Handle artifact selection cancellation
     Input('artifact-start-input', 'value'), # Handle artifact selection start input value 
     Input('artifact-end-input', 'value')], # Handle artifact selection end input value 
    [State('upload-data', 'filename'), # Include filename
     State('data-store', 'data'), # Include saved data state
     State('peaks-store', 'data'), # Include saved peaks state
     State('ppg-plot', 'figure'), # Include saved figure state
     State('peak-change-store', 'data'), # Include saved peak changes state
     State('hidden-filename', 'children'), # Include hidden filename
     State('mode-store', 'data'), # Include saved mode state
     State('artifact-windows-store', 'data'), # Include saved artifact windows state
     State('figure-store', 'data')]  # Include saved figure state
)
def update_plot(contents, clickData, n_clicks_confirm, n_clicks_cancel, start_input, end_input, filename, data_json, valid_peaks, existing_figure, peak_changes, hidden_filename, mode_store, existing_artifact_windows, saved_figure_json):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        logging.info("No trigger for the callback")
        raise dash.exceptions.PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    artifact_output = ""  # Define artifact_output to ensure it's always available for return
    cancel_button_style = {'display': 'none'}  # Initialize cancel button style
    #client_mode_data = dash.no_update  # Initialize client-side mode data
    trigger_mode_change = dash.no_update  # Initialize trigger mode change
    
    # Initialize the figure
    fig = go.Figure(existing_figure) if existing_figure else go.Figure()

    try:
        # Handling file upload
        if triggered_id == 'upload-data' and contents:
            logging.info("Handling file upload")
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            fig = create_figure(df, valid_peaks)
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks)}
            new_filename = filename if isinstance(filename, str) else hidden_filename
            return [fig, df.to_json(date_format='iso', orient='split'), valid_peaks, new_filename, peak_changes, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update]

        # TODO: Fix double peak correction bug

        # Handling peak correction via plot clicks
        if mode_store['mode'] == 'peak_correction' and triggered_id == 'ppg-plot' and clickData:
            logging.info(f"Handling peak correction")
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
        
            fig = create_figure(df, valid_peaks)
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(xaxis=current_layout['xaxis'], yaxis=current_layout['yaxis'])
            return [fig, dash.no_update, valid_peaks, dash.no_update, peak_changes, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update]
        
        # Handle artifact selection
        if mode_store['mode'] == 'artifact_selection':
            artifact_start = start_input
            artifact_end = end_input
            
            # TODO: Enable post artifact selection mode toggle back to peak detection. 
            # TODO: Verify artifact-store content for multiple artifact windows.
            # TODO: Implement artifact correction.
            # TODO: Fix file save after artifact selection. 
            
            # Add shape only when confirmation button is clicked
            if 'confirm-artifact-button' in triggered_id and n_clicks_confirm > 0:
                logging.info(f"Artifact window confirmed: {artifact_start} to {artifact_end}")
                
                # Update the figure
                existing_artifact_windows.append({'start': artifact_start, 'end': artifact_end})
                
                # Show the cancel button
                cancel_button_style = {'display': 'block'}
                
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
                new_artifact_windows = existing_artifact_windows.copy()
                new_artifact_windows.append({'start': artifact_start, 'end': artifact_end})
                
            # Cancelling the last confirmed artifact window
            elif 'cancel-artifact-button' in triggered_id and n_clicks_cancel > 0:
                logging.info("Cancelling last artifact window")
                
                if saved_figure_json:
                    fig = go.Figure(saved_figure_json)  # Restore the saved figure state
                    existing_artifact_windows.clear()  # Clear the artifact windows
                
                # Hide the cancel button again
                cancel_button_style = {'display': 'none'}
                
                # Reset the artifact window start and end input values to zero
                artifact_start = artifact_end = 0
                artifact_output = "Artifact window cancelled."
                
                # Trigger the client-side callback to switch mode back to peak correction
                trigger_mode_change = 'Switch to Peak Correction'

            return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, artifact_output, existing_artifact_windows, artifact_start, artifact_end, cancel_button_style, trigger_mode_change]

        else:
            cancel_button_style = {'display': 'none'} if not existing_artifact_windows else {'display': 'block'}

        artifact_output = f"Selected Window: {artifact_start} to {artifact_end}"
        return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, artifact_output, existing_artifact_windows, artifact_start, artifact_end, cancel_button_style, dash.no_update]
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Error in plot updating", existing_artifact_windows, start_input, end_input, cancel_button_style, dash.no_update]

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

def create_figure(df, valid_peaks):
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
