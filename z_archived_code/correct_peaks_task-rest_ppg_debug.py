import base64
import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import Input, Output, State
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
import flask
import json

# conda activate nipype
# TODO: implement artifact selection and correction
# TODO: implement recalculation and saving of PPG and HRV statistics
# TODO: implement subject and run specific archived logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='data-store'),  # To store the DataFrame
    dcc.Store(id='peaks-store'),  # To store the valid peaks
    dcc.Store(id='filename-store'),  # To store the original filename
    dcc.Store(id='peak-change-store', data={'added': 0, 'deleted': 0, 'original': 0}),
    dcc.Store(id='rr-intervals-store'),  # New store for R-R intervals
    dcc.Store(id='artifact-store', data={'start': None, 'end': None, 'marking_active': False}),
    dcc.Store(id='mode-store', data={'mode': 'peak_correction'}),  # To store the current mode
    html.Div(id='hidden-filename', style={'display': 'none'}),  # Hidden div for filename
    html.Div(id='js-trigger', style={'display': 'none'}), 
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select _processed.tsv.gz File to Correct Peaks'),
        style={},  # Add necessary style properties
        multiple=False
    ),
    dcc.Graph(
    id='ppg-plot',
    config={
        'displayModeBar': True,
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawrect']
    },
    figure={
        'data': [],  # Your data here
        'layout': {
            'dragmode': 'pan'
        }}),
    html.Button('Save Corrected Data', id='save-button'),
    html.Div([
        html.Button('Toggle Mode', id='toggle-mode-button'),
        html.Div(id='mode-indicator')
    ]),
    html.Div([
        html.Button('Mark Artifact Start', id='mark-artifact-start', style={'display': 'none'}),
        html.Button('Mark Artifact End', id='mark-artifact-end', style={'display': 'none'}),
        dcc.Input(id='start-point-input', type='number', placeholder='Start Point', step=1, style={'display': 'none'}),
        dcc.Input(id='end-point-input', type='number', placeholder='End Point', step=1, style={'display': 'none'}),
        dcc.Input(id='hidden-selection-range', style={'display': 'none'}),
        html.Button('Confirm Artifact', id='confirm-artifact', style={'display': 'none'}),
        html.Button('Cancel Artifact', id='cancel-artifact', style={'display': 'none'}),
    ]),
    html.Div(id='save-status'),  # To display the status of the save operation
])

app.clientside_callback(
    "window.dash_clientside.clientside.setupGraphEventListener",
    Output('js-trigger', 'children'),
    [Input('js-trigger', 'children')],
    prevent_initial_call=True
)

@app.callback(
    [Output('mode-store', 'data'),
     Output('mode-indicator', 'children'),
     Output('mark-artifact-start', 'style'),
     Output('mark-artifact-end', 'style'),
     Output('start-point-input', 'style'),
     Output('end-point-input', 'style'),
     Output('confirm-artifact', 'style'),
     Output('cancel-artifact', 'style')],
    [Input('toggle-mode-button', 'n_clicks')],
    [State('mode-store', 'data')]
)
def toggle_mode(n_clicks, mode_data):
    if n_clicks is None:
        # No button click yet, return default mode
        return mode_data, "Current Mode: Peak Correction", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    # Check the current mode and toggle it
    if mode_data['mode'] == 'peak_correction':
        new_mode = 'artifact_selection'
        mode_text = "Current Mode: Artifact Selection"
        artifact_style = {'display': 'block'}  # Show artifact components
    else:
        new_mode = 'peak_correction'
        mode_text = "Current Mode: Peak Correction"
        artifact_style = {'display': 'none'}  # Hide artifact components

    return {'mode': new_mode}, mode_text, artifact_style, artifact_style, artifact_style, artifact_style, artifact_style, artifact_style

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
     Output('hidden-filename', 'children'),
     Output('peak-change-store', 'data'),
     Output('start-point-input', 'value'),
     Output('end-point-input', 'value'),
     Output('artifact-store', 'data'),
     Output('js-trigger', 'children')],
    [Input('upload-data', 'contents'),
     Input('ppg-plot', 'clickData'),
     Input('hidden-selection-range', 'value'),
     Input('mark-artifact-start', 'n_clicks'),
     Input('start-point-input', 'value'),
     Input('mark-artifact-end', 'n_clicks'),
     Input('end-point-input', 'value')],
    [State('upload-data', 'filename'),
     State('data-store', 'data'),
     State('peaks-store', 'data'),
     State('ppg-plot', 'figure'),
     State('peak-change-store', 'data'),
     State('hidden-filename', 'children'),
     State('artifact-store', 'data'),
     State('mode-store', 'data')]
)
def update_plot(contents, clickData, selection_range_json, mark_artifact_start_clicks, start_point_input_value, mark_artifact_end_clicks, end_point_input_value, filename, data_json, valid_peaks, existing_figure, peak_changes, hidden_filename, artifact_store, mode_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        logging.info("No trigger for the callback")
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logging.info(f"Triggered ID: {triggered_id}")

    fig = go.Figure(existing_figure) if existing_figure else go.Figure()
    artifact_data = artifact_store if artifact_store else {'start': None, 'end': None, 'marking_active': False}

    try:
        # Handling file upload 
        if triggered_id == 'upload-data' and contents:
            logging.info("Handling file upload")
            df = parse_contents(contents)
            valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
            fig = create_figure(df, valid_peaks)
            peak_changes = {'added': 0, 'deleted': 0, 'original': len(valid_peaks)}
            new_filename = filename if isinstance(filename, str) else hidden_filename
            return fig, df.to_json(date_format='iso', orient='split'), valid_peaks, new_filename, peak_changes, artifact_data['start'], artifact_data['end'], artifact_data, dash.no_update

        # TODO: Fix double peak labeling issue
        
        # Handling peak correction via plot clicks
        elif mode_store['mode'] == 'peak_correction' and triggered_id == 'ppg-plot' and clickData:
            logging.info("Handling peak correction")
            clicked_x = clickData['points'][0]['x']
            df = pd.read_json(data_json, orient='split')
            peak_indices, _ = find_peaks(df['PPG_Clean'])
            
            if clicked_x in valid_peaks:
                logging.info("Deleting a peak")
                valid_peaks.remove(clicked_x)
                peak_changes['deleted'] += 1
            else:
                logging.info("Adding a new peak")
                peak_indices, _ = find_peaks(df['PPG_Clean'])
                nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
                valid_peaks.append(nearest_peak)
                peak_changes['added'] += 1
                valid_peaks.sort()
        
            fig = create_figure(df, valid_peaks)
            current_layout = existing_figure['layout'] if existing_figure else None
            if current_layout:
                fig.update_layout(xaxis=current_layout['xaxis'], yaxis=current_layout['yaxis'])
            return fig, dash.no_update, valid_peaks, dash.no_update, peak_changes, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
        # Toggle artifact marking active/inactive
        if triggered_id == 'mark-artifact-start':
            logging.info("Mark Artifact Start button pressed - Activating box selection tool")
            artifact_store['marking_active'] = True
            return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, 'execute-js'
        elif triggered_id == 'mark-artifact-end':
            artifact_store['marking_active'] = False
            return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, dash.no_update

        # Handle artifact selection from box select tool
        if mode_store['mode'] == 'artifact_selection' and triggered_id == 'hidden-selection-range' and artifact_store['marking_active']:
            if selection_range_json:
                logging.info(f"Selection range: {selection_range_json}")
                selection_range = json.loads(selection_range_json)
                start, end = selection_range
                logging.info(f"Start: {start}, End: {end}")
                artifact_store['start'] = start
                artifact_store['end'] = end
                fig.update_layout(shapes=[create_rectangle_shape(start, end)])
                artifact_store['marking_active'] = False
                return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, 'execute-js'

        # Handle manual input adjustments
        if triggered_id in ['start-point-input', 'end-point-input']:
            start = start_point_input_value if start_point_input_value is not None else start
            end = end_point_input_value if end_point_input_value is not None else end
            fig.update_layout(shapes=[create_rectangle_shape(start, end)])
            return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, dash.no_update

        return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, dash.no_update
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, start, end, artifact_store, dash.no_update

def create_rectangle_shape(start, end):
    """Create a rectangle shape for artifact marking."""
    return dict(
        type="rect",
        xref="x", yref="paper",
        x0=start, y0=0, x1=end, y1=1,
        fillcolor="LightSkyBlue", opacity=0.5,
        layer="below", line_width=0
    )
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
