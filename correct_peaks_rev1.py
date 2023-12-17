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

# conda activate nipype
# TODO: track number of corrections made and save to a file

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout with Store components
app.layout = html.Div([
    dcc.Store(id='data-store'),  # To store the DataFrame
    dcc.Store(id='peaks-store'),  # To store the valid peaks
    dcc.Store(id='filename-store'),  # To store the original filename
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            # Style properties
        },
        multiple=False  # Allow only single file to be uploaded
    ),
    dcc.Graph(id='ppg-plot'),
    html.Button('Save Corrected Data', id='save-button'),
    html.Div(id='save-status')  # To display the status of the save operation

])

def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.BytesIO(decoded), sep='\t', compression='gzip')
    return df

def upload_data(contents):
    if contents:
        df = parse_contents(contents)
        valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
        return df.to_json(date_format='iso', orient='split'), valid_peaks
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('ppg-plot', 'figure'),
     Output('data-store', 'data'),
     Output('peaks-store', 'data')],
    [Input('upload-data', 'contents'),
     Input('ppg-plot', 'clickData')],
    [State('data-store', 'data'),
     State('peaks-store', 'data'),
     State('ppg-plot', 'figure')]
)

def update_plot_and_peaks(contents, clickData, data_json, valid_peaks, existing_figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        logging.info("No trigger for the callback")
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-data' and contents:
        logging.info("Handling file upload")
        df = parse_contents(contents)
        valid_peaks = df[df['PPG_Peaks_elgendi'] == 1].index.tolist()
        fig = create_figure(df, valid_peaks)
        return fig, df.to_json(date_format='iso', orient='split'), valid_peaks

    elif trigger_id == 'ppg-plot' and clickData:
        logging.info("Handling click event on plot")
        clicked_x = clickData['points'][0]['x']
        df = pd.read_json(data_json, orient='split')

        if clicked_x in valid_peaks:
            logging.info("Deleting a peak")
            valid_peaks.remove(clicked_x)
        else:
            logging.info("Adding a new peak")
            peak_indices, _ = find_peaks(df['PPG_Clean'])
            nearest_peak = min(peak_indices, key=lambda peak: abs(peak - clicked_x))
            valid_peaks.append(nearest_peak)
            valid_peaks.sort()

        fig = create_figure(df, valid_peaks)
        current_layout = existing_figure['layout'] if existing_figure else None
        if current_layout:
            fig.update_layout(
                xaxis=current_layout['xaxis'],
                yaxis=current_layout['yaxis']
            )
        return fig, dash.no_update, valid_peaks

    logging.error("Unexpected trigger in callback")
    raise dash.exceptions.PreventUpdate

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
    fig.update_yaxes(title_text='Amplitude (Volts)', row=2, col=1)
    fig.update_yaxes(title_text='R-R Interval (ms)', row=3, col=1)
    fig.update_yaxes(title_text='FD (mm)', row=4, col=1)

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

@app.callback(
    Output('save-status', 'children'),
    [State('data-store', 'data'),
     State('peaks-store', 'data'),
     State('filename-store', 'data')],
    [Input('save-button', 'n_clicks')]
)
def save_corrected_data(n_clicks, data_json, valid_peaks, original_filepath):
    if n_clicks is None or not data_json or not valid_peaks or not original_filepath:
        raise dash.exceptions.PreventUpdate

    try:
        # Load the data from JSON
        df = pd.read_json(data_json, orient='split')

        # Update the DataFrame with the corrected peak data
        df['PPG_Peaks_elgendi_corrected'] = 0
        df.loc[valid_peaks, 'PPG_Peaks_elgendi_corrected'] = 1

        # Extracting subject_id and run_id from the original filepath
        file_parts = original_filepath.split('/')
        subject_id = file_parts[-3]
        run_id = file_parts[-2]
        
        derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/ppg/')

        # Constructing the new filename
        original_filename = os.path.basename(original_filepath)
        base_name = os.path.splitext(original_filename)[0]
        new_filename = f"{base_name}_corrected.tsv.gz"

        # Constructing the full path for the new file
        full_new_path = os.path.join(derivatives_dir, subject_id, run_id, new_filename)

        # Save the DataFrame to a new tsv.gz file
        df.to_csv(full_new_path, sep='\t', compression='gzip', index=False)
        return f"Data saved to {full_new_path}"
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        return "An error occurred while saving data."
    
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
        app.run_server(debug=True)
    except Exception as e:
        logging.error(f"Error running the app: {e}")
