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

# conda activate nipype

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Allow only single file to be uploaded
    ),
    dcc.Graph(id='ppg-plot'),
])

def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.BytesIO(decoded), sep='\t', compression='gzip')
    return df

@app.callback(
    Output('ppg-plot', 'figure'),
    [Input('upload-data', 'contents')]
)
def update_plot(contents):
    if contents:
        df = parse_contents(contents)
        # Check if the necessary columns are in the dataframe
        if 'PPG_Clean' in df.columns and 'PPG_Peaks_elgendi' and 'RR_interval' and 'FD_Upsampled' in df.columns:
            sampling_rate = 100 # Hz
            fig = make_subplots(rows=3, cols=1, shared_xaxes=False, shared_yaxes=False,
                                subplot_titles=('PPG with R Peaks', 'R-R Intervals Tachogram',
                                                'Framewise Displacement'),
                                vertical_spacing=0.065)
            
            # Add Scatter Plot for PPG
            fig.add_trace(go.Scatter(y=df['PPG_Clean'], mode='lines', name='Filtered Cleaned PPG', line=dict(color='green')),
                          row=1, col=1)
            
            # Extract valid peaks that are within the dataframe
            #valid_peaks = df['PPG_Peaks_elgendi'].dropna().astype(int)
            #valid_peaks = valid_peaks[valid_peaks < len(df)]
            
            valid_peaks = df.index[df['PPG_Peaks_elgendi'] == 1].tolist()
            valid_peaks = np.array(valid_peaks)
            y_values = df.loc[valid_peaks, 'PPG_Clean'].tolist()
            
            # Add Scatter Plot for R Peaks
            fig.add_trace(go.Scatter(x=valid_peaks, y=y_values, mode='markers', name='R Peaks',
                                     marker=dict(color='red')), row=1, col=1)
            
            # Calculate R-R intervals in milliseconds
            rr_intervals = np.diff(valid_peaks) / sampling_rate * 1000  # in ms

            # Calculate midpoints between R peaks in terms of the sample indices
            #midpoint_samples = (valid_peaks[:-1] + valid_peaks[1:]) // 2
            midpoint_samples = ((valid_peaks[:-1] + valid_peaks[1:]) // 2).tolist()
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

            return fig
        else:
            return go.Figure()
    return go.Figure()

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
