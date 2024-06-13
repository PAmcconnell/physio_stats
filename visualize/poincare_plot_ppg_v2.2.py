import pandas as pd
import numpy as np
import argparse
import os
import neurokit2 as nk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from neurokit2.hrv.hrv_utils import _hrv_format_input
from neurokit2.hrv.intervals_utils import _intervals_successive
import scipy.stats

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_density_plot(ax1, ax2, ax1_min, ax1_max, ax2_min, ax2_max):
    xx, yy = np.mgrid[ax1_min:ax1_max:100j, ax2_min:ax2_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ax1, ax2])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f

# Main function to generate Poincare plots
def main(subject_dir):
    logging.info(f"Starting Poincare plot generation for directory: {subject_dir}")
    sampling_rate = 100  # Sampling rate for PPG data downsampled
    try:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Method 1: Original Uncorrected", "Method 2: Kubios Corrected",
                                                            "Method 3: Dash Corrected", "Method 4: Censored"))

        methods = [
            ('PPG_Peaks_elgendi', 'RR_interval_interpolated', 'Original Uncorrected'),
            ('Peaks_Kubios', 'RR_interval_interpolated_Kubios', 'Kubios Corrected'),
            ('PPG_Peaks_elgendi_corrected', 'RR_interval_interpolated_corrected', 'Dash Corrected'),
            ('PPG_Peaks_elgendi_corrected_censored', 'RR_interval_interpolated_corrected_censored', 'Censored')
        ]

        subject_id = os.path.basename(subject_dir)
        run_folders = sorted([f for f in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, f))])
        logging.info(f"Found run folders: {run_folders}")

        # Define colors for each run
        run_colors = {
            'run-01': ('blue', 'yellow'),
            'run-02': ('green', 'purple'),
            'run-03': ('red', 'cyan'),
            'run-04': ('purple', 'orange')
        }

        # Track legend entries
        added_legends = set()

        for method_idx, method in enumerate(methods):
            for run_folder in run_folders:
                run_path = os.path.join(subject_dir, run_folder)
                tsv_file = f'{subject_id}_ses-1_task-rest_{run_folder}_physio_filtered_cleaned_ppg_processed_corrected.tsv.gz'
                tsv_path = os.path.join(run_path, tsv_file)

                logging.info(f"Checking for file: {tsv_path}")
                if os.path.isfile(tsv_path):
                    logging.info(f"Processing file: {tsv_file}")
                    df = pd.read_csv(tsv_path, sep='\t')
                    logging.info(f"Columns in DataFrame: {df.columns.tolist()}")

                    peaks_col, rr_col, method_name = method
                    logging.info(f"Processing method: {method_name}")
                    logging.info(f"Peaks column: {peaks_col}, RR column: {rr_col}")

                    if peaks_col not in df.columns:
                        logging.warning(f"Column {peaks_col} not found in {tsv_file}")
                    else:
                        peaks = df[peaks_col].to_numpy()
                        rr = df[rr_col].to_numpy()

                        if method_name == 'Censored':
                            # For censored data, filter the peaks based on non-NaN RR intervals
                            rr_missing = np.isnan(rr)
                            valid_indices = np.where(~rr_missing)[0]
                            peaks = peaks[valid_indices]
                            peaks_indices = np.where(peaks == 1)[0]
                        else:
                            peaks_indices = np.where(peaks == 1)[0]

                        if len(peaks_indices) > 1:
                            # Use neurokit2 to compute SD1 and SD2, and RRI
                            hrv_indices = nk.hrv_nonlinear(peaks, sampling_rate)
                            sd1 = hrv_indices["HRV_SD1"][0]
                            sd2 = hrv_indices["HRV_SD2"][0]

                            if isinstance(sd1, pd.Series):
                                sd1 = float(sd1.iloc[0])
                            if isinstance(sd2, pd.Series):
                                sd2 = float(sd2.iloc[0])

                            rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate)

                            # Extract ax1 and ax2 for Poincare plot
                            ax1 = rri[:-1]
                            ax2 = rri[1:]

                            logging.warning(f"Missing values in RRI: {rri_missing}")

                            if rri_missing:
                                logging.warning(f"Missing values found in {tsv_file} for method {method_name}")
                                # Only include successive differences
                                ax1 = ax1[_intervals_successive(rri, intervals_time=rri_time)]
                                ax2 = ax2[_intervals_successive(rri, intervals_time=rri_time)]

                            # Debug: Print head and tail of ax1 and ax2
                            logging.debug(f"ax1 head: {ax1[:5]}, tail: {ax1[-5:]}")
                            logging.debug(f"ax2 head: {ax2[:5]}, tail: {ax2[-5:]}")

                            if method_name == 'Censored':
                                logging.debug(f"ax1 values: {ax1}")
                                logging.debug(f"ax2 values: {ax2}")

                            # Set grid boundaries
                            ax1_lim = (max(ax1) - min(ax1)) / 10
                            ax2_lim = (max(ax2) - min(ax2)) / 10
                            ax1_min = min(ax1) - ax1_lim
                            ax1_max = max(ax1) + ax1_lim
                            ax2_min = min(ax2) - ax2_lim
                            ax2_max = max(ax2) + ax2_lim

                            # Density plot
                            xx, yy, f = generate_density_plot(ax1, ax2, ax1_min, ax1_max, ax2_min, ax2_max)

                            row = (method_idx // 2) + 1
                            col = (method_idx % 2) + 1

                            run_color, complementary_color = run_colors[run_folder]

                            logging.info(f"Adding trace to subplot at row {row}, col {col}")

                            fig.add_trace(
                                go.Scatter(x=ax1, y=ax2, mode='markers', name=run_folder if run_folder not in added_legends else None,
                                           marker=dict(size=5, opacity=0.5, color=run_color)),
                                row=row, col=col
                            )

                            fig.add_trace(
                                go.Scatter(x=[None], y=[None], mode='markers',
                                           marker=dict(size=5, color=complementary_color),
                                           showlegend=True if f'{run_folder} - Ellipse' not in added_legends else False,
                                           name=f'{run_folder} - Ellipse'),
                                row=row, col=col
                            )

                            added_legends.add(run_folder)
                            added_legends.add(f'{run_folder} - Ellipse')

                            # Compute dynamic range for the axes
                            x_min, x_max = ax1.min(), ax1.max()
                            y_min, y_max = ax2.min(), ax2.max()

                            fig.update_xaxes(title_text="RR<sub>n</sub> (ms)", range=[x_min - 500, x_max + 500], row=row, col=col)
                            fig.update_yaxes(title_text="RR<sub>n+1</sub> (ms)", range=[y_min - 500, y_max + 500], row=row, col=col)
                            logging.info(f"Added Poincaré plot for method {method_name} to subplot {row}, {col}")

                            # Ellipse plot
                            angle = 45
                            width = 2 * sd2 + 1
                            height = 2 * sd1 + 1
                            mean_heart_period = np.mean(rri)
                            fig.add_shape(type="Ellipse", x0=mean_heart_period - sd2, y0=mean_heart_period - sd1,
                                          x1=mean_heart_period + sd2, y1=mean_heart_period + sd1,
                                          line=dict(color=complementary_color), row=row, col=col)
                        else:
                            logging.info(f"Not enough peaks found in {tsv_file} for method {method_name}")
                else:
                    logging.error(f"File not found: {tsv_path}")
                    
        # Remove unwanted legend entries
        for trace in fig['data']:
            logging.info(f"Trace name: {trace['name']}")
            if trace['name'] is None:
                trace['showlegend'] = False    

        fig.update_layout(title_text=f"Poincaré Plots of HRV Data for Subject {subject_id}", showlegend=True)
        fig.show()
        logging.info("Plot generation completed")
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Poincare plots for PPG data.')
    parser.add_argument('subject_dir', type=str, help='Path to the subject directory')
    args = parser.parse_args()

    main(args.subject_dir)
