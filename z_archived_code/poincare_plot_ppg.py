import pandas as pd
import numpy as np
import argparse
import os
import neurokit2 as nk
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from neurokit2.hrv.hrv_utils import _hrv_format_input

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
                        peaks_indices = np.where(peaks == 1)[0]

                        if len(peaks_indices) > 1:
                            # Use neurokit2 to compute SD1 and SD2, and RRI
                            hrv_indices = nk.hrv_nonlinear(peaks, sampling_rate)
                            sd1 = hrv_indices["HRV_SD1"][0]
                            sd2 = hrv_indices["HRV_SD2"][0]
                            rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate)

                            # Extract ax1 and ax2 for Poincare plot
                            ax1 = rri[:-1]
                            ax2 = rri[1:]

                            # Debug: Print head and tail of ax1 and ax2
                            logging.debug(f"ax1 head: {ax1[:5]}, tail: {ax1[-5:]}")
                            logging.debug(f"ax2 head: {ax2[:5]}, tail: {ax2[-5:]}")

                            row = (method_idx // 2) + 1
                            col = (method_idx % 2) + 1

                            logging.info(f"Adding trace to subplot at row {row}, col {col}")

                            fig.add_trace(
                                go.Scatter(x=ax1, y=ax2, mode='markers', name=f'{method_name} - {run_folder}',
                                           marker=dict(size=5, opacity=0.5)),
                                row=row, col=col
                            )

                            # Compute dynamic range for the axes
                            x_min, x_max = ax1.min(), ax1.max()
                            y_min, y_max = ax2.min(), ax2.max()

                            fig.update_xaxes(title_text="RR<sub>n</sub> (ms)", range=[x_min - 500, x_max + 500], row=row, col=col)
                            fig.update_yaxes(title_text="RR<sub>n+1</sub> (ms)", range=[y_min - 500, y_max + 500], row=row, col=col)
                            logging.info(f"Added Poincare plot for method {method_name} to subplot {row}, {col}")

                            # Add SD1 and SD2 annotations
                            fig.add_annotation(x=x_min, y=y_max, text=f'SD1: {sd1:.2f}', showarrow=False, row=row, col=col)
                            fig.add_annotation(x=x_min, y=y_max - 250, text=f'SD2: {sd2:.2f}', showarrow=False, row=row, col=col)
                        else:
                            logging.info(f"Not enough peaks found in {tsv_file} for method {method_name}")
                else:
                    logging.error(f"File not found: {tsv_path}")

        fig.update_layout(title_text="Poincare Plots of HRV Data", showlegend=True)
        fig.show()
        logging.info("Plot generation completed")
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Poincare plots for PPG data.')
    parser.add_argument('subject_dir', type=str, help='Path to the subject directory')
    args = parser.parse_args()

    main(args.subject_dir)