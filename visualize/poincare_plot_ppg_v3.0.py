import pandas as pd
import numpy as np
import argparse
import os
import neurokit2 as nk  # noqa: F401
import matplotlib.pyplot as plt
import logging
import matplotlib.font_manager as fm  # noqa: F401
from neurokit2.hrv.hrv_utils import _hrv_format_input
from neurokit2.hrv.intervals_utils import _intervals_successive
import scipy.stats

#! Plot code adapted from: https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_nonlinear.html#hrv_nonlinear on about 20240614 - PAMcConnell

# Configure logging
def setup_logging(subject_dir):
    log_file = os.path.join(subject_dir, 'poincare_plot.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
# Function to generate density plot
def generate_density_plot(ax1, ax2, ax1_min, ax1_max, ax2_min, ax2_max):
    # Create meshgrid
    xx, yy = np.mgrid[ax1_min:ax1_max:100j, ax2_min:ax2_max:100j]
    # Fit Gaussian Kernel
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ax1, ax2])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f

# Main function to generate Poincare plot
def main(subject_dir):
    setup_logging(subject_dir)
    logging.info(f"Starting Poincare plot generation for directory: {subject_dir}")
    sampling_rate = 100  # Sampling rate for PPG data downsampled
    
    # Define methods of interest for non-linear HRV analysis comparison
    methods = [
        ('PPG_Peaks_elgendi', 'RR_interval_interpolated', 'Original_Uncorrected'),
        ('Peaks_Kubios', 'RR_interval_interpolated_Kubios', 'Kubios_Corrected'),
        ('PPG_Peaks_elgendi_corrected', 'RR_interval_interpolated_corrected', 'Dash_Corrected'),
        ('PPG_Peaks_elgendi_corrected_censored', 'RR_interval_interpolated_corrected_censored', 'Dash_Corrected_Censored')
    ]
    
    # Define runs of interest
    run_folders = ['run-01', 'run-02', 'run-03', 'run-04']  # Add all run folders here

    """
    # Define primary and complementary colors for plot
    run_colors = {
        'run-01': 'blue',
        'run-02': 'green',
        'run-03': 'red',
        'run-04': 'purple'
    }
    """
    
    # Define primary and complementary colors for plot
    run_colors = {
        'run-01': 'blue',
        'run-02': 'blue',
        'run-03': 'blue',
        'run-04': 'blue'
    }
    
    
    for method in methods:
        all_ax1 = []
        all_ax2 = []
        run_labels = []
        
        try:
            for run_folder in run_folders:
                # Configure path locations
                run_path = os.path.join(subject_dir, run_folder)
                tsv_file = f'{os.path.basename(subject_dir)}_ses-1_task-rest_{run_folder}_physio_filtered_cleaned_ppg_processed_corrected.tsv.gz'
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
                        peaks_indices = np.where(peaks == 1)[0]

                        if method_name == 'Dash_Corrected_Censored':
                            # For censored data, filter the peaks based on non-NaN RR intervals
                            rr_missing = np.isnan(rr)
                            valid_indices = np.where(~rr_missing)[0]
                            peaks = peaks[valid_indices]
                            peaks_indices = np.where(peaks == 1)[0]
                        else:
                            peaks_indices = np.where(peaks == 1)[0]

                        if len(peaks_indices) > 1:
                            
                            # Format RRI data for Poincare plot
                            rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate)
                            ax1 = rri[:-1]
                            ax2 = rri[1:]

                            # Check for missing values in RRI data
                            if rri_missing:
                                logging.warning(f"Missing values found in {tsv_file} for method {method_name}")
                                ax1 = ax1[_intervals_successive(rri, intervals_time=rri_time)]
                                ax2 = ax2[_intervals_successive(rri, intervals_time=rri_time)]

                            if method_name == 'Censored':
                                logging.debug(f"ax1 values: {ax1}")
                                logging.debug(f"ax2 values: {ax2}")

                            # Mean center the data for the current run
                            ax1_centered = ax1 - np.mean(ax1)
                            ax2_centered = ax2 - np.mean(ax2)
                            
                            # Mean heart period for the current run
                            mean_heart_period = np.mean([ax1_centered, ax2_centered])
                            logging.info(f"Mean heart period for {run_folder}: {mean_heart_period}")
                            
                            # Concatenate the RRI data from each run
                            all_ax1.extend(ax1_centered)
                            all_ax2.extend(ax2_centered)
                            run_labels.extend([run_folder] * len(ax1_centered))

            # Convert lists to numpy arrays
            all_ax1 = np.array(all_ax1)
            all_ax2 = np.array(all_ax2)
            run_labels = np.array(run_labels)
            
            # Compute descriptive statistics for SD1 and SD2
            sd1 = np.std(all_ax1 - all_ax2) / np.sqrt(2)
            sd2 = np.std(all_ax1 + all_ax2) / np.sqrt(2)
            logging.info(f"SD1: {sd1}, SD2: {sd2}")
            
            # Set grid boundaries
            ax1_lim = (max(all_ax1) - min(all_ax1)) / 10
            ax2_lim = (max(all_ax2) - min(all_ax2)) / 10
            ax1_min = min(all_ax1) - ax1_lim
            ax1_max = max(all_ax1) + ax1_lim
            ax2_min = min(all_ax2) - ax2_lim
            ax2_max = max(all_ax2) + ax2_lim
            
            # Hard code the grid boundaries for now 
            ax1_min = -300
            ax1_max = 300
            ax2_min = -300
            ax2_max = 300
            
            logging.info(f"Axis limits: ax1_min={ax1_min}, ax1_max={ax1_max}, ax2_min={ax2_min}, ax2_max={ax2_max}")

            # Prepare matplotlib figure
            fig = plt.figure(figsize=(10, 10), facecolor='white')  # Set figure background to white  # noqa: F841
            gs = plt.GridSpec(4, 4)
            ax_marg_x = plt.subplot(gs[0, 0:3], facecolor='white')  # Set axes background to white
            ax_marg_y = plt.subplot(gs[1:4, 3], facecolor='white')  # Set axes background to white
            ax = plt.subplot(gs[1:4, 0:3], facecolor='white')  # Set axes background to white
            gs.update(wspace=0.025, hspace=0.05)  # Reduce spaces
            
            # Configure plot for current method
            plt.suptitle(f"Method: {method_name}")

            # Generate density plot
            xx, yy, f = generate_density_plot(all_ax1, all_ax2, ax1_min, ax1_max, ax2_min, ax2_max)
            cmap = plt.get_cmap("Blues").resampled(10)
            ax.contourf(xx, yy, f, cmap=cmap, alpha=1)

            # Marginal densities
            ax_marg_x.hist(all_ax1, bins=int(len(all_ax1) / 10), density=True, alpha=1, color="#ccdff0", edgecolor="none")
            ax_marg_y.hist(
                all_ax2,
                bins=int(len(all_ax2) / 10),
                density=True,
                alpha=1,
                color="#ccdff0",
                edgecolor="none",
                orientation="horizontal",
                zorder=1,
            )
            kde1 = scipy.stats.gaussian_kde(all_ax1)
            x1_plot = np.linspace(ax1_min, ax1_max, len(all_ax1))
            x1_dens = kde1.evaluate(x1_plot)
            ax_marg_x.fill(x1_plot, x1_dens, facecolor="none", edgecolor="#1b6aaf", alpha=0.5, linewidth=2)

            kde2 = scipy.stats.gaussian_kde(all_ax2)
            x2_plot = np.linspace(ax2_min, ax2_max, len(all_ax2))
            x2_dens = kde2.evaluate(x2_plot)
            ax_marg_y.fill_betweenx(x2_plot, x2_dens, facecolor="none", edgecolor="#1b6aaf", linewidth=2, alpha=0.5, zorder=2)

            # Turn off marginal axes labels
            ax_marg_x.axis("off")
            ax_marg_y.axis("off")

            # Plot ellipse on top of density
            angle = 45
            width = 2 * sd2 + 1
            height = 2 * sd1 + 1
            mean_heart_period = np.mean([all_ax1, all_ax2])
            logging.info(f"Mean heart period across all runs: {mean_heart_period}")
            xy = (mean_heart_period, mean_heart_period)
            ellipse = plt.matplotlib.patches.Ellipse(xy=xy, width=width, height=height, angle=angle, linewidth=2, fill=False)
            ellipse.set_alpha(0.5)
            ellipse.set_edgecolor("black")
            ax.add_patch(ellipse)

            # Plot points outside ellipse with respective run colors
            cos_angle = np.cos(np.radians(180.0 - angle))
            sin_angle = np.sin(np.radians(180.0 - angle))
            xc = all_ax1 - xy[0]
            yc = all_ax2 - xy[1]
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle
            rad_cc = (xct**2 / (width / 2.0) ** 2) + (yct**2 / (height / 2.0) ** 2)
            points = np.where(rad_cc > 1)[0]

            for run_folder in run_folders:
                run_points = points[run_labels[points] == run_folder]
                ax.plot(all_ax1[run_points], all_ax2[run_points], 'o', color=run_colors[run_folder], alpha=0.25, markersize=4)
            
            # SD1 and SD2 arrows on top of ellipse
            ax.arrow(
                mean_heart_period,
                mean_heart_period,
                -sd1 * np.sqrt(2) / 2,
                sd1 * np.sqrt(2) / 2,
                linewidth=3,
                ec="#E91E63",
                fc="#E91E63",
                label="SD1",
            )
            ax.arrow(
                mean_heart_period,
                mean_heart_period,
                sd2 * np.sqrt(2) / 2,
                sd2 * np.sqrt(2) / 2,
                linewidth=3,
                ec="#FF9800",
                fc="#FF9800",
                label="SD2",
            )
            
            # Set the aspect ratio of the plot to be equal
            ax.set_aspect('equal', 'box')
            
            # Apply the hardcoded limits again to ensure they are respected
            ax.set_xlim(ax1_min, ax1_max)
            ax.set_ylim(ax2_min, ax2_max)
            
            ax.set_xlabel("RR$_n$ (ms)")
            ax.set_ylabel("RR$_{n+1}$ (ms)")
            
            # Save high-resolution plot
            plot_file = f'{os.path.basename(subject_dir)}_poincare_plot_{method_name}.png'
            plot_path = os.path.join(subject_dir, plot_file)
            plt.savefig(plot_path, dpi=300)
            logging.info(f"Saved plot to: {plot_path}")
            
            plt.show()
        except Exception as e:
            logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Poincare plots for PPG data.')
    parser.add_argument('subject_dir', type=str, help='Path to the subject directory')
    args = parser.parse_args()

    main(args.subject_dir)