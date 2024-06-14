import pandas as pd
import numpy as np
import argparse
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
import logging
import matplotlib.font_manager as fm
from neurokit2.hrv.hrv_utils import _hrv_format_input
from neurokit2.hrv.intervals_utils import _intervals_successive
import scipy.stats

#! Plot code adapted from: https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_nonlinear.html#hrv_nonlinear on about 20240614 - PAMcConnell

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

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
    logging.info(f"Starting Poincare plot generation for directory: {subject_dir}")
    sampling_rate = 100  # Sampling rate for PPG data downsampled
    try:
        
        # Define methods of interest for non-linear HRV analysis comparison
        method = ('PPG_Peaks_elgendi', 'RR_interval_interpolated', 'Original Uncorrected')
        
        # Define runs of interest
        run_folder = 'run-01'

        # Define primary and complementary colors for plot
        run_colors = {
            'run-01': 'blue',
        }

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

                if len(peaks_indices) > 1:
                    # Compute sd1 and sd2 via neurokit2
                    hrv_indices = nk.hrv_nonlinear(peaks, sampling_rate)
                    sd1 = hrv_indices["HRV_SD1"][0]
                    sd2 = hrv_indices["HRV_SD2"][0]

                    if isinstance(sd1, pd.Series):
                        sd1 = float(sd1.iloc[0])
                    if isinstance(sd2, pd.Series):
                        sd2 = float(sd2.iloc[0])

                    # Format RRI data for Poincare plot
                    rri, rri_time, rri_missing = _hrv_format_input(peaks, sampling_rate)
                    ax1 = rri[:-1]
                    ax2 = rri[1:]

                    # Check for missing values in RRI data
                    if rri_missing:
                        logging.warning(f"Missing values found in {tsv_file} for method {method_name}")
                        ax1 = ax1[_intervals_successive(rri, intervals_time=rri_time)]
                        ax2 = ax2[_intervals_successive(rri, intervals_time=rri_time)]

                    # Set grid boundaries
                    ax1_lim = (max(ax1) - min(ax1)) / 10
                    ax2_lim = (max(ax2) - min(ax2)) / 10
                    ax1_min = min(ax1) - ax1_lim
                    ax1_max = max(ax1) + ax1_lim
                    ax2_min = min(ax2) - ax2_lim
                    ax2_max = max(ax2) + ax2_lim
                    
                    # Prepare matplotlib figure
                    fig = plt.figure(figsize=(10, 10), facecolor='white')  # Set figure background to white
                    gs = plt.GridSpec(4, 4)
                    ax_marg_x = plt.subplot(gs[0, 0:3], facecolor='white')  # Set axes background to white
                    ax_marg_y = plt.subplot(gs[1:4, 3], facecolor='white')  # Set axes background to white
                    ax = plt.subplot(gs[1:4, 0:3], facecolor='white')  # Set axes background to white
                    gs.update(wspace=0.025, hspace=0.05)  # Reduce spaces
                    
                    # Configure plot for current method
                    plt.suptitle(f"Method: {method_name}")

                    # Generate density plot
                    xx, yy, f = generate_density_plot(ax1, ax2, ax1_min, ax1_max, ax2_min, ax2_max)
                    cmap = plt.get_cmap("Blues").resampled(10)
                    ax.contourf(xx, yy, f, cmap=cmap, alpha=0.75)
                    # ax.imshow(np.rot90(f), extent=[ax1_min, ax1_max, ax2_min, ax2_max], aspect="auto")

                    # Marginal densities
                    ax_marg_x.hist(ax1, bins=int(len(ax1) / 10), density=True, alpha=1, color="#ccdff0", edgecolor="none")
                    ax_marg_y.hist(
                        ax2,
                        bins=int(len(ax2) / 10),
                        density=True,
                        alpha=1,
                        color="#ccdff0",
                        edgecolor="none",
                        orientation="horizontal",
                        zorder=1,
                    )
                    kde1 = scipy.stats.gaussian_kde(ax1)
                    x1_plot = np.linspace(ax1_min, ax1_max, len(ax1))
                    x1_dens = kde1.evaluate(x1_plot)
                    ax_marg_x.fill(x1_plot, x1_dens, facecolor="none", edgecolor="#1b6aaf", alpha=0.8, linewidth=2)

                    kde2 = scipy.stats.gaussian_kde(ax2)
                    x2_plot = np.linspace(ax2_min, ax2_max, len(ax2))
                    x2_dens = kde2.evaluate(x2_plot)
                    ax_marg_y.fill_betweenx(x2_plot, x2_dens, facecolor="none", edgecolor="#1b6aaf", linewidth=2, alpha=0.8, zorder=2)

                    # Turn off marginal axes labels
                    ax_marg_x.axis("off")
                    ax_marg_y.axis("off")

                    # Plot ellipse on top of density
                    angle = 45
                    width = 2 * sd2 + 1
                    height = 2 * sd1 + 1
                    mean_heart_period = np.mean([ax1, ax2])
                    xy = (mean_heart_period, mean_heart_period)
                    ellipse = plt.matplotlib.patches.Ellipse(xy=xy, width=width, height=height, angle=angle, linewidth=2, fill=False)
                    ellipse.set_alpha(0.5)
                    ellipse.set_edgecolor("black")
                    ax.add_patch(ellipse)

                    # Plot points outside ellipse
                    cos_angle = np.cos(np.radians(180.0 - angle))
                    sin_angle = np.sin(np.radians(180.0 - angle))
                    xc = ax1 - xy[0]
                    yc = ax2 - xy[1]
                    xct = xc * cos_angle - yc * sin_angle
                    yct = xc * sin_angle + yc * cos_angle
                    rad_cc = (xct**2 / (width / 2.0) ** 2) + (yct**2 / (height / 2.0) ** 2)

                    points = np.where(rad_cc > 1)[0]
                    ax.plot(ax1[points], ax2[points], 'o', color='k', alpha=0.5, markersize=4)
                    
                    # Plot points
                    # ax.plot(ax1, ax2, 'o', markersize=4, color=run_colors[run_folder], alpha=0.5, label=run_folder)

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


        ax.set_xlabel("RR$_n$ (ms)")
        ax.set_ylabel("RR$_{n+1}$ (ms)")
        ax.legend(fontsize=12, loc="best")
        plt.show()
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Poincare plots for PPG data.')
    parser.add_argument('subject_dir', type=str, help='Path to the subject directory')
    args = parser.parse_args()

    main(args.subject_dir)