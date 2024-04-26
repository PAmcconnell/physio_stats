# Function to calculate the correlation between FD and PPG
def calculate_fd_ppg_correlation(fd, ppg):
    """
    Calculate the Pearson correlation coefficient and p-value between framewise displacement (FD) and PPG.

    Parameters:
    fd (array-like): The framewise displacement timeseries.
    ppg (array-like): The ppg activity timeseries.

    Returns:
    float: The Pearson correlation coefficient.
    float: The p-value indicating statistical significance.

    The function assumes both inputs are numpy arrays of the same length.
    The Pearson correlation coefficient measures the linear correlation between the two arrays,
    returning a value between -1 and 1, where 1 means total positive linear correlation,
    0 means no linear correlation, and -1 means total negative linear correlation.
    The p-value roughly indicates the probability of an uncorrelated system producing datasets
    that have a Pearson correlation at least as extreme as the one computed from these datasets.
    """

    # Ensure that FD and PPG are of the same length
    if len(fd) != len(ppg):
        logging.error("FD and PPG timeseries must be of the same length.")
        return None, None

    try:
        # Calculate Pearson correlation
        r_value, p_value = pearsonr(fd, ppg)
        logging.info(f"Calculated Pearson correlation: {r_value}, p-value: {p_value}")
        
        # Round p-value to 3 decimal places
        p_value_rounded = round(p_value, 3)
        p_value = p_value_rounded

        # Check for very large sample sizes which might render p-value as 0
        if p_value == 0:
            logging.info("P-value returned as 0, possibly due to large sample size and high precision of correlation.")

        return r_value, p_value
    except Exception as e:
        logging.error(f"Error in calculating correlation: {e}")
        return None, None

# Function to plot the correlation between FD and PPG
def plot_fd_ppg_correlation(fd, ppg, file_name):
    
    if len(fd) != len(ppg):
        logging.warning("Error: FD and PPG timeseries must be of the same length.")
        return
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(fd, ppg)
        fit_line = slope * fd + intercept
        r_squared = r_value**2
        
        # Calculate the confidence interval of the fit line
        t_val = t.ppf(1-0.05/2, len(fd)-2)  # t-score for 95% confidence interval & degrees of freedom
        conf_interval = t_val * std_err * np.sqrt(1/len(fd) + (fd - np.mean(fd))**2 / np.sum((fd - np.mean(fd))**2))

        # Upper and lower bounds of the confidence interval
        lower_bound = fit_line - conf_interval
        upper_bound = fit_line + conf_interval

        plt.figure(figsize=(10, 6))
        plt.scatter(fd, ppg, alpha=0.5, label='Data Points')
        plt.plot(fd, fit_line, color='red', label=f'Fit Line (R = {r_value:.3f}, p = {p_value:.3f})')
        plt.fill_between(fd, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.ylabel('PPG (Volts)')
        plt.xlabel('Framewise Displacement (mm)')
        plt.title('Correlation between FD and PPG with Linear Fit and Confidence Interval')
        plt.legend(loc='upper left')
        plt.savefig(file_name, dpi=300)  # Save the figure with increased DPI for higher resolution
        plt.close()

    except Exception as e:
        logging.warning(f"An error occurred: {e}")

def compute_hrv_stats(df, valid_peaks, filename, save_directory):
    
    """
    draft code for HRV statistical recomputation
    """
    
    #%% #! SECTION 1: Variables setup
    
    sampling_rate = 100   # downsampled sampling rate
    logging.info(f"Downsampled PPG sampling rate: {sampling_rate} Hz")

    # FIXME: base_name and path handling
    
    # Generate a base filename by removing the '.tsv.gz' extension
    base_filename = os.path.basename(file).replace('.tsv.gz', '')
    # Construct the base path
    base_path = os.path.join(derivatives_dir, 'ppg', participant_id, run_id)

    # Define the frequency bands
    frequency_bands = {
    'ULF': (0, 0.003), # Associated with: Very slow breathing, vasomotion, myogenic activity, thermoregulation, metabolism, and the renin-angiotensin system; Dominated by: Peak around 0.003 Hz
    'VLF': (0.003, 0.04), # Associated with: Thermoregulation, hormonal fluctuations, slow changes in blood pressure, peripheral sympathetic; Motion Artifacts
    'LF': (0.04, 0.15), # Associated with: Sympathetic and Parasympathetic nervous system activity, baroreflex function, stress; Dominated by: Peak around 0.1 Hz
    'HF': (0.15, 0.4), # Associated with: Parasympathetic nervous system activity, respiratory sinus arrhythmia; Dominated by: Peak around 0.25-0.3 Hz
    'VHF': (0.4, 1.0) # Associated with: Very high frequency oscillations; Dominated by: Peak around 0.75 Hz
    }
    
    # Define the frequency bands and their corresponding colors
    frequency_bands_plot = {
        'ULF': (0, 0.003, 'grey'),  # Ultra Low Frequency
        'VLF': (0.003, 0.04, 'red'),  # Very Low Frequency
        'LF': (0.04, 0.15, 'green'),  # Low Frequency
        'HF': (0.15, 0.4, 'blue'),  # High Frequency
        'VHF': (0.4, 1.0, 'purple')  # Very High Frequency
    }
    
    #%% #! SECTION 2: PSD Plotly Plots (0 - 8 Hz full range)  
    #! Here we need to load the corrected ppg timeseries into a pandas dataframe for neurokit2 functions
    # Compute Power Spectral Density 0 - 8 Hz for PPG
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG using multitapers hann windowing.")
    ppg_filtered_psd = nk.signal_psd(ppg_cleaned_df['PPG_Clean'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                        min_frequency=0, max_frequency=8.0, window=None, window_type='hann',
                        silent=False, t=None)

    # Plotting Power Spectral Density
    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 8 Hz for filtered cleaned PPG using multitapers hann windowing.")

    # Create a Plotly figure
    fig = go.Figure()

    # Create a figure with a secondary x-axis using make_subplots
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add the Power Spectral Density trace to the figure
    fig.add_trace(go.Scatter(x=ppg_filtered_psd['Frequency'], y=ppg_filtered_psd['Power'],
                            mode='lines', name='Normalized PPG PSD',
                            line=dict(color='blue'), fill='tozeroy'))

    # Update layout for the primary x-axis (Frequency in Hz)
    fig.update_layout(
        title='Power Spectral Density (PSD) (Multitapers with Hanning Window) for Filtered Cleaned PPG',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Normalized Power',
        template='plotly_white',
        width=1200, height=800
    )

    # Create a secondary x-axis for Heart Rate in BPM
    fig.update_layout(
        xaxis=dict(
            title='Frequency (Hz)',
            range=[0, 8]  # Set the x-axis range from 0 to 8 Hz for frequency
        ),
        xaxis2=dict(
            title='Heart Rate (BPM)',
            overlaying='x',
            side='top',
            range=[0, 8*60],  # Convert frequency to BPM by multiplying by 60 (since 1 Hz = 60 BPM)
            scaleanchor='x',
            scaleratio=60
        )
    )

    # Save the plot as an HTML file
    plot_append = (f"corrected")
    plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_psd_{plot_append}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 3: PSD Plotly Plots (0 - 1 Hz HRV range) 
    # Compute Power Spectral Density 0 - 1 Hz for PPG
    logging.info(f"Computing Power Spectral Density (PSD) for filtered PPG HRV using multitapers hann windowing.")
    ppg_filtered_psd_hrv = nk.signal_psd(ppg_cleaned_df['RR_interval_interpolated'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                        silent=False, t=None)

    # Plotting Power Spectral Density
    logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned PPG HRV using multitapers hann windowing.")

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the Power Spectral Density for each band with its own color
    for band, (low_freq, high_freq, color) in frequency_bands_plot.items():
        # Filter the PSD data for the current band
        band_mask = (ppg_filtered_psd_hrv['Frequency'] >= low_freq) & (ppg_filtered_psd_hrv['Frequency'] < high_freq)
        band_psd = ppg_filtered_psd_hrv[band_mask]
        
        # Add the PSD trace for the current band
        fig.add_trace(go.Scatter(
            x=band_psd['Frequency'],
            y=band_psd['Power'],
            mode='lines',
            name=f'{band} Band',
            line=dict(color=color, width=2),
            fill='tozeroy'
        ))

    # Update layout for the primary x-axis (Frequency in Hz)
    fig.update_layout(
        title='Power Spectral Density (PSD) (Multitapers with Hanning Window) for R-R Interval Tachogram',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Normalized Power',
        template='plotly_white',
        width=1200, height=800
    )

    # Create a secondary x-axis for Heart Rate in BPM
    fig.update_layout(
        xaxis=dict(
            title='Frequency (Hz)',
            range=[0, 1]  # Set the x-axis range from 0 to 8 Hz for frequency
        ),
        xaxis2=dict(
            title='Heart Rate (BPM)',
            overlaying='x',
            side='top',
            range=[0, 1*60],  # Convert frequency to BPM by multiplying by 60 (since 1 Hz = 60 BPM)
            scaleanchor='x',
            scaleratio=60
        )
    )

    # Save the plot as an HTML file
    plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_psd_hrv_{plot_append}.html")
    plotly.offline.plot(fig, filename=plot_filename, auto_open=False)
    
    #%% #! SECTION 4: Re-calculate HRV Stats
    
    # Create a mask for FD values less than or equal to 0.5
    mask_ppg = fd_upsampled_ppg <= 0.5

    # Apply the mask to both FD and PPG data
    filtered_fd_ppg = fd_upsampled_ppg[mask_ppg]
    filtered_ppg = ppg_cleaned[mask_ppg]

    # Initialize correlation and above threshold FD variables as NaN
    r_value_ppg = np.nan
    p_value_ppg = np.nan
    r_value_ppg_thresh = np.nan
    p_value_ppg_thresh = np.nan

    num_samples_above_threshold = np.nan
    percent_samples_above_threshold = np.nan
    mean_fd_above_threshold = np.nan
    std_dev_fd_above_threshold = np.nan
    
    # Check if there are any FD values above the threshold
    if np.any(fd > voxel_threshold):
        # Check if filtered data is not empty
        if len(filtered_fd_ppg) > 0 and len(filtered_ppg) > 0:
            
            # Calculate above threshold FD statistics
            num_samples_above_threshold = np.sum(fd_upsampled_ppg > voxel_threshold)
            percent_samples_above_threshold = num_samples_above_threshold / len(fd_upsampled_ppg) * 100
            mean_fd_above_threshold = np.mean(fd_upsampled_ppg[fd_upsampled_ppg > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
            std_dev_fd_above_threshold = np.std(fd_upsampled_ppg[fd_upsampled_ppg > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
            
            # Calculate the correlation between filtered FD and PPG
            r_value_ppg, p_value_ppg = calculate_fd_ppg_correlation(filtered_fd_ppg, filtered_ppg)
            logging.info(f"Correlation between FD (filtered) and filtered cleaned PPG timeseries < {voxel_threshold} mm: {r_value_ppg}, p-value: {p_value_ppg}")

            plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_ppg_correlation_filtered_{plot_append}.png"
            plot_filepath = os.path.join(base_path, plot_filename)
            plot_fd_ppg_correlation(filtered_fd_ppg, filtered_ppg, plot_filepath)
            logging.info(f"FD-PPG filtered correlation plot saved to {plot_filepath}")
            
        else:
            # Log a warning if there are no FD values below the threshold after filtering
            logging.warning(f"No FD values below {voxel_threshold} mm. Correlation cannot be calculated.")
    else:
        # Log a warning if there are no FD values above the threshold
        logging.warning(f"No FD values above {voxel_threshold} mm. No need to filter and calculate correlation.")

    # Calculate statistics related to framewise displacement
    mean_fd_below_threshold = np.mean(fd_upsampled_ppg[fd_upsampled_ppg < voxel_threshold])
    std_dev_fd_below_threshold = np.std(fd_upsampled_ppg[fd_upsampled_ppg < voxel_threshold])

    # Average ppg Frequency (counts/min)
    total_time_minutes = len(ppg_cleaned) / (sampling_rate * 60)
    average_ppg_frequency = len(valid_peaks) / total_time_minutes

    # Average, Std Deviation, Max, Min Inter-ppg Interval (sec)
    inter_ppg_intervals = np.diff(valid_peaks) / sampling_rate
    average_inter_ppg_interval = inter_ppg_intervals.mean()
    std_inter_ppg_interval = inter_ppg_intervals.std()
    max_inter_ppg_interval = inter_ppg_intervals.max()
    min_inter_ppg_interval = inter_ppg_intervals.min()

    logging.info(f"Calculating HRV Indices via Neurokit")
    hrv_indices = nk.hrv(valid_peaks, sampling_rate=sampling_rate, show=True)
    
    # Update the ppg_stats dictionary
    ppg_stats = {
        'R-Peak Count (# peaks)': len(valid_peaks),
        'Average R-Peak Frequency (counts/min)': average_ppg_frequency,
        'Average Inter-Peak Interval (sec)': average_inter_ppg_interval,
        'Std Deviation Inter-Peak Interval (sec)': std_inter_ppg_interval,
        'Max Inter-Peak Interval (sec)': max_inter_ppg_interval,
        'Min Inter-Peak Interval (sec)': min_inter_ppg_interval,
        'Mean Framewise Displacement (mm)': fd_upsampled_ppg.mean(),
        'Std Deviation Framewise Displacement (mm)': fd_upsampled_ppg.std(),
        'Max Framewise Displacement (mm)': fd_upsampled_ppg.max(),
        'Min Framewise Displacement (mm)': fd_upsampled_ppg.min(),
        'Number of samples with FD > 0.5 mm': num_samples_above_threshold,
        'Percent of samples with FD > 0.5 mm': percent_samples_above_threshold,
        'Mean FD > 0.5 mm': mean_fd_above_threshold,
        'Std Deviation FD > 0.5 mm': std_dev_fd_above_threshold,
        'Mean FD < 0.5 mm': mean_fd_below_threshold,
        'Std Deviation FD < 0.5 mm': std_dev_fd_below_threshold,
        'Framewise Displacement - PPG Correlation R-Value': r_value_ppg,
        'Framewise Displacement - PPG Correlation P-Value': p_value_ppg,
        'Framewise Displacement - PPG Correlation R-Value (FD < 0.5 mm)': r_value_ppg_thresh,
        'Framewise Displacement - PPG Correlation P-Value (FD < 0.5 mm)': p_value_ppg_thresh,
        }
        
    # Assume hrv_indices is a dictionary with each value being a Series or a single value
    hrv_stats = {}

    # Loop through each HRV metric in hrv_indices
    for metric, value in hrv_indices.items():
        # Check if the value is a Series and take the first value
        if isinstance(value, pd.Series):
            hrv_stats[metric] = value.iloc[0]
        else:
            hrv_stats[metric] = value

    # Debug: Check the updated PPG statistics
    logging.info(f"PPG Stats: {ppg_stats}")

    # Assume ppg_stats are dictionaries containing the statistics
    ppg_stats_df = pd.DataFrame(ppg_stats.items(), columns=['Statistic', 'Value'])
    hrv_stats_df = pd.DataFrame(hrv_stats.items(), columns=['Statistic', 'Value'])
    
    # Concatenate DataFrames vertically, ensuring the order is maintained
    ppg_summary_stats_df = pd.concat([ppg_stats_df, hrv_stats_df], axis=0, ignore_index=True)
    
    # Add a column to indicate the category of each statistic
    ppg_summary_stats_df.insert(0, 'Category', '')
    
    # Assign 'PPG Stats' to the first part and 'HRV Stats' to the second part
    ppg_summary_stats_df.loc[:len(ppg_stats_df) - 1, 'Category'] = 'PPG Stats'
    ppg_summary_stats_df.loc[len(ppg_stats_df):, 'Category'] = 'HRV Stats'
    
    # Save the summary statistics to a TSV file, with headers and without the index
    summary_stats_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_ppg_{peak_method}_summary_statistics_{plot_append}.tsv")
    ppg_summary_stats_df.to_csv(summary_stats_filename, sep='\t', header=True, index=False)
    logging.info(f"Saving summary statistics to TSV file: {summary_stats_filename}")
                                