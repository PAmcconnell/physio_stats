### Step 1: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz ###
                            logging.info(f"Step 1: Prefilter gradient artifact from EDA using comb-band stop filter at 1/TR (0.5 Hz) and downsample to 2000 Hz.")

                            # Pre-filter gradient artifact.
                            eda_filtered_array = comb_band_stop_filter(eda, stop_freq, sampling_rate, visualize=False)

                            # Downsample the filtered data.
                            eda_filtered_ds = nk.signal_resample(eda_filtered_array, desired_length=None, sampling_rate=sampling_rate, desired_sampling_rate=50, method='pandas')

                            # Hanlde the index for the downsampled data
                            new_length = len(eda_filtered_ds)
                            new_index = pd.RangeIndex(start=0, stop=new_length, step=1)  # or np.arange(new_length)

                            # Convert the filtered data back into a Pandas Series
                            eda_filtered = pd.Series(eda_filtered_ds, index=new_index)

                            if eda_filtered.empty:
                                logging.error(f"Error: 'eda_filtered' is empty.")
                                # Log stack trace for debugging purposes
                                logging.error(traceback.format_exc())
                            else:
                                logging.info(f"'eda_filtered' is not empty, length: {len(eda_filtered)}")

                            sampling_rate = 50    # downsampled sampling rate
                            
                            # Calculate the time vector in minutes - Length of EDA data divided by the sampling rate gives time in seconds, convert to minutes
                            time_vector = np.arange(new_length) / sampling_rate / 60

                            ### Step 2: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods. ###
                            logging.info(f"Step 2: Clean prefiltered EDA for non-default phasic decomposition and peak detection methods.")

                            # First, clean the EDA signal
                            eda_cleaned = nk.eda_clean(eda_filtered, sampling_rate=sampling_rate)
                            logging.info(f"Prefiltered EDA signal cleaned using NeuroKit's eda_clean.")
                            
                            logging.info(f"Starting phasic decomposition and peak detection for prefiltered EDA signal.")
                            logging.info(f"Sampling rate: {sampling_rate} Hz")
                            logging.info(f"Size of prefiltered EDA signal: {eda_cleaned.size}")

                            # Define methods for phasic decomposition and peak detection.
                            methods = ['cvxEDA'] #sparsEDA 'sparse'? 'smoothmedian', 'highpass'
                            logging.info(f"Using the following methods for phasic decomposition: {methods}")

                            peak_methods = ["gamboa2008"] # "kim2004", "neurokit", "vanhalem2020", "nabian2018"
                            logging.info(f"Using the following methods for peak detection: {peak_methods}")

                            # Initialize psd_full_stats dictionary
                            filtered_psd_full_stats_by_method = {}    
                            filtered_psd_symp_full_stats_by_method = {}    
                            
                            ### Step 3: Perform cvxEDA phasic decomposition with Gamboa 2008 peak detection. ###
                            logging.info(f"Step 3: Perform cvxEDA phasic decomposition with Gamboa 2008 peak detection.")

                            # Process phasic decomposition of EDA signal using NeuroKit2
                            for method in methods:
                                logging.info(f"Starting processing for method: {method}")

                                # Decompose EDA signal using the specified method
                                try:
                                    decomposed = nk.eda_phasic(eda_cleaned, method=method, sampling_rate=sampling_rate)
                                    
                                    # Ensure 'decomposed' is a DataFrame with the required column.
                                    if not isinstance(decomposed, pd.DataFrame) or 'EDA_Phasic' not in decomposed.columns:
                                        logging.error(f"'decomposed' is not a DataFrame or 'EDA_Phasic' column is missing for method {method}.")
                                        # Log stack trace for debugging purposes
                                        logging.error(traceback.format_exc())
                                        continue
                                    
                                    logging.info(f"Decomposed EDA using {method} method. Size of decomposed data: {decomposed.size}")

                                except ValueError as e:
                                    logging.error(f"ValueError encountered: {e}")
                                    logging.error(f"Method: {method}, Sampling Rate: {sampling_rate}")
                                    logging.error(f"EDA Cleaned details: Range: {eda_cleaned.min()} - {eda_cleaned.max()}, NaNs: {np.isnan(eda_cleaned).sum()}, Infs: {np.isinf(eda_cleaned).sum()}")
                                    raise  # Optionally re-raise the exception if you want the program to stop
                                
                                except Exception as e:
                                    logging.error(f"Error in EDA decomposition with method {method}: {e}")
                                    # Log stack trace for debugging purposes
                                    logging.error(traceback.format_exc())                        
                                    continue
                                
                                # Process FD

                                fd_timeseries = fd  # Assuming 'fd' is your framewise displacement timeseries
                                eda_phasic = decomposed['EDA_Phasic'].values
                                eda_tonic = decomposed['EDA_Tonic'].values

                                eda_sampling_rate = sampling_rate
                                eda_duration_phasic = len(eda_phasic) / eda_sampling_rate  # Total duration in seconds
                                eda_duration_tonic = len(eda_tonic) / eda_sampling_rate  # Total duration in seconds

                                # Create a time array for the FD timeseries
                                fd_time_phasic = np.linspace(0, eda_duration_phasic, len(fd_timeseries))
                                fd_time_tonic = np.linspace(0, eda_duration_tonic, len(fd_timeseries))

                                # The new time array for the upsampled FD timeseries should match the EDA timeseries length
                                # Make sure the last time point in upsampled_time does not exceed the last point in fd_time
                                upsampled_time_phasic = np.linspace(0, eda_duration_phasic, len(eda_phasic))
                                upsampled_time_tonic = np.linspace(0, eda_duration_tonic, len(eda_tonic))

                                # Use linear interpolation with bounds_error set to False to prevent extrapolation
                                fd_interpolator_phasic = interp1d(fd_time_phasic, fd_timeseries, kind='linear', bounds_error=False, fill_value='extrapolate')
                                fd_upsampled_phasic = fd_interpolator_phasic(upsampled_time_phasic)

                                fd_interpolator_tonic = interp1d(fd_time_tonic, fd_timeseries, kind='linear', bounds_error=False, fill_value='extrapolate')
                                fd_upsampled_tonic = fd_interpolator_tonic(upsampled_time_tonic)

                                # Handle any NaN values that might have been introduced due to the bounds_error setting
                                fd_upsampled_phasic[np.isnan(fd_upsampled_phasic)] = fd_timeseries[-1]  # Replace NaNs with the last valid FD value
                                fd_upsampled_tonic[np.isnan(fd_upsampled_tonic)] = fd_timeseries[-1]  # Replace NaNs with the last valid FD value

                                #fd_upsampled_corrected_tonic = fd_upsampled_tonic / 100  # Correct the scale
                                #fd_upsampled_tonic = fd_upsampled_corrected_tonic

                                # Calculate Phasic FD-EDA correlation
                                r_value_phasic, p_value_phasic = calculate_fd_eda_correlation(fd_upsampled_phasic, eda_phasic)
                                logging.info(f"Correlation between FD and filtered cleaned Phasic EDA timeseries: {r_value_phasic}, p-value: {p_value_phasic}")
                                plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_eda_phasic_correlation.png"
                                plot_filepath = os.path.join(base_path, plot_filename)
                                plot_fd_eda_correlation(fd_upsampled_phasic, eda_phasic, plot_filepath)
                                logging.info(f"FD-EDA Phasic correlation plot saved to {plot_filepath}")

                                # Calculate Tonic FD-EDA correlation
                                r_value_tonic, p_value_tonic = calculate_fd_eda_correlation(fd_upsampled_tonic, eda_tonic)
                                logging.info(f"Correlation between FD and filtered cleaned Tonic EDA timeseries: {r_value_tonic}, p-value: {p_value_tonic}")
                                plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_eda_tonic_correlation.png"
                                plot_filepath = os.path.join(base_path, plot_filename)
                                plot_fd_eda_correlation(fd_upsampled_tonic, eda_tonic, plot_filepath)
                                logging.info(f"FD-EDA Tonic correlation plot saved to {plot_filepath}")

                                # After decomposing and before peak detection, calculate sympathetic indices by Posada method.
                                for symp_method in sympathetic_methods:
                                    try:
                                        # Calculate phasic sympathetic indices
                                        eda_symp_decomposed_phasic = nk.eda_sympathetic(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=symp_method, show=False)
            
                                        # Log the phasic sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Phasic_{symp_method}: {eda_symp_decomposed_phasic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned phasic sympathetic indices using {symp_method} method for {method}.")

                                    except Exception as e:
                                        logging.error(f"Error in sympathetic index calculation ({method}, {symp_method}): {e}")
                                
                                # Compute Power Spectral Density 0 - 1 Hz for Phasic EDA
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered Phasic EDA {method} using multitapers hann windowing.")
                                eda_psd_filt_phasic = nk.signal_psd(decomposed['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_filt_phasic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_filt_phasic[(eda_psd_filt_phasic['Frequency'] >= low_freq) & (eda_psd_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_filt_phasic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_filt_phasic_full_stats.update({
                                    'Phasic Mean (Normalized Power)': eda_psd_filt_phasic['Power'].mean(),
                                    'Phasic Median (Normalized Power)': eda_psd_filt_phasic['Power'].median(),
                                    'Phasic Total Power (Normalized Power)': eda_psd_filt_phasic['Power'].sum(),
                                    'Phasic Peak Frequency (Hz)': eda_psd_filt_phasic.loc[eda_psd_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Phasic Standard Deviation (Normalized Power)': eda_psd_filt_phasic['Power'].std(),
                                    'Phasic Variance (Normalized Power)': eda_psd_filt_phasic['Power'].var(),
                                    'Phasic Skewness': eda_psd_filt_phasic['Power'].skew(),
                                    'Phasic Kurtosis': eda_psd_filt_phasic['Power'].kurtosis(),
                                    'Phasic Peak Power (Normalized Power)': eda_psd_filt_phasic['Power'].max(),
                                    'Phasic Bandwidth (Hz)': eda_psd_filt_phasic['Frequency'].iloc[-1] - eda_psd_filt_phasic['Frequency'].iloc[0],
                                    'Phasic PSD Area (Normalized Power)': np.trapz(eda_psd_filt_phasic['Power'], eda_psd_filt_phasic['Frequency']),
                                    'Phasic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_phasic['EDA_Sympathetic'],
                                    'Phasic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_phasic['EDA_SympatheticN']
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned Phasic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_filt_phasic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned Phasic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_filt_phasic['Frequency'], eda_psd_filt_phasic['Power'], color='blue', label='Normalized Phasic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned Phasic EDA from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}.tsv")
                                eda_psd_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")
                            
                                # Assuming eda_psd_filt_phasic_full_stats is your dictionary of statistics
                                eda_psd_filt_phasic_stats_df = pd.DataFrame(eda_psd_filt_phasic_full_stats, index=[0])

                                # Transpose the DataFrame
                                eda_psd_filt_phasic_stats_df = eda_psd_filt_phasic_stats_df.transpose().reset_index()
                                eda_psd_filt_phasic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_{method}_summary_statistics.tsv")
                                eda_psd_filt_phasic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD summary statistics to {full_file_path}")
                                
                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned Phasic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_phasic = nk.signal_psd(decomposed['EDA_Phasic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.045, max_frequency=0.25, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_symp_filt_phasic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_symp_filt_phasic[(eda_psd_symp_filt_phasic['Frequency'] >= low_freq) & (eda_psd_symp_filt_phasic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_phasic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_phasic_full_stats.update({
                                    'Phasic Mean (Normalized Power)': eda_psd_symp_filt_phasic['Power'].mean(),
                                    'Phasic Median (Normalized Power)': eda_psd_symp_filt_phasic['Power'].median(),
                                    'Phasic Total Power (Normalized Power)': eda_psd_symp_filt_phasic['Power'].sum(),
                                    'Phasic Peak Frequency (Hz)': eda_psd_symp_filt_phasic.loc[eda_psd_symp_filt_phasic['Power'].idxmax(), 'Frequency'],
                                    'Phasic Standard Deviation (Normalized Power)': eda_psd_symp_filt_phasic['Power'].std(),
                                    'Phasic Variance (Normalized Power)': eda_psd_symp_filt_phasic['Power'].var(),
                                    'Phasic Skewness': eda_psd_symp_filt_phasic['Power'].skew(),
                                    'Phasic Kurtosis': eda_psd_symp_filt_phasic['Power'].kurtosis(),
                                    'Phasic Peak Power (Normalized Power)': eda_psd_symp_filt_phasic['Power'].max(),
                                    'Phasic Bandwidth (Hz)': eda_psd_symp_filt_phasic['Frequency'].iloc[-1] - eda_psd_symp_filt_phasic['Frequency'].iloc[0],
                                    'Phasic PSD Area (Normalized Power)': np.trapz(eda_psd_symp_filt_phasic['Power'], eda_psd_symp_filt_phasic['Frequency']),
                                    'Phasic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_phasic['EDA_Sympathetic'],
                                    'Phasic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_phasic['EDA_SympatheticN']
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned Phasic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_symp_filt_phasic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned Phasic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_symp_filt_phasic['Frequency'], eda_psd_symp_filt_phasic['Power'], color='blue', label='Normalized Phasic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned Phasic EDA in Symp band from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}.tsv")
                                eda_psd_symp_filt_phasic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD Symp data to {full_file_path}")
                            
                                # Assuming eda_psd_symp_filt_phasic_full_stats is your dictionary of statistics
                                eda_psd_symp_filt_phasic_stats_df = pd.DataFrame(eda_psd_symp_filt_phasic_full_stats, index=[0])

                                # Transpose the DataFrame
                                eda_psd_symp_filt_phasic_stats_df = eda_psd_symp_filt_phasic_stats_df.transpose().reset_index()
                                eda_psd_symp_filt_phasic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the correctly formatted DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_phasic_symp_{method}_summary_statistics.tsv")
                                eda_psd_symp_filt_phasic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Phasic PSD summary statistics to {full_file_path}")

                                # Compute Power Spectral Density 0 - 1 Hz
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} using multitapers hann windowing.")
                                eda_psd_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0, max_frequency=1.0, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_filt_tonic_full_stats = {}  
                                
                                # After decomposing and before peak detection, calculate sympathetic indices
                                for symp_method in sympathetic_methods:
                                    try:
                                        # Calculate tonic sympathetic indices
                                        eda_symp_decomposed_tonic = nk.eda_sympathetic(decomposed["EDA_Tonic"], sampling_rate=sampling_rate, method=symp_method, show=False)

                                        # Log the sympathetic index for phasic decomposition methods. 
                                        logging.info(f"EDA_Sympathetic_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_Sympathetic']}")
                                        logging.info(f"EDA_Sympathetic_Normalized_Tonic_{symp_method}: {eda_symp_decomposed_tonic['EDA_SympatheticN']}")
                                        logging.info(f"Calculated filtered cleaned tonic sympathetic indices using {symp_method} method for {method}.")

                                    except Exception as e:
                                        logging.error(f"Error in sympathetic index calculation ({method}, {symp_method}): {e}")
                                
                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    band_power = eda_psd_filt_tonic[(eda_psd_filt_tonic['Frequency'] >= low_freq) & (eda_psd_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_filt_tonic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_filt_tonic_full_stats.update({
                                    'Tonic Mean (Normalized Power)': eda_psd_filt_tonic['Power'].mean(),
                                    'Tonic Median (Normalized Power)': eda_psd_filt_tonic['Power'].median(),
                                    'Tonic Total Power (Normalized Power)': eda_psd_filt_tonic['Power'].sum(),
                                    'Tonic Peak Frequency (Hz)': eda_psd_filt_tonic.loc[eda_psd_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Tonic Standard Deviation (Normalized Power)': eda_psd_filt_tonic['Power'].std(),
                                    'Tonic Variance (Normalized Power)': eda_psd_filt_tonic['Power'].var(),
                                    'Tonic Skewness': eda_psd_filt_tonic['Power'].skew(),
                                    'Tonic Kurtosis': eda_psd_filt_tonic['Power'].kurtosis(),
                                    'Tonic Peak Power (Normalized Power)': eda_psd_filt_tonic['Power'].max(),
                                    'Tonic Bandwidth (Hz)': eda_psd_filt_tonic['Frequency'].iloc[-1] - eda_psd_filt_tonic['Frequency'].iloc[0],
                                    'Tonic PSD Area (Normalized Power)': np.trapz(eda_psd_filt_tonic['Power'], eda_psd_filt_tonic['Frequency']),
                                    'Tonic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_tonic['EDA_Sympathetic'],
                                    'Tonic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_tonic['EDA_SympatheticN']
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned tonic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_filt_tonic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned tonic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_filt_tonic['Frequency'], eda_psd_filt_tonic['Power'], color='blue', label='Normalized tonic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned tonic EDA from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}.tsv")
                                eda_psd_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                                # Assuming eda_psd_filt_tonic_full_stats is your dictionary of statistics
                                eda_psd_filt_tonic_stats_df = pd.DataFrame(eda_psd_filt_tonic_full_stats, index=[0])

                                # Transpose the DataFrame to get labels as rows and values in the second column
                                eda_psd_filt_tonic_stats_df = eda_psd_filt_tonic_stats_df.transpose().reset_index()
                                eda_psd_filt_tonic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the transposed DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_{method}_summary_statistics.tsv")
                                eda_psd_filt_tonic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved full range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")
                                
                                # Compute Power Spectral Density in Sympathetic band
                                logging.info(f"Computing Power Spectral Density (PSD) for filtered cleaned tonic EDA {method} in Sympathetic band using multitapers hann windowing.")
                                eda_psd_symp_filt_tonic = nk.signal_psd(decomposed['EDA_Tonic'], sampling_rate=sampling_rate, method='multitapers', show=False, normalize=True, 
                                                        min_frequency=0.045, max_frequency=0.25, window=None, window_type='hann',
                                                        silent=False, t=None)
                            
                                # Initialize psd_full_stats dictionary
                                eda_psd_symp_filt_tonic_full_stats = {}  

                                # Define the frequency bands
                                frequency_bands = {
                                    'VLF': (0, 0.045),
                                    'LF': (0.045, 0.15),
                                    'HF1': (0.15, 0.25),
                                    'HF2': (0.25, 0.4),
                                    'VHF': (0.4, 0.5)
                                }

                                # Calculate the power in each band
                                for band, (low_freq, high_freq) in frequency_bands.items():
                                    # Corrected line of code
                                    band_power = eda_psd_symp_filt_tonic[(eda_psd_symp_filt_tonic['Frequency'] >= low_freq) & (eda_psd_symp_filt_tonic['Frequency'] < high_freq)]['Power'].sum()
                                    eda_psd_symp_filt_tonic_full_stats[f'{band} Power'] = band_power

                                # Calculate and save summary statistics for the full range PSD
                                eda_psd_symp_filt_tonic_full_stats.update({
                                    'Tonic Mean (Normalized Power)': eda_psd_symp_filt_tonic['Power'].mean(),
                                    'Tonic Median (Normalized Power)': eda_psd_symp_filt_tonic['Power'].median(),
                                    'Tonic Total Power (Normalized Power)': eda_psd_symp_filt_tonic['Power'].sum(),
                                    'Tonic Peak Frequency (Hz)': eda_psd_symp_filt_tonic.loc[eda_psd_symp_filt_tonic['Power'].idxmax(), 'Frequency'],
                                    'Tonic Standard Deviation (Normalized Power)': eda_psd_symp_filt_tonic['Power'].std(),
                                    'Tonic Variance (Normalized Power)': eda_psd_symp_filt_tonic['Power'].var(),
                                    'Tonic Skewness': eda_psd_symp_filt_tonic['Power'].skew(),
                                    'Tonic Kurtosis': eda_psd_symp_filt_tonic['Power'].kurtosis(),
                                    'Tonic Peak Power (Normalized Power)': eda_psd_symp_filt_tonic['Power'].max(),
                                    'Tonic Bandwidth (Hz)': eda_psd_symp_filt_tonic['Frequency'].iloc[-1] - eda_psd_symp_filt_tonic['Frequency'].iloc[0],
                                    'Tonic PSD Area (Normalized Power)': np.trapz(eda_psd_symp_filt_tonic['Power'], eda_psd_symp_filt_tonic['Frequency']),
                                    'Tonic Sympathetic Power Posada Method (μS2/Hz)': eda_symp_decomposed_tonic['EDA_Sympathetic'],
                                    'Tonic Sympathetic Power Normalized Posada Method': eda_symp_decomposed_tonic['EDA_SympatheticN']
                                })

                                # Log the summary statistics
                                logging.info(f'filtered cleaned tonic PSD Summary Statistics for method {method}')
                                for stat, value in eda_psd_symp_filt_tonic_full_stats.items():
                                    logging.info(f"{stat}: {value}")

                                # Plotting Power Spectral Density
                                logging.info(f"Plotting Power Spectral Density (PSD) 0 - 1 Hz for filtered cleaned tonic EDA using multitapers hann windowing.")
                                plt.figure(figsize=(12, 6))
                                plt.fill_between(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], color='blue', alpha=0.3)  # alpha controls the transparency
                                plt.plot(eda_psd_symp_filt_tonic['Frequency'], eda_psd_symp_filt_tonic['Power'], color='blue', label='Normalized tonic PSD (Multitapers with Hanning Window)')
                                plt.title(f'Power Spectral Density (PSD) (Multitapers with Hanning Window) for filtered cleaned tonic EDA in Symp band from {method} method')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Normalized Power')
                                plt.legend()
                                plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}.png")
                                plt.savefig(plot_filename, dpi=dpi_value)
                                #plt.show()
                                plt.close()

                                # Save the full range PSD data to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}.tsv")
                                eda_psd_symp_filt_tonic.to_csv(full_file_path, sep='\t', index=False)
                                logging.info(f"Saved full range Filtered Cleaned PSD data to {full_file_path}")

                                # Assuming eda_psd_symp_filt_tonic_full_stats is your dictionary of statistics
                                eda_psd_symp_filt_tonic_stats_df = pd.DataFrame(eda_psd_symp_filt_tonic_full_stats, index=[0])

                                # Transpose the DataFrame to get labels as rows and values in the second column
                                eda_psd_symp_filt_tonic_stats_df = eda_psd_symp_filt_tonic_stats_df.transpose().reset_index()
                                eda_psd_symp_filt_tonic_stats_df.columns = ['Statistic', 'Value']  # Set new headers

                                # Save the transposed DataFrame to a TSV file
                                full_file_path = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_psd_tonic_symp_{method}_summary_statistics.tsv")
                                eda_psd_symp_filt_tonic_stats_df.to_csv(full_file_path, sep='\t', header=True, index=False)
                                logging.info(f"Saved Symp range Filtered Cleaned Tonic PSD summary statistics to {full_file_path}")

                                # # Calculate amplitude_min from the phasic component for peak detection
                                # amplitude_min = 0.001 * decomposed["EDA_Phasic"].max()
                                # logging.info(f"Calculated amplitude_min: {amplitude_min} for {method} method.")

                                for peak_method in peak_methods:
                                    try:
                                       
                                        if not pd.to_numeric(decomposed["EDA_Phasic"], errors='coerce').notnull().all():
                                            logging.warning("Non-numeric values found in EDA_Phasic. Attempting to clean...")
                                            # Implement appropriate cleaning or transformation here

                                        # Check if the phasic component is not empty
                                        if decomposed["EDA_Phasic"].size == 0:
                                            logging.warning(f"The phasic component is empty for {method}. Skipping peak detection.")
                                            continue
                                        
                                        if len(decomposed["EDA_Phasic"]) > 0:
                                            logging.info(f"Array is not empty")
                                        else:
                                            logging.warning(f"Array is empty")

                                        # Additional check for non-empty and valid data
                                        if decomposed["EDA_Phasic"].isnull().all() or decomposed["EDA_Phasic"].empty:
                                            logging.warning(f"No valid data in phasic component for {method} using {peak_method}. Skipping peak detection.")
                                            continue
                                            
                                        # Check if there are NaN values
                                        if np.isnan(decomposed["EDA_Phasic"]).any():
                                            logging.warning(f"Array contains NaN values")

                                            # Fill NaN values with a specific value, for example, 0
                                            logging.info(f"Filling NaN values with 0")
                                            decomposed["EDA_Phasic"].fillna(0, inplace=True)
                                        
                                        if not pd.to_numeric(decomposed["EDA_Phasic"], errors='coerce').notnull().all():
                                            logging.warning(f"Array contains non-numeric values")

                                        if decomposed["EDA_Phasic"].nunique() == 1:
                                            logging.warning(f"Array contains constant values")

                                        # Detect peaks using the specified method
                                        logging.info(f"Detecting peaks using method: {peak_method}")    
                                        _, peaks = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=sampling_rate, method=peak_method)
                                        
                                        # Log the detected SCR onsets and recoveries for inspection
                                        logging.info(f"SCR Onsets: {peaks['SCR_Onsets']}")
                                        logging.info(f"SCR Rise Time: {peaks['SCR_RiseTime']}")
                                        logging.info(f"SCR Peaks: {peaks['SCR_Peaks']}")
                                        logging.info(f"SCR Amplitude: {peaks['SCR_Amplitude']}")
                                        logging.info(f"SCR Height: {peaks['SCR_Height']}")
                                        logging.info(f"SCR Recovery: {peaks['SCR_Recovery']}")
                                        logging.info(f"SCR Recovery Time: {peaks['SCR_RecoveryTime']}")
                                        
                                        # Initialize the columns for SCR events in the DataFrame
                                        decomposed['SCR_Onsets'] = 0
                                        decomposed['SCR_Peaks'] = 0
                                        decomposed['SCR_RiseTime'] = np.nan
                                        decomposed['SCR_Amplitude'] = np.nan
                                        decomposed['SCR_Recovery'] = 0
                                        decomposed['SCR_RecoveryTime'] = np.nan

                                        # Convert to 0-based indexing if your data is 1-based indexed
                                        valid_onsets = peaks['SCR_Onsets'] - 1
                                        valid_peaks = peaks['SCR_Peaks'] - 1

                                        # Filter peaks that come after the first valid onset
                                        first_valid_onset = valid_onsets[0]
                                        valid_peaks_indices = valid_peaks[valid_peaks > first_valid_onset]
                                        valid_peaks_after_onset = valid_peaks[valid_peaks > first_valid_onset]

                                        # Update SCR Onsets in the DataFrame
                                        decomposed.loc[valid_onsets, 'SCR_Onsets'] = 1

                                        # Update SCR Peaks, Amplitude, and Rise Time in the DataFrame, only for peaks that occur after the first valid onset
                                        decomposed.loc[valid_peaks_after_onset, 'SCR_Peaks'] = 1
                                        decomposed.loc[valid_peaks_after_onset, 'SCR_Amplitude'] = peaks['SCR_Amplitude'][valid_peaks > first_valid_onset]
                                        decomposed.loc[valid_peaks_after_onset, 'SCR_RiseTime'] = peaks['SCR_RiseTime'][valid_peaks > first_valid_onset]

                                        # Handle NaN values for Amplitude, Rise Time and Recovery Time where no events are detected
                                        decomposed['SCR_Amplitude'].fillna(0, inplace=True)
                                        decomposed['SCR_RecoveryTime'].fillna(0, inplace=True)
                                        decomposed['SCR_RiseTime'].fillna(0, inplace=True)

                                        # Filter out the valid recovery indices where recovery times are not NaN
                                        valid_recovery_indices = peaks['SCR_Recovery'][~np.isnan(peaks['SCR_Recovery'])].astype(int) - 1  # Adjust indexing if needed

                                        # Update the decomposed DataFrame with valid recoveries
                                        decomposed.loc[valid_recovery_indices, 'SCR_Recovery'] = 1
                                        decomposed.loc[valid_recovery_indices, 'SCR_RecoveryTime'] = peaks['SCR_RecoveryTime'][~np.isnan(peaks['SCR_RecoveryTime'])]
                                        
                                        # Initialize the columns for invalid SCR events in the DataFrame
                                        decomposed['SCR_Invalid_Recovery'] = 0

                                        # Create and configure plots
                                        fig, axes = plt.subplots(4, 1, figsize=(20, 10))

                                        # Plot 1: Overlay of Raw and Cleaned EDA
                                        axes[0].plot(eda_filtered, label='Raw Filtered EDA')
                                        axes[0].plot(eda_cleaned, label='Filtered Cleaned EDA', color='orange')
                                        axes[0].set_title('Filtered Raw and Cleaned EDA Signal')
                                        axes[0].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                        axes[0].set_ylabel('Amplitude (µS)')
                                        axes[0].legend()

                                        # Plot 2: Phasic Component with SCR Events
                                        axes[1].plot(decomposed["EDA_Phasic"], label='Phasic Component', color='green')
                                        axes[1].scatter(valid_onsets, decomposed.loc[valid_onsets, "EDA_Phasic"], color='blue', label='SCR Onsets')
                                        axes[1].scatter(valid_peaks_indices, decomposed.loc[valid_peaks_indices, "EDA_Phasic"], color='red', label='SCR Peaks')
                                        axes[1].scatter(valid_recovery_indices, decomposed.loc[valid_recovery_indices, "EDA_Phasic"], color='purple', label='SCR Recovery')

                                        # Mark invalid recoveries with black x marks and update the DataFrame
                                        invalid_recovery_indices = []  # Initialize an empty list to keep track of invalid recovery indices
                                        for peak_idx, recovery_time in zip(peaks['SCR_Peaks'], peaks['SCR_RecoveryTime']):
                                            if np.isnan(recovery_time):
                                                peak_idx_adjusted = peak_idx - 1 if peak_idx > 0 else peak_idx
                                                if peak_idx_adjusted < len(decomposed):
                                                    axes[1].scatter(peak_idx_adjusted, decomposed.loc[peak_idx_adjusted, "EDA_Phasic"], color='black', marker='x')
                                                    decomposed.at[peak_idx_adjusted, 'SCR_Invalid_Recovery'] = 1
                                                    invalid_recovery_indices.append(peak_idx_adjusted)  # Add the index to the list
                                        invalid_recovery_count = len(invalid_recovery_indices)  # Get the count of invalid recoveries

                                        # Add a single legend entry for invalid recoveries
                                        axes[1].scatter([], [], color='black', marker='x', label='Invalid Recovery')

                                        # Add legend and set titles
                                        axes[1].set_title(f'Phasic EDA ({method}) with {peak_method} Peaks')
                                        axes[1].set_xlabel(f'Samples ({sampling_rate} Hz)')
                                        axes[1].set_ylabel('Amplitude (µS)')
                                        axes[1].legend()

                                        # Plot 3: Tonic Component
                                        axes[2].plot(decomposed.index / sampling_rate / 60, decomposed["EDA_Tonic"], label='Tonic Component', color='brown')
                                        axes[2].set_title(f'Tonic EDA ({method})')
                                        axes[2].set_xlabel('Time (minutes)')
                                        axes[2].set_ylabel('Amplitude (µS)')
                                        axes[2].legend()

                                        # Plot 4: Framewise Displacement
                                        voxel_threshold = 0.5 # mm
                                        axes[3].plot(fd_upsampled_phasic, label='Framewise Displacement', color='blue')
                                        axes[3].axhline(y=voxel_threshold, color='r', linestyle='--')
                                        
                                        # Calculate the number of volumes (assuming 2 sec TR and given sampling rate)
                                        num_volumes = len(fd_upsampled_phasic) / (sampling_rate * 2)

                                        # Generate the volume numbers for the x-axis
                                        volume_numbers = np.arange(0, num_volumes)

                                        # Set x-axis ticks to display volume numbers at regular intervals
                                        # The interval for ticks can be adjusted (e.g., every 10 volumes)
                                        tick_interval = 10  # Adjust this value as needed
                                        axes[3].set_xticks(np.arange(0, len(fd_upsampled_phasic), tick_interval * sampling_rate * 2))
                                        axes[3].set_xticklabels([f"{int(vol)}" for vol in volume_numbers[::tick_interval]])

                                        axes[3].set_title('Framewise Displacement')
                                        axes[3].set_xlabel('Volume Number (2 sec TR)')
                                        axes[3].set_ylabel('FD (mm)')

                                        # # Add shading where FD is above threshold across all subplots
                                        # for ax in axes[:-1]: # Exclude the last axis which is for FD plot
                                        #     ax.fill_between(decomposed.index / sampling_rate / 60, 0, voxel_threshold, where=fd_upsampled_phasic > voxel_threshold, color='red', alpha=0.3)

                                        # Save the combined plot
                                        plt.tight_layout()
                                        combo_plot_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_{method}_{peak_method}_subplots.png")
                                        plt.savefig(combo_plot_filename, dpi=dpi_value)
                                        plt.close()
                                        logging.info(f"Saved EDA subplots for {method} with {peak_method} to {combo_plot_filename}")
                                        
                                        # Create a mask for FD values less than or equal to 0.5
                                        mask_phasic = fd_upsampled_phasic <= 0.5
                                        mask_tonic = fd_upsampled_tonic <= 0.5

                                        # Apply the mask to both FD and EDA data
                                        filtered_fd_phasic = fd_upsampled_phasic[mask_phasic]
                                        filtered_fd_tonic = fd_upsampled_tonic[mask_tonic]
                                        filtered_eda_phasic = eda_phasic[mask_phasic]
                                        filtered_eda_tonic = eda_tonic[mask_tonic]

                                        # Initialize correlation and above threshold FD variables as NaN
                                        r_value_phasic_thresh = np.nan
                                        p_value_phasic_thresh = np.nan
                                        r_value_tonic_thresh = np.nan
                                        p_value_tonic_thresh = np.nan

                                        num_samples_above_threshold = np.nan
                                        percent_samples_above_threshold = np.nan
                                        mean_fd_above_threshold = np.nan
                                        std_dev_fd_above_threshold = np.nan
                                        
                                        # Check if there are any FD values above the threshold
                                        if np.any(fd > voxel_threshold):
                                            # # Filter FD and EDA data based on the threshold
                                            # filtered_fd = fd[fd < voxel_threshold]
                                            # filtered_eda_phasic = eda[fd < voxel_threshold]  # Assuming eda is aligned with fd and same length
                                            
                                            # Check if filtered data is not empty
                                            if len(filtered_fd_phasic) > 0 and len(filtered_eda_phasic) > 0:
                                                
                                                # Calculate above threshold FD statistics
                                                num_samples_above_threshold = np.sum(fd_upsampled_phasic > voxel_threshold)
                                                percent_samples_above_threshold = num_samples_above_threshold / len(fd_upsampled_phasic) * 100
                                                mean_fd_above_threshold = np.mean(fd_upsampled_phasic[fd_upsampled_phasic > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
                                                std_dev_fd_above_threshold = np.std(fd_upsampled_phasic[fd_upsampled_phasic > voxel_threshold]) if num_samples_above_threshold > 0 else np.nan
                                                
                                                # Calculate the correlation between filtered FD and Phasic EDA
                                                r_value_phasic_thresh, p_value_phasic_thresh = calculate_fd_eda_correlation(filtered_fd_phasic, filtered_eda_phasic)
                                                logging.info(f"Correlation between FD (filtered) and filtered cleaned Phasic EDA timeseries < {voxel_threshold} mm: {r_value_phasic_thresh}, p-value: {p_value_phasic_thresh}")

                                                plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_eda_phasic_correlation_filtered.png"
                                                plot_filepath = os.path.join(base_path, plot_filename)
                                                plot_fd_eda_correlation(fd_upsampled_phasic, filtered_eda_phasic, plot_filepath)
                                                logging.info(f"FD-EDA Phasic filtered correlation plot saved to {plot_filepath}")
                                            
                                                # Calculate the correlation between filtered FD and Tonic EDA
                                                r_value_tonic_thresh, p_value_tonic_thresh = calculate_fd_eda_correlation(filtered_fd_tonic, filtered_eda_tonic)
                                                logging.info(f"Correlation between FD (filtered) and filtered cleaned Tonic EDA timeseries < {voxel_threshold} mm: {r_value_tonic_thresh}, p-value: {p_value_tonic_thresh}")

                                                plot_filename = f"{participant_id}_{session_id}_task-{task_name}_{run_id}_fd_eda_tonic_correlation_filtered.png"
                                                plot_filepath = os.path.join(base_path, plot_filename)
                                                plot_fd_eda_correlation(fd_upsampled_tonic, filtered_eda_tonic, plot_filepath)
                                                logging.info(f"FD-EDA Tonic filtered correlation plot saved to {plot_filepath}")
                                            else:
                                                # Log a warning if there are no FD values below the threshold after filtering
                                                logging.warning(f"No FD values below {voxel_threshold} mm. Correlation cannot be calculated.")
                                        else:
                                            # Log a warning if there are no FD values above the threshold
                                            logging.warning(f"No FD values above {voxel_threshold} mm. No need to filter and calculate correlation.")

                                        # Calculate statistics related to framewise displacement
                                        mean_fd_below_threshold = np.mean(fd_upsampled_phasic[fd_upsampled_phasic < voxel_threshold])
                                        std_dev_fd_below_threshold = np.std(fd_upsampled_phasic[fd_upsampled_phasic < voxel_threshold])

                                        # Assign existing correlation R-value and P-value
                                        # (Assuming r_value and p_value_thresh are already calculated with checks)
                                        correlation_r_value_phasic = r_value_phasic
                                        correlation_p_value_phasic = p_value_phasic
                                        correlation_r_value_below_threshold_phasic = r_value_phasic_thresh
                                        correlation_p_value_below_threshold_phasic = p_value_phasic_thresh

                                        correlation_r_value_tonic = r_value_tonic
                                        correlation_p_value_tonic = p_value_tonic
                                        correlation_r_value_below_threshold_tonic = r_value_tonic_thresh
                                        correlation_p_value_below_threshold_tonic = p_value_tonic_thresh

                                        # Filter SCR amplitudes for valid peaks only
                                        valid_scr_amplitudes = decomposed.loc[valid_peaks_indices, 'SCR_Amplitude']

                                        # Calculate the required statistics using the filtered amplitudes
                                        non_response_scrs = np.sum(valid_scr_amplitudes < 0.01)  # SCRs less than 0.01 microsiemens
                                        mean_scr_amplitude = valid_scr_amplitudes.mean()  # Mean SCR amplitude
                                        std_dev_scr_amplitude = valid_scr_amplitudes.std()  # Standard deviation of SCR amplitude
                                        max_scr_amplitude = valid_scr_amplitudes.max()  # Maximum SCR amplitude
                                        min_scr_amplitude = valid_scr_amplitudes.min()  # Minimum SCR amplitude
                                        amplitude_range = max_scr_amplitude - min_scr_amplitude  # Range of SCR amplitudes

                                        # Average SCR Frequency (counts/min)
                                        total_time_minutes = len(decomposed) / (sampling_rate * 60)
                                        average_scr_frequency = len(valid_peaks_indices) / total_time_minutes

                                        # Average, Std Deviation, Max, Min Inter-SCR Interval (sec)
                                        inter_scr_intervals = np.diff(valid_peaks_indices) / sampling_rate
                                        average_inter_scr_interval = inter_scr_intervals.mean()
                                        std_inter_scr_interval = inter_scr_intervals.std()
                                        max_inter_scr_interval = inter_scr_intervals.max()
                                        min_inter_scr_interval = inter_scr_intervals.min()

                                        # Mean, Std Deviation, Max, Min RiseTime (sec)
                                        mean_risetime = decomposed['SCR_RiseTime'][decomposed['SCR_RiseTime'] > 0].mean()
                                        std_risetime = decomposed['SCR_RiseTime'][decomposed['SCR_RiseTime'] > 0].std()
                                        max_risetime = decomposed['SCR_RiseTime'].max()
                                        min_risetime = decomposed['SCR_RiseTime'][decomposed['SCR_RiseTime'] > 0].min()

                                        # Mean, Std Deviation Half RecoveryTime (sec)
                                        valid_recovery_times = decomposed['SCR_RecoveryTime'][decomposed['SCR_Recovery'] == 1]
                                        mean_recoverytime = valid_recovery_times.mean()
                                        std_recoverytime = valid_recovery_times.std()

                                        # Update the scr_stats dictionary
                                        scr_stats = {
                                            'SCR Count (# peaks)': len(valid_peaks_indices),
                                            'Valid Recovery Count': len(valid_recovery_indices),
                                            'Invalid Recovery Count': invalid_recovery_count,
                                            'Count of Non-Response SCRs (< 0.01 µS)': non_response_scrs,
                                            'Mean SCR Amplitude (µS)': mean_scr_amplitude,
                                            'Std Deviation SCR Amplitude (µS)': std_dev_scr_amplitude,
                                            'Max SCR Amplitude (µS)': max_scr_amplitude,
                                            'Min SCR Amplitude (µS)': min_scr_amplitude,
                                            'Amplitude Range of SCRs (µS)': amplitude_range,
                                            'Average SCR Frequency (counts/min)': average_scr_frequency,
                                            'Average Inter-SCR Interval (sec)': average_inter_scr_interval,
                                            'Std Deviation Inter-SCR Interval (sec)': std_inter_scr_interval,
                                            'Max Inter-SCR Interval (sec)': max_inter_scr_interval,
                                            'Min Inter-SCR Interval (sec)': min_inter_scr_interval,
                                            'Mean RiseTime (sec)': mean_risetime,
                                            'Std Deviation RiseTime (sec)': std_risetime,
                                            'Mean Half RecoveryTime (sec)': mean_recoverytime,
                                            'Std Deviation Half RecoveryTime (sec)': std_recoverytime,
                                            'Max RiseTime (sec)': max_risetime,
                                            'Min RiseTime (sec)': min_risetime,
                                            'Mean Framewise Displacement (mm)': fd_upsampled_phasic.mean(),
                                            'Std Deviation Framewise Displacement (mm)': fd_upsampled_phasic.std(),
                                            'Max Framewise Displacement (mm)': fd_upsampled_phasic.max(),
                                            'Min Framewise Displacement (mm)': fd_upsampled_phasic.min(),
                                            'Number of samples with FD > 0.5 mm': num_samples_above_threshold,
                                            'Percent of samples with FD > 0.5 mm': percent_samples_above_threshold,
                                            'Mean FD > 0.5 mm': mean_fd_above_threshold,
                                            'Std Deviation FD > 0.5 mm': std_dev_fd_above_threshold,
                                            'Mean FD < 0.5 mm': mean_fd_below_threshold,
                                            'Std Deviation FD < 0.5 mm': std_dev_fd_below_threshold,
                                            'Framewise Displacement - Phasic EDA Correlation R-Value': r_value_phasic,
                                            'Framewise Displacement - Phasic EDA Correlation P-Value': p_value_phasic,
                                            'Framewise Displacement - Tonic EDA Correlation R-Value': r_value_tonic,
                                            'Framewise Displacement - Tonic EDA Correlation P-Value': p_value_tonic,
                                            'Framewise Displacement - Phasic EDA Correlation R-Value (FD < 0.5 mm)': r_value_phasic_thresh,
                                            'Framewise Displacement - Phasic EDA Correlation P-Value (FD < 0.5 mm)': p_value_phasic_thresh,
                                            'Framewise Displacement - Tonic EDA Correlation R-Value (FD < 0.5 mm)': r_value_tonic_thresh,
                                            'Framewise Displacement - Tonic EDA Correlation P-Value (FD < 0.5 mm)': p_value_tonic_thresh
                                        }
                                       
                                        # Assuming decomposed is a DataFrame containing the decomposed EDA components
                                        phasic_component = decomposed['EDA_Phasic']
                                        tonic_component = decomposed['EDA_Tonic']

                                        # Basic statistics for Phasic Component
                                        phasic_stats = {
                                            'Phasic Mean (µS)': phasic_component.mean(),
                                            'Phasic Median (µS)': phasic_component.median(),
                                            'Phasic Std Deviation (µS)': phasic_component.std(),
                                            'Phasic Variance (µS)': phasic_component.var(),
                                            'Phasic Skewness': phasic_component.skew(),
                                            'Phasic Kurtosis': phasic_component.kurtosis(),
                                            'Phasic Range (µS)': phasic_component.max() - phasic_component.min(),
                                            'Phasic Total Absolute Sum (µS)': phasic_component.abs().sum(),
                                            'Phasic 25th Percentile (µS)': phasic_component.quantile(0.25),
                                            'Phasic 75th Percentile (µS)': phasic_component.quantile(0.75),
                                            'Phasic IQR': phasic_component.quantile(0.75) - phasic_component.quantile(0.25),
                                            'Phasic 10th Percentile (µS)': phasic_component.quantile(0.10),
                                            'Phasic 90th Percentile (µS)': phasic_component.quantile(0.90)
                                        }

                                        # Basic statistics for Tonic Component
                                        tonic_stats = {
                                            'Tonic Mean (µS)': tonic_component.mean(),
                                            'Tonic Median (µS)': tonic_component.median(),
                                            'Tonic Std Deviation (µS)': tonic_component.std(),
                                            'Tonic Variance (µS)': tonic_component.var(),
                                            'Tonic Skewness': tonic_component.skew(),
                                            'Tonic Kurtosis': tonic_component.kurtosis(),
                                            'Tonic Range (µS)': tonic_component.max() - tonic_component.min(),
                                            'Tonic Total Absolute Sum (µS)': tonic_component.abs().sum(),
                                            'Tonic 25th Percentile (µS)': tonic_component.quantile(0.25),
                                            'Tonic 75th Percentile (µS)': tonic_component.quantile(0.75),
                                            'Tonic IQR': tonic_component.quantile(0.75) - tonic_component.quantile(0.25),
                                            'Tonic 10th Percentile (µS)': tonic_component.quantile(0.10),
                                            'Tonic 90th Percentile (µS)': tonic_component.quantile(0.90)
                                        }

                                        # Debug: Check the updated SCR statistics
                                        logging.info(f"SCR Stats: {scr_stats}")

                                        # Assume scr_stats, phasic_stats, and tonic_stats are dictionaries containing the statistics
                                        scr_stats_df = pd.DataFrame(scr_stats.items(), columns=['Statistic', 'Value'])
                                        phasic_stats_df = pd.DataFrame(phasic_stats.items(), columns=['Statistic', 'Value'])
                                        tonic_stats_df = pd.DataFrame(tonic_stats.items(), columns=['Statistic', 'Value'])

                                        # Concatenate the three DataFrames vertically, ensuring the order is maintained
                                        eda_summary_stats_df = pd.concat([scr_stats_df, phasic_stats_df, tonic_stats_df], axis=0, ignore_index=True)

                                        # Add a column to indicate the category of each statistic
                                        eda_summary_stats_df.insert(0, 'Category', '')
                                        eda_summary_stats_df.loc[:len(scr_stats)-1, 'Category'] = 'SCR Stats'
                                        eda_summary_stats_df.loc[len(scr_stats):len(scr_stats)+len(phasic_stats)-1, 'Category'] = 'Phasic Stats'
                                        eda_summary_stats_df.loc[len(scr_stats)+len(phasic_stats):, 'Category'] = 'Tonic Stats'

                                        # Save the summary statistics to a TSV file, with headers and without the index
                                        summary_stats_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_{method}_{peak_method}_summary_statistics.tsv")
                                        eda_summary_stats_df.to_csv(summary_stats_filename, sep='\t', header=True, index=False)
                                        logging.info(f"Saving summary statistics to TSV file: {summary_stats_filename}")
                                    
                                    except Exception as e:
                                        logging.error(f"Error in peak detection ({method}, {peak_method}): {e}")
                                        # Log stack trace for debugging purposes
                                        logging.error(traceback.format_exc())
                                        continue

                                # Save decomposed data to a TSV file
                                logging.info(f"Saving decomposed data to TSV file for {method}.")
                                decomposed_filename = os.path.join(base_path, f"{base_filename}_filtered_cleaned_eda_processed_{method}.tsv")
                                decomposed.to_csv(decomposed_filename, sep='\t', index=False)
                                
                                # Compress the TSV file
                                with open(decomposed_filename, 'rb') as f_in:
                                    with gzip.open(f"{decomposed_filename}.gz", 'wb') as f_out:
                                        f_out.writelines(f_in)
                                
                                # Remove the uncompressed file
                                os.remove(decomposed_filename)
                                logging.info(f"Saved and compressed decomposed data to {decomposed_filename}.gz")