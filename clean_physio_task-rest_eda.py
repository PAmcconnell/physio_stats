# Script name: clean_physio_task-rest_eda.py

import neurokit2 as nk
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import logging
import os
import traceback
from datetime import datetime

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main script logic
def main():
    """
    Main function to clean ecg data from a BIDS dataset.
    """
    logging.info("Starting main function")
    start_time = datetime.now()

    # Define and check BIDS root directory
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    logging.info(f"Checking BIDS root directory: {bids_root_dir}")
    if not os.path.exists(bids_root_dir):
        logging.warning(f"Directory not found: {bids_root_dir}")
        return

    # Define and check derivatives directory
    derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    logging.info(f"Checking derivatives directory: {derivatives_dir}")
    if not os.path.exists(derivatives_dir):
        logging.warning(f"Directory not found: {derivatives_dir}")
        return

    # Define and check participants file
    participants_file = os.path.join(bids_root_dir, 'participants.tsv')
    logging.info(f"Checking participants file: {participants_file}")
    if not os.path.exists(participants_file):
        logging.warning(f"Participants file not found: {participants_file}")
        return

    try: 
        # Read participants file and process groups
        participants_df = pd.read_csv(participants_file, delimiter='\t')
        
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error reading participants file: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading participants file: {e}")
        return    
    
    logging.info(f"Found {len(participants_df)} participants")
#  # Process each run for each participant
#     #for run_number in range(1, 5):  # Assuming 4 runs
#     for run_number in range(1, 1):  # Test 1 run
#         for i, participant_id in enumerate(participants_df['participant_id']):
            
    # Select the first participant for testing
    participant_id = participants_df['participant_id'].iloc[2]
    #logging.info(f"Testing for participant: {participant_id}")
    task_name = 'rest'

    # Process the first run for the selected participant
    run_number = 1
    logging.info(f"Testing run task {task_name} run-0{run_number} for participant {participant_id}")
    physio_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
    for file in physio_files:
        logging.info(f"Processing file: {file}")
        try:
            base_filename = os.path.splitext(os.path.splitext(file)[0])[0]  # Removes .tsv.gz

            with gzip.open(file, 'rt') as f:
                physio_data = pd.read_csv(f, delimiter='\t')
                logging.info(f"Data shape after reading CSV: {physio_data.shape}")
                eda = physio_data['eda']
                logging.info(f"EDA Series type: {type(eda)}, shape: {eda.shape}")

                eda_signals_neurokit, info_eda_neurokit = nk.eda_process(eda, sampling_rate=5000, method='neurokit')
                logging.info(f"Processed EDA shape: {eda_signals_neurokit.shape}")
                nk.eda_plot(eda_signals_neurokit)
                plot_filename = f"{base_filename}_processed_eda_plot.png"
                plt.savefig(plot_filename)
                plt.close()
                logging.info(f"Saved processed EDA plot to {plot_filename}")

                methods = ['smoothmedian', 'highpass', 'cvxEDA']
                for method in methods:
                    decomposed = nk.eda_phasic(eda, method=method, sampling_rate=5000)
                    logging.info(f"Decomposed EDA shape for {method}: {decomposed.shape}")

                    for component in ["EDA_Tonic", "EDA_Phasic"]:
                        plt.figure()
                        nk.signal_plot(decomposed[component])
                        component_plot_filename = f"{base_filename}_{method}_{component}_plot.png"
                        plt.savefig(component_plot_filename)
                        plt.close()
                        logging.info(f"Saved {component} plot for {method} to {component_plot_filename}")

                    peak_methods = ["kim2004", "neurokit", "nabian2018"]
                    for peak_method in peak_methods:
                        _, peaks = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=5000, method=peak_method)
                        plt.figure()
                        nk.events_plot(peaks["SCR_Peaks"], decomposed["EDA_Phasic"])
                        peaks_plot_filename = f"{base_filename}_{method}_peaks_{peak_method}_plot.png"
                        plt.savefig(peaks_plot_filename)
                        plt.close()
                        logging.info(f"Saved peaks plot for {method} using {peak_method} to {peaks_plot_filename}")

                    decomposed_filename = f"{base_filename}_{method}_decomposed.tsv"
                    with open(decomposed_filename, 'w') as f_out:
                        decomposed.to_csv(f_out, sep='\t', index=False)
                    with open(decomposed_filename, 'rb') as f_in:
                        with gzip.open(f"{decomposed_filename}.gz", 'wb') as f_out:
                            f_out.writelines(f_in)
                    os.remove(decomposed_filename)
                    logging.info(f"Saved decomposed data to {decomposed_filename}.gz")

                processed_signals_filename = f"{base_filename}_processed_eda.tsv"
                eda_signals_neurokit.to_csv(processed_signals_filename, sep='\t', index=False)
                with open(processed_signals_filename, 'rb') as f_in:
                    with gzip.open(f"{base_filename}_processed_eda.tsv.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(processed_signals_filename)
                logging.info(f"Saved processed signals to {processed_signals_filename}.gz")

        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
            traceback.print_exc()

    end_time = datetime.now()
    script_runtime = (end_time - start_time).total_seconds() / 60
    logging.info(f"Main function completed. Script runtime: {script_runtime} minutes")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()