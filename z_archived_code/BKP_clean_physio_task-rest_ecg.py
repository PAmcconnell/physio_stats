# Script name: clean_physio_task-restecg.py

import os
import gzip
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import logging
from bids import BIDSLayout

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main script logic
def main():
    """
    Main function to clean ecg data from a BIDS dataset.
    """
    logging.info("Starting main function")

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

    # Initialize BIDS Layout
    layout = BIDSLayout(bids_root_dir, validate=True)
    
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
    participant_id = participants_df['participant_id'].iloc[0]
    logging.info(f"Testing for participant: {participant_id}")

    # Process the first run for the selected participant
    run_number = 1
    physio_files = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f'{participant_id}_ses-1_task-rest_run-{run_number:02d}_physio.tsv.gz' in f]
    for file in physio_files:
        try:
            with gzip.open(file, 'rt') as f:
                physio_data = pd.read_csv(f, delimiter='\t')
                ecg = physio_data['cardiac']
                
                # Clean ECG signal using Pan-Tompkins method
                ecg_cleaned_pan_tompkins = nk.ecg_clean(ecg, sampling_rate=5000, method="pantompkins1985")

                # clean ecg using other available methods
                ecg_cleaned_neurokit = nk.ecg_clean(ecg, sampling_rate=5000, method="neurokit")
                ecg_cleaned_biosppy = nk.ecg_clean(ecg, sampling_rate=5000, method="biosppy")
                ecg_cleaned_hamilton = nk.ecg_clean(ecg, sampling_rate=5000, method="hamilton2002")
                ecg_cleaned_elgendi = nk.ecg_clean(ecg, sampling_rate=5000, method="elgendi2010")
                ecg_cleaned_engzeemod = nk.ecg_clean(ecg, sampling_rate=5000, method="engzeemod2012")

                # process the ecg signals to find r-peaks
                signals, info_raw = nk.ecg_process(ecg, sampling_rate=5000)
                signals_pan_tompkins, info_pan_tompkins = nk.ecg_process(ecg_cleaned_pan_tompkins, sampling_rate=5000)
                signals_neurokit, info_neurokit = nk.ecg_process(ecg_cleaned_neurokit, sampling_rate=5000)
                signals_biosppy, info_biosppy = nk.ecg_process(ecg_cleaned_biosppy, sampling_rate=5000)
                signals_hamilton, info_hamilton = nk.ecg_process(ecg_cleaned_hamilton, sampling_rate=5000)
                signals_elgendi, info_elgendi = nk.ecg_process(ecg_cleaned_elgendi, sampling_rate=5000)
                signals_engzeemod, info_engzeemod = nk.ecg_process(ecg_cleaned_engzeemod, sampling_rate=5000)

                # assess the quality of the raw and cleaned ecg signals
                quality_raw = nk.ecg_quality(ecg, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_pan_tompkins = nk.ecg_quality(ecg_cleaned_pan_tompkins, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_neurokit = nk.ecg_quality(ecg_cleaned_neurokit, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_biosppy = nk.ecg_quality(ecg_cleaned_biosppy, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_hamilton = nk.ecg_quality(ecg_cleaned_hamilton, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_elgendi = nk.ecg_quality(ecg_cleaned_elgendi, sampling_rate=5000, method='zhao2018', approach='fuzzy')
                quality_cleaned_engzeemod = nk.ecg_quality(ecg_cleaned_engzeemod, sampling_rate=5000, method='zhao2018', approach='fuzzy')

                # log the quality scores
                logging.info(f"quality of raw ecg (zhao2018-fuzzy): {quality_raw}")
                logging.info(f"quality of pan-tompkins cleaned ecg (zhao2018-fuzzy): {quality_cleaned_pan_tompkins}")
                logging.info(f"quality of neurokit cleaned ecg (zhao2018-fuzzy): {quality_cleaned_neurokit}")
                logging.info(f"quality of biosppy cleaned ecg (zhao2018-fuzzy): {quality_cleaned_biosppy}")
                logging.info(f"quality of hamilton cleaned ecg (zhao2018-fuzzy): {quality_cleaned_hamilton}")
                logging.info(f"quality of elgendi cleaned ecg (zhao2018-fuzzy): {quality_cleaned_elgendi}")
                logging.info(f"quality of engzeemod cleaned ecg (zhao2018-fuzzy): {quality_cleaned_engzeemod}")

                # create subplots with linked x-axes
                fig, axs = plt.subplots(7, 1, figsize=(15, 10), sharex=True)

                # plot raw ecg
                axs[0].plot(ecg)
                axs[0].scatter(info_raw['ECG_R_Peaks'], ecg[info_raw['ECG_R_Peaks']], color='red')
                axs[0].set_title("raw ecg with r-peaks")

                # plot pan-tompkins cleaned ecg
                axs[1].plot(ecg_cleaned_pan_tompkins)
                axs[1].scatter(info_pan_tompkins['ECG_R_Peaks'], ecg_cleaned_pan_tompkins[info_pan_tompkins['ECG_R_Peaks']], color='red')
                axs[1].set_title("pan-tompkins cleaned ecg with r-peaks")

                # plot neurokit cleaned ecg
                axs[2].plot(ecg_cleaned_neurokit)
                axs[2].scatter(info_neurokit['ECG_R_Peaks'], ecg_cleaned_neurokit[info_neurokit['ECG_R_Peaks']], color='red')
                axs[2].set_title("neurokit cleaned ecg with r-peaks")

                # plot biosppy cleaned ecg
                axs[3].plot(ecg_cleaned_biosppy)
                axs[3].scatter(info_biosppy['ECG_R_Peaks'], ecg_cleaned_biosppy[info_biosppy['ECG_R_Peaks']], color='red')
                axs[3].set_title("biosppy cleaned ecg with r-peaks")
                
                # plot hamilton cleaned ecg
                axs[4].plot(ecg_cleaned_hamilton)
                axs[4].scatter(info_hamilton['ECG_R_Peaks'], ecg_cleaned_hamilton[info_hamilton['ECG_R_Peaks']], color='red')
                axs[4].set_title("hamilton cleaned ecg with r-peaks")

                # plot elgendi cleaned ecg
                axs[5].plot(ecg_cleaned_elgendi)
                axs[5].scatter(info_elgendi['ECG_R_Peaks'], ecg_cleaned_elgendi[info_elgendi['ECG_R_Peaks']], color='red')
                axs[5].set_title("elgendi cleaned ecg with r-peaks")
                
                # plot engzeemod cleaned ecg
                axs[6].plot(ecg_cleaned_engzeemod)
                axs[6].scatter(info_engzeemod['ECG_R_Peaks'], ecg_cleaned_engzeemod[info_engzeemod['ECG_R_Peaks']], color='red')
                axs[6].set_title("engzeemod cleaned ecg with r-peaks")

                plt.show()
                # further analysis and plotting can be added here

        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")

    logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()