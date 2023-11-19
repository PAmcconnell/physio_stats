# Script name: visualize_physio.py

import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main script logic
def main():
    logging.info("Starting main function")
    
    bids_root_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/dataset/')
    derivatives_dir = os.path.expanduser('~/Documents/MRI/LEARN/BIDS_test/derivatives/physio/rest/')
    logging.info(f"Checking derivatives directory: {derivatives_dir}")

    if not os.path.exists(derivatives_dir):
        logging.warning(f"Directory not found: {derivatives_dir}")
        # Handle the error or exit
        return # or exit the script

    layout = BIDSLayout(bids_root_dir, validate=True)

    # Derivatives
    ecg_files_derivatives = [os.path.join(derivatives_dir, f) for f in os.listdir(derivatives_dir) if f.endswith('run-01_physio.tsv.gz')]
    logging.info(f"Found {len(ecg_files_derivatives)} ECG files")
        
    all_ecg_data = []

    for file in ecg_files_derivatives:
        # Read ECG data from a gzipped tsv file
        with gzip.open(file, 'rt') as f:
            ecg_data = pd.read_csv(f, delimiter='\t')
    
        # Check if 'cardiac' column exists and plot the data
        if 'cardiac' in ecg_data.columns:
            all_ecg_data.append(ecg_data['cardiac'])
    
    if all_ecg_data:
        
        # Convert to a DataFrame for easier manipulation
        all_ecg_data_df = pd.concat(all_ecg_data, axis=1)

        # Compute mean and standard deviation
        mean_ecg = all_ecg_data_df.mean(axis=1)
        std_ecg = all_ecg_data_df.std(axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(mean_ecg, label='Group Mean ECG')
        plt.fill_between(mean_ecg.index, mean_ecg - std_ecg, mean_ecg + std_ecg, color='gray', alpha=0.5, label='Standard Deviation')
        plt.title('Group Mean ECG with Standard Deviation')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.show()
    else:
        logging.warning("No ECG data found.")
    
logging.info("Main function completed")

# If this script is run as the main script, execute the main function.
if __name__ == '__main__':
    
    # Call the main function.
    main()