## Name: edf_to_csv.py
##
## About: This file is used to extract only the motor imagery files from the PhysioNet database and convert the data
##        from a .edf file to a .csv file. The data is extracted into a eeg data file and a labels file. The original
#         .edf files are removed to save space upon conclusion. An additional channels.csv is created for reference.

import numpy as np
import mne
import os

cwd = os.getcwd()                                               # Gets the current working directory
my_dir = cwd + '/data'                                          # Set the data directory path

all_data = []                                                   # Location to save all data objects

for fname in os.listdir(my_dir):                                # Parse through the data directory
    if 'S1' in fname or 'S0' in fname:                          # For any subject id folder
        f_dir = os.path.join(my_dir, fname)                     # Set the subdirectory as the current filepath
        for f in os.listdir(f_dir):                             # Look through all files in subdirectory
            f_path = os.path.join(f_dir, f)                     # Set file as the current path
            if 'event' not in f_path and '.csv' not in f_path:  # Exclude event files

                eeg_f_name = f[0:7] + '.csv'
                eeg_out = os.path.join(f_dir, eeg_f_name)       # EEG data save location

                data = mne.io.read_raw_edf(f_path,
                                           verbose=False,
                                           preload=True)        # Load in the data

                data_ = data._data                              # extract the raw eeg data
                np.savetxt(eeg_out,
                           data_,
                           delimiter=", ",
                           fmt='% s')                           # Save the data to a csv file

                labels = np.zeros(data_.shape[1])               # Make a list to label every sample
                an = mne.events_from_annotations(data)          # Extract the annotations
                an = an[0]                                      # Only keep the label portion

                for a in an:
                    lab = a[2]
                    labels[a[0]:] = lab

                labels_name = f[0:7] + '_labels.csv'
                labels_out = os.path.join(f_dir, labels_name)
                np.savetxt(labels_out,
                           labels,
                           delimiter=", ",
                           fmt='% s')                           # Save the labels to a csv file

            if '.csv' not in f:
                os.remove(f_path)


channels = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'Cp5', 'Cp3',
            'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5',
            'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7',
            'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2',
            'Iz']

np.savetxt('channels.csv',
           channels,
           delimiter=", ",
           fmt='% s')  # Save the labels to a csv file


