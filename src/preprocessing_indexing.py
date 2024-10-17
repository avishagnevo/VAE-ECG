#!/usr/bin/env python
import os
import numpy as np
from scipy.io import loadmat
import torch
import csv, json
import pandas as pd
import h5py
from helper_code import *

'''
Key Features of preprocessing_indexing.py:
-Normalization: A function to normalize ECG signals based on the header data.
-Label Collection and Indexing: Functions to collect unique labels across multiple directories and create a label index stored in a JSON file.
-Multi-Hot Encoding: Functions that encode the labels as multi-hot vectors and store them in an HDF5 file for efficient access.
-HDF5 Management: Functions to create and manage the HDF5 file, storing the encodings and retrieving them by record ID or in batches.
-Main Routine: The main routine initializes everything, prepares the HDF5 file, and demonstrates how to access multi-hot encoded batches from the file.
'''


# Adjusted for optional signal normalization.
def normalize_signals(recording, header, leads):
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        # Avoid division by zero or by very small numbers
        if adc_gains[i] > 1e-5:  # Example threshold, adjust based on domain knowledge
            recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]
        else:
            print(f"Warning: ADC gain for lead {leads[i]} is too low ({adc_gains[i]}), skipping normalization.")
    return recording


# Collect all unique labels and create their index mapping
def collect_labels_and_create_index(base_dir):
    """
    Collect all unique labels from the .hea files in the base directory 
    and create a label-to-index mapping.
    """
    all_labels = set()

    # Find all header files across all subdirectories
    header_files, _ = find_challenge_files(base_dir)

    # Iterate through each header file and collect the labels
    for header_file in header_files:
        header = load_header(header_file)
        labels = get_labels(header)
        all_labels.update(labels)

    # Create a sorted label-to-index mapping
    label_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    return label_index


# Save the label index to a JSON file.
def save_labels_to_json(label_index, filepath):
    with open(filepath, 'w') as f:
        json.dump(label_index, f, indent=4)

# Load the label index from a JSON file.
def load_labels_from_json(filepath):
    with open(filepath, 'r') as f:
        label_index = json.load(f)
    return label_index

def get_label_index(label_index_path):
    # Collect labels and create an index only if the JSON does not already exist
    if not os.path.exists(label_index_path):
        label_index = collect_labels_and_create_index(base_dir = base_dir)
        save_labels_to_json(label_index, label_index_path)
    else:
        label_index = load_labels_from_json(label_index_path)
    return label_index


# Encode labels using the provided index from JSON.
def encode_labels(labels, label_index):
    encoded = np.zeros(len(label_index), dtype=int)
    for label in labels:
        if label in label_index:
            encoded[label_index[label]] = 1
    return encoded

# Create an HDF5 file to store multi-hot encodings
def create_hdf5_file(hdf5_filepath):
    with h5py.File(hdf5_filepath, 'w') as hdf5_file:
        print(f"HDF5 file created: {hdf5_filepath}")

# Add multi-hot encoding to HDF5 file for a specific record ID
def add_multi_hot_encoding_to_hdf5(hdf5_filepath, record_id, encoding):
    with h5py.File(hdf5_filepath, 'a') as hdf5_file:
        # Store the multi-hot encoding using the record_id as the key
        hdf5_file.create_dataset(record_id, data=encoding)
        print(f"Added encoding for {record_id} to HDF5 file")


def encode_and_store_labels_in_hdf5(hdf5_filepath, label_index, base_dir):
    """
    Encode labels for each record in the dataset and store them in an HDF5 file.
    """
    # Find all header files in the base directory
    header_files, _ = find_challenge_files(base_dir)

    # Loop through each header file, encode labels, and store in HDF5
    for header_file in header_files:
        # Get the record ID (e.g., JS00001 from JS00001.hea)
        record_id = os.path.splitext(os.path.basename(header_file))[0]

        # Load header and extract labels
        header = load_header(header_file)
        labels = get_labels(header)

        # Encode the labels using the label index
        encoded = encode_labels(labels, label_index).astype('float32')

        # Add the multi-hot encoding to the HDF5 file
        add_multi_hot_encoding_to_hdf5(hdf5_filepath, record_id, encoded)



def initialize_hdf5_file(hdf5_filepath, label_index, base_dir):
    if not os.path.exists(hdf5_filepath):
        # Create an HDF5 file to store the multi-hot encodings
        create_hdf5_file(hdf5_filepath)
        # Encode the labels and store them in the HDF5 file
        encode_and_store_labels_in_hdf5(hdf5_filepath, label_index, base_dir)    


# Retrieve multi-hot encoding for a batch of records
def get_batch_encodings_from_hdf5(hdf5_filepath, record_ids):
    encodings = []
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        for record_id in record_ids:
            if record_id in hdf5_file:
                encodings.append(hdf5_file[record_id][:])
            else:
                print(f"Record {record_id} not found in HDF5 file.")
    return np.array(encodings)


def main():
    #examine_single(file = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/g1/JS00001")
    #examine_dir(dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/g1")
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    label_index_path = "label_index.json"
    hdf5_filepath = "multi_hot_encodings.h5"

    # Collect labels and create an index only if the JSON does not already exist
    label_index = get_label_index(label_index_path)    

    # Initialize the HDF5 file with multi-hot encodings
    initialize_hdf5_file(hdf5_filepath, label_index, base_dir)

    # Example batch access for training:
    batch_record_ids = ['JS00001', 'JS00004', 'JS00007']
    batch_encodings = get_batch_encodings_from_hdf5(hdf5_filepath, batch_record_ids)
    print("Batch of multi-hot encodings:", batch_encodings)



if __name__ == "__main__":
    main()
