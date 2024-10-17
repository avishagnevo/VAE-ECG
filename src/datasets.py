import torch
from torch.utils.data import Dataset
import os
import h5py
from helper_code import load_recording, load_header, get_age, get_sex, find_challenge_files, visualize_one_record, visualize_same_encoding_close_age
from preprocessing_indexing import get_batch_encodings_from_hdf5, load_labels_from_json
from torch.utils.data import DataLoader
import json
import numpy as np
import random
from scipy.signal.windows import hamming
from scipy.signal import convolve

'''
Carefully structure the data so that:
-ECG Data is kept separate from the multi-hot encoded labels and age data.
-Sex information is excluded when constructing the dataset.
-The dataset is efficient for CPU usage, considering that training will be computationally extensive.
-The design is flexible to allow for training on male and female ECG records separately, aligning with your research focus.
Dataset Structure:
-ECG Data: Time-series data from .mat files.
-Multi-Hot Encoded Labels: Retrieved from the HDF5 file.
-Age Data: Extracted from the .hea file.
-Sex Data: Excluded from the dataset but used for separating male and female records.
Dimentions are: ECG Data: torch.Size([batch, 12, 5000]), Multi-hot Encodings: torch.Size([batch, 54]), Age: torch.Size([batch])
Female have 4514 records in the dataset.
Male have 5733 records in the dataset.

Use a Combined Scaler for Both:
Since I want the male and female datasets to be scaled consistently across both groups, I apply a global scaler for both datasets.
Why: This approach ensures that the model's inputs are on the same scale when comparing the reconstruction error between sexes. 
This might make more sense if your hypothesis is focused on comparing the reconstruction behavior of the VAE across sexes, 
rather than the differences in the raw ECG data between sexes.

Dynamic Windowing in Dataset: allows the model to generalize better by varying the portions of the ECG sequences seen during training.
'''

import torch
import random
from scipy.signal import find_peaks

def detect_qrs_complex(ecg_data, lead_idx=0, height_threshold=0.5, distance=150):
    """
    Detect QRS complexes in the ECG data using peak detection.
    
    lead_idx: Which ECG lead to use for detecting QRS complexes (default: 0).
    height_threshold: Minimum height for a peak to be considered a QRS complex.
    distance: Minimum distance between detected peaks (in samples).
    """
    # Use the selected lead (typically Lead I or Lead II) to detect QRS complexes
    ecg_lead = ecg_data[lead_idx]
    
    # Detect peaks (QRS complexes) using find_peaks
    peaks, _ = find_peaks(ecg_lead, height=height_threshold, distance=distance)
    
    return peaks

def select_window_around_peak(ecg_data, peaks, window_size):
    """
    Select a window around the detected peaks. If multiple peaks are detected, select one peak randomly.
    
    ecg_data: The full ECG data.
    peaks: Indices of detected peaks.
    window_size: The size of the window to extract around the peak.
    """
    # Remove peaks that are outside the valid range
    peaks = [peak for peak in peaks if peak >= window_size // 2 and peak <= ecg_data.shape[1] - window_size // 2]

    # Randomly select a peak to center the window around
    if len(peaks) > 0:
        # Remove peaks that are outside the valid range
        selected_peak = random.choice(peaks)
    else:
        # If no peaks are detected, fall back to random window selection
        selected_peak = random.randint(0, ecg_data.shape[1] - window_size)
    
    # Determine start and end indices of the window
    start_idx = max(0, selected_peak - window_size // 2)
    end_idx = min(ecg_data.shape[1], start_idx + window_size)
    
    # Slice the window
    windowed_ecg_data = ecg_data[:, start_idx:end_idx]
    
    return windowed_ecg_data



# Function to calculate per-lead min-max values for ECG data and age
def calculate_scaling_params(ecg_directory, save_path="scaling_params.json"):
    age_min, age_max = float('inf'), float('-inf')
    ecg_min = np.full(12, float('inf'))  # Assuming 12 leads
    ecg_max = np.full(12, float('-inf'))

    header_files, _ = find_challenge_files(ecg_directory)
    for header_file in header_files:
        header = load_header(header_file)
        
        # Get age and update min-max for age
        age = get_age(header)
        if age is not None:
            age_min = min(age_min, age)
            age_max = max(age_max, age)

        # Get ECG data and update min-max for each lead
        record_id = os.path.splitext(os.path.basename(header_file))[0]
        recording_file = header_file.replace('.hea', '.mat')
        ecg_data = load_recording(recording_file, header=header)

        # Ensure ecg_data is in torch tensor format
        if isinstance(ecg_data, torch.Tensor):
            ecg_data_min = ecg_data.min(dim=1).values.numpy()  # Min for each lead
            ecg_data_max = ecg_data.max(dim=1).values.numpy()  # Max for each lead

            # Update min-max for each lead separately
            ecg_min = np.minimum(ecg_min, ecg_data_min)
            ecg_max = np.maximum(ecg_max, ecg_data_max)

    # Save the scaling parameters
    scaling_params = {
        "age_min": age_min,
        "age_max": age_max,
        "ecg_min": ecg_min.tolist(), 
        "ecg_max": ecg_max.tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(scaling_params, f, indent=4)

    print(f"Scaling parameters saved to {save_path}")

# Load the scaling parameters from the saved file
def load_scaling_params(scaling_params_path="scaling_params.json"):
    with open(scaling_params_path, 'r') as f:
        return json.load(f)

# Min-Max scaling function for each ECG lead
def min_max_scale_ecg(ecg_data, ecg_min, ecg_max):
    """Scales the ECG data for each lead using the per-lead min and max."""
    scaled_ecg = (ecg_data - torch.tensor(ecg_min).view(-1, 1)) / (torch.tensor(ecg_max).view(-1, 1) - torch.tensor(ecg_min).view(-1, 1))
    return scaled_ecg


def standardize_ecg(ecg_data):
    """Standardizes ECG data by normalizing using twice the standard deviation.
    Why It Helps:
Normalization reduces variability: Standardizing the signal helps normalize the data distribution across different records and leads. By dividing by twice the standard deviation, you are scaling the signal to have smaller variance, which is useful because ECG signals can vary widely in amplitude between patients or recordings.
Keeps signal within a common range: Standardization ensures that the values of the ECG signal stay in a well-defined range, preventing large amplitude differences that might hinder learning.
Prevents extreme values from dominating: Since you’re reducing the signal's range, it prevents peaks from dominating the model’s attention over time, making the learning process smoother and helping the model focus on more than just the largest peaks.
    """
    mean = torch.mean(ecg_data, dim=1, keepdim=True)
    std = torch.std(ecg_data, dim=1, keepdim=True)
    std[std == 0] = 1
    standardized_ecg = (ecg_data - mean) / (2 * std)
    return standardized_ecg

def power_transform_ecg(ecg_data, power=3):
    """Raises the ECG data to a given power to squash secondary local extrema.
    
    Why It Helps:

    Squashing secondary peaks: Raising the signal to the power of 3 or 4 helps reduce the impact of smaller secondary peaks while preserving the primary peaks, which are usually more significant in diagnosing ECG-related abnormalities. The non-linear transformation reduces the contribution of weaker signals, allowing the model to focus more on the dominant features (main peaks like QRS complexes).
    Prevents noise interference: In ECG data, small fluctuations or noise can create irrelevant peaks that interfere with the model’s learning. This transformation reduces the effect of minor variations and noise without fully removing them.
    Emphasizes key features: By squashing minor local extrema, you give more prominence to primary features, improving the model’s ability to focus on relevant signal components.
        """
    transformed_ecg = torch.sign(ecg_data) * torch.pow(torch.abs(ecg_data), power)
    return transformed_ecg

def apply_hamming_filter(ecg_data):
    """Applies a Hamming window to smooth the ECG signal.
    Smoothing reduces noise: ECG signals often contain high-frequency noise (e.g., from muscle activity or measurement devices). A Hamming filter smooths the signal by applying a low-pass filter, reducing sharp, high-frequency noise that could mislead the model during training.
    Preserving the waveform: Unlike more aggressive filters, the Hamming filter reduces noise without distorting the shape of the primary ECG components (like P, QRS, and T waves). This ensures that the important signal structure remains intact while reducing distractions from noise.
    Improving generalization: By smoothing the signal, the model is less likely to overfit to minor, noise-related features and more likely to generalize to the key structure of the ECG waveform.
    Most references to the Hamming window come from the signal processing literature, where it is used as one of many windowing functions for smoothing values. It is also known as an apodization (which means “removing the foot”, i.e. smoothing discontinuities at the beginning and end of the sampled signal) or tapering function. [R186]	Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra, Dover Publications, New York.
        """
    hamming_window = hamming(ecg_data.shape[1])  # Create Hamming window for the time axis (2000 samples)
    smoothed_ecg = []
    
    # Apply the Hamming filter along the time axis for each lead
    for lead in range(ecg_data.shape[0]):  # Loop over each lead
        smoothed_lead = torch.tensor(
            convolve(ecg_data[lead].cpu().numpy(), hamming_window, mode='same')
        )
        smoothed_ecg.append(smoothed_lead)
    
    # Stack the smoothed leads back into a tensor
    smoothed_ecg = torch.stack(smoothed_ecg)
    return smoothed_ecg

def log_transform_ecg(ecg_data):
    """Applies a log(1 + x) transformation to the ECG signal.
    Why It Helps:

    Re-scaling large values: Logarithmic transformations reduce the range of values, making large values smaller and more manageable. For ECG signals, this helps scale down large peaks (like R-waves in QRS complexes) while maintaining their prominence relative to other parts of the signal.
    Restoring relative amplitudes: Applying a logarithmic transformation helps restore the balance between small and large signal amplitudes. This helps the model see more uniform differences between peaks and valleys and avoid focusing too much on extreme values.
    Preventing overfitting to large peaks: Without the log transformation, large peaks could dominate the loss function during training, leading the model to overfit these features while ignoring smaller but potentially important variations in the ECG signal.

    """
    log_ecg = torch.log1p(ecg_data)
    return log_ecg



class ECGVAEDataset(Dataset):
    def __init__(self, ecg_directory, hdf5_filepath, label_index_path, sex_filter=None, window = 2000, scaling_params_path="scaling_params.json", model_type=None):
        """
        ecg_directory: Directory containing the ECG files (.mat and .hea)
        hdf5_filepath: Path to the HDF5 file containing multi-hot encoded labels
        label_index_path: Path t0 a dictionary containing label-to-index mapping
        sex_filter: 'Male', 'Female', or None to filter records by sex
        scaling_params_path: Path to the file containing the GLOBAL min-max scaling parameters
        """
        self.header_files, self.ecg_files = find_challenge_files(ecg_directory)
        self.hdf5_filepath = hdf5_filepath
        self.label_index = load_labels_from_json(label_index_path)
        self.sex_filter = sex_filter
        self.window = window
        self.model_type = model_type

        # Check if scaling_params.json exists, if not, calculate and save scaling params
        if not os.path.exists(scaling_params_path):
            print(f"{scaling_params_path} not found. Calculating scaling parameters...")
            calculate_scaling_params(ecg_directory, scaling_params_path)
        
        self.scaling_params = load_scaling_params(scaling_params_path)

        # Filter ECG records based on sex (if filter is provided)
        self.filtered_files = self.filter_by_sex()


    def filter_by_sex(self):
        """Filters records by sex if sex_filter is provided ('Male' or 'Female')."""
        filtered_files = []
        for header_file in self.header_files:
            header = load_header(header_file)
            sex = get_sex(header)
            if (self.sex_filter is None) or (sex == self.sex_filter):
                filtered_files.append(header_file)
        return filtered_files

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        # Get header and recording file paths
        header_file = self.filtered_files[idx]
        record_id = os.path.splitext(os.path.basename(header_file))[0]
        recording_file = header_file.replace('.hea', '.mat')

        # Load ECG data
        ecg_data = load_recording(recording_file, header=load_header(header_file))

        # Dynamic window selection
        ecg_length = ecg_data.shape[1]  # Get the full length of the ECG data
        if ecg_length > self.window:
            # Detect QRS complexes or important features in the ECG data
            peaks = detect_qrs_complex(ecg_data, lead_idx=0, height_threshold=0.5, distance=150)
        
            # Select a window around the detected QRS peak or other important features
            ecg_data = select_window_around_peak(ecg_data, peaks, self.window)
        
        ''' '''
        if self.model_type == None:
            ecg_data = self.scale_ecg_data(ecg_data, power = 3) # .half() to convert ECG data to float16 to save memory and improve speed
        
        ecg_data = self.scale_ecg_data_min_max(ecg_data).permute(1,0) # .half() to convert ECG data to float16 to save memory and improve speed
        
        if self.model_type == 'ConvVAE':
            ecg_data = ecg_data[:, 0].unsqueeze(1) # Use only the first lead for ConvVAE 
        
        
        # Retrieve the multi-hot encoding from HDF5
        multi_hot_encoding = self.get_multi_hot_encoding(record_id)

        # Get and scale age
        age = get_age(load_header(header_file))
        #age = self.scale_age(age)

        age = torch.tensor(age, dtype=torch.float32)

        return {
            'ecg_data': ecg_data,              # Scaled ECG time-series data 
            'multi_hot_encoding': multi_hot_encoding,  # Multi-hot encoded labels
            'age': age  
        }


    def scale_ecg_data(self, ecg_data, power=3):
        """ Scales and processes ECG data with custom transformations.
        1. Standardizes by twice the standard deviation.
        2. Raises the signal to the power of 'power'.
        3. Applies a Hamming filter to smooth the signal.
        4. Applies log(1 + x) transformation.

        Final Explanation for Each Step:
        Standardization reduces the variability of the data, ensuring the model can learn consistently across different patients.
        Power transformation suppresses minor local peaks and highlights the major, diagnostically relevant peaks.
        Smoothing through the Hamming filter reduces noise, making the signal cleaner without losing critical information.
        Logarithmic transformation adjusts the scale of the signal, ensuring that both large and small features are equally considered during training.
        Why This Approach Can Help with the Problem:
        These transformations target key problems with ECG signals, such as noise, large amplitude variations, and the prominence of irrelevant minor peaks. Together, they ensure that the ECG signal fed to the model is both smooth and representative of the main features (like P, QRS, and T waves), which should help the model learn the correct patterns and improve performance.
        """
        # Step 1: Standardization by twice the standard deviation
        ecg = standardize_ecg(ecg_data)
        
        # Step 2: Raise to the power (e.g., cubic or quartic transformation)
        ecg = power_transform_ecg(ecg, power=power)
        
        # Step 3: Apply Hamming filter
        #ecg = apply_hamming_filter(ecg)
        
        # Step 4: Logarithmic transformation
        #ecg = log_transform_ecg(ecg)
        
        return ecg

    def scale_ecg_data_min_max(self, ecg_data):
        """Scales ECG data for each lead using the global min-max values."""
        ecg_min = self.scaling_params['ecg_min']
        ecg_max = self.scaling_params['ecg_max']
        return min_max_scale_ecg(ecg_data, ecg_min, ecg_max)

    def scale_age(self, age):
        """Scales age using the global min-max values."""
        age_min = self.scaling_params['age_min']
        age_max = self.scaling_params['age_max']
        return (age - age_min) / (age_max - age_min)

    def get_multi_hot_encoding(self, record_id):
        """Retrieve the multi-hot encoding from the HDF5 file for the given record ID."""
        return get_batch_encodings_from_hdf5(self.hdf5_filepath, [record_id])[0]


# Example of how to use the dataset
if __name__ == "__main__":
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    hdf5_filepath = "multi_hot_encodings.h5"
    label_index_path = "label_index.json"
    scaling_params_path = "scaling_params.json"

    # Create dataset instance for female ECG records
    dataset = ECGVAEDataset(base_dir, hdf5_filepath, label_index_path, sex_filter=None, window = 512, scaling_params_path=scaling_params_path, model_type='ConvVAE')

    # Example DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Print the dataset length
    print(f"Dataset length: {len(dataset)}")

    # Visualize one record (first record in the dataset)
    for idx in range(1):
        visualize_one_record(dataset, idx=idx)
    

    
    '''
    for batch in dataloader:
        ecg_data = batch['ecg_data']
        multi_hot_encoding = batch['multi_hot_encoding']
        age = batch['age']
        print(f"ECG Data: {ecg_data.shape}, Multi-hot Encodings: {multi_hot_encoding.shape}, Age: {age.shape}")
    '''
