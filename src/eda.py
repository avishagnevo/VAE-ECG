import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helper_code import get_age, get_sex, visualize_same_encoding_close_age
from preprocessing_indexing import get_batch_encodings_from_hdf5, load_labels_from_json
from datasets import ECGVAEDataset
import numpy as np
import matplotlib.pyplot as plt


"""
Female Mean Age: 58.93066016836509, Median Age: 60.0
Male Mean Age: 60.99947671376243, Median Age: 63.0
"""

def calculate_age_statistics(dataset, sex_filter=None):
    """Calculate mean and median age for the given dataset filtered by sex."""
    ages = []
    for idx in range(len(dataset)):
        record = dataset[idx]
        age = record['age'].item()
        ages.append(age)

    ages = np.array(ages)
    mean_age = np.mean(ages)
    median_age = np.median(ages)
    
    return mean_age, median_age

def calculate_median_dx_vector(dataset):
    """Calculate the median entry-wise multi-hot encoding for the #Dx labels."""
    all_encodings = []
    
    for idx in range(len(dataset)):
        record = dataset[idx]
        multi_hot_encoding = record['multi_hot_encoding'] # Get multi-hot encoding
        all_encodings.append(multi_hot_encoding)
    
    # Convert list of multi-hot vectors to a numpy array
    all_encodings = np.array(all_encodings)
    
    # Calculate median across all samples (entry-wise median)
    median_encoding = np.median(all_encodings, axis=0)
    
    return median_encoding

def visualise_age_distribution(female_dataset, male_dataset):
    """Visualize the age distribution"""
    # Additional EDA: Age distribution histograms
    plt.figure(figsize=(10, 5))
    sns.histplot([record['age'].item() for record in female_dataset], label='Female', color='red', kde=True)
    sns.histplot([record['age'].item() for record in male_dataset], label='Male', color='blue', kde=True)
    plt.legend()
    plt.title("Age Distribution by Sex")
    plt.show()


def calculate_mean_dx_vector(dataset):
    """Calculate the mean entry-wise multi-hot encoding for the #Dx labels."""
    all_encodings = []
    
    for idx in range(len(dataset)):
        record = dataset[idx]
        multi_hot_encoding = record['multi_hot_encoding']  # Already a NumPy array
        all_encodings.append(multi_hot_encoding)
    
    # Convert list of multi-hot vectors to a numpy array
    all_encodings = np.array(all_encodings)
    
    # Calculate mean across all samples (entry-wise mean)
    mean_encoding = np.mean(all_encodings, axis=0)
    
    return mean_encoding

def visualize_mean_dx_vector(mean_vector_female, mean_vector_male, label_index):
    """Visualize the differences between the male and female #Dx mean multi-hot vectors."""
    # Filter labels where both male and female mean values are zero (optional)
    non_zero_indices = (mean_vector_female != 0) | (mean_vector_male != 0)
    
    # Filter the mean vectors and corresponding labels
    mean_vector_female = mean_vector_female[non_zero_indices]
    mean_vector_male = mean_vector_male[non_zero_indices]
    
    labels = sorted(label_index.keys(), key=lambda x: label_index[x])
    filtered_labels = np.array(labels)[non_zero_indices]  # Filter the labels
    
    # Visualize the filtered mean vectors
    indices = np.arange(len(mean_vector_female))
    
    plt.figure(figsize=(15, 6))
    
    plt.bar(indices - 0.2, mean_vector_female, width=0.4, label='Female', color='red', align='center')
    plt.bar(indices + 0.2, mean_vector_male, width=0.4, label='Male', color='blue', align='center')
    
    plt.xticks(indices, filtered_labels, rotation=90)
    plt.xlabel('Diagnosis #Dx')
    plt.ylabel('Mean Multi-Hot Encoding')
    plt.title('Comparison of Mean Multi-Hot Encoding Vectors by Sex')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_absolute_mean_diffs(mean_vector_female, mean_vector_male, label_index):
    """Visualize the absolute mean differences between male and female #Dx mean multi-hot vectors."""
    
    # Calculate absolute differences
    abs_diffs = np.abs(mean_vector_female - mean_vector_male)
    
    # Filter labels where the absolute differences are non-zero (optional)
    non_zero_indices = abs_diffs != 0
    
    # Filter the absolute differences and corresponding labels
    abs_diffs = abs_diffs[non_zero_indices]
    labels = sorted(label_index.keys(), key=lambda x: label_index[x])
    filtered_labels = np.array(labels)[non_zero_indices]  # Filter the labels
    
    # Visualize the filtered absolute differences
    indices = np.arange(len(abs_diffs))
    
    plt.figure(figsize=(15, 6))
    
    plt.bar(indices, abs_diffs, width=0.5, color='purple', align='center')
    
    plt.xticks(indices, filtered_labels, rotation=90)
    plt.xlabel('Diagnosis #Dx')
    plt.ylabel('Absolute Mean Difference')
    plt.title('Absolute Mean Differences in Multi-Hot Encoding Vectors by Sex')
    plt.tight_layout()
    plt.show()    

# Example of how to use the dataset and perform the EDA
if __name__ == "__main__":
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    hdf5_filepath = "multi_hot_encodings.h5"
    label_index_path = "label_index.json"
    scaling_params_path = "scaling_params.json"

    # Load the label index
    label_index = load_labels_from_json(label_index_path)

    # Create dataset instances for male and female ECG records
    female_dataset = ECGVAEDataset(base_dir, hdf5_filepath, label_index_path, sex_filter='Female', window = 512, scaling_params_path=scaling_params_path, model_type=None)
    male_dataset = ECGVAEDataset(base_dir, hdf5_filepath, label_index_path, sex_filter='Male', window = 512, scaling_params_path=scaling_params_path, model_type=None)

    visualize_same_encoding_close_age(male_dataset, female_dataset, age_threshold=5)

    # Calculate mean and median age for female and male datasets
    #mean_age_female, median_age_female = calculate_age_statistics(female_dataset)
    #mean_age_male, median_age_male = calculate_age_statistics(male_dataset)

    #print(f"Female Mean Age: {mean_age_female}, Median Age: {median_age_female}")
    #print(f"Male Mean Age: {mean_age_male}, Median Age: {median_age_male}")

    #visualise_age_distribution(female_dataset, male_dataset)

    # Calculate mean Dx vector for female and male datasets
    #mean_dx_female = calculate_mean_dx_vector(female_dataset)
    #mean_dx_male = calculate_mean_dx_vector(male_dataset)

    # Visualize the difference in Dx vectors between male and female groups
    #visualize_mean_dx_vector(mean_dx_female, mean_dx_male, label_index)

    # Visualize the absolute mean differences in Dx vectors between male and female groups
    #visualize_absolute_mean_diffs(mean_dx_female, mean_dx_male, label_index)



    
