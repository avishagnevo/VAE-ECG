#!/usr/bin/env python
import os
import numpy as np
from scipy.io import loadmat
import torch
import csv, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns



# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    leads = sorted(leads, key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)))
    return tuple(leads)

def find_challenge_files(data_directory):
    """
    Recursively find all header (.hea) and recording (.mat) files in the given data directory.
    """
    header_files = []
    recording_files = []
    
    # Traverse all subdirectories in the data directory
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            # Check if the file is a header (.hea) file
            if file.endswith('.hea'):
                header_file = os.path.join(root, file)
                recording_file = os.path.join(root, file.replace('.hea', '.mat'))
                
                # Check if corresponding recording file exists
                if os.path.isfile(recording_file):
                    header_files.append(header_file)
                    recording_files.append(recording_file)
    
    return header_files, recording_files

'''# Find header and recording files.
def find_challenge_files(data_directory):
    header_files = list()
    recording_files = list()
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            recording_file = os.path.join(data_directory, root + '.mat')
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files
'''

# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

#load recording file as an array
def load_recording(recording_file, header=None, leads=None, normalize=True):
    recording = loadmat(recording_file)['val']
    if header and leads:
        recording = choose_leads(recording, header, leads)
        if normalize:
            recording = normalize_signals(recording, header, leads)
    return torch.tensor(recording, dtype=torch.float32)        



def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = recording.shape[1]
    chosen_recording = np.zeros((num_leads, num_samples), dtype=recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording    

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    for line in header.split('\n'):
        if line.startswith('# Age'):
            try:
                return float(line.split(': ')[1])
            except ValueError:
                return float('nan')  # Return NaN if conversion fails

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('# Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_leads = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains

# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split('/')[0])
                except:
                    pass
        else:
            break
    return baselines

# Get labels from header.
def get_labels(header):
    labels = []
    for line in header.split('\n'):
        if line.startswith('# Dx'):
            try:
                entries = line.split(': ')[1].split(',')
                labels.extend([entry.strip() for entry in entries])
            except:
                pass
    return labels

# Save outputs from model.
def save_outputs(output_file, recording_id, classes, labels, probabilities):
    # Format the model outputs.
    recording_string = '#{}'.format(recording_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Save the model outputs.
    with open(output_file, 'w') as f:
        f.write(output_string)

# Load outputs from model.
def load_outputs(output_file):
    with open(output_file, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                recording_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(entry.strip() for entry in l.split(','))
            elif i==3:
                probabilities = tuple(float(entry) if is_finite_number(entry) else float('nan') for entry in l.split(','))
            else:
                break
    return recording_id, classes, labels, probabilities


# Load recording file as an array.
def load_recording_numpy(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads_numpy(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording


def examine_single(file):
    # Define paths to the ECG data files
    header_file = file + ".hea"
    recording_file = file + ".mat"

    # Load header and print content
    header = load_header(header_file)
    print("Header Content:\n", header)

    # Extract and print demographic data
    age = get_age(header)
    sex = get_sex(header)
    print("Demographic Data - Age:", age, ", Sex:", sex)

    # Get leads available in the recording
    leads = get_leads(header)
    print("Available Leads:", leads)

    # Load and normalize the recording data
    recording = load_recording(recording_file, header=header, leads=leads, normalize=True)
    print("Loaded and Normalized Recording Data (Shape):", recording.shape)
    print("Sample Data from every Lead:", recording[:, :10])  # Display first 10 samples from every lead

def examine_dir(dir):
    directory = dir
    header_files, _ = find_challenge_files(directory)

    # Iterate over all header files found
    for header_file in header_files:
        base_file_path = os.path.splitext(header_file)[0]  # Remove the .hea extension
        print("Examining file:", os.path.basename(header_file))
        examine_single(file=base_file_path)  # Pass the correct base file path
        break


# Function to visualize one record
def visualize_one_record(dataset, idx=0):
    """
    Visualizes the ECG data, multi-hot encodings, and age for one record in the dataset.
    """
    record = dataset[idx]
    
    ecg_data = record['ecg_data'].permute(1,0)  # ECG data (12 leads)
    multi_hot_encoding = record['multi_hot_encoding']  # Multi-hot encoding
    age = record['age']  # Age
    
    print(f"ECG Data Shape: {ecg_data.shape}")
    print(f"Multi-hot Encodings Shape: {multi_hot_encoding.shape}")
    print(f"Age: {age.item()}")
    
    # Plot the ECG data (all 12 leads)
    plt.figure(figsize=(15, 10))
    for lead_idx in range(ecg_data.shape[0]):  # Iterate through all 12 leads
        plt.subplot(6, 2, lead_idx + 1)
        plt.plot(ecg_data[lead_idx].numpy())
        plt.title(f"Lead {lead_idx + 1}")
    plt.tight_layout()
    plt.show()    


def visualize_same_encoding_close_age(male_dataset, female_dataset, age_threshold=5):
    # Collect records with the same multi_hot_encoding and close age
    close_age_records = []

    # Iterate over male records
    for male_record in male_dataset:
        male_encoding = torch.from_numpy(male_record['multi_hot_encoding'])
        male_age = male_record['age']

        # Iterate over female records
        for female_record in female_dataset:
            female_encoding = torch.from_numpy(female_record['multi_hot_encoding'])
            female_age = female_record['age']

            # Check if multi_hot_encodings are the same
            if torch.equal(male_encoding, female_encoding):
                age_difference = abs(male_age - female_age)
                # Check if ages are within the threshold
                if age_difference <= age_threshold:
                    close_age_records.append((male_record, female_record))
                    print(f"Found matching pair:")
                    print(f"Male Age: {male_age}, Female Age: {female_age}, Age Difference: {age_difference}")
                    print(f"Multi-hot Encoding: {male_encoding}")
                    break

        if close_age_records:
            break

    if not close_age_records:
        print(f"No records found with same multi-hot encoding and age difference <= {age_threshold}.")
        return

    # Visualize the selected records
    for male_record, female_record in close_age_records:
        ecg_data_male = male_record['ecg_data']
        ecg_data_female = female_record['ecg_data']

        plt.figure(figsize=(15, 20))  # Adjusted figure size
        for lead_idx in range(ecg_data_male.shape[0]):  # Iterate through all 12 leads
            plt.subplot(6, 2, lead_idx + 1)
            plt.plot(ecg_data_male[lead_idx].numpy(), color='blue', label='Male' if lead_idx == 0 else "")
            plt.plot(ecg_data_female[lead_idx].numpy(), color='red', label='Female' if lead_idx == 0 else "")
            plt.title(f"Lead {lead_idx + 1}")
            if lead_idx == 0:
                plt.legend(loc='upper right')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        break


def visualize_ecg_comparison(data_loader_wrapper, model ,training_sex = 'Female', data_type='val_same_sex', idx=0, device='cpu', model_type=None):
    """
    Visualizes the original and reconstructed ECG data for one record from a specified data type in the data loader.
    Compares the original 12-lead ECG signal with the reconstructed one.
    
    data_loader_wrapper: The DataLoaderWrapper instance that holds the train/val loaders.
    model: The trained model.
    data_type: Type of data to visualize ('train', 'val_same_sex', 'val_other_sex').
    idx: The index of the record in the dataset.
    device: The device on which the model is running ('cpu' or 'cuda').
    """
    # Set model to evaluation mode
    model.eval()

    if training_sex == 'Female':
            train_loader, val_loader_same_sex, _, val_loader_other_sex = data_loader_wrapper.get_dataloaders()
    elif training_sex == 'Male':
        _, val_loader_other_sex, train_loader, val_loader_same_sex = data_loader_wrapper.get_dataloaders()
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

    # Select the appropriate dataset based on the data_type
    if data_type == 'train':
        dataset = train_loader.dataset
    elif data_type == 'val_same_sex':
        dataset = val_loader_same_sex.dataset
    elif data_type == 'val_other_sex':
        dataset = val_loader_other_sex.dataset
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Choose from 'train', 'val_same_sex', 'val_other_sex'.")

    # Get the original record from the selected dataset
    record = dataset[idx]
    
    # Move data to the specified device
    ecg_data = record['ecg_data'].unsqueeze(0).to(device)  # ECG data (12 leads), add batch dimension
    multi_hot_encoding = torch.tensor(record['multi_hot_encoding']).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    age = record['age'].unsqueeze(0).to(device)  # Age
    ecg_seq_len = ecg_data.shape[1]  # Sequence length of the ECG signal

    # Print shapes for debugging
    #print(f"Original ECG Data Shape: {ecg_data.shape}")
    #print(f"Multi-hot Encodings Shape: {multi_hot_encoding.shape}")
    #print(f"Age: {age.item()}")
    
    # Forward pass to get the reconstructed ECG from the model
    with torch.no_grad():
        try:
            ecg_reconstructed, _, _ = model(ecg_data, age, multi_hot_encoding, ecg_seq_len)
        except:
            try:
                ecg_reconstructed = model(ecg_data, age, multi_hot_encoding)   
            except:
                ecg_reconstructed, _, _ = model(ecg_data.permute(0, 2, 1))    
 
    
    # Squeeze to remove batch dimension for visualization
    ecg_data = ecg_data.squeeze(0).permute(1, 0).cpu()
    ecg_reconstructed = ecg_reconstructed.squeeze(0).permute(1, 0).cpu()

    # Plot the original and reconstructed ECG data side by side
    plt.figure(figsize=(15, 12))
    
    for lead_idx in range(ecg_data.shape[0]):  # Iterate through all 12 leads
        
        lead_data = ecg_data[lead_idx].numpy()
        lead_mean = np.mean(lead_data)
        lead_min = lead_mean - 0.05
        lead_max = lead_mean + 0.05

        plt.subplot(6, 2, lead_idx + 1)
        plt.plot(lead_data, label='Original', color='blue')
        plt.plot(ecg_reconstructed[lead_idx].numpy(), label='Reconstructed', color='red') #linestyle='--',
        plt.ylim(lead_min, lead_max)  # Set y-axis limits
        plt.title(f"Lead {lead_idx + 1}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_training_history(history, save_path):
    """
    Visualizes the training history including the training loss, validation loss (same sex), and validation loss (other sex).

    Parameters:
    history (dict): The history dictionary returned by the Trainer, containing train_loss, val_loss_same_sex, and val_loss_other_sex.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training reconstruction loss
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, history['train_loss'], label='Train Reconstruction Loss', color='blue')
    
    # Plot validation loss for the same sex
    if any(val_loss is not None for val_loss in history['val_loss_same_sex']):
        plt.plot(epochs, history['val_loss_same_sex'], label='Validation Loss (Same Sex)', color='green')

    # Plot validation loss for the opposite sex
    if any(val_loss is not None for val_loss in history['val_loss_other_sex']):
        plt.plot(epochs, history['val_loss_other_sex'], label='Validation Loss (Opposite Sex)', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{save_path}\nTraining and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_latent_space_age(model, data_loader_wrapper, save_path, training_sex = 'Female', device='cpu', method='pca'):
    """
    Visualizes the latent space using PCA or t-SNE.
    """
    model.eval()
    latents = []
    labels = []

    if training_sex == 'Female':
        data_loader, _, _, _ = data_loader_wrapper.get_dataloaders()
    elif training_sex == 'Male':
        _, _, data_loader, _ = data_loader_wrapper.get_dataloaders()
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")
    
    with torch.no_grad():
        for batch in data_loader:
            ecg_data = batch['ecg_data'].to(device)
            age = batch['age'].to(device)
            multi_hot = batch['multi_hot_encoding'].to(device)
            
            try:
                _ ,mu, logvar = model.encode(ecg_data, age, multi_hot)
            except:
                mu, logvar = model.encode(ecg_data.permute(0, 2, 1))    
            latent = model.reparameterize(mu, logvar).cpu().numpy()
            latents.append(latent)
            labels.append(age.cpu().numpy())  # Using age as a label for visualization

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    
    reduced_latents = reducer.fit_transform(latents)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_latents[:, 0], reduced_latents[:, 1], c=labels, cmap='viridis')
    plt.colorbar(label='Age')
    plt.title(f"{save_path}\nLatent Space Visualization using {method.upper()}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()



def visualize_latent_space_categories(model, data_loader_wrapper, save_path, training_sex='Female', device='cpu', method='pca'):
    """
    Visualizes the latent space using PCA or t-SNE, where points are colored based on multi-hot encoded categories.
    Each category has a distinct color, and multi-label instances are plotted in each corresponding category.

    model: Trained ConditionalLSTMVAE model.
    data_loader_wrapper: DataLoaderWrapper instance to retrieve the dataset.
    save_path: Filepath for saving the visualization.
    training_sex: Which sex's data to visualize ('Female' or 'Male').
    device: Device to run inference ('cpu' or 'cuda').
    method: Dimensionality reduction method ('pca' or 'tsne').
    """
    model.eval()
    latents = []
    multi_hot_labels = []

    # Select the appropriate DataLoader for the training sex
    if training_sex == 'Female':
        data_loader, _, _, _ = data_loader_wrapper.get_dataloaders()
    elif training_sex == 'Male':
        _, _, data_loader, _ = data_loader_wrapper.get_dataloaders()
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")
    
    with torch.no_grad():
        for batch in data_loader:
            ecg_data = batch['ecg_data'].to(device)
            age = batch['age'].to(device)  # We can still use age if needed later
            multi_hot = batch['multi_hot_encoding'].to(device)
            
            # Encode and get latent space
            try:
                _, mu, logvar = model.encode(ecg_data, age, multi_hot)
            except:
                mu, logvar = model.encode(ecg_data.permute(0, 2, 1))    
            latent = model.reparameterize(mu, logvar).cpu().numpy()
            latents.append(latent)
            multi_hot_labels.append(multi_hot.cpu().numpy())  # Store multi-hot labels

    # Concatenate all latent representations and labels
    latents = np.concatenate(latents, axis=0)
    multi_hot_labels = np.concatenate(multi_hot_labels, axis=0)

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    reduced_latents = reducer.fit_transform(latents)

    # Prepare the plot
    plt.figure(figsize=(12, 8))

    # Overlay scatter plots for each active category
    num_categories = multi_hot_labels.shape[1]  # Number of categories in the multi-hot vector
    colors = sns.color_palette("husl", num_categories)  # Use a color palette for distinct categories

    # Plot each category in the multi-hot encoding
    for category_idx in range(num_categories):
        # Get points that have the current category as active (value = 1 in multi-hot encoding)
        active_points = multi_hot_labels[:, category_idx] == 1
        plt.scatter(
            reduced_latents[active_points, 0], 
            reduced_latents[active_points, 1], 
            label=f"Category {category_idx + 1}",
            color=colors[category_idx], 
            alpha=0.6, 
            #edgecolor='k'
        )

    # Set the title and labels
    plt.title(f"{save_path}\nLatent Space Visualization by Categories using {method.upper()}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Categories')
    plt.tight_layout()

    # Save or show the plot
    #plt.savefig(f"{save_path}_latent_space_{method}.png", bbox_inches='tight')
    plt.show()



def compare_reconstructions(model, data_loader_wrapper, training_sex = 'Female', device='cpu', num_samples=5):
    """
    Compare original ECG data with reconstructed data for a number of samples.
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

    if training_sex == 'Female':
        data_loader, _, _, _ = data_loader_wrapper.get_dataloaders()
    elif training_sex == 'Male':
        _, _, data_loader, _ = data_loader_wrapper.get_dataloaders()
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            ecg_data = batch['ecg_data'].to(device)
            age = batch['age'].to(device)
            multi_hot = batch['multi_hot_encoding'].to(device)
            ecg_seq_len = ecg_data.shape[1]

            # Forward pass
            ecg_reconstructed, mu, logvar = model(ecg_data, age, multi_hot, ecg_seq_len)

            # Plot original ECG
            axes[i, 0].plot(ecg_data[0, 0, :].cpu().numpy())  # Plot first lead of original ECG
            axes[i, 0].set_title(f"Original ECG (Sample {i+1})")

            # Plot reconstructed ECG
            axes[i, 1].plot(ecg_reconstructed[0, 0, :].cpu().numpy())  # Plot first lead of reconstructed ECG
            axes[i, 1].set_title(f"Reconstructed ECG (Sample {i+1})")

    plt.tight_layout()
    plt.show()
