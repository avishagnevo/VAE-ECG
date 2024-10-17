import torch
from torch.utils.data import DataLoader, random_split
from datasets import ECGVAEDataset  # Assuming ECGVAEDataset is defined in datasets.py
from preprocessing_indexing import load_labels_from_json

class DataLoaderWrapper:
    def __init__(self, base_dir, hdf5_filepath, label_index_path, batch_size=32, window_size=2000, split_ratio=0.8, model_type=None):
        """
        Initializes the DataLoader wrapper to create train and validation data loaders.

        base_dir: Directory where the ECG files are located.
        hdf5_filepath: Path to the HDF5 file containing the multi-hot encodings.
        label_index_path: Path to the label index JSON file.
        batch_size: Batch size for the DataLoader.
        window_size: Size of the ECG window (e.g., 2000).
        split_ratio: The ratio to split the dataset (e.g., 0.9 for 90% train, 10% validation).
        """
        self.base_dir = base_dir
        self.hdf5_filepath = hdf5_filepath
        self.batch_size = batch_size
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.model_type = model_type

        # Load the label index
        self.label_index_path = label_index_path

    def split_dataset(self, dataset):
        """Splits the dataset into train and validation sets according to split_ratio."""
        dataset_len = len(dataset)
        train_len = int(self.split_ratio * dataset_len)
        val_len = dataset_len - train_len
        return random_split(dataset, [train_len, val_len])

    def get_dataloaders(self):
        """Creates and returns DataLoader objects for training and validation."""
        # Create datasets for female and male
        dataset_female = ECGVAEDataset(self.base_dir, self.hdf5_filepath, self.label_index_path, sex_filter='Female', window=self.window_size, model_type=self.model_type)
        dataset_male = ECGVAEDataset(self.base_dir, self.hdf5_filepath, self.label_index_path, sex_filter='Male', window=self.window_size, model_type=self.model_type)

        # Split the datasets into train and validation sets
        train_dataset_female, val_dataset_female = self.split_dataset(dataset_female)
        train_dataset_male, val_dataset_male = self.split_dataset(dataset_male)

        # Create DataLoaders for the datasets
        train_loader_female = DataLoader(train_dataset_female, batch_size=self.batch_size, shuffle=True)
        val_loader_female = DataLoader(val_dataset_female, batch_size=self.batch_size, shuffle=False)

        train_loader_male = DataLoader(train_dataset_male, batch_size=self.batch_size, shuffle=True)
        val_loader_male = DataLoader(val_dataset_male, batch_size=self.batch_size, shuffle=False)

        return train_loader_female, val_loader_female, train_loader_male, val_loader_male

if __name__ == "__main__":
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    hdf5_filepath = "multi_hot_encodings.h5"
    label_index_path = "label_index.json"

    # Create the DataLoaderWrapper
    data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size=32, window_size=2000)

    # Get the DataLoaders
    train_loader_female, val_loader_female, train_loader_male, val_loader_male = data_loader_wrapper.get_dataloaders()
    
