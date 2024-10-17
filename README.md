# VAE-ECG
A VAE-based framework for reconstructing ECG signals and conducting statistical hypothesis testing. Detailed methodology and results can be found in the report.pdf.

To download the dataset used for training, run the following command in the project directory:
"""
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/
"""

## File Overview (src directory)
### .py
- **dataloaders.py**: Handles the loading and batching of ECG data for model training.
- **datasets.py**: Defines custom datasets and prepares the data structure for input to the models.
- **eda.py**: Contains exploratory data analysis scripts for visualizing and understanding the dataset.
- **helper_code.py**: Utility functions to support various tasks throughout the project.
- **hypoth_testing.py**: Performs hypothesis testing, including statistical tests like Mann-Whitney U.
- **models.py**: Defines the VAE and other neural network architectures used for ECG reconstruction.
- **preprocessing_indexing.py**: Preprocesses and indexes the ECG data before training.
- **trainer.py**: Manages the training loop, validation, and testing phases for the models.


### constants
- **label_index.json**: Maps labels to their corresponding index in the dataset.
- **labels.csv**: Contains the diagnosis labels for each ECG record.
- **multi_hot_encodings.h5**: Stores the multi-hot encoded diagnoses used for training.
- **scaling_params.json**: Stores global min-max values for scaling the ECG data and age.
