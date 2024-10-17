import numpy as np
from scipy.stats import ranksums, mannwhitneyu
import matplotlib.pyplot as plt
import torch
from models import ConvVAE, CRVAE, CRAE
from trainer import vae_loss_function
from dataloaders import DataLoaderWrapper

'''
The statistical hypothesis testing is added for the reconstruction errors between the In-Distribution (ID) and Out-of-Distribution (OOD) groups using the permutation test.
The following aspects of the code are covered:
Permutation Test Setup: The PermutationTest class and the hypothesis_testing function will perform a permutation test to evaluate the stochastic differences between the two groups of reconstruction errors.
Model Loading and Error Calculation: The functions for loading the model, computing reconstruction errors, and running the hypothesis test are clearly defined.
Visualization: The null distribution of the permutation test is visualized using a histogram, with a vertical line representing the observed test statistic and another line for the 0.95 quantile to assist in hypothesis testing.
Testing Framework: The pipeline will calculate errors from the trained model and run the permutation test to verify whether the null hypothesis should be rejected.
'''


class PermutationTest:
    def __init__(self, errors_id, errors_ood, training_sex = 'Female', num_permutations=1000, type='wilcoxon'):
        """
        errors_id: List or array of reconstruction errors from the In-Distribution (ID) group.
        errors_ood: List or array of reconstruction errors from the Out-of-Distribution (OOD) group.
        num_permutations: Number of permutations to run for the test.
        """
        self.errors_id = np.array(errors_id)
        self.errors_ood = np.array(errors_ood)
        self.num_permutations = num_permutations
        self.observed_stat = None
        self.p_value = None
        self.null_distribution = []
        self.type = 'wilcoxon'

    def compute_test_statistic(self, errors_id, errors_ood):
        """
        Computes the test statistic for the current group of reconstruction errors.
        Uses the Wilcoxon Rank-Sum test as the test statistic.
        """
        if self.type == 'wilcoxon':
            stat, _ = ranksums(errors_id, errors_ood)
        else :
            stat, _ = mannwhitneyu(errors_id, errors_ood)    
        return stat

    def run_permutation_test(self):
        """
        Runs the permutation test to estimate the p-value.
        """
        # Compute the observed test statistic
        self.observed_stat = self.compute_test_statistic(self.errors_id, self.errors_ood)

        # Combine both groups
        combined_errors = np.concatenate([self.errors_id, self.errors_ood])
        n_id = len(self.errors_id)

        # Permutation loop
        for _ in range(self.num_permutations):
            np.random.shuffle(combined_errors)
            permuted_id = combined_errors[:n_id]
            permuted_ood = combined_errors[n_id:]
            permuted_stat = self.compute_test_statistic(permuted_id, permuted_ood)
            self.null_distribution.append(permuted_stat)

        self.null_distribution = np.array(self.null_distribution)

        # Calculate p-value
        #print('self.null_distribution:', self.null_distribution)
        #print('self.observed_stat:', self.observed_stat)
        self.p_value = np.mean(self.null_distribution >= self.observed_stat)

    def visualize_permutation_distribution(self):
        """
        Visualizes the null distribution from the permutation test with the observed statistic
        and the 0.95 quantile for a one-sided test.
        """
        # Calculate the 0.95 quantile of the null distribution
        quantile_5 = np.percentile(self.null_distribution, 5)

        # Plot the null distribution
        plt.hist(self.null_distribution, bins=40, alpha=0.7, label='Null Distribution', color='yellow')

        # Add a vertical line for the observed statistic
        plt.axvline(self.observed_stat, color='green', linestyle='dashed', linewidth=2, label='Observed Statistic')

        # Add a vertical line for the 0.5 quantile
        plt.axvline(quantile_5, color='black', linestyle='dotted', linewidth=2, label='0.95CI RR upper bound')

        # Add labels and titles
        plt.title('Permutation Test Null Distribution')
        plt.xlabel('Test Statistic')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def hypothesis_testing(errors_id, errors_ood, num_permutations=1000, test_type='wilcoxon'):
    """
    Function to run the permutation test and return the p-value.
    errors_id: List or array of reconstruction errors from the In-Distribution (ID) group.
    errors_ood: List or array of reconstruction errors from the Out-of-Distribution (OOD) group.
    num_permutations: Number of permutations to run for the test.
    """
    permutation_test = PermutationTest(errors_id, errors_ood, num_permutations, type= test_type)    
    permutation_test.run_permutation_test()

    # Visualize the results
    permutation_test.visualize_permutation_distribution()
    return permutation_test.p_value


def load_trained_model(model_path, data_loader_wrapper,hidden_dim=None, hidden_dims=[16, 32, 48], latent_dim=None, embedding_dim = None, device='cpu', model_type='ConvVAE'):
    """
    Function to load the trained model and return it.
    """
    if model_type == 'CRVAE':
        model = CRVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
        model.initialize_model_dims(data_loader_wrapper.get_dataloaders()[0])
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_type == 'CRAE':
        model = CRAE(data_loader_wrapper.get_dataloaders()[0], embedding_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))    
    else:
        seq_len = data_loader_wrapper.get_dataloaders()[0].dataset[0]['ecg_data'].shape[0]
        model = ConvVAE(seq_len, latent_dim, hidden_dims, kernel_size=9, dropout=0.1)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()    
    return model


def get_reconstruction_errors(model, data_loader, device='cpu'):
    """
    Function to compute reconstruction errors for each record in the data loader.
    """
    errors = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            ecg_data = batch['ecg_data'].to(device)
            age = batch['age'].to(device)
            multi_hot = batch['multi_hot_encoding'].to(device)
            ecg_seq_len = ecg_data.shape[1]

            # Forward pass to get reconstructed ECG
            try:
                ecg_reconstructed, mu, logvar = model(ecg_data, age, multi_hot_encoding, ecg_seq_len)
            except:
                try:
                    ecg_reconstructed = model(ecg_data, age, multi_hot_encoding)   
                except:
                    ecg_reconstructed, mu, logvar = model(ecg_data.permute(0, 2, 1))    
 
    
            # Compute reconstruction loss (using MAE/L1 loss)
            recon_loss, _ = vae_loss_function(ecg_reconstructed, ecg_data, mu, logvar)
            # Store reconstruction error for each record
            errors.append(recon_loss.cpu().numpy())

    return errors



def visualize_error_distributions(errors_id, errors_ood, training_sex='Female'):
    """
    Visualizes the distributions of reconstruction errors for In-Distribution (ID) and Out-of-Distribution (OOD) groups.

    errors_id: List or array of reconstruction errors from the In-Distribution (ID) group.
    errors_ood: List or array of reconstruction errors from the Out-of-Distribution (OOD) group.
    """
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot PDF on the first subplot
    ax = axs[0]
    if training_sex == 'Female':
        ax.hist(errors_id, bins=40, alpha=0.6, color='red', label='In-Distribution (Female)', density=True)
        ax.hist(errors_ood, bins=40, alpha=0.6, color='blue', label='Out-of-Distribution (Male)', density=True)
    elif training_sex == 'Male':
        ax.hist(errors_id, bins=40, alpha=0.6, color='blue', label='In-Distribution (Male)', density=True)
        ax.hist(errors_ood, bins=40, alpha=0.6, color='red', label='Out-of-Distribution (Female)', density=True)
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

    ax.set_title('Reconstruction Error PDF (ID vs. OOD)')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.legend()

    # Plot CDF on the second subplot
    ax = axs[1]
    if training_sex == 'Female':
        ax.hist(errors_id, bins=40, alpha=0.6, color='red', label='In-Distribution (Female)',
                density=True, cumulative=True, histtype='step')
        ax.hist(errors_ood, bins=40, alpha=0.6, color='blue', label='Out-of-Distribution (Male)',
                density=True, cumulative=True, histtype='step')
    elif training_sex == 'Male':
        ax.hist(errors_id, bins=40, alpha=0.6, color='blue', label='In-Distribution (Male)',
                density=True, cumulative=True, histtype='step')
        ax.hist(errors_ood, bins=40, alpha=0.6, color='red', label='Out-of-Distribution (Female)',
                density=True, cumulative=True, histtype='step')
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

    ax.set_title('Reconstruction Error CDF (ID vs. OOD)')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Cumulative Density')
    ax.legend()

    plt.tight_layout()
    plt.show()



def run_hypothesis_test(model_path, data_loader_wrapper,training_sex = 'Female',hidden_dim=None, hidden_dims=[16, 32, 48], latent_dim=None, embedding_dim = None, device='cpu', model_type='ConvVAE', num_permutations=1000, test_type='wilcoxon'):
    """
    Main pipeline to load the model, compute reconstruction errors, and run the permutation test.
    """
    # Load trained model
    model = load_trained_model( model_path, data_loader_wrapper,hidden_dim, hidden_dims, latent_dim, embedding_dim, device, model_type)

    # Get data loaders for same sex (ID) and opposite sex (OOD)
    if training_sex == 'Female':
        _ , val_loader_same_sex, _, val_loader_other_sex = data_loader_wrapper.get_dataloaders()
    elif training_sex == 'Male':
        _, val_loader_other_sex, _ , val_loader_same_sex = data_loader_wrapper.get_dataloaders()
    else:
        raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

    #_, val_loader_same_sex, _, val_loader_other_sex = data_loader_wrapper.get_dataloaders()

    # Compute reconstruction errors for ID (same sex) and OOD (other sex)
    errors_id = get_reconstruction_errors(model, val_loader_same_sex, device)
    errors_ood = get_reconstruction_errors(model, val_loader_other_sex, device)

    # Visualize reconstruction error distributions
    visualize_error_distributions(errors_id, errors_ood, training_sex)

    # Run permutation test
    p_value = hypothesis_testing(errors_id, errors_ood, num_permutations, test_type=test_type)
    result = "Reject" if p_value > 0.95 else "Fail to reject"
    print(f"Hypothesis test completed. {result} the null hypothesis.")
    print('p-value:', p_value)

    num_bootestrap=1000
    p_value_std = bootstrap(errors_id, errors_ood, num_permutations=1000, test_type=test_type, num_bootestrap=num_bootestrap)
    print(f'CI for p-value: {p_value} +/- {p_value_std}, estimated with {num_bootestrap} bootstrap samples')


def bootstrap(errors_id, errors_ood, num_permutations=1000, test_type='wilcoxon', num_bootestrap=100):
    """
    Function to run the permutation test and return the p-value.
    errors_id: List or array of reconstruction errors from the In-Distribution (ID) group.
    errors_ood: List or array of reconstruction errors from the Out-of-Distribution (OOD) group.
    num_permutations: Number of permutations to run for the test.
    """
    p_values = []
    for i in range(num_bootestrap):
        permutation_test = PermutationTest(errors_id, errors_ood, num_permutations, type= test_type)    
        permutation_test.run_permutation_test()
        p_values.append(permutation_test.p_value)

    # Visualize p-values distribution
    plt.hist(p_values, bins=40, alpha=1, color='green', label='p-values')
    plt.axvline(x=0.05, color='black', linestyle='--', label='0.05')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')
    plt.legend()
    plt.show()

    return np.std(p_values)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    hdf5_filepath = "multi_hot_encodings.h5"
    label_index_path = "label_index.json"
    training_sex='Female'
    model_type='ConvVAE'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim=64
    latent_dim=60
    lr=1e-5
    kl_weight=0.2
    batch_size=32
    window_size=512
    save_path='models/'+ f'_ts{training_sex}_hd{hidden_dim}_ld{latent_dim}_lr{lr}_kl{kl_weight}_bs{batch_size}_ws{window_size}_att_pp_latent.pth'
    embedding_dim = 64
    hidden_dims=[16, 32, 48]

    if model_type == 'CRVAE' or model_type == 'CRAE':
        data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size, window_size)
    else:
        latent_dim=60
        data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size, window_size, model_type='ConvVAE')
        save_path='models/'+ model_type +f'_ts{training_sex}_hd{hidden_dims}_ld{latent_dim}_lr{lr}_kl{kl_weight}_bs{batch_size}_ws{window_size}_night.pth'

    run_hypothesis_test(save_path, data_loader_wrapper,training_sex ,hidden_dim, hidden_dims, latent_dim, embedding_dim, device, model_type='ConvVAE', num_permutations=10000, test_type='mannwhitneyu')

    
