import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import copy
from models import CRVAE, CRAE, ConvVAE
from dataloaders import DataLoaderWrapper
import matplotlib.pyplot as plt
from helper_code import *


class Trainer:
    def __init__(self, model, data_loader_wrapper, training_sex='Female', lr=0.001, kl_weight=None):
        """
        Initializes the trainer for Conditional LSTM VAE.
        
        model: The ConditionalLSTMVAE model.
        data_loader_wrapper: A DataLoaderWrapper instance that provides data loaders.
        training_sex: The sex to use for training ('Female' or 'Male').
        device: 'cpu' or 'cuda' (detected automatically if not provided).
        lr: Learning rate for the optimizer.
        kl_weight: Weighting coefficient for KL-divergence in the loss.
        """
        # Automatically choose device if not specified
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        self.model = model.to(self.device)
        self.kl_weight = kl_weight
        self.data_loader_wrapper = data_loader_wrapper
        self.training_sex = training_sex
        self.num_epochs = None

        # Set up train and validation loaders based on the training sex
        if training_sex == 'Female':
            self.train_loader, self.val_loader_same_sex, _, self.val_loader_other_sex = data_loader_wrapper.get_dataloaders()
        elif training_sex == 'Male':
            _, self.val_loader_other_sex, self.train_loader, self.val_loader_same_sex = data_loader_wrapper.get_dataloaders()
        else:
            raise ValueError(f"Invalid value for training_sex: {training_sex}. Must be 'Female' or 'Male'.")

        self.optimizer = None # Initialize optimizer after the first forward pass when model parameters are available
        self.lr = lr
        self.best_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.history = {'train_loss': [], 'val_loss_same_sex': [], 'val_loss_other_sex': []}


    def initialize_optimizer(self):
        """Initialize the optimizer after the model has been used for the first time and the parameters are available."""
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            print("Optimizer initialized")    


    def train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train()
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        # Define dynamic KL weight scaling (adjust this logic based on your preference)
        kl_weight = min(1.0, self.kl_weight * (epoch / (self.num_epochs // 2)))

        for batch in self.train_loader:
            ecg_data = batch['ecg_data'].to(self.device)
            age = batch['age'].to(self.device)
            multi_hot = batch['multi_hot_encoding'].to(self.device)
            ecg_seq_len = ecg_data.shape[1]

            #print('ecg_data.shape', ecg_data.shape)
            #print('age.shape', age.shape)
            #print('multi_hot.shape', multi_hot.shape)
            #print('ecg_seq_len', ecg_seq_len)

            # Forward pass
            try:
                ecg_reconstructed, mu, logvar = self.model(ecg_data, age, multi_hot, ecg_seq_len)
            except:
                ecg_reconstructed, mu, logvar = self.model(ecg_data.permute(0, 2, 1))    

            #print('ecg_reconstructed.shape',ecg_reconstructed.shape)
            #print('ecg_reconstructed',ecg_reconstructed)
            
            # Initialize the optimizer after the first forward pass
            if self.optimizer is None:
                self.initialize_optimizer()

            self.optimizer.zero_grad()

            # Compute loss
            recon_loss, kl_divergence = vae_loss_function(ecg_reconstructed, ecg_data, mu, logvar)
            #print('recon_loss', recon_loss)
            #print('kl_divergence', kl_divergence)
            loss = recon_loss + kl_weight * kl_divergence

            #print('loss', loss)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_recon_loss += recon_loss.item() * ecg_data.size(0)
            total_kl_loss += kl_divergence.item() * ecg_data.size(0)
            #if total_recon_loss != total_recon_loss:
            #    break
            #print('total_recon_loss', total_recon_loss)


        #print('total_recon_loss', total_recon_loss)
        #print('len(self.train_loader.dataset)', len(self.train_loader.dataset))
        #print('total_kl_loss', total_kl_loss)
        #print('len(self.train_loader.dataset)', len(self.train_loader.dataset))
        epoch_recon_loss = total_recon_loss / len(self.train_loader.dataset)
        epoch_kl_loss = total_kl_loss / len(self.train_loader.dataset)
        #print('epoch_recon_loss', epoch_recon_loss)
        return epoch_recon_loss, epoch_kl_loss


    def validate_epoch(self, val_loader):
        """Evaluates the model on a validation set."""
        self.model.eval()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ecg_data = batch['ecg_data'].to(self.device)
                age = batch['age'].to(self.device)
                multi_hot = batch['multi_hot_encoding'].to(self.device)
                ecg_seq_len = ecg_data.shape[1]

                # Forward pass
                try:
                    ecg_reconstructed, mu, logvar = self.model(ecg_data, age, multi_hot, ecg_seq_len)
                except:
                    ecg_reconstructed, mu, logvar = self.model(ecg_data.permute(0, 2, 1))    


                # Compute loss
                recon_loss, kl_divergence = vae_loss_function(ecg_reconstructed, ecg_data, mu, logvar)
                loss = recon_loss + self.kl_weight * kl_divergence

                total_recon_loss += recon_loss.item() * ecg_data.size(0)
                total_kl_loss += kl_divergence.item() * ecg_data.size(0)

        epoch_recon_loss = total_recon_loss / len(val_loader.dataset)
        epoch_kl_loss = total_kl_loss / len(val_loader.dataset)
        return epoch_recon_loss, epoch_kl_loss

    def train(self, num_epochs=50, save_path='best_model.pth', validate_every_n_epochs=5):
        """Main training loop.
        
        num_epochs: Total number of epochs to train.
        save_path: Path to save the best model.
        validate_every_n_epochs: Run validation after every 'n' epochs.
        """
        self.num_epochs = num_epochs

        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_recon_loss, train_kl_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}/{num_epochs} - Train recon loss: {train_recon_loss:.4f}, KL loss: {train_kl_loss:.4f}')

            # Perform validation every 'n' epochs
            if epoch % validate_every_n_epochs == 0 or epoch == num_epochs or epoch == 1:
                # Validate on same sex dataset
                val_recon_loss_same_sex, val_kl_loss_same_sex = self.validate_epoch(self.val_loader_same_sex)
                print(f'Epoch {epoch} - Validation (same sex) recon loss: {val_recon_loss_same_sex:.4f}, KL loss: {val_kl_loss_same_sex:.4f}')

                # Validate on opposite sex dataset
                val_recon_loss_other_sex, val_kl_loss_other_sex = self.validate_epoch(self.val_loader_other_sex)
                print(f'Epoch {epoch} - Validation (other sex) recon loss: {val_recon_loss_other_sex:.4f}, KL loss: {val_kl_loss_other_sex:.4f}')

                # Save best model based on validation loss (same sex)
                train_loss = train_recon_loss + self.kl_weight * train_kl_loss
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_model_wts, save_path)
                    print(f'Saved best model at epoch {epoch} with validation loss {train_loss:.4f}')

                # Record the history for validation
                self.history['val_loss_same_sex'].append(val_recon_loss_same_sex)
                self.history['val_loss_other_sex'].append(val_recon_loss_other_sex)

                visualize_ecg_comparison(self.data_loader_wrapper, self.model, self.training_sex, data_type='train', idx=5, device='cpu')

            else:
                # Still record the training history even when skipping validation
                self.history['val_loss_same_sex'].append(self.history['val_loss_same_sex'][-1])
                self.history['val_loss_other_sex'].append(self.history['val_loss_other_sex'][-1])

            # Record the history for training
            self.history['train_loss'].append(train_recon_loss)

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model.eval(), self.history


    def train_CRA_model(self, model, n_epochs, save_path):
        '''
        model, history = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        n_epochs=150
        )

        
        '''
        model = model.to(self.device)
        train_loader = self.train_loader 
        val_loader= self.val_loader_same_sex
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.L1Loss(reduction='sum').to(self.device)
        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000.0

        for epoch in range(1, n_epochs + 1):
            model = model.train()

            train_losses = []
            for batch in train_loader:
                ecg = batch['ecg_data'].to(self.device)
                age = batch['age'].to(self.device)
                multi_hot = batch['multi_hot_encoding'].to(self.device)
                #ecg_seq_len = ecg.shape[2]

                optimizer.zero_grad()

                ecg_reconstructed = model(ecg, age, multi_hot)

                loss = criterion(ecg_reconstructed, ecg)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    ecg = batch['ecg_data'].to(self.device)
                    age = batch['age'].to(self.device)
                    multi_hot = batch['multi_hot_encoding'].to(self.device)

                    optimizer.zero_grad()

                    ecg_reconstructed = model(ecg, age, multi_hot)

                    loss = criterion(ecg_reconstructed, ecg)

                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(self.best_model_wts, save_path.replace('last', 'best'))
                print(f'Saved best model at epoch {epoch} with validation loss {val_loss:.4f}')

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')


        last_model_wts = copy.deepcopy(model.state_dict())
        torch.save(last_model_wts, save_path.replace('best', 'last'))
        print(f'Saved last model at epoch {epoch} with validation loss {val_loss:.4f}')

        model.load_state_dict(best_model_wts)
        return model.eval(), history 


# VAE loss function
def vae_loss_function(reconstructed_ecg, ecg, mu, logvar):
    """Calculates the VAE loss (reconstruction + KL-divergence)."""
    # Reconstruction loss (MAE L1 loss)
    #recon_loss = F.l1_loss(reconstructed_ecg, ecg, reduction='sum')
    #print('reconstructed_ecg', reconstructed_ecg)
    #print('ecg', ecg)
    recon_loss = F.mse_loss(reconstructed_ecg, ecg, reduction='sum') 
    #print('recon_loss', recon_loss)
    
    # KL-divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss, kl_divergence


def train_CRVAE(hidden_dim, latent_dim, data_loader_wrapper, training_sex='Female', lr=1e-3, kl_weight=0.2, num_epochs=50, save_path='best_model.pth', validate_every_n_epochs=5):
    """Trains a Conditional LSTM VAE model.
    
    hidden_dim: Hidden dimension for the LSTM layers.
    latent_dim: Latent dimension for the VAE.
    data_loader_wrapper: DataLoaderWrapper instance."""
    # Initialize the model
    model = CRVAE(hidden_dim, latent_dim)
    model.initialize_model_dims(data_loader_wrapper.get_dataloaders()[0]) #initilize dimentions
    
    try:
        model.load_state_dict(torch.load(save_path))
    except:
        print('No model found, training from scratch')    

    # Initialize Trainer with chosen sex ('Female' in this case)
    trainer = Trainer(model, data_loader_wrapper, training_sex, lr, kl_weight) #device=device,

    # Train the model and validate every 5 epochs
    trained_model, history = trainer.train(num_epochs, save_path, validate_every_n_epochs)

    return trained_model, history


def train_ConvVAE(latent_dim, hidden_dims,  data_loader_wrapper, training_sex='Female', lr=1e-3, kl_weight=0.2, num_epochs=50, save_path='best_model.pth', validate_every_n_epochs=5):
    # Initialize the model
    seq_len = data_loader_wrapper.get_dataloaders()[0].dataset[0]['ecg_data'].shape[0]
    model = ConvVAE(input_dim=seq_len, latent_dim=60, hidden_dims=[16, 32, 48], kernel_size=9, dropout=0.1)
    
    try:
        model.load_state_dict(torch.load(save_path))
    except:
        print('No model found, training from scratch')    

    # Initialize Trainer with chosen sex ('Female' in this case)
    trainer = Trainer(model, data_loader_wrapper, training_sex, lr, kl_weight) #device=device,

    # Train the model and validate every 5 epochs
    trained_model, history = trainer.train(num_epochs, save_path, validate_every_n_epochs)

    return trained_model, history


def train_CRAE(
    data_loader_wrapper,
    embedding_dim = 64,
    training_sex='Female',
    lr=1e-3, 
    kl_weight=0.2,
    num_epochs=50, 
    save_path=f'best_model.pth', 
    validate_every_n_epochs=5
    ):

    # Assuming:
    # x: (batch_size, seq_len, n_features)
    # age: (batch_size, 1)
    # multi_hot: (batch_size, 54)

    # Initialize model
    model = CRAE(data_loader_wrapper.get_dataloaders()[0], embedding_dim)
    model.load_state_dict(torch.load(save_path))
    
    ''''''
    trainer = Trainer(model, data_loader_wrapper, training_sex, lr, kl_weight) #device=device,

    model, history=trainer.train_CRA_model(model, num_epochs, save_path)

    ax = plt.figure().gca()

    ax.plot(history['train'])
    ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.show()
    
    
    model = model.to('cpu')
    model.load_state_dict(torch.load(save_path))
    visualize_one_record(data_loader_wrapper.get_dataloaders()[0].dataset, idx=0)
    visualize_ecg_comparison(data_loader_wrapper, model, training_sex, data_type='train', idx=0, device='cpu')
    
    return model, history



if __name__ == "__main__":
    base_dir = "physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"
    hdf5_filepath = "multi_hot_encodings.h5"
    label_index_path = "label_index.json"
    training_sex='Female'
    model_type='CRVAE'
    hidden_dim=64
    latent_dim=32 #60
    lr=1e-5
    kl_weight=0.2
    num_epochs= 0
    batch_size=32
    window_size=216
    save_path='models/' + f'_ts{training_sex}_hd{hidden_dim}_ld{latent_dim}_lr{lr}_kl{kl_weight}_bs{batch_size}_ws{window_size}_att_pp_latent.pth'
    #save_path = 'best_model_Female_win150_b8_ld128_hd32_dykl.pth'
    validate_every_n_epochs=70
    embedding_dim = 64
    
    if model_type == 'CRVAE':
        #window_size=512
        batch_size=32
        hidden_dim=64
        latent_dim=32
        kl_weight=0.5
        save_path='models/'+ model_type + f'_ts{training_sex}_hd{hidden_dim}_ld{latent_dim}_lr{lr}_kl{kl_weight}_bs{batch_size}_ws{window_size}_att_pp_latent.pth'
        #save_path = 'models/CRVAE_tsFemale_hd64_ld32_lr1e-05_kl0.5_bs16_ws500_att_pp_latent.pth'
        data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size, window_size)
        model, history =train_CRVAE(hidden_dim, latent_dim, data_loader_wrapper, training_sex, lr, kl_weight, num_epochs, save_path, validate_every_n_epochs)
    elif model_type == 'CRAE':
        data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size, window_size)
        model, history = train_CRAE( data_loader_wrapper,embedding_dim,training_sex,lr, kl_weight,num_epochs, f'best_model_{training_sex}_CRA.pth', validate_every_n_epochs)
    else:
        data_loader_wrapper = DataLoaderWrapper(base_dir, hdf5_filepath, label_index_path, batch_size, window_size, model_type='ConvVAE')
        hidden_dims=[16, 32, 48]
        window_size=512
        batch_size=32
        save_path='models/'+ model_type +f'_ts{training_sex}_hd{hidden_dims}_ld{latent_dim}_lr{lr}_kl{kl_weight}_bs{batch_size}_ws{window_size}_night2.pth'
        model, history = train_ConvVAE( hidden_dims, latent_dim, data_loader_wrapper, training_sex, lr, kl_weight, num_epochs, save_path, validate_every_n_epochs)

    #visualize_training_history(history, save_path)

    for idx in range(1):
        visualize_ecg_comparison(data_loader_wrapper, model, training_sex,  data_type='train', idx=idx, device='cpu')
        

    #visualize_latent_space_age(model, data_loader_wrapper,save_path, training_sex, device='cpu', method='pca')

    #visualize_latent_space_categories(model, data_loader_wrapper,save_path, training_sex, device='cpu', method='pca')

    #compare_reconstructions(model, data_loader_wrapper, training_sex , device='cpu', num_samples=12)
