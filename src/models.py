import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

"""
Meaning: The reconstruction loss measures how well the model can reproduce the ECG signal based solely on the latent space. 
It evaluates how accurately the model decodes the ECG from the latent representation, age, and multi-hot encoding.
If the goal is to compare stochastic differences in ECG reconstruction quality between sexes, we focus on comparing the reconstruction loss (e.g., MAE) only. 
The KL-divergence applies to the latent space regularization and not directly to the reconstruction quality, which might not be relevant to this comparison.
Use reconstruction loss only if you want to test how well the model reconstructs ECGs from different groups.


"""

class ConvVAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=60, hidden_dims=[16, 32, 48, 64, 80], kernel_size=19, dropout=0.1):
        """
        Convolutional Variational Autoencoder (CVAE) for 1D time-series data (ECG).
        
        input_dim: The length of the input ECG signal (e.g., window_size).
        latent_dim: The dimensionality of the latent space (e.g., 60).
        hidden_dims: List of hidden dimensions for the convolutional layers.
        kernel_size: The size of the convolutional filter.
        dropout: Dropout probability.
        """
        super(ConvVAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        in_channels = 1  # Starting with 1 channel (ECG input)
        self.input_dim = input_dim
        
        for h_dim in hidden_dims:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Conv1d(h_dim, h_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = h_dim
        
        self.flatten = nn.Flatten()

        # Fully connected layers to learn mean and variance
        #print('hidden_dims[-1]',hidden_dims[-1]  )
        #print('input_dim ',input_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1] * input_dim , latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * input_dim , latent_dim)

        # Decoder layers
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * input_dim)
        
        self.decoder = nn.ModuleList()
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i+1], kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.ConvTranspose1d(hidden_dims[i+1], hidden_dims[i+1], kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # Output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def encode(self, x, **kwargs):
        """
        Encode the input into the latent space, returning mu and logvar.
        """
        for layer in self.encoder:
            x = layer(x)
        # x.shape: torch.Size([16, 48, 1024])    
        x = self.flatten(x)
        # x.shape: torch.Size([16, 49152])
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode the latent vector z back to the reconstructed input.
        """
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, self.input_dim)  # Reshape to match the decoder input
        for layer in self.decoder:
            x = layer(x)
        x = self.final_layer(x)
        return x.permute(0,2,1)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



class CRVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        """
        LSTM VAE model with conditional inputs for age and multi-hot encoding.
        
        hidden_dim: Dimension of the LSTM hidden state.
        latent_dim: Dimension of the latent space.
        """
        super(CRVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM for ECG (input dim will be inferred during forward pass)
        self.encoder_lstm = None  # Initialize LSTM dynamically later

        # Linear layers to encode into latent space
        self.hidden_to_mu = None  # Will initialize after inferring input dim
        self.hidden_to_logvar = None  # Will initialize after inferring input dim

        # Conditional layers for age and multi-hot encoding
        self.age_embed = None  # To be initialized later
        self.multi_hot_embed = None  # To be initialized later

        # Decoder LSTM (input dim will be inferred during forward pass)
        self.decoder_lstm = None  # Initialize LSTM dynamically later
        self.hidden_to_ecg = None  # Output layer for ECG reconstruction will be initialized dynamically

        # Attention mechanism
        self.attention = None  # To be initialized after LSTM dimensions are known


    def initialize_model_dims(self, data_loader):
        """
        Initializes LSTM layers and linear layers based on inferred input dimensions.
        This is called once we know the input dims.
        """
        batch = next(iter(data_loader))
        ecg_data = batch['ecg_data']
        ecg_input_dim = ecg_data.shape[2]
        multi_hot = batch['multi_hot_encoding']
        multi_hot_dim = multi_hot.shape[-1]

        #print('ecg_data.shape:', ecg_data.shape)
        #print('ecg_data', ecg_data)
        #print('multi_hot.shape:', multi_hot.shape)
        #print('multi_hot', multi_hot)
        #print('ecg_input_dim 12!:', ecg_data.shape[2])


        # Encoder LSTM for ECG
        self.encoder_lstm = nn.LSTM(ecg_input_dim, self.hidden_dim, batch_first=True)

        #print('self.encoder_lstm:', self.encoder_lstm)

        # Linear layers to encode into latent space
        self.hidden_to_mu = nn.Linear(2*self.hidden_dim, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(2*self.hidden_dim, self.latent_dim)
        
        # Linear layers for conditioning inputs (age and multi-hot encoding)
        self.age_embed = nn.Linear(1, 1) #todo
        self.multi_hot_embed = nn.Linear(multi_hot_dim, self.hidden_dim - 1) #todo

        # Decoder LSTM for ECG reconstruction
        #self.decoder_lstm = nn.LSTM(self.latent_dim + 2 * self.hidden_dim, self.hidden_dim, batch_first=True, dropout=0.3)
        self.decoder_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True) # self.latent_dim -> self.hidden
        self.hidden_to_ecg = nn.Linear(self.hidden_dim, ecg_input_dim)

        # Attention mechanism
        self.attention = nn.Linear(self.hidden_dim, 1)  # Attention scores


    def encode(self, ecg, age, multi_hot):
        """Encodes the ECG data into the latent space.
        """
        # Embed age and multi-hot encoding
        age = torch.unsqueeze(age, 1)
        age_embedding = F.tanh(self.age_embed(age))
        multi_hot_embedding = F.relu(self.multi_hot_embed(multi_hot))

        # Create an initial hidden state for the LSTM
        init_hidden = torch.cat([age_embedding, multi_hot_embedding], dim=-1).unsqueeze(0).repeat(self.encoder_lstm.num_layers, 1, 1)
        init_cell = torch.zeros_like(init_hidden)  # Initialize the cell state as zeros
    
        encoder_outputs, (hidden_state, _) = self.encoder_lstm(ecg , (init_hidden, init_hidden))
        hidden_state = hidden_state[-1]  # Get the last hidden state

        hidden_state_categorical = torch.cat([hidden_state, age_embedding, multi_hot_embedding], dim=-1) #added
        
        # Map hidden state to latent space (mean and log variance)
        #mu = self.hidden_to_mu(hidden_state)
        #logvar = self.hidden_to_logvar(hidden_state)
        mu = self.hidden_to_mu(hidden_state_categorical)
        logvar = self.hidden_to_logvar(hidden_state_categorical)

        return encoder_outputs, mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def apply_attention(self, encoder_outputs, decoder_hidden):
        """
        Apply attention mechanism on the encoder hidden states to focus on important parts.
        """
        # Compute attention scores
        attn_scores = torch.tanh(self.attention(encoder_outputs))  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize over sequence length

        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)  # [batch_size, hidden_dim]
        return context_vector  
    

    def decode(self, latent, encoder_outputs, age, multi_hot, ecg_seq_len):
        """
        Decodes the latent vector back to ECG, conditioned on age and multi-hot encoding.
        """
        # Embed age and multi-hot encoding
        age = torch.unsqueeze(age, 1)
        age_embedding = F.tanh(self.age_embed(age))
        multi_hot_embedding = F.relu(self.multi_hot_embed(multi_hot))

        '''
        # Concatenate latent vector with conditioning inputs
        latent_cond = torch.cat([latent, age_embedding, multi_hot_embedding], dim=-1)
        latent_cond = latent_cond.unsqueeze(1).repeat(1, ecg_seq_len, 1)  # Repeat latent for the entire sequence
        # Decode with LSTM
        decoded, _ = self.decoder_lstm(latent_cond)
        '''

        # Create an initial hidden state for the LSTM
        init_hidden = torch.cat([age_embedding, multi_hot_embedding], dim=-1).unsqueeze(0).repeat(self.encoder_lstm.num_layers, 1, 1)
        #init_cell = torch.zeros_like(init_hidden)  # Initialize the cell state as zeros
        
        # Apply attention mechanism to encoder outputs
        context_vector = self.apply_attention(encoder_outputs, latent)
        #init_cell = copy.deepcopy(context_vector).unsqueeze(1)

        context_vector = context_vector.unsqueeze(1).repeat(1, ecg_seq_len, 1)  # Repeat latent for the entire sequence
        
        # Decode with LSTM
        decoded, _ = self.decoder_lstm(context_vector, (init_hidden, init_hidden)) 
        ecg_reconstructed = self.hidden_to_ecg(decoded)
        
        return ecg_reconstructed #.permute(0, 2, 1)  Reshape to (batch, features, seq_len)


    def forward(self, ecg, age, multi_hot, ecg_seq_len):
        """Forward pass for training the VAE."""
        # Encode
        encoder_outputs, mu, logvar = self.encode(ecg, age, multi_hot)
        
        # Sample from the latent space
        latent = self.reparameterize(mu, logvar)
        
        # Decode and reconstruct ECG
        ecg_reconstructed = self.decode(latent, encoder_outputs, age, multi_hot, ecg_seq_len)
        
        return ecg_reconstructed, mu, logvar    
    

'''
Explanation
Encoder Adjustments:
Input Dimension: Increased to include n_features (ECG data), age_dim, and multi_hot_dim.
Forward Method: Concatenates the ECG data with age and multi-hot vectors at each time step.
Decoder Adjustments:
Input Dimension: Includes embedding_dim (latent vector), age_dim, and multi_hot_dim.
Forward Method:
The latent vector is repeated across the sequence length.
Concatenated with age and multi-hot vectors.
Passed through the LSTM to reconstruct the sequence.
RecurrentAutoencoder:
Passes x, age, and multi_hot through the encoder and decoder.
Benefits of This Approach
Simplicity: By concatenating the conditional inputs directly to the data at each time step, we avoid complex modifications to the architecture.
Effectiveness: This method allows the model to use conditional information throughout the sequence, improving its ability to reconstruct the input based on these conditions.
'''

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features=1, embedding_dim=64, age_dim=1, multi_hot_dim=54):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = n_features + age_dim + multi_hot_dim  # Adjust input_dim
        self.hidden_dim = embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, ecg, age, multi_hot):
        # ecg: (batch_size, seq_len, n_features)
        batch_size = ecg.size(0)

        # Repeat age and multi_hot across the sequence length
        age = age.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, age_dim)
        multi_hot = multi_hot.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, multi_hot_dim)

        # Concatenate ecg with age and multi_hot
        x = torch.cat([ecg, age, multi_hot], dim=2)  # (batch_size, seq_len, input_dim)

        # Pass through LSTM
        _, (hidden_n, _) = self.rnn1(x)

        # hidden_n: (num_layers, batch_size, hidden_dim)
        return hidden_n[-1]  # (batch_size, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, seq_len, n_features=1, embedding_dim=64, age_dim=1, multi_hot_dim=54):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = embedding_dim + age_dim + multi_hot_dim  # Adjust input_dim
        self.hidden_dim = embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, latent, age, multi_hot):
        # latent: (batch_size, embedding_dim)
        batch_size = latent.size(0)

        # Prepare initial input for LSTM (could be zeros or a learned parameter)
        initial_input = torch.zeros(batch_size, self.seq_len, self.hidden_dim).to(latent.device)

        # Repeat age and multi_hot across the sequence length
        age = age.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, age_dim)
        multi_hot = multi_hot.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, multi_hot_dim)

        # Concatenate initial_input with latent, age, and multi_hot
        latent = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch_size, seq_len, embedding_dim)
        x = torch.cat([latent, age, multi_hot], dim=2)  # (batch_size, seq_len, input_dim)

        # Pass through LSTM
        output, (hidden_n, _) = self.rnn1(x)

        # Map output to original feature space
        ecg_reconstructed = self.output_layer(output)  # (batch_size, seq_len, n_features)

        return ecg_reconstructed



class CRAE(nn.Module):
    def __init__(self,data_loader, embedding_dim=64):
        super(CRAE, self).__init__()

        batch = next(iter(data_loader))
        ecg_data = batch['ecg_data']
        n_features = ecg_data.shape[2]
        seq_len = ecg_data.shape[1]
        multi_hot = batch['multi_hot_encoding']
        multi_hot_dim = multi_hot.shape[-1]
        age = batch['age']
        age_dim = age.shape[-1]

        self.encoder = Encoder(seq_len, n_features, embedding_dim, age_dim, multi_hot_dim)
        self.decoder = Decoder(seq_len, n_features, embedding_dim, age_dim, multi_hot_dim)

    def forward(self, ecg, age, multi_hot):
        latent = self.encoder(ecg, age, multi_hot)
        reconstructed = self.decoder(latent, age, multi_hot)
        return reconstructed
