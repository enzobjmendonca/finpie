"""
This module implements a transformer-based language model for financial market data forecasting.
It includes tools for tokenizing market data, training transformer models, and generating forecasts.

The module provides the following main components:
- MarketTokenizer: Converts market data into discrete tokens
- ReturnTokenDataset: PyTorch dataset for handling tokenized market data
- MarketTransformer: Transformer model architecture for market data
- LLMForecaster: High-level interface for training and using the model
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union

class MarketTokenizer:
    """
    A utility class for converting continuous market data into discrete tokens.
    This tokenization process is essential for applying language model techniques to market data.
    
    The class provides methods for:
    - Converting market data series into discrete tokens
    - Converting tokens back to approximate market values
    """

    @staticmethod
    def series_to_tokens(
    series: pd.Series,
    num_bins: int,
    method: str = 'equal_width',
    first_n: Union[int, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a pandas Series into tokens using either equal-width or equal-frequency binning.
        
        Args:
            series (pd.Series): Input series to be tokenized
            num_bins (int): Number of bins to create
            method (str): Binning method - 'equal_width' or 'equal_freq'
            first_n (int, optional): If provided, only use first N samples to determine bin edges
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - tokens: Array of token indices
                - bins: Array of bin edges used for tokenization
        """
        if first_n is not None:
            data_for_bins = series.iloc[:first_n]
        else:
            data_for_bins = series
            
        if method == 'equal_width':
            # Equal width binning
            min_val = data_for_bins.min()
            max_val = data_for_bins.max()
            bins = np.linspace(min_val, max_val, num_bins)
            
        elif method == 'equal_freq':
            # Equal frequency binning using quantiles
            # First get unique values to avoid duplicate bin edges
            unique_vals = np.sort(data_for_bins.unique())
            if len(unique_vals) <= num_bins:
                # If we have fewer unique values than requested bins,
                # create bins that include all unique values
                bins = np.concatenate([unique_vals, [unique_vals[-1]]])
            else:
                # Use quantiles to create bins
                bins = np.quantile(unique_vals, np.linspace(0, 1, num_bins))
            
        else:
            raise ValueError("method must be either 'equal_width' or 'equal_freq'")
        
        # Ensure bins are unique and sorted
        bins = np.unique(bins)
        
        # If we still have fewer bins than requested, adjust the last bin
        while len(bins) < num_bins:
            last_bin = bins[-1]
            next_bin = last_bin + (last_bin - bins[-2])
            bins = np.append(bins, next_bin)
        
        # Digitize the data to get tokens
        tokens = np.digitize(series, bins[:-1])
        
        # Digitize the data to get tokens
        tokens = np.digitize(series, bins[:-1])
        
        return tokens, bins

    @staticmethod
    def tokens_to_values(tokens: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Convert tokens back to approximate values using bin centers."""
        bin_centers = (bins[:-1] + bins[1:]) / 2
        tokens = np.asarray(tokens)
        
        # Clip tokens to valid range
        tokens = np.clip(tokens, 0, len(bin_centers) - 1)
        
        return bin_centers[tokens]

class ReturnTokenDataset(Dataset):
    """
    PyTorch Dataset class for handling tokenized market return data.
    Creates sequences of tokens for training the transformer model.
    
    Attributes:
        tokens (torch.Tensor): The tokenized market data
        seq_len (int): Length of sequences to generate
    """

    def __init__(self, tokens, seq_len):
        """
        Initialize the dataset with tokenized data and sequence length.
        
        Args:
            tokens: Tokenized market data
            seq_len: Length of sequences to generate
        """
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        """Return the number of possible sequences in the dataset."""
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        """
        Get a sequence of tokens and its corresponding target sequence.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and target sequence
        """
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]  # next-token at each pos
        return x, y

class MarketTransformer(nn.Module):
    """
    Transformer model architecture for market data forecasting.
    Implements a causal transformer that can predict future market movements
    based on historical tokenized data.
    
    Attributes:
        token_embedding (nn.Embedding): Embedding layer for tokens
        pos_embedding (nn.Embedding): Positional encoding layer
        transformer (nn.TransformerEncoder): Transformer encoder layers
        fc_out (nn.Linear): Output projection layer
    """

    def __init__(self, vocab_size, emb_dim, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size (int): Size of the token vocabulary
            emb_dim (int): Dimension of token embeddings
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(1024, emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        # Ensure input is 2D (batch_size, seq_len)
        if len(x.shape) > 2:
            x = x.squeeze()
        
        # Create position indices
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_embedding(x)  # (batch_size, seq_len, emb_dim)
        pos_emb = self.pos_embedding(pos)  # (1, seq_len, emb_dim)
        
        # Add positional embeddings
        x = x + pos_emb
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project to vocabulary
        return self.fc_out(x)

class LLMForecaster:
    """
    High-level interface for training and using the market transformer model.
    Handles data preprocessing, model training, and forecasting.
    
    Attributes:
        device (torch.device): Device to run the model on (CPU/GPU)
        seq_len (int): Length of input sequences
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        vocab_size (int): Size of the token vocabulary
        emb_dim (int): Dimension of token embeddings
        lr (float): Learning rate for training
    """

    def __init__(self, seq_len=60, batch_size=64, epochs=5, vocab_size=128, emb_dim=64, lr=3e-4):
        """
        Initialize the forecaster with model parameters.
        
        Args:
            seq_len (int): Length of input sequences
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            vocab_size (int): Size of the token vocabulary
            emb_dim (int): Dimension of token embeddings
            lr (float): Learning rate for training
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lr = lr

    def build_dataset(self, returns: pd.Series):
        """
        Build the dataset from market returns.
        
        Args:
            returns (pd.Series): Series of market returns to tokenize
        """
        self.returns = returns
        self.tokens, self.bins = MarketTokenizer.series_to_tokens(self.returns, self.vocab_size)
        self.dataset = ReturnTokenDataset(self.tokens, self.seq_len)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        """Initialize the transformer model, loss function, and optimizer."""
        self.model = MarketTransformer(self.vocab_size, self.emb_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """Train the model for the specified number of epochs."""
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits.view(-1, self.vocab_size), batch_y.view(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.dataloader):.4f}")

    def fit(self, prompt, steps=10):
        """
        Generate predictions using the trained model.
        
        Args:
            prompt (torch.Tensor): Initial sequence to start generation from
            steps (int): Number of tokens to generate
            
        Returns:
            list: Generated sequence of tokens
        """
        self.model.eval()
        generated = prompt.clone().to(self.device)
        
        for _ in range(steps):
            if generated.size(1) > self.seq_len:
                generated = generated[:, -self.seq_len:]
            
            with torch.no_grad():
                logits = self.model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
        
        return generated.squeeze().tolist()