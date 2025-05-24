"""
LLM-based forecasting tools for financial time series data.

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
    The class provides methods for converting market data series into discrete tokens and
    converting tokens back to approximate market values.
    
    Example:
        >>> tokenizer = MarketTokenizer()
        >>> tokens, bins = tokenizer.series_to_tokens(returns, num_bins=128)
        >>> values = tokenizer.tokens_to_values(tokens, bins)
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
            series: Input series to be tokenized
            num_bins: Number of bins to create
            method: Binning method - 'equal_width' or 'equal_freq'
            first_n: If provided, only use first N samples to determine bin edges
            
        Returns:
            Tuple containing:
                - tokens: Array of token indices
                - bins: Array of bin edges used for tokenization
                
        Example:
            >>> tokens, bins = tokenizer.series_to_tokens(returns, num_bins=128)
            >>> # tokens contains the discrete token indices
            >>> # bins contains the bin edges used for tokenization
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
        
        return tokens, bins

    @staticmethod
    def tokens_to_values(tokens: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Convert tokens back to approximate values using bin centers.
        
        Args:
            tokens: Array of token indices
            bins: Array of bin edges used for tokenization
            
        Returns:
            Array of approximate values corresponding to the tokens
            
        Example:
            >>> values = tokenizer.tokens_to_values(tokens, bins)
            >>> # values contains the approximate market values
        """
        bin_centers = (bins[:-1] + bins[1:]) / 2
        tokens = np.asarray(tokens)
        
        # Clip tokens to valid range
        tokens = np.clip(tokens, 0, len(bin_centers) - 1)
        
        return bin_centers[tokens]

class ReturnTokenDataset(Dataset):
    """
    PyTorch Dataset class for handling tokenized market return data.
    
    This class creates sequences of tokens for training the transformer model.
    It handles the creation of input sequences and their corresponding target sequences.
    
    Attributes:
        tokens (torch.Tensor): The tokenized market data
        seq_len (int): Length of sequences to generate
        
    Example:
        >>> dataset = ReturnTokenDataset(tokens, seq_len=60)
        >>> x, y = dataset[0]  # Get first sequence and target
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
        """
        Return the number of possible sequences in the dataset.
        
        Returns:
            Number of sequences
        """
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        """
        Get a sequence of tokens and its corresponding target sequence.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tuple containing:
                - x: Input sequence of tokens
                - y: Target sequence (next token at each position)
                
        Example:
            >>> x, y = dataset[0]
            >>> # x contains the input sequence
            >>> # y contains the target sequence
        """
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]  # next-token at each pos
        return x, y

class MarketTransformer(nn.Module):
    """
    Transformer model architecture for market data forecasting.
    
    This class implements a causal transformer that can predict future market movements
    based on historical tokenized data. It uses a standard transformer architecture with
    token embeddings, positional encodings, and multi-head attention.
    
    Attributes:
        token_embedding (nn.Embedding): Embedding layer for tokens
        pos_embedding (nn.Embedding): Positional encoding layer
        transformer (nn.TransformerEncoder): Transformer encoder layers
        fc_out (nn.Linear): Output projection layer
        
    Example:
        >>> model = MarketTransformer(vocab_size=128, emb_dim=64)
        >>> output = model(input_tokens)
    """

    def __init__(self, vocab_size, emb_dim, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Size of the token vocabulary
            emb_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
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
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
            
        Example:
            >>> output = model(input_tokens)
            >>> # output contains logits for each token position
        """
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
        x = self.fc_out(x)
        
        return x

class LLMForecaster:
    """
    High-level interface for training and using the transformer model for market forecasting.
    
    This class provides a convenient interface for:
    - Building and training the transformer model
    - Generating market forecasts
    - Handling data preprocessing and tokenization
    
    Attributes:
        seq_len (int): Length of input sequences
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        vocab_size (int): Size of the token vocabulary
        emb_dim (int): Dimension of token embeddings
        lr (float): Learning rate
        
    Example:
        >>> forecaster = LLMForecaster(seq_len=60, batch_size=64)
        >>> forecaster.fit(returns)
        >>> predictions = forecaster.predict(prompt)
    """

    def __init__(self, seq_len=60, batch_size=64, epochs=5, vocab_size=128, emb_dim=64, lr=3e-4):
        """
        Initialize the LLM forecaster.
        
        Args:
            seq_len: Length of input sequences
            batch_size: Batch size for training
            epochs: Number of training epochs
            vocab_size: Size of the token vocabulary
            emb_dim: Dimension of token embeddings
            lr: Learning rate
        """
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lr = lr
        
        self.tokenizer = MarketTokenizer()
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.dataloader = None

    def build_dataset(self, returns: pd.Series):
        """
        Build the dataset for training.
        
        Args:
            returns: Series of market returns
            
        Example:
            >>> forecaster.build_dataset(returns)
        """
        tokens, self.bins = self.tokenizer.series_to_tokens(returns, self.vocab_size)
        self.dataset = ReturnTokenDataset(tokens, self.seq_len)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        """
        Build the transformer model.
        
        Example:
            >>> forecaster.build_model()
        """
        self.model = MarketTransformer(self.vocab_size, self.emb_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """
        Train the model on the prepared dataset.
        
        Example:
            >>> forecaster.train()
        """
        if self.model is None:
            self.build_model()
            
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output.view(-1, self.vocab_size), batch_y.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def fit(self, prompt, steps=10):
        """
        Generate a forecast sequence from a prompt.
        
        Args:
            prompt: Initial sequence of returns
            steps: Number of steps to forecast
            
        Returns:
            Series of forecasted returns
            
        Example:
            >>> forecast = forecaster.fit(prompt, steps=10)
            >>> # forecast contains the predicted returns
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.eval()
        tokens, _ = self.tokenizer.series_to_tokens(prompt, self.vocab_size, first_n=len(prompt))
        tokens = torch.tensor(tokens[-self.seq_len:], dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(steps):
                output = self.model(tokens)
                next_token = output[0, -1].argmax()
                tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                tokens = tokens[:, -self.seq_len:]
                
        forecast_tokens = tokens[0, -steps:].numpy()
        forecast_values = self.tokenizer.tokens_to_values(forecast_tokens, self.bins)
        
        return pd.Series(forecast_values, index=pd.date_range(start=prompt.index[-1], periods=steps+1)[1:])