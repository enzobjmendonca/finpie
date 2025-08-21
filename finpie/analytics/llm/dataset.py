"""
PyTorch dataset classes for LLM-based financial forecasting.

This module provides dataset classes for handling tokenized market data in PyTorch.
It creates sequences of tokens suitable for training transformer models on financial data.
"""

import torch
from torch.utils.data import Dataset


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
        self.tokens = torch.tensor(tokens, dtype=torch.long)  # Keep on CPU for DataLoader
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
