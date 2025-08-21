"""
Transformer model architecture for financial forecasting.

This module implements a causal transformer model designed specifically for
financial market data forecasting. It uses proper causal attention masking
to ensure autoregressive generation without lookahead bias.
"""

import torch
import torch.nn as nn


def get_device():
    """
    Get the best available device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        # Use the first available GPU (usually the main one)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device


# Global device variable
DEVICE = get_device()


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
        Initialize the transformer model with proper causal attention.
        
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
        
        # Use TransformerEncoder with explicit causal masking for autoregressive generation
        # This is more reliable than TransformerDecoder for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        
        # Move model to GPU
        self.to(DEVICE)

    def forward(self, x):
        """
        Forward pass through the transformer model with causal attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
            
        Example:
            >>> output = model(input_tokens)
            >>> # output contains logits for each token position
        """
        # Ensure input is on the correct device
        x = x.to(DEVICE)
        
        # Ensure input is 2D (batch_size, seq_len)
        if len(x.shape) > 2:
            x = x.squeeze()
        
        seq_len = x.size(1)
        
        # Create position indices
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_embedding(x)  # (batch_size, seq_len, emb_dim)
        pos_emb = self.pos_embedding(pos)  # (1, seq_len, emb_dim)
        
        # Add positional embeddings
        x = x + pos_emb
        
        # Create causal mask for autoregressive generation
        # Upper triangular mask: 1s above diagonal are masked (can't see future)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Apply transformer encoder with explicit causal masking
        x = self.transformer(x, mask=causal_mask)
        
        # Project to vocabulary
        x = self.fc_out(x)
        
        return x
