"""
Market data tokenization utilities for LLM-based financial forecasting.

This module provides tools for converting continuous market data into discrete tokens
that can be processed by transformer models. The tokenization process is essential for
applying language model techniques to financial time series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union


class MarketTokenizer:
    """
    A utility class for converting continuous market data into discrete tokens.
    
    This tokenization process is essential for applying language model techniques to market data.
    The class provides methods for converting market data series into discrete tokens and
    converting tokens back to approximate market values.
    
    Example:
        >>> tokenizer = MarketTokenizer()
        >>> tokens, bins, mean = tokenizer.series_to_tokens(returns, num_bins=128)
        >>> values = tokenizer.tokens_to_values(tokens, bins, mean)
    """

    @staticmethod
    def series_to_tokens(
        series: pd.Series,
        num_bins: int,
        method: str = 'equal_width',
        first_n: Union[int, None] = None,
        zero_centered: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert a pandas Series into tokens using either equal-width or equal-frequency binning.
        
        Args:
            series: Input series to be tokenized
            num_bins: Number of bins to create
            method: Binning method - 'equal_width' or 'equal_freq'
            first_n: If provided, only use first N samples to determine bin edges
            zero_centered: If True, center data around zero before tokenization
            
        Returns:
            Tuple containing:
                - tokens: Array of token indices
                - bins: Array of bin edges used for tokenization
                - original_mean: Original mean (for reconstruction)
                
        Example:
            >>> tokens, bins, mean = tokenizer.series_to_tokens(returns, num_bins=128)
            >>> # tokens contains the discrete token indices
            >>> # bins contains the bin edges used for tokenization
        """
        # Store original mean for bias correction
        original_mean = series.mean()
        
        # Center data around zero if requested
        if zero_centered:
            centered_series = series - original_mean
        else:
            centered_series = series
            original_mean = 0.0  # No centering applied
        
        if first_n is not None:
            data_for_bins = centered_series.iloc[:first_n]
        else:
            data_for_bins = centered_series
            
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
        
        # Digitize the centered data to get tokens
        tokens = np.digitize(centered_series, bins[:-1])
        
        return tokens, bins, original_mean

    @staticmethod
    def tokens_to_values(tokens: np.ndarray, bins: np.ndarray, original_mean: float = 0.0) -> np.ndarray:
        """
        Convert tokens back to approximate values using bin centers.
        
        Args:
            tokens: Array of token indices
            bins: Array of bin edges used for tokenization
            original_mean: Original mean to add back (for zero-centered tokenization)
            
        Returns:
            Array of approximate values corresponding to the tokens
            
        Example:
            >>> values = tokenizer.tokens_to_values(tokens, bins, original_mean)
            >>> # values contains the approximate market values
        """
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        tokens = np.asarray(tokens)
        
        # Clip tokens to valid range
        tokens = np.clip(tokens, 0, len(bin_centers) - 1)
        
        # Reconstruct values and add back original mean
        values = bin_centers[tokens] + original_mean
        
        return values
