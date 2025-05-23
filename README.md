# finpie

A Python library for quantitative finance, providing tools for data handling, analysis, and trading strategies.

## Installation

### Regular Installation
```bash
pip install finpie
```

### Development Setup

1. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   
   # Linux/MacOS
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

3. Install in development mode:
   ```bash
   # Install in development mode with all dependencies
   pip install -e .
   
   # Install with development tools
   pip install -e ".[dev]"
   
   # Install with notebook support
   pip install -e ".[notebooks]"
   
   # Install with both development tools and notebook support
   pip install -e ".[dev,notebooks]"
   ```

4. Running Jupyter Notebooks:
   ```bash
   # Start Jupyter
   jupyter notebook
   
   # Navigate to the finpie/notebooks directory
   # Open the desired notebook (e.g., data_examples.ipynb)
   ```

## Quick Start

### Basic Time Series Usage

```python
from finpie.data import TimeSeries, TimeSeriesMetadata
import pandas as pd
from datetime import datetime

# Create a simple time series
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [103, 104, 105],
    'low': [99, 100, 101],
    'close': [102, 103, 104],
    'volume': [1000, 1100, 1200]
}, index=pd.date_range('2024-01-01', periods=3))

metadata = TimeSeriesMetadata(
    symbol='AAPL',
    source='yahoo',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 3),
    frequency='1D',
    currency='USD',
    additional_info={'sector': 'Technology'}
)

ts = TimeSeries(data, metadata)

# Basic operations
returns = ts.returns()  # Calculate returns
resampled = ts.resample('1W')  # Resample to weekly frequency
rolling_stats = ts.rolling(window=20)  # Calculate rolling statistics
```

### Statistical Analysis

```python
from finpie.analytics import StatisticalAnalytics

# Create analytics instance
analytics = StatisticalAnalytics(ts, column='close')

# Calculate various statistics
zscore = analytics.zscore(window=20)  # Calculate z-scores
half_life = analytics.half_life()  # Calculate half-life of mean reversion
hurst = analytics.hurst_exponent()  # Calculate Hurst exponent
spread = analytics.spread_to_ma(window=20)  # Calculate spread to moving average

# Generate trading signals
signals = analytics.trading_signals(zscore_threshold=2.0, window=20)
```

### Ratio Analysis

```python
from finpie.data import RatioTimeSeries

# Create ratio time series (e.g., for pair trading)
ratio_ts = RatioTimeSeries(numerator_ts, denominator_ts)

# Access the ratio data
ratio_data = ratio_ts.data

# Create analytics instance for the ratio
ratio_analytics = StatisticalAnalytics(ratio_ts, column='ratio')

# Calculate statistics
zscore = ratio_analytics.zscore()
mean_reversion = ratio_analytics.spread_to_ma() / ratio_ts.data['ratio']
signals = ratio_analytics.trading_signals()
```

### Spread Analysis

```python
from finpie.data import SpreadTimeSeries

# Create spread time series (e.g., for statistical arbitrage)
spread_ts = SpreadTimeSeries(series1_ts, series2_ts)

# Access the spread data and hedge ratio
spread_data = spread_ts.data
hedge_ratio = spread_ts.hedge_ratio

# Create analytics instance for the spread
spread_analytics = StatisticalAnalytics(spread_ts, column='spread')

# Calculate statistics
zscore = spread_analytics.zscore()
half_life = spread_analytics.half_life()
hurst = spread_analytics.hurst_exponent()
signals = spread_analytics.trading_signals()
```

### Multiple Time Series Analysis

```python
from finpie.data import MultiTimeSeries
from finpie.data.service import DataService

ds = DataService.create_default_service()
ts1 = = service.get_close_prices(
    'AAPL',
    'yahoo_finance',
    '2025-01-01',
    '2025-01-10',
    interval='1d',
)
ts2 = = service.get_close_prices(
    'GOOGL',
    'yahoo_finance',
    '2025-01-01',
    '2025-01-10',
    interval='1d',
)
ts2 = = service.get_close_prices(
    'MSFT',
    'yahoo_finance',
    '2025-01-01',
    '2025-01-10',
    interval='1d',
)
# Create multi time series for portfolio analysis
multi_ts = MultiTimeSeries([ts1, ts2, ts3])

# Calculate correlation and covariance
correlation = multi_ts.correlation()
covariance = multi_ts.covariance()

# Calculate portfolio returns
weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
portfolio_returns = multi_ts.portfolio_returns(weights)

# Calculate rolling correlations
rolling_correlations = multi_ts.rolling_correlation(window=20)
```

## Project Structure

```
finpie/
├── data/           # Data handling module
├── analytics/      # Analytics and modeling module
├── datasource/     # Data request module
├── notebooks/      # Usage examples and tutorials
└── docs/           # Documentation
```

## Features

- **Time Series Management**
  - Flexible data structure with metadata support
  - Built-in data validation and alignment
  - Support for various frequencies and resampling
  - Comprehensive statistical calculations

- **Statistical Analysis**
  - Z-score calculations
  - Mean reversion metrics (half-life, Hurst exponent)
  - Moving average analysis
  - Trading signal generation

- **Specialized Time Series**
  - Ratio analysis for pair trading
  - Spread analysis for statistical arbitrage
  - Multi-time series for portfolio analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request :).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 