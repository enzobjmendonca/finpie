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

## Examples

### 1. Fetching and Displaying Time Series Data

```python
from finpie.datasource.service import DataService

data_service = DataService.create_default_service()

symbol = "WIN$N"  # Change to your desired symbol
start_date = "2022-01-01"
end_date = "2022-01-10"
interval = "1m"  # 1-minute data

ts_object = data_service.get_close_prices(
    symbol=symbol,
    source='mt5',
    start_date=start_date,
    end_date=end_date,
    interval=interval
)

print(ts_object.data.head())  # Display the first rows of the time series
```

### 2. Statistical and Technical Analytics

```python
from finpie.analytics import Statistical, Technical
from finpie.data import TimeSeries

# Assume ts_object is a TimeSeries as above
stat = Statistical(ts_object, column='close')
tech = Technical(ts_object, column='close')

# Statistical analytics
zscore = stat.zscore(window=20)
half_life = stat.half_life()
hurst = stat.hurst_exponent()

# Technical analytics
rsi = tech.rsi(window=14)
macd = tech.macd()

print('Z-score:', zscore.tail())
print('RSI:', rsi.tail())
```

### 3. LLM-based Market Forecasting

```python
import torch
import pandas as pd
from finpie.analytics.llm import LLMForecaster, MarketTokenizer
from finpie.data.timeseries import TimeSeries

# Load your time series data (example: from a parquet file)
ts = TimeSeries(pd.read_parquet('finpie/notebooks/win_returns.parquet'))
ret = ts.returns(intraday_only=True)
returns = ret.data[(ret.data.index.hour >= 10) & (ret.data.index.hour < 17)]

# Initialize and train the LLM forecaster
forecaster = LLMForecaster()
forecaster.build_dataset(returns['close'])
forecaster.build_model()
forecaster.train()

# Generate predictions for the next 5 periods
data_prompt = torch.tensor(
    MarketTokenizer.series_to_tokens(
        returns['close'].iloc[-60:],
        forecaster.vocab_size
    )[0],
    dtype=torch.long
).unsqueeze(0)
predicted_tokens = forecaster.fit(data_prompt, steps=5)
predicted_returns = MarketTokenizer.tokens_to_values(
    predicted_tokens, forecaster.bins
)
print('Predicted returns:', predicted_returns)
```

### 4. Fetching Fundamentals Data

```python
from finpie.datasource.sources.schemas.status_invest import FundamentalsParams
from finpie.datasource.sources.status_invest import StatusInvestSource

si = StatusInvestSource()
params = FundamentalsParams(max_pe_ratio=10, items_per_page=15)
fundamentals = si.get_fundamentals(params)
print(fundamentals.head())
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