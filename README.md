# QuantLib

A comprehensive Python library for quantitative finance research and analysis.

## Overview

QuantLib is designed to provide robust tools and utilities for quantitative finance research, focusing on two main components:

1. **Data Module**: Handles financial data acquisition, processing, and management
2. **Analytics Module**: Provides advanced financial analysis and modeling capabilities

## Features

- Modern Python implementation
- Comprehensive financial data handling
- Advanced analytics and modeling tools
- Clean and maintainable codebase
- MIT licensed

## Setup

### Virtual Environment Setup

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

### Installation Options

#### 1. Development Mode Installation
This is recommended for developers who want to modify the code:
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

#### 2. Regular Installation
For users who just want to use the library:
```bash
# Install with all dependencies
pip install .

# Install with specific extras
pip install ".[notebooks]"
```

#### 3. Building Distribution Packages
To create distributable packages:
```bash
# Install build tools
pip install build

# Build the package
python -m build

# This will create:
# - dist/quantlib-0.1.0.tar.gz (source distribution)
# - dist/quantlib-0.1.0-py3-none-any.whl (wheel distribution)
```

### Running Jupyter Notebooks

1. Ensure the virtual environment is activated
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Navigate to the `quantlib/notebooks` directory
4. Open the desired notebook (e.g., `data_examples.ipynb`)

## Project Structure

```
quantlib/
├── data/           # Data handling module
├── analytics/      # Analytics and modeling module
├── tests/          # Test suite
├── examples/       # Usage examples
└── docs/           # Documentation
```

## Usage

```python
# Example usage will be added as the library develops
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 