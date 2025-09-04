"""
Simple performance test for the new MultiTimeSeries implementation.
"""

import pandas as pd
import numpy as np
import time
import sys
import os

# Add the data_new path
sys.path.append('../data_new')

def test_data_loading():
    """Test loading the parquet file"""
    print("Loading mts2.parquet...")
    
    if not os.path.exists('mts2.parquet'):
        print("Error: mts2.parquet not found")
        return None
    
    start_time = time.time()
    data = pd.read_parquet('mts2.parquet')
    load_time = time.time() - start_time
    
    print(f"✓ Data loaded in {load_time:.4f} seconds")
    print(f"✓ Data shape: {data.shape}")
    print(f"✓ Data memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"✓ Columns: {list(data.columns)}")
    print(f"✓ Date range: {data.index[0]} to {data.index[-1]}")
    
    return data

def test_original_pandas_operations(data):
    """Test basic pandas operations on the raw data"""
    print("\nTesting Basic Pandas Operations:")
    print("-" * 40)
    
    # Test correlation
    start_time = time.time()
    corr = data.corr()
    corr_time = time.time() - start_time
    print(f"✓ Pandas correlation: {corr_time:.4f} seconds")
    
    # Test returns calculation
    start_time = time.time()
    returns = data.pct_change()
    returns_time = time.time() - start_time
    print(f"✓ Pandas pct_change: {returns_time:.4f} seconds")
    
    # Test slicing
    start_time = time.time()
    mid_point = len(data) // 2
    data_slice1 = data.iloc[:mid_point]
    data_slice2 = data.iloc[mid_point:]
    slice_time = time.time() - start_time
    print(f"✓ Pandas slicing: {slice_time:.4f} seconds")
    
    return {
        'correlation': corr_time,
        'returns': returns_time,
        'slicing': slice_time
    }

def test_new_multitimeseries(data):
    """Test the new MultiTimeSeries implementation"""
    print("\nTesting New MultiTimeSeries Implementation:")
    print("-" * 50)
    
    try:
        # Import with proper error handling
        try:
            from timeseries import TimeSeries, TimeSeriesMetadata
            from multitimeseries import MultiTimeSeries
        except ImportError as e:
            print(f"Import error: {e}")
            return None
        
        # Test creation
        print("Creating MultiTimeSeries...")
        start_time = time.time()
        mts = MultiTimeSeries(data)
        creation_time = time.time() - start_time
        print(f"✓ MultiTimeSeries creation: {creation_time:.4f} seconds")
        print(f"✓ Result shape: {mts.shape}")
        
        # Test slice operation
        print("Testing slice operation...")
        start_time = time.time()
        mts_is, mts_os = mts.slice(0.5)
        slice_time = time.time() - start_time
        print(f"✓ MultiTimeSeries slice: {slice_time:.4f} seconds")
        print(f"✓ In-sample shape: {mts_is.shape}")
        print(f"✓ Out-sample shape: {mts_os.shape}")
        
        # Test correlation
        print("Testing correlation...")
        start_time = time.time()
        corr = mts.correlation()
        corr_time = time.time() - start_time
        print(f"✓ MultiTimeSeries correlation: {corr_time:.4f} seconds")
        print(f"✓ Correlation shape: {corr.shape}")
        
        return {
            'creation': creation_time,
            'slice': slice_time,
            'correlation': corr_time,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ MultiTimeSeries test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def analyze_performance_bottlenecks(pandas_results, mts_results):
    """Analyze where the performance bottlenecks are"""
    print("\nPerformance Bottleneck Analysis:")
    print("=" * 50)
    
    if not mts_results or not mts_results.get('success', False):
        print("Cannot analyze - MultiTimeSeries failed")
        return
    
    print(f"MultiTimeSeries creation overhead: {mts_results['creation']:.4f} seconds")
    print(f"This includes:")
    print(f"  - DataFrame processing")
    print(f"  - Individual TimeSeries creation")
    print(f"  - Data alignment")
    print(f"  - Metadata creation")
    print()
    
    # Compare correlation performance
    pandas_corr = pandas_results['correlation']
    mts_corr = mts_results['correlation']
    corr_ratio = mts_corr / pandas_corr if pandas_corr > 0 else float('inf')
    
    print(f"Correlation Performance:")
    print(f"  Raw pandas: {pandas_corr:.4f}s")
    print(f"  MultiTimeSeries: {mts_corr:.4f}s")
    print(f"  Overhead: {corr_ratio:.2f}x")
    
    if corr_ratio > 2:
        print(f"  ⚠️  SIGNIFICANT OVERHEAD in correlation method")
    
    # Compare slicing performance  
    pandas_slice = pandas_results['slicing']
    mts_slice = mts_results['slice']
    slice_ratio = mts_slice / pandas_slice if pandas_slice > 0 else float('inf')
    
    print(f"\nSlicing Performance:")
    print(f"  Raw pandas: {pandas_slice:.4f}s")
    print(f"  MultiTimeSeries: {mts_slice:.4f}s")
    print(f"  Overhead: {slice_ratio:.2f}x")
    
    if slice_ratio > 5:
        print(f"  ⚠️  MAJOR OVERHEAD in slice method")

if __name__ == "__main__":
    print("MultiTimeSeries Performance Analysis")
    print("=" * 60)
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        exit(1)
    
    # Test basic pandas operations
    pandas_results = test_original_pandas_operations(data)
    
    # Test new MultiTimeSeries
    mts_results = test_new_multitimeseries(data)
    
    # Analyze performance
    analyze_performance_bottlenecks(pandas_results, mts_results)
