"""
Performance comparison test between old and new MultiTimeSeries implementations.
"""

import pandas as pd
import time
import sys
import os

# Add paths for both implementations
sys.path.append('../data')  # Original implementation
sys.path.append('../data_new')  # New implementation

def test_original_implementation():
    """Test performance of original MultiTimeSeries"""
    print("Testing Original MultiTimeSeries Implementation:")
    print("-" * 50)
    
    try:
        # Import original implementation
        from finpie.data import MultiTimeSeries as OriginalMTS
        
        # Load data
        data = pd.read_parquet('mts2.parquet')
        print(f"Data shape: {data.shape}")
        
        # Time creation
        start_time = time.time()
        mts_original = OriginalMTS(data)
        creation_time = time.time() - start_time
        print(f"✓ Creation time: {creation_time:.4f} seconds")
        
        # Time slice operation
        start_time = time.time()
        mts_is, mts_os = mts_original.slice(0.5)
        slice_time = time.time() - start_time
        print(f"✓ Slice time: {slice_time:.4f} seconds")
        
        # Time correlation
        start_time = time.time()
        corr = mts_original.correlation()
        corr_time = time.time() - start_time
        print(f"✓ Correlation time: {corr_time:.4f} seconds")
        print(f"✓ Correlation shape: {corr.shape}")
        
        return {
            'creation': creation_time,
            'slice': slice_time,
            'correlation': corr_time,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ Original implementation failed: {e}")
        return {'success': False, 'error': str(e)}

def test_new_implementation():
    """Test performance of new MultiTimeSeries"""
    print("\nTesting New MultiTimeSeries Implementation:")
    print("-" * 50)
    
    try:
        # Import new implementation
        from multitimeseries import MultiTimeSeries as NewMTS
        
        # Load data
        data = pd.read_parquet('mts2.parquet')
        print(f"Data shape: {data.shape}")
        
        # Time creation
        start_time = time.time()
        mts_new = NewMTS(data)
        creation_time = time.time() - start_time
        print(f"✓ Creation time: {creation_time:.4f} seconds")
        
        # Time slice operation
        start_time = time.time()
        mts_is, mts_os = mts_new.slice(0.5)
        slice_time = time.time() - start_time
        print(f"✓ Slice time: {slice_time:.4f} seconds")
        
        # Time correlation
        start_time = time.time()
        corr = mts_new.correlation()
        corr_time = time.time() - start_time
        print(f"✓ Correlation time: {corr_time:.4f} seconds")
        print(f"✓ Correlation shape: {corr.shape}")
        
        return {
            'creation': creation_time,
            'slice': slice_time,
            'correlation': corr_time,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ New implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def compare_performance(original_results, new_results):
    """Compare performance between implementations"""
    print("\nPerformance Comparison:")
    print("=" * 50)
    
    if not original_results['success'] or not new_results['success']:
        print("Cannot compare - one or both implementations failed")
        return
    
    operations = ['creation', 'slice', 'correlation']
    
    for op in operations:
        original_time = original_results[op]
        new_time = new_results[op]
        
        if original_time > 0:
            ratio = new_time / original_time
            change = ((new_time - original_time) / original_time) * 100
            
            print(f"{op.capitalize()}:")
            print(f"  Original: {original_time:.4f}s")
            print(f"  New:      {new_time:.4f}s")
            print(f"  Ratio:    {ratio:.2f}x")
            print(f"  Change:   {change:+.1f}%")
            
            if ratio > 1.5:
                print(f"  ⚠️  NEW IS {ratio:.1f}x SLOWER")
            elif ratio < 0.7:
                print(f"  ✓ New is {1/ratio:.1f}x faster")
            else:
                print(f"  ≈ Similar performance")
            print()

if __name__ == "__main__":
    print("MultiTimeSeries Performance Analysis")
    print("=" * 60)
    
    # Check if parquet file exists
    if not os.path.exists('mts2.parquet'):
        print("Error: mts2.parquet not found in current directory")
        sys.exit(1)
    
    # Test original implementation
    original_results = test_original_implementation()
    
    # Test new implementation
    new_results = test_new_implementation()
    
    # Compare results
    compare_performance(original_results, new_results)
