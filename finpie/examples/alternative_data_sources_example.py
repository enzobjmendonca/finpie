"""
Example demonstrating alternative data sources for minute-level historical data.

This example shows how to use various data sources that overcome the limitations
of Yahoo Finance (30-day limit) and MT5 (requires terminal running).
"""

from finpie.datasource.service import DataService
from finpie.datasource.sources.twelve_data import TwelveDataSource
from finpie.datasource.sources.polygon_io import PolygonIOSource
from finpie.datasource.sources.iex_cloud import IEXCloudSource
from datetime import datetime, timedelta

def example_twelve_data():
    """Example using Twelve Data API"""
    print("=== Twelve Data Example ===")
    
    # Initialize with your API key
    api_key = "your_twelve_data_api_key"
    twelve_data = TwelveDataSource(api_key)
    
    # Get 1-minute data for the last 5 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    try:
        # Get minute-level data
        ts = twelve_data.get_prices(
            symbol="AAPL",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval="1m",
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        print(f"Fetched {len(ts.data)} minutes of data for AAPL")
        print(f"Date range: {ts.metadata.start_date} to {ts.metadata.end_date}")
        print(f"Latest prices:\n{ts.data.tail()}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_polygon_io():
    """Example using Polygon.io API"""
    print("\n=== Polygon.io Example ===")
    
    # Initialize with your API key
    api_key = "your_polygon_io_api_key"
    polygon = PolygonIOSource(api_key)
    
    # Get 1-minute data for the last 2 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    try:
        # Get minute-level data
        ts = polygon.get_prices(
            symbol="MSFT",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval="1m",
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        print(f"Fetched {len(ts.data)} minutes of data for MSFT")
        print(f"Date range: {ts.metadata.start_date} to {ts.metadata.end_date}")
        print(f"Sample data:\n{ts.data.head()}")
        
        # Get real-time quote
        quote = polygon.get_real_time_quote("MSFT")
        print(f"Real-time quote: {quote}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_iex_cloud():
    """Example using IEX Cloud API"""
    print("\n=== IEX Cloud Example ===")
    
    # Initialize with your API key
    api_key = "your_iex_cloud_api_key"
    iex = IEXCloudSource(api_key)
    
    try:
        # Get recent intraday data (last few days)
        ts = iex.get_prices(
            symbol="GOOGL",
            interval="1m",
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        print(f"Fetched {len(ts.data)} minutes of data for GOOGL")
        print(f"Date range: {ts.metadata.start_date} to {ts.metadata.end_date}")
        print(f"Sample data:\n{ts.data.tail()}")
        
        # Get company information
        metadata = iex.get_metadata("GOOGL")
        print(f"Company: {metadata.get('companyName')}")
        print(f"Sector: {metadata.get('sector')}")
        print(f"Market Cap: ${metadata.get('marketcap', 0):,}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_data_service_with_alternatives():
    """Example using DataService with multiple alternative sources"""
    print("\n=== DataService with Alternatives ===")
    
    # Create service with multiple API keys
    service = DataService.create_default_service(
        alpha_vantage_key="your_alpha_vantage_key",
        twelve_data_key="your_twelve_data_key",
        polygon_io_key="your_polygon_io_key",
        iex_cloud_key="your_iex_cloud_key"
    )
    
    print("Available data sources:")
    for source_name in service.list_sources():
        print(f"  - {source_name}")
    
    # Compare data from different sources
    symbol = "AAPL"
    interval = "1d"  # Use daily for compatibility
    
    for source in ['yahoo_finance', 'twelve_data', 'polygon_io', 'iex_cloud']:
        try:
            ts = service.get_close_prices(
                symbol=symbol,
                source=source,
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                interval=interval
            )
            
            print(f"\n{source}: {len(ts.data)} days of data")
            print(f"Latest close: ${ts.data.iloc[-1]['close']:.2f}")
            
        except Exception as e:
            print(f"{source}: Error - {e}")

def comparison_summary():
    """Summary of data source capabilities"""
    print("\n=== Data Source Comparison ===")
    
    sources = {
        "Yahoo Finance": {
            "1-minute data": "30 days only",
            "Cost": "Free",
            "Rate limits": "Moderate",
            "Global coverage": "Excellent"
        },
        "Alpha Vantage": {
            "1-minute data": "Recent data",
            "Cost": "Free tier available",
            "Rate limits": "5 calls/min, 500/day",
            "Global coverage": "Good"
        },
        "Twelve Data": {
            "1-minute data": "Historical data available",
            "Cost": "$8/month+",
            "Rate limits": "800 calls/day free",
            "Global coverage": "Excellent"
        },
        "Polygon.io": {
            "1-minute data": "2004+ for US stocks",
            "Cost": "$99/month+",
            "Rate limits": "High limits",
            "Global coverage": "US-focused"
        },
        "IEX Cloud": {
            "1-minute data": "Recent intraday",
            "Cost": "Generous free tier",
            "Rate limits": "500k credits/month",
            "Global coverage": "US stocks only"
        },
        "MT5": {
            "1-minute data": "Extensive historical",
            "Cost": "Free with broker",
            "Rate limits": "None",
            "Global coverage": "Broker-dependent"
        }
    }
    
    for source, features in sources.items():
        print(f"\n{source}:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")

if __name__ == "__main__":
    print("Alternative Data Sources for Minute-Level Historical Data")
    print("=" * 60)
    
    # Note: Replace API keys with actual keys to run examples
    print("Note: Replace API keys with actual keys to run these examples\n")
    
    # Run examples (commented out to avoid API errors without keys)
    # example_twelve_data()
    # example_polygon_io()
    # example_iex_cloud()
    # example_data_service_with_alternatives()
    
    # Show comparison
    comparison_summary()
    
    print("\n=== Recommendations ===")
    print("1. For budget-conscious users: IEX Cloud (generous free tier)")
    print("2. For comprehensive data: Twelve Data (good value at $8/month)")
    print("3. For professional use: Polygon.io (premium but comprehensive)")
    print("4. For global markets: Yahoo Finance + Twelve Data combination")
    print("5. For forex: Keep MT5 but add TraderMade or similar for automation")

