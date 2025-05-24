Examples
========

This section contains examples of how to use FinPie.

Basic Usage
----------

.. code-block:: python

    from finpie import TimeSeries, YahooFinanceSource

    # Create a data source
    yahoo = YahooFinanceSource()

    # Get historical data
    data = yahoo.get_historical_data("AAPL", "2020-01-01", "2021-01-01")

    # Create a TimeSeries object
    ts = TimeSeries(data)

    # Calculate returns
    returns = ts.returns()

    # Plot the data
    ts.plot()

Advanced Usage
-------------

.. code-block:: python

    from finpie import MultiTimeSeries, RatioTimeSeries, SpreadTimeSeries

    # Create multiple time series
    mts = MultiTimeSeries([ts1, ts2, ts3])

    # Calculate correlation matrix
    corr = mts.correlation()

    # Create a ratio time series
    ratio = RatioTimeSeries(ts1, ts2)

    # Create a spread time series
    spread = SpreadTimeSeries(ts1, ts2) 