from finpie.data.timeseries import TimeSeries
from abc import ABC, abstractmethod

class Indicator(ABC):
    def __init__(self, timeseries: TimeSeries, column: str = 'close'):
        self.timeseries = timeseries
        self.column = column
    
    @abstractmethod
    def calculate(self) -> TimeSeries:
        pass