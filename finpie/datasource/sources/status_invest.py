from typing import Optional
import pandas as pd
import requests
import time
from datetime import datetime
from finpie.datasource.sources.schemas.status_invest import FundamentalsParams
import logging

logger = logging.getLogger(__name__)

FUNDAMENTALS_URL = "https://statusinvest.com.br/category/advancedsearchresultpaginated"
MIN_REQUEST_INTERVAL = 2  # Minimum seconds between requests

class StatusInvestSource:
    def __init__(self):
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://statusinvest.com.br",
            "Referer": "https://statusinvest.com.br/"
        })

    def _respect_rate_limit(self):
        """Ensure minimum time between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last_request)
        self.last_request_time = time.time()

    def get_fundamentals_table(self, params: FundamentalsParams) -> pd.DataFrame:
        """
        Fetch fundamentals data based on query parameters and return as DataFrame.
        Includes rate limiting and error handling.
        
        Args:
            params: FundamentalsParams object containing search criteria
            
        Returns:
            pd.DataFrame containing the fundamentals data
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response cannot be parsed
        """
        try:
            self._respect_rate_limit()
            
            search_params = params.to_query_params()

            # Print final URL for troubleshooting
            final_url = f"{FUNDAMENTALS_URL}?search={search_params}&page={params.page}&take={params.items_per_page}&CategoryType={params.category_type}"
            final_url = final_url.replace("None", "null").replace(" ","")

            logger.info(f"Making request to: {final_url}")
            # Make request
            response = self.session.get(final_url)            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', MIN_REQUEST_INTERVAL))
                time.sleep(retry_after)
                return self.get_fundamentals_table(params)
            
            # Handle other errors
            response.raise_for_status()
            
            # Convert response to DataFrame
            data = response.json()

            if not data:
                raise ValueError("Empty response received")
            
            if data['totalResults'] > params.items_per_page:
                logger.warning(f"Total results ({data['totalResults']}) exceed the items per page ({params.items_per_page}). Some results may be truncated.")
            
            return pd.DataFrame(data['list'])
            
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"Failed to fetch data from Status Invest: {str(e)}\n"
                f"URL: {response.url}\n"
                f"Status Code: {response.status_code}\n"
                f"Response: {response.text[:500]}"  # Limit response text in error message
            )
        except ValueError as e:
            raise ValueError(f"Failed to parse response: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def get_dividends_history(self, ticker: str, by_year: bool = False) -> pd.DataFrame:
        """
        Fetch dividends history for a given ticker.
        
        Args:
            ticker: The ticker symbol of the stock
            by_year: If True, return the dividends by year, otherwise return the dividends as paid
        """
        try:
            self._respect_rate_limit()

            url = f"https://statusinvest.com.br/acao/companytickerprovents?ticker={ticker}&chartProventsType=2"

            logger.info(f"Making request to: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            if not data:
                raise ValueError("Empty response received")
            
            if by_year:
                return pd.DataFrame(data['assetEarningsYearlyModels'])
            else:
                return pd.DataFrame(data['assetEarningsModels'])
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch dividends history: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Failed to parse dividends history: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def get_revenue_history(self, ticker: str, by_year: bool = False) -> pd.DataFrame:
        """
        Fetch revenue history for a given ticker.
        
        Args:
            ticker: The ticker symbol of the stock
            by_year: If True, return the revenue by year, otherwise return the revenue by quarter
        """
        try:
            self._respect_rate_limit()

            view_type = 0 if by_year else 1

            url = f"https://statusinvest.com.br/acao/getrevenue?code={ticker}&type=2&viewType={view_type}"

            logger.info(f"Making request to: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            if not data:
                raise ValueError("Empty response received")
            
            return pd.DataFrame(data)     
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch revenue history: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Failed to parse revenue history: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def get_margins_history(self, ticker: str) -> pd.DataFrame:
        """
        Fetch margins (%) history for a given ticker.
        
        Args:
            ticker: The ticker symbol of the stock
        """
        
        try:
            self._respect_rate_limit()
            
            url = f"https://statusinvest.com.br/acao/getmargins?code={ticker}&type=2"

            logger.info(f"Making request to: {url}")    

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            return pd.DataFrame(data)
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch margins history: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Failed to parse margins history: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def get_balance_sheet_history(self, ticker: str) -> pd.DataFrame:
        """
        Fetch balance sheet history for a given ticker.
        
        Args:
            ticker: The ticker symbol of the stock
        """
        try:
            self._respect_rate_limit()
            
            url = f"https://statusinvest.com.br/acao/getbsactivepassivechart?code={ticker}&type=2"

            logger.info(f"Making request to: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            return pd.DataFrame(data)
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch balance sheet history: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Failed to parse balance sheet history: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")