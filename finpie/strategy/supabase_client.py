from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
from finpie.strategy.strategy import StrategyParams, StrategyData, Fill

class SupabaseClient:
    """
    Handles database operations using Supabase.
    
    Attributes:
        client (Client): Supabase client instance for database operations
    """
    
    def __init__(self, url: str, key: str):
        """
        Initialize the Supabase client.
        
        Args:
            url (str): Supabase project URL
            key (str): Supabase API key
        """
        self.client: Client = create_client(url, key)

    def get_strategy_params(self, name: Optional[str] = None, instance: Optional[str] = None) -> List[StrategyParams]:
        """
        Fetch StrategyParams from the database.
        If name is provided, fetch only the matching strategy.
        """
        query = self.client.table("strategy_params").select("*")
        if name:
            query = query.eq("name", name)
        if instance:
            query = query.eq("instance", instance)
        response = query.execute()
        results = response.data if hasattr(response, "data") else response
        return [StrategyParams(**item) for item in results]

    def upsert_strategy_params(self, params: List[StrategyParams]):
        """
        Bulk upsert StrategyParams into the database.
        """
        data = [vars(p) for p in params]
        if len(data) > 0:
            self.client.table("strategy_params").upsert(data).execute()

    def upsert_strategy_data(self, data_list: List[StrategyData]):
        """
        Bulk upsert StrategyData into the database.
        Accepts datetime objects for the 'time' field and converts them to ISO format for the database.
        """
        data = []
        for d in data_list:
            d_dict = vars(d).copy()
            if isinstance(d_dict.get("time"), datetime):
                d_dict["time"] = d_dict["time"].isoformat()
            data.append(d_dict)
        if len(data) > 0:
            self.client.table("strategy_data").upsert(data).execute()

    def get_strategy_data(self, name: Optional[str] = None) -> List[StrategyData]:
        """
        Fetch StrategyData from the database.
        If name is provided, fetch only the matching strategy data.
        Converts 'time' field to datetime if it is a string.
        """
        query = self.client.table("strategy_data").select("*")
        if name:
            query = query.eq("name", name)
        response = query.execute()
        results = response.data if hasattr(response, "data") else response
        strategy_data_list = []
        for item in results:
            if isinstance(item.get("time"), str):
                try:
                    item["time"] = datetime.fromisoformat(item["time"])
                except Exception:
                    pass  # Leave as is if conversion fails
            strategy_data_list.append(StrategyData(**item))
        return strategy_data_list

    def get_fills(self, name: Optional[str] = None) -> List[Fill]:
        """
        Fetch Fill records from the database.
        If name is provided, fetch only the matching fills.
        """
        query = self.client.table("fills").select("*")
        if name:
            query = query.eq("name", name)
        response = query.execute()
        results = response.data if hasattr(response, "data") else response
        # Convert 'time' field to datetime if needed
        fills = []
        for item in results:
            if isinstance(item.get("time"), str):
                item["time"] = datetime.fromisoformat(item["time"])
            fills.append(Fill(**item))
        return fills

    def upsert_fills(self, fills: List[Fill]):
        """
        Bulk upsert Fill records into the database.
        """
        data = []
        for f in fills:
            d = vars(f).copy()
            # Convert datetime to ISO string for DB
            if isinstance(d["time"], datetime):
                d["time"] = d["time"].isoformat()
            data.append(d)
        if len(data) > 0:
            self.client.table("fills").upsert(data).execute()
    
    