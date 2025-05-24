from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class FundamentalsParams:
    """Parameters for querying stock fundamentals"""
    # Core filters
    # Dividend Yield
    min_dividend_yield: Optional[float] = None
    max_dividend_yield: Optional[float] = None
    # Price/Earnings
    min_pe_ratio: Optional[float] = None
    max_pe_ratio: Optional[float] = None
    # Price/Book Value
    min_pb_ratio: Optional[float] = None
    max_pb_ratio: Optional[float] = None
    # Price/Earnings Growth
    min_peg_ratio: Optional[float] = None
    max_peg_ratio: Optional[float] = None
    # Price/Assets
    min_price_assets: Optional[float] = None
    max_price_assets: Optional[float] = None
    # Gross Margin
    min_gross_margin: Optional[float] = None
    max_gross_margin: Optional[float] = None
    # EBIT Margin
    min_ebit_margin: Optional[float] = None
    max_ebit_margin: Optional[float] = None
    # Net Margin
    min_net_margin: Optional[float] = None
    max_net_margin: Optional[float] = None
    # Price/EBIT
    min_price_ebit: Optional[float] = None
    max_price_ebit: Optional[float] = None
    # EV/EBIT
    min_ev_ebit: Optional[float] = None
    max_ev_ebit: Optional[float] = None
    # Net Debt/EBIT
    min_net_debt_ebit: Optional[float] = None
    max_net_debt_ebit: Optional[float] = None
    # Net Debt/Equity
    min_net_debt_equity: Optional[float] = None
    max_net_debt_equity: Optional[float] = None
    # Price/Sales
    min_price_sales: Optional[float] = None
    max_price_sales: Optional[float] = None
    # Price/Working Capital
    min_price_working_capital: Optional[float] = None
    max_price_working_capital: Optional[float] = None
    # Price/Current Assets
    min_price_current_assets: Optional[float] = None
    max_price_current_assets: Optional[float] = None
    # ROE
    min_roe: Optional[float] = None
    max_roe: Optional[float] = None
    # ROIC
    min_roic: Optional[float] = None
    max_roic: Optional[float] = None
    # ROA
    min_roa: Optional[float] = None
    max_roa: Optional[float] = None
    # Current Liquidity
    min_current_liquidity: Optional[float] = None
    max_current_liquidity: Optional[float] = None
    # Equity/Assets
    min_equity_assets: Optional[float] = None
    max_equity_assets: Optional[float] = None
    # Liabilities/Assets
    min_liabilities_assets: Optional[float] = None
    max_liabilities_assets: Optional[float] = None
    # Asset Turnover
    min_asset_turnover: Optional[float] = None
    max_asset_turnover: Optional[float] = None
    # Revenue CAGR 5Y
    min_revenue_cagr_5y: Optional[float] = None
    max_revenue_cagr_5y: Optional[float] = None
    # Profit CAGR 5Y
    min_profit_cagr_5y: Optional[float] = None
    max_profit_cagr_5y: Optional[float] = None
    # Daily trading volume
    min_daily_liquidity: Optional[float] = None
    max_daily_liquidity: Optional[float] = None
    # Book value per share
    min_book_value_per_share: Optional[float] = None
    max_book_value_per_share: Optional[float] = None
    # Earnings per share
    min_earnings_per_share: Optional[float] = None
    max_earnings_per_share: Optional[float] = None
    # Market value
    min_market_value: Optional[float] = None
    max_market_value: Optional[float] = None

    # Pagination parameters
    page: int = 0
    items_per_page: int = 100
    category_type: int = 1

    def to_query_params(self) -> Dict[str, Any]:
        search_params = {
        "dy": {"Item1": self.min_dividend_yield, "Item2": self.max_dividend_yield},
        "p_l": {"Item1": self.min_pe_ratio, "Item2": self.max_pe_ratio},
        "p_vp": {"Item1": self.min_pb_ratio, "Item2": self.max_pb_ratio},
        "peg_ratio": {"Item1": self.min_peg_ratio, "Item2": self.max_peg_ratio},
        "price_assets": {"Item1": self.min_price_assets, "Item2": self.max_price_assets},
        "gross_margin": {"Item1": self.min_gross_margin, "Item2": self.max_gross_margin},
        "ebit_margin": {"Item1": self.min_ebit_margin, "Item2": self.max_ebit_margin},
        "net_margin": {"Item1": self.min_net_margin, "Item2": self.max_net_margin},
        "price_ebit": {"Item1": self.min_price_ebit, "Item2": self.max_price_ebit}, 
        "ev_ebit": {"Item1": self.min_ev_ebit, "Item2": self.max_ev_ebit},
        "net_debt_ebit": {"Item1": self.min_net_debt_ebit, "Item2": self.max_net_debt_ebit},
        "net_debt_equity": {"Item1": self.min_net_debt_equity, "Item2": self.max_net_debt_equity},
        "price_sales": {"Item1": self.min_price_sales, "Item2": self.max_price_sales},
        "price_working_capital": {"Item1": self.min_price_working_capital, "Item2": self.max_price_working_capital},
        "price_current_assets": {"Item1": self.min_price_current_assets, "Item2": self.max_price_current_assets},
        "roe": {"Item1": self.min_roe, "Item2": self.max_roe},
        "roic": {"Item1": self.min_roic, "Item2": self.max_roic},
        "roa": {"Item1": self.min_roa, "Item2": self.max_roa},
        "current_liquidity": {"Item1": self.min_current_liquidity, "Item2": self.max_current_liquidity},
        "equity_assets": {"Item1": self.min_equity_assets, "Item2": self.max_equity_assets},
        "liabilities_assets": {"Item1": self.min_liabilities_assets, "Item2": self.max_liabilities_assets},
        "asset_turnover": {"Item1": self.min_asset_turnover, "Item2": self.max_asset_turnover},
        "revenue_cagr_5y": {"Item1": self.min_revenue_cagr_5y, "Item2": self.max_revenue_cagr_5y},
        "profit_cagr_5y": {"Item1": self.min_profit_cagr_5y, "Item2": self.max_profit_cagr_5y},
        "daily_liquidity": {"Item1": self.min_daily_liquidity, "Item2": self.max_daily_liquidity},
        "book_value_per_share": {"Item1": self.min_book_value_per_share, "Item2": self.max_book_value_per_share},
        "earnings_per_share": {"Item1": self.min_earnings_per_share, "Item2": self.max_earnings_per_share},
        "market_value": {"Item1": self.min_market_value, "Item2": self.max_market_value}
        }
    
        # Remove parameters where both min and max are None
        search_params = {k: v for k, v in search_params.items() 
                        if v["Item1"] is not None or v["Item2"] is not None}
        
        return search_params