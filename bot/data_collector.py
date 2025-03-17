import os
import logging
import pandas as pd
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests
from urllib3.exceptions import HTTPError
import http.client
from functools import wraps
from .config import BotConfig

logger = logging.getLogger(__name__)

def retry_on_connection_error(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator to retry functions on connection errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (BinanceAPIException, requests.exceptions.RequestException, 
                        http.client.RemoteDisconnected, ConnectionError, HTTPError) as e:
                    last_exception = e
                    
                    if retry < max_retries:
                        wait_time = delay * (backoff_factor ** retry)
                        logger.warning(f"Connection error: {e}. Retrying in {wait_time:.2f}s ({retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise last_exception
            
            return None  # This line should not be reached
        return wrapper
    return decorator

class BinanceDataCollector:
    """Collects data from Binance for the trading bot."""
    
    def __init__(self, config: BotConfig):
        """Initialize data collector with bot configuration."""
        self.config = config
        self.client = self._initialize_client()
    
    @retry_on_connection_error()
    def _initialize_client(self):
        """Initialize Binance client based on the trading mode."""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Use testnet credentials for test mode
        if self.config.mode == 'test':
            api_key = os.getenv('BINANCE_TESTNET_API_KEY', api_key)
            api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', api_secret)
            return Client(api_key, api_secret, testnet=True)
        
        # Use regular API for live or backtest
        return Client(api_key, api_secret)
    
    @retry_on_connection_error(max_retries=5, initial_delay=2)
    def get_historical_data(self, limit=500):
        """Get historical klines data from Binance."""
        logger.info(f"Getting historical data for {self.config.symbol} with interval {self.config.interval}")
        
        try:
            klines = self.client.get_klines(
                symbol=self.config.symbol,
                interval=self.config.interval,
                limit=limit
            )
            
            # Convert to dataframe
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} historical data points")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    @retry_on_connection_error(max_retries=5, initial_delay=2)
    def get_latest_data(self, lookback=100):
        """Get the latest market data."""
        return self.get_historical_data(limit=lookback)
