import os
import logging
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from .config import BotConfig

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """Collects data from Binance for the trading bot."""
    
    def __init__(self, config: BotConfig):
        """Initialize data collector with bot configuration."""
        self.config = config
        self.client = self._initialize_client()
    
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
    
    def get_latest_data(self, lookback=100):
        """Get the latest market data."""
        return self.get_historical_data(limit=lookback)
