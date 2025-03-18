import pandas as pd
import numpy as np
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, client):
        """Initialize the MarketData class with a Binance client."""
        self.client = client
        
    def get_historical_klines(self, symbol, interval, start_str=None, end_str=None, limit=1000):
        """
        Fetch historical klines (candlestick data) for a given symbol and interval.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1h', '15m')
            start_str: Start time as string
            end_str: End time as string
            limit: Maximum number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Get klines from Binance API
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=self._str_to_timestamp(start_str) if start_str else None,
                    endTime=self._str_to_timestamp(end_str) if end_str else None,
                    limit=limit
                )
                
                # Convert to DataFrame and process
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                return df
                
            except BinanceAPIException as e:
                retry_count += 1
                logger.warning(f"API error: {e}, retrying {retry_count}/{max_retries}")
                time.sleep(2 * retry_count)  # Exponential backoff
            
            except Exception as e:
                logger.error(f"Error fetching market data: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error
                
        logger.error(f"Failed to get historical data after {max_retries} attempts")
        return pd.DataFrame()
    
    def get_recent_trades(self, symbol, limit=100):
        """Get recent trades for a symbol."""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return []

    def get_ticker(self, symbol):
        """Get current ticker information."""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}
            
    def get_order_book(self, symbol, limit=10):
        """Get order book for a symbol."""
        try:
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {"bids": [], "asks": []}
            
    def calculate_volatility(self, symbol, interval='1d', window=14):
        """Calculate price volatility for a symbol."""
        try:
            df = self.get_historical_klines(symbol, interval, limit=window+1)
            if df.empty:
                return 0
                
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = df['returns'].std()
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def _str_to_timestamp(self, time_str):
        """Convert time string to millisecond timestamp."""
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except:
            return None
