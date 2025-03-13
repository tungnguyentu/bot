import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self, api_key, api_secret, symbol='BTCUSDT', interval='1h'):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.interval = interval
        self.timeframe_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
    def get_historical_data(self, lookback='500 days ago UTC'):
        """Fetch historical klines/candlestick data"""
        try:
            logger.info(f"Fetching historical data for {self.symbol} at {self.interval} interval")
            klines = self.client.get_historical_klines(
                self.symbol, 
                self.timeframe_map[self.interval], 
                lookback
            )
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_latest_data(self, limit=100):
        """Get the latest candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe_map[self.interval],
                limit=limit
            )
            
            # Process data similar to historical data
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error fetching latest data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bollinger_middle'] = df['close'].rolling(window=20).mean()
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * 2)
        df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # VWAP
        df['vwap'] = (df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)).cumsum() / df['volume'].cumsum()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = true_range
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr.ewm(alpha=1/14, adjust=False).mean())
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr.ewm(alpha=1/14, adjust=False).mean()))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
