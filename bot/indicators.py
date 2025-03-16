import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""
    
    def calculate_all(self, df):
        """Calculate all technical indicators and add them to the dataframe."""
        if df.empty:
            logger.warning("Empty dataframe, cannot calculate indicators")
            return df
        
        # Create a copy to avoid modifying the original dataframe
        df_with_indicators = df.copy()
        
        # Calculate Bollinger Bands (14, 2)
        df_with_indicators = self.add_bollinger_bands(df_with_indicators, window=14, num_std=2)
        
        # Calculate RSI (6)
        df_with_indicators = self.add_rsi(df_with_indicators, window=6)
        
        # Calculate MACD (5, 13, 1)
        df_with_indicators = self.add_macd(df_with_indicators, fast=5, slow=13, signal=1)
        
        # Calculate VWAP
        df_with_indicators = self.add_vwap(df_with_indicators)
        
        # Calculate ATR (14)
        df_with_indicators = self.add_atr(df_with_indicators, window=14)
        
        # Handle NaN values
        df_with_indicators.dropna(inplace=True)
        
        return df_with_indicators
    
    def add_bollinger_bands(self, df, window=14, num_std=2):
        """Add Bollinger Bands to the dataframe."""
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        df['bb_std'] = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df
    
    def add_rsi(self, df, window=6):
        """Add Relative Strength Index to the dataframe."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RSI using SMA
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def add_macd(self, df, fast=5, slow=13, signal=1):
        """Add MACD to the dataframe."""
        df['ema_fast'] = df['close'].ewm(span=fast).mean()
        df['ema_slow'] = df['close'].ewm(span=slow).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def add_vwap(self, df):
        """Add Volume Weighted Average Price to the dataframe."""
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        # Calculate VWAP
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    def add_atr(self, df, window=14):
        """Add Average True Range to the dataframe."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        df['atr'] = true_range.rolling(window=window).mean()
        return df
