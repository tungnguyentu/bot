"""
Technical indicators for the AI Trading Bot.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame or pd.Series): DataFrame with 'close' price column or Series of close prices
        period (int): RSI period

    Returns:
        pd.Series: RSI values
    """
    try:
        # Check if data is a Series or DataFrame
        if isinstance(data, pd.Series):
            close_prices = data
        elif isinstance(data, pd.DataFrame) and 'close' in data.columns:
            close_prices = data['close']
        else:
            raise ValueError("Input must be a DataFrame with a 'close' column or a Series of close prices")
        
        rsi_indicator = RSIIndicator(close=close_prices, window=period)
        return rsi_indicator.rsi()
    except Exception as e:
        raise Exception(f"Error calculating RSI: {e}")


def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).

    Args:
        data (pd.DataFrame or dict): DataFrame with 'high', 'low', 'close' columns or dict with these keys
        period (int): ATR period

    Returns:
        pd.Series: ATR values
    """
    try:
        # Check if data is a DataFrame or dict
        if isinstance(data, pd.DataFrame):
            high = data["high"]
            low = data["low"]
            close = data["close"]
        elif isinstance(data, dict) and all(k in data for k in ["high", "low", "close"]):
            high = data["high"]
            low = data["low"]
            close = data["close"]
        else:
            raise ValueError("Input must be a DataFrame or dict with 'high', 'low', 'close' columns/keys")
        
        atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=period)
        return atr_indicator.average_true_range()
    except Exception as e:
        raise Exception(f"Error calculating ATR: {e}")


def calculate_vwap(data, period=14):
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        data (pd.DataFrame or dict): DataFrame with 'high', 'low', 'close', 'volume' columns or dict with these keys
        period (int): VWAP period

    Returns:
        pd.Series: VWAP values
    """
    try:
        # Check if data is a DataFrame or dict
        if isinstance(data, pd.DataFrame):
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
        elif isinstance(data, dict) and all(k in data for k in ["high", "low", "close", "volume"]):
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
        else:
            raise ValueError("Input must be a DataFrame or dict with 'high', 'low', 'close', 'volume' columns/keys")
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwap
    except Exception as e:
        raise Exception(f"Error calculating VWAP: {e}")


def detect_volume_spike(data, threshold=1.5, period=20):
    """
    Detect volume spikes.

    Args:
        data (pd.DataFrame or pd.Series): DataFrame with 'volume' column or Series of volume values
        threshold (float): Threshold for volume spike (e.g., 1.5 = 50% above average)
        period (int): Period for calculating average volume

    Returns:
        pd.Series: Boolean series indicating volume spikes
    """
    try:
        # Check if data is a Series or DataFrame
        if isinstance(data, pd.Series):
            volume = data
        elif isinstance(data, pd.DataFrame) and "volume" in data.columns:
            volume = data["volume"]
        else:
            raise ValueError("Input must be a DataFrame with a 'volume' column or a Series of volume values")
        
        avg_volume = volume.rolling(window=period).mean()
        return volume > (avg_volume * threshold)
    except Exception as e:
        raise Exception(f"Error detecting volume spikes: {e}")


def calculate_order_book_imbalance(order_book, depth=10):
    """
    Calculate order book imbalance.

    Args:
        order_book (dict): Order book data from Binance API
        depth (int): Depth of order book to consider

    Returns:
        float: Order book imbalance ratio (>1 means more buy pressure, <1 means more sell pressure)
    """
    bids = order_book["bids"][:depth]
    asks = order_book["asks"][:depth]

    bid_volume = sum(float(bid[1]) for bid in bids)
    ask_volume = sum(float(ask[1]) for ask in asks)

    if ask_volume == 0:
        return float("inf")

    return bid_volume / ask_volume


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Args:
        data (pd.DataFrame or pd.Series): DataFrame with 'close' price column or Series of close prices
        period (int): Period for moving average
        std_dev (int): Number of standard deviations

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    try:
        # Check if data is a Series or DataFrame
        if isinstance(data, pd.Series):
            close_prices = data
        elif isinstance(data, pd.DataFrame) and "close" in data.columns:
            close_prices = data["close"]
        else:
            raise ValueError("Input must be a DataFrame with a 'close' column or a Series of close prices")
        
        middle_band = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band
    except Exception as e:
        raise Exception(f"Error calculating Bollinger Bands: {e}")


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        data (pd.DataFrame or pd.Series): DataFrame with 'close' price column or Series of close prices
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    try:
        # Check if data is a Series or DataFrame
        if isinstance(data, pd.Series):
            close_prices = data
        elif isinstance(data, pd.DataFrame) and "close" in data.columns:
            close_prices = data["close"]
        else:
            raise ValueError("Input must be a DataFrame with a 'close' column or a Series of close prices")
        
        fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
    except Exception as e:
        raise Exception(f"Error calculating MACD: {e}")
