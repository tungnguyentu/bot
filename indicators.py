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
        data (pd.DataFrame): DataFrame with 'close' price column
        period (int): RSI period

    Returns:
        pd.Series: RSI values
    """
    rsi_indicator = RSIIndicator(close=data["close"], window=period)
    return rsi_indicator.rsi()


def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns
        period (int): ATR period

    Returns:
        pd.Series: ATR values
    """
    atr_indicator = AverageTrueRange(
        high=data["high"], low=data["low"], close=data["close"], window=period
    )
    return atr_indicator.average_true_range()


def calculate_vwap(data, period=14):
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', 'close', 'volume' columns
        period (int): VWAP period

    Returns:
        pd.Series: VWAP values
    """
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    vwap = (typical_price * data["volume"]).rolling(window=period).sum() / data[
        "volume"
    ].rolling(window=period).sum()
    return vwap


def detect_volume_spike(data, threshold=1.5, period=20):
    """
    Detect volume spikes.

    Args:
        data (pd.DataFrame): DataFrame with 'volume' column
        threshold (float): Threshold for volume spike (e.g., 1.5 = 50% above average)
        period (int): Period for calculating average volume

    Returns:
        pd.Series: Boolean series indicating volume spikes
    """
    avg_volume = data["volume"].rolling(window=period).mean()
    return data["volume"] > (avg_volume * threshold)


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
        data (pd.DataFrame): DataFrame with 'close' price column
        period (int): Period for moving average
        std_dev (int): Number of standard deviations

    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    middle_band = data["close"].rolling(window=period).mean()
    std = data["close"].rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return middle_band, upper_band, lower_band


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        data (pd.DataFrame): DataFrame with 'close' price column
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    fast_ema = data["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data["close"].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
