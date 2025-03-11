"""
Utility functions for the AI Trading Bot.
"""

import os
import time
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
def setup_logger(log_level="INFO"):
    """
    Set up logger for the trading bot.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(getattr(logging, log_level))
    
    # Create file handler
    log_filename = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def convert_to_dataframe(klines):
    """
    Convert Binance klines data to pandas DataFrame.
    
    Args:
        klines (list): List of klines from Binance API or CCXT
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Check if data is in CCXT format (6 columns) or Binance API format (12 columns)
    if len(klines) > 0 and len(klines[0]) == 6:
        # CCXT format: [timestamp, open, high, low, close, volume]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        # Binance API format
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df


def calculate_position_size(balance, risk_percent, entry_price, stop_loss_price):
    """
    Calculate position size based on risk management.
    
    Args:
        balance (float): Account balance
        risk_percent (float): Percentage of balance to risk (e.g., 0.01 for 1%)
        entry_price (float): Entry price
        stop_loss_price (float): Stop loss price
        
    Returns:
        float: Position size in quote currency
    """
    # Calculate risk amount
    risk_amount = balance * risk_percent
    
    # Calculate price difference
    price_diff = abs(entry_price - stop_loss_price)
    
    # Calculate position size
    if price_diff == 0:
        return 0
    
    position_size = risk_amount / price_diff
    
    return position_size


def calculate_take_profit_price(entry_price, take_profit_percent, position_type):
    """
    Calculate take profit price.
    
    Args:
        entry_price (float): Entry price
        take_profit_percent (float): Take profit percentage (e.g., 0.01 for 1%)
        position_type (str): Position type ('long' or 'short')
        
    Returns:
        float: Take profit price
    """
    if position_type == 'long':
        return entry_price * (1 + take_profit_percent)
    else:  # short
        return entry_price * (1 - take_profit_percent)


def calculate_stop_loss_price(entry_price, stop_loss_percent, position_type):
    """
    Calculate stop loss price.
    
    Args:
        entry_price (float): Entry price
        stop_loss_percent (float): Stop loss percentage (e.g., 0.01 for 1%)
        position_type (str): Position type ('long' or 'short')
        
    Returns:
        float: Stop loss price
    """
    if position_type == 'long':
        return entry_price * (1 - stop_loss_percent)
    else:  # short
        return entry_price * (1 + stop_loss_percent)


def calculate_atr_stop_loss(entry_price, atr_value, atr_multiplier, position_type):
    """
    Calculate ATR-based stop loss price.
    
    Args:
        entry_price (float): Entry price
        atr_value (float): ATR value
        atr_multiplier (float): ATR multiplier
        position_type (str): Position type ('long' or 'short')
        
    Returns:
        float: ATR-based stop loss price
    """
    atr_distance = atr_value * atr_multiplier
    
    if position_type == 'long':
        return entry_price - atr_distance
    else:  # short
        return entry_price + atr_distance


def save_trade_history(trade_data, filename='trade_history.json'):
    """
    Save trade history to JSON file.
    
    Args:
        trade_data (dict): Trade data
        filename (str): Filename to save trade history
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filepath = os.path.join('data', filename)
    
    # Load existing trade history if file exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            trade_history = json.load(f)
    else:
        trade_history = []
    
    # Add new trade data
    trade_history.append(trade_data)
    
    # Save updated trade history
    with open(filepath, 'w') as f:
        json.dump(trade_history, f, indent=4)


def rate_limit_handler(max_requests, time_window=60, buffer=0.8):
    """
    Rate limit handler to avoid API rate limits.
    
    Args:
        max_requests (int): Maximum number of requests allowed
        time_window (int): Time window in seconds
        buffer (float): Buffer to stay below the limit (e.g., 0.8 for 80%)
        
    Returns:
        function: Decorator function
    """
    request_timestamps = []
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal request_timestamps
            
            # Remove timestamps older than time_window
            current_time = time.time()
            request_timestamps = [ts for ts in request_timestamps if current_time - ts <= time_window]
            
            # Check if we're approaching the rate limit
            effective_limit = max_requests * buffer
            if len(request_timestamps) >= effective_limit:
                # Calculate time to wait
                oldest_timestamp = min(request_timestamps)
                wait_time = time_window - (current_time - oldest_timestamp)
                
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Add current timestamp
            request_timestamps.append(time.time())
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator 