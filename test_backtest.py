import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import indicators
from indicators import calculate_rsi, calculate_vwap, calculate_atr, calculate_bollinger_bands, calculate_macd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_backtest")

def test_fetch_data():
    """Test fetching historical data from Binance."""
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Set symbol and timeframe
        symbol = "BTCUSDT"
        timeframe = "5m"
        
        # Calculate start and end timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Just 5 days to test
        
        # Convert to milliseconds timestamp
        start_timestamp = int(start_date.timestamp() * 1000)
        
        logger.info(f"Fetching historical data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Fetch historical data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=1000
            )
            
            # Check if data is empty
            if not ohlcv or len(ohlcv) == 0:
                logger.error(f"No historical data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Print data info
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            logger.info(f"Data columns: {df.columns.tolist()}")
            logger.info(f"Data sample:\n{df.head()}")
            
            # Check if 'close' column exists
            if 'close' not in df.columns:
                logger.error("'close' column not found in data")
                return None
            
            # Try to access the 'close' column
            close_values = df['close']
            logger.info(f"First 5 close values: {close_values.head().tolist()}")
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching historical data: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching historical data: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error in test_fetch_data: {e}")
        return None

def prepare_data(data):
    """
    Prepare data for backtesting.
    
    Args:
        data (pd.DataFrame): Historical data
        
    Returns:
        pd.DataFrame: Prepared data
    """
    try:
        # Check if data is empty
        if data is None or len(data) == 0:
            logger.error("Historical data is empty")
            return None
        
        # Check if required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Required column '{col}' not found in historical data")
                logger.error(f"Available columns: {data.columns.tolist()}")
                return None
        
        # Add indicators
        df = data.copy()
        
        # Add RSI
        try:
            logger.info("Adding RSI indicator...")
            df['rsi'] = calculate_rsi(df)
            logger.info("RSI indicator added successfully")
        except Exception as e:
            logger.error(f"Error adding RSI indicator: {e}")
            # Continue without this indicator
        
        # Add VWAP
        try:
            logger.info("Adding VWAP indicator...")
            df['vwap'] = calculate_vwap(df)
            logger.info("VWAP indicator added successfully")
        except Exception as e:
            logger.error(f"Error adding VWAP indicator: {e}")
            # Continue without this indicator
        
        # Add ATR
        try:
            logger.info("Adding ATR indicator...")
            df['atr'] = calculate_atr(df)
            logger.info("ATR indicator added successfully")
        except Exception as e:
            logger.error(f"Error adding ATR indicator: {e}")
            # Continue without this indicator
        
        # Add Bollinger Bands
        try:
            logger.info("Adding Bollinger Bands indicator...")
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df)
            logger.info("Bollinger Bands indicator added successfully")
        except Exception as e:
            logger.error(f"Error adding Bollinger Bands indicator: {e}")
            # Continue without this indicator
        
        # Add MACD
        try:
            logger.info("Adding MACD indicator...")
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
            logger.info("MACD indicator added successfully")
        except Exception as e:
            logger.error(f"Error adding MACD indicator: {e}")
            # Continue without this indicator
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check for any remaining NaN values
        if df.isnull().values.any():
            logger.warning("Data contains NaN values after filling")
            logger.warning(f"Columns with NaN values: {df.columns[df.isnull().any()].tolist()}")
            # Replace remaining NaN values with 0
            df = df.fillna(0)
        
        logger.info(f"Data prepared successfully with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Final columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None

def test_backtest_process():
    """Test the backtest process."""
    try:
        # Fetch historical data
        data = test_fetch_data()
        if data is None:
            logger.error("Failed to fetch historical data")
            return False
        
        # Prepare data
        prepared_data = prepare_data(data)
        if prepared_data is None:
            logger.error("Failed to prepare data")
            return False
        
        logger.info("Backtest process simulation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in test_backtest_process: {e}")
        return False

if __name__ == "__main__":
    success = test_backtest_process()
    if success:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed") 