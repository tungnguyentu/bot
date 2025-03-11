"""
Backtesting module for the AI Trading Bot.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt
from dotenv import load_dotenv

import config
from indicators import (
    calculate_rsi, 
    calculate_vwap, 
    calculate_atr, 
    detect_volume_spike,
    calculate_bollinger_bands,
    calculate_macd
)
from utils import (
    calculate_take_profit_price,
    calculate_stop_loss_price,
    calculate_atr_stop_loss,
    setup_logger,
    convert_to_dataframe
)

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(config.LOG_LEVEL)


class Backtester:
    """
    Backtester for the trading strategy.
    """
    
    def __init__(self, symbol=None, timeframe=None, start_date=None, end_date=None, strategy_class=None):
        """
        Initialize the backtester.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            strategy_class (class): Strategy class to use
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.start_date = start_date or config.BACKTEST_START_DATE
        self.end_date = end_date or config.BACKTEST_END_DATE
        self.strategy_class = strategy_class
        
        # Initialize exchange for historical data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Initialize strategy
        self.strategy = self.strategy_class(
            binance_client=None,  # Not needed for backtesting
            telegram_notifier=None,  # Not needed for backtesting
            symbol=self.symbol,
            timeframe=self.timeframe,
            leverage=config.LEVERAGE
        )
        
        # Initialize results
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.active_positions = {}
        
        logger.info(f"Backtester initialized for {self.symbol} ({self.timeframe}) from {self.start_date} to {self.end_date}.")
        logger.info(f"Strategy: {self.strategy_class.__name__}")
    
    def fetch_historical_data(self):
        """
        Fetch historical data for backtesting.
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch historical data
            logger.info(f"Fetching historical data for {self.symbol} ({self.timeframe})...")
            
            # Initialize empty list for all klines
            all_klines = []
            
            # Fetch data in chunks to avoid rate limits
            current_timestamp = start_timestamp
            while current_timestamp < end_timestamp:
                # Fetch klines
                klines = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not klines:
                    break
                
                # Add klines to list
                all_klines.extend(klines)
                
                # Update current timestamp
                current_timestamp = klines[-1][0] + 1
                
                # Sleep to avoid rate limits
                self.exchange.sleep(self.exchange.rateLimit / 1000)
            
            # Convert to DataFrame
            df = convert_to_dataframe(all_klines)
            
            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol} ({self.timeframe}).")
            
            self.data = df
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def prepare_data(self):
        """
        Prepare data for backtesting by calculating indicators.
        
        Returns:
            pd.DataFrame: Prepared data
        """
        if self.data is None:
            self.fetch_historical_data()
        
        try:
            # Calculate common indicators
            self.data['atr'] = calculate_atr(self.data, period=config.ATR_PERIOD)
            
            # Calculate strategy-specific indicators
            if isinstance(self.strategy, ScalpingStrategy):
                self.data['rsi'] = calculate_rsi(self.data, period=config.RSI_PERIOD)
                self.data['vwap'] = calculate_vwap(self.data, period=config.VWAP_PERIOD)
                self.data['volume_spike'] = detect_volume_spike(self.data, threshold=config.VOLUME_THRESHOLD)
                self.data['trend'] = np.where(self.data['close'] > self.data['vwap'], 'bullish', 'bearish')
                
                # Generate signals
                self.data['long_signal'] = (
                    (self.data['trend'] == 'bullish') &
                    (self.data['rsi'] < config.RSI_OVERSOLD)
                )
                
                self.data['short_signal'] = (
                    (self.data['trend'] == 'bearish') &
                    (self.data['rsi'] > config.RSI_OVERBOUGHT)
                )
            
            elif isinstance(self.strategy, SwingStrategy):
                # Calculate MACD
                macd_data = calculate_macd(
                    self.data,
                    fast_period=config.MACD_FAST_PERIOD,
                    slow_period=config.MACD_SLOW_PERIOD,
                    signal_period=config.MACD_SIGNAL_PERIOD
                )
                self.data['macd'] = macd_data['macd']
                self.data['macd_signal'] = macd_data['signal']
                
                # Calculate Bollinger Bands
                bb_data = calculate_bollinger_bands(
                    self.data,
                    period=config.BB_PERIOD,
                    std=config.BB_STD
                )
                self.data['bb_upper'] = bb_data['upper_band']
                self.data['bb_lower'] = bb_data['lower_band']
                
                # Generate signals
                self.data['long_signal'] = (
                    (self.data['macd'] > self.data['macd_signal']) &
                    (self.data['close'] < self.data['bb_lower'])
                )
                
                self.data['short_signal'] = (
                    (self.data['macd'] < self.data['macd_signal']) &
                    (self.data['close'] > self.data['bb_upper'])
                )
            
            elif isinstance(self.strategy, BreakoutStrategy):
                # Calculate recent high/low
                self.data['recent_high'] = self.data['high'].rolling(window=config.BREAKOUT_PERIOD).max()
                self.data['recent_low'] = self.data['low'].rolling(window=config.BREAKOUT_PERIOD).min()
                self.data['volume_spike'] = detect_volume_spike(self.data, threshold=config.VOLUME_THRESHOLD)
                
                # Generate signals
                self.data['long_signal'] = (
                    (self.data['close'] > self.data['recent_high'].shift(1)) &
                    self.data['volume_spike'] &
                    (self.data['close'] > self.data['open'])
                )
                
                self.data['short_signal'] = (
                    (self.data['close'] < self.data['recent_low'].shift(1)) &
                    self.data['volume_spike'] &
                    (self.data['close'] < self.data['open'])
                )
            
            # Count signals
            long_signals = self.data['long_signal'].sum()
            short_signals = self.data['short_signal'].sum()
            
            logger.info(f"Data prepared for backtesting.")
            logger.info(f"Signal counts - Long: {long_signals}, Short: {short_signals}")
            
            return self.data
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def run_backtest(self, initial_balance=10000):
        """
        Run backtest on historical data.
        
        Args:
            initial_balance (float): Initial balance
            
        Returns:
            dict: Backtest results
        """
        if self.data is None or 'long_signal' not in self.data.columns:
            self.prepare_data()
        
        try:
            # Initialize backtest variables
            balance = initial_balance
            self.equity_curve = [{'timestamp': self.data.index[0], 'equity': balance}]
            self.trades = []
            self.active_positions = {}
            
            # Set strategy-specific parameters
            if isinstance(self.strategy, ScalpingStrategy):
                take_profit = config.SCALPING_TAKE_PROFIT / 100
                stop_loss = config.SCALPING_STOP_LOSS / 100
            elif isinstance(self.strategy, SwingStrategy):
                take_profit = config.SWING_TAKE_PROFIT / 100
                stop_loss = config.SWING_STOP_LOSS / 100
            elif isinstance(self.strategy, BreakoutStrategy):
                take_profit = config.BREAKOUT_TAKE_PROFIT / 100
                stop_loss = config.BREAKOUT_STOP_LOSS / 100
            
            # Iterate through data
            for i in range(1, len(self.data)):
                current_row = self.data.iloc[i]
                previous_row = self.data.iloc[i-1]
                
                # Check for entry signals
                if len(self.active_positions) < self.strategy.max_active_positions:
                    # Check for long signal
                    if previous_row['long_signal']:
                        self._open_long_position(current_row, balance, take_profit, stop_loss)
                    
                    # Check for short signal
                    elif previous_row['short_signal']:
                        self._open_short_position(current_row, balance, take_profit, stop_loss)
                
                # Manage open positions
                self._manage_positions(current_row)
                
                # Update equity curve
                total_value = balance
                for position in self.active_positions.values():
                    if position['type'] == 'long':
                        total_value += (current_row['close'] - position['entry_price']) * position['quantity']
                    else:  # short
                        total_value += (position['entry_price'] - current_row['close']) * position['quantity']
                
                self.equity_curve.append({
                    'timestamp': current_row.name,
                    'equity': total_value
                })
            
            # Close any remaining positions at the end
            final_price = self.data.iloc[-1]['close']
            for position_id in list(self.active_positions.keys()):
                self._close_position(position_id, 'end_of_backtest', final_price)
            
            # Calculate results
            results = self._calculate_results(initial_balance)
            
            return results
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _open_long_position(self, current_row, balance, take_profit, stop_loss):
        """
        Open a long position.
        """
        # ... existing code ...
    
    def _open_short_position(self, current_row, balance, take_profit, stop_loss):
        """
        Open a short position.
        """
        # ... existing code ...
    
    def _manage_positions(self, current_row):
        """
        Manage open positions.
        """
        # ... existing code ...
    
    def _close_position(self, position_id, reason, exit_price):
        """
        Close a position.
        """
        # ... existing code ...
    
    def _calculate_results(self, initial_balance):
        """
        Calculate backtest results.
        """
        # ... existing code ...
    
    def plot_results(self, results):
        """
        Plot backtest results.
        """
        # ... existing code ...


if __name__ == '__main__':
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(initial_balance=10000)
    backtester.plot_results(results) 