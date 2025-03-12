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
    calculate_macd,
)
from strategy import BreakoutStrategy, ScalpingStrategy, SwingStrategy
from ai_strategy import AIStrategy, AIScalpingStrategy, AISwingStrategy, AIBreakoutStrategy
from utils import (
    calculate_take_profit_price,
    calculate_stop_loss_price,
    calculate_atr_stop_loss,
    setup_logger,
    convert_to_dataframe,
)

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(config.LOG_LEVEL)


class Backtester:
    """
    Backtester for the trading strategy.
    """

    def __init__(
        self,
        symbol=None,
        timeframe=None,
        start_date=None,
        end_date=None,
        strategy_class=None,
        initial_balance=10000,
        leverage=config.LEVERAGE,
        transaction_fee=0.0004,
        slippage=0.0002,
        is_ai_strategy=False,
    ):
        """
        Initialize the backtester.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            strategy_class (class): Strategy class
            initial_balance (float): Initial balance
            leverage (int): Leverage
            transaction_fee (float): Transaction fee
            slippage (float): Slippage
            is_ai_strategy (bool): Whether the strategy is an AI strategy
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_class = strategy_class
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.is_ai_strategy = is_ai_strategy
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Initialize strategy
        self.strategy = None
        
        # Initialize data
        self.data = None
        
        # Initialize results
        self.results = {
            'balance': [],
            'equity': [],
            'trades': [],
            'positions': [],
            'returns': [],
            'drawdowns': [],
        }
        
        logger.info(f"Backtester initialized for {self.symbol} ({self.timeframe})")
        
    def fetch_historical_data(self):
        """
        Fetch historical data for backtesting.
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Calculate start and end timestamps
            if self.end_date:
                end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now()
            
            if self.start_date:
                start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
            else:
                # Default to 30 days before end date
                start_date = end_date - timedelta(days=30)
            
            # Convert to milliseconds timestamp
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            logger.info(f"Fetching historical data for {self.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Fetch historical data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=start_timestamp,
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def prepare_data(self, data):
        """
        Prepare data for backtesting.
        
        Args:
            data (pd.DataFrame): Historical data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        try:
            # Add indicators
            df = data.copy()
            
            # Add RSI
            df['rsi'] = calculate_rsi(df['close'])
            
            # Add VWAP
            df['vwap'] = calculate_vwap(df)
            
            # Add ATR
            df['atr'] = calculate_atr(df)
            
            # Add volume spike
            df['volume_spike'] = detect_volume_spike(df['volume'])
            
            # Add Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
            
            # Add MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
            
            # Add returns
            df['return'] = df['close'].pct_change()
            
            # Add target for AI models
            df['return_1'] = df['close'].pct_change(1).shift(-1)  # Next candle's return
            df['return_5'] = df['close'].pct_change(5).shift(-5)  # 5 candles ahead return
            df['return_10'] = df['close'].pct_change(10).shift(-10)  # 10 candles ahead return
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def run_backtest(self):
        """
        Run backtest.
        
        Returns:
            dict: Backtest results
        """
        try:
            # Fetch historical data
            data = self.fetch_historical_data()
            
            # Prepare data
            self.data = self.prepare_data(data)
            
            # Initialize strategy
            if self.strategy_class:
                # Create a mock binance client for the strategy
                mock_client = type('MockBinanceClient', (), {
                    'get_klines': lambda symbol, timeframe, limit: self.data.iloc[-limit:].copy() if limit else self.data.copy(),
                    'create_market_order': lambda symbol, side, amount: {'orderId': 'backtest'},
                    'get_position': lambda symbol: None,
                    'get_balance': lambda: {'total': self.initial_balance},
                    'set_leverage': lambda symbol, leverage: None
                })
                
                # Initialize strategy
                self.strategy = self.strategy_class(mock_client, None, self.symbol, self.timeframe, self.leverage)
                
                # If it's an AI strategy, we need to train the models
                if self.is_ai_strategy and hasattr(self.strategy, 'train_models'):
                    logger.info("Training AI models for backtesting...")
                    self.strategy.train_models()
            else:
                logger.error("No strategy class provided")
                return None
            
            # Initialize backtest variables
            balance = self.initial_balance
            equity = self.initial_balance
            positions = {}
            trades = []
            
            # Run backtest
            logger.info(f"Running backtest for {self.symbol} ({self.timeframe})")
            
            # Iterate through data
            for i in range(len(self.data)):
                # Get current candle
                current_data = self.data.iloc[:i+1]
                current_candle = current_data.iloc[-1]
                current_price = current_candle['close']
                current_time = current_data.index[-1]
                
                # Update mock client data
                mock_client.get_klines = lambda symbol, timeframe, limit: current_data.iloc[-limit:].copy() if limit else current_data.copy()
                
                # Analyze market
                analysis = None
                try:
                    analysis = self.strategy.analyze_market()
                except Exception as e:
                    logger.error(f"Error analyzing market: {e}")
                    continue
                
                # Execute signals
                if analysis:
                    # Check for long signal
                    if analysis.get('long_signal', False) and len(positions) < self.strategy.max_active_positions:
                        # Calculate position size
                        position_size = self.strategy.position_size
                        
                        # Adjust position size for AI strategies
                        if self.is_ai_strategy and 'adjusted_position_size' in analysis:
                            position_size = analysis['adjusted_position_size']
                        
                        # Calculate entry price with slippage
                        entry_price = current_price * (1 + self.slippage)
                        
                        # Calculate take profit and stop loss
                        take_profit_percent = self.strategy.take_profit_percent
                        stop_loss_percent = self.strategy.stop_loss_percent
                        
                        # Adjust take profit and stop loss for AI strategies
                        if self.is_ai_strategy:
                            if 'adjusted_take_profit' in analysis:
                                take_profit_percent = analysis['adjusted_take_profit']
                            if 'adjusted_stop_loss' in analysis:
                                stop_loss_percent = analysis['adjusted_stop_loss']
                        
                        take_profit_price = calculate_take_profit_price(entry_price, take_profit_percent, 'long')
                        
                        if self.strategy.use_atr_for_sl and 'atr' in current_candle:
                            stop_loss_price = calculate_atr_stop_loss(
                                entry_price, current_candle['atr'], self.strategy.atr_multiplier, 'long'
                            )
                        else:
                            stop_loss_price = calculate_stop_loss_price(entry_price, stop_loss_percent, 'long')
                        
                        # Calculate cost and fees
                        cost = position_size * entry_price / self.leverage
                        fee = cost * self.transaction_fee
                        
                        # Check if enough balance
                        if cost + fee <= balance:
                            # Create position
                            position_id = f"long_{self.symbol}_{i}"
                            positions[position_id] = {
                                'id': position_id,
                                'symbol': self.symbol,
                                'type': 'long',
                                'entry_price': entry_price,
                                'take_profit': take_profit_price,
                                'stop_loss': stop_loss_price,
                                'quantity': position_size,
                                'entry_time': current_time,
                                'cost': cost,
                                'fee': fee
                            }
                            
                            # Update balance
                            balance -= (cost + fee)
                            
                            logger.info(f"Opened long position at {entry_price:.2f}")
                    
                    # Check for short signal
                    elif analysis.get('short_signal', False) and len(positions) < self.strategy.max_active_positions:
                        # Calculate position size
                        position_size = self.strategy.position_size
                        
                        # Adjust position size for AI strategies
                        if self.is_ai_strategy and 'adjusted_position_size' in analysis:
                            position_size = analysis['adjusted_position_size']
                        
                        # Calculate entry price with slippage
                        entry_price = current_price * (1 - self.slippage)
                        
                        # Calculate take profit and stop loss
                        take_profit_percent = self.strategy.take_profit_percent
                        stop_loss_percent = self.strategy.stop_loss_percent
                        
                        # Adjust take profit and stop loss for AI strategies
                        if self.is_ai_strategy:
                            if 'adjusted_take_profit' in analysis:
                                take_profit_percent = analysis['adjusted_take_profit']
                            if 'adjusted_stop_loss' in analysis:
                                stop_loss_percent = analysis['adjusted_stop_loss']
                        
                        take_profit_price = calculate_take_profit_price(entry_price, take_profit_percent, 'short')
                        
                        if self.strategy.use_atr_for_sl and 'atr' in current_candle:
                            stop_loss_price = calculate_atr_stop_loss(
                                entry_price, current_candle['atr'], self.strategy.atr_multiplier, 'short'
                            )
                        else:
                            stop_loss_price = calculate_stop_loss_price(entry_price, stop_loss_percent, 'short')
                        
                        # Calculate cost and fees
                        cost = position_size * entry_price / self.leverage
                        fee = cost * self.transaction_fee
                        
                        # Check if enough balance
                        if cost + fee <= balance:
                            # Create position
                            position_id = f"short_{self.symbol}_{i}"
                            positions[position_id] = {
                                'id': position_id,
                                'symbol': self.symbol,
                                'type': 'short',
                                'entry_price': entry_price,
                                'take_profit': take_profit_price,
                                'stop_loss': stop_loss_price,
                                'quantity': position_size,
                                'entry_time': current_time,
                                'cost': cost,
                                'fee': fee
                            }
                            
                            # Update balance
                            balance -= (cost + fee)
                            
                            logger.info(f"Opened short position at {entry_price:.2f}")
                
                # Manage positions
                for position_id in list(positions.keys()):
                    position = positions[position_id]
                    
                    # Check if position should be closed
                    if position['type'] == 'long':
                        # Check take profit
                        if current_price >= position['take_profit']:
                            # Calculate profit
                            profit = position['quantity'] * (current_price - position['entry_price'])
                            
                            # Calculate fee
                            fee = position['quantity'] * current_price * self.transaction_fee / self.leverage
                            
                            # Update balance
                            balance += (position['cost'] + profit - fee)
                            
                            # Add trade
                            trades.append({
                                'position_id': position_id,
                                'symbol': position['symbol'],
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': position['quantity'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'profit': profit,
                                'fee': position['fee'] + fee,
                                'result': 'take_profit'
                            })
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed long position at {current_price:.2f} (take profit)")
                        
                        # Check stop loss
                        elif current_price <= position['stop_loss']:
                            # Calculate profit (negative)
                            profit = position['quantity'] * (current_price - position['entry_price'])
                            
                            # Calculate fee
                            fee = position['quantity'] * current_price * self.transaction_fee / self.leverage
                            
                            # Update balance
                            balance += (position['cost'] + profit - fee)
                            
                            # Add trade
                            trades.append({
                                'position_id': position_id,
                                'symbol': position['symbol'],
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': position['quantity'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'profit': profit,
                                'fee': position['fee'] + fee,
                                'result': 'stop_loss'
                            })
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed long position at {current_price:.2f} (stop loss)")
                        
                        # Check trailing stop
                        elif self.strategy.use_trailing_stop:
                            # Calculate price movement
                            price_movement = (current_price - position['entry_price']) / position['entry_price']
                            
                            # Check if trailing stop should be activated
                            if price_movement >= self.strategy.trailing_stop_activation:
                                # Calculate new stop loss
                                new_stop_loss = max(
                                    position['stop_loss'],
                                    current_price * (1 - self.strategy.trailing_stop_callback)
                                )
                                
                                # Update stop loss if it has changed
                                if new_stop_loss > position['stop_loss']:
                                    position['stop_loss'] = new_stop_loss
                    
                    elif position['type'] == 'short':
                        # Check take profit
                        if current_price <= position['take_profit']:
                            # Calculate profit
                            profit = position['quantity'] * (position['entry_price'] - current_price)
                            
                            # Calculate fee
                            fee = position['quantity'] * current_price * self.transaction_fee / self.leverage
                            
                            # Update balance
                            balance += (position['cost'] + profit - fee)
                            
                            # Add trade
                            trades.append({
                                'position_id': position_id,
                                'symbol': position['symbol'],
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': position['quantity'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'profit': profit,
                                'fee': position['fee'] + fee,
                                'result': 'take_profit'
                            })
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed short position at {current_price:.2f} (take profit)")
                        
                        # Check stop loss
                        elif current_price >= position['stop_loss']:
                            # Calculate profit (negative)
                            profit = position['quantity'] * (position['entry_price'] - current_price)
                            
                            # Calculate fee
                            fee = position['quantity'] * current_price * self.transaction_fee / self.leverage
                            
                            # Update balance
                            balance += (position['cost'] + profit - fee)
                            
                            # Add trade
                            trades.append({
                                'position_id': position_id,
                                'symbol': position['symbol'],
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': position['quantity'],
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'profit': profit,
                                'fee': position['fee'] + fee,
                                'result': 'stop_loss'
                            })
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed short position at {current_price:.2f} (stop loss)")
                        
                        # Check trailing stop
                        elif self.strategy.use_trailing_stop:
                            # Calculate price movement
                            price_movement = (position['entry_price'] - current_price) / position['entry_price']
                            
                            # Check if trailing stop should be activated
                            if price_movement >= self.strategy.trailing_stop_activation:
                                # Calculate new stop loss
                                new_stop_loss = min(
                                    position['stop_loss'],
                                    current_price * (1 + self.strategy.trailing_stop_callback)
                                )
                                
                                # Update stop loss if it has changed
                                if new_stop_loss < position['stop_loss']:
                                    position['stop_loss'] = new_stop_loss
                
                # Calculate equity
                equity = balance
                for position_id, position in positions.items():
                    if position['type'] == 'long':
                        equity += position['cost'] + position['quantity'] * (current_price - position['entry_price'])
                    elif position['type'] == 'short':
                        equity += position['cost'] + position['quantity'] * (position['entry_price'] - current_price)
                
                # Store results
                self.results['balance'].append(balance)
                self.results['equity'].append(equity)
                self.results['positions'].append(len(positions))
            
            # Close any remaining positions at the last price
            last_price = self.data.iloc[-1]['close']
            last_time = self.data.index[-1]
            
            for position_id, position in positions.items():
                if position['type'] == 'long':
                    # Calculate profit
                    profit = position['quantity'] * (last_price - position['entry_price'])
                    
                    # Calculate fee
                    fee = position['quantity'] * last_price * self.transaction_fee / self.leverage
                    
                    # Update balance
                    balance += (position['cost'] + profit - fee)
                    
                    # Add trade
                    trades.append({
                        'position_id': position_id,
                        'symbol': position['symbol'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': last_price,
                        'quantity': position['quantity'],
                        'entry_time': position['entry_time'],
                        'exit_time': last_time,
                        'profit': profit,
                        'fee': position['fee'] + fee,
                        'result': 'end_of_backtest'
                    })
                    
                    logger.info(f"Closed long position at {last_price:.2f} (end of backtest)")
                
                elif position['type'] == 'short':
                    # Calculate profit
                    profit = position['quantity'] * (position['entry_price'] - last_price)
                    
                    # Calculate fee
                    fee = position['quantity'] * last_price * self.transaction_fee / self.leverage
                    
                    # Update balance
                    balance += (position['cost'] + profit - fee)
                    
                    # Add trade
                    trades.append({
                        'position_id': position_id,
                        'symbol': position['symbol'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': last_price,
                        'quantity': position['quantity'],
                        'entry_time': position['entry_time'],
                        'exit_time': last_time,
                        'profit': profit,
                        'fee': position['fee'] + fee,
                        'result': 'end_of_backtest'
                    })
                    
                    logger.info(f"Closed short position at {last_price:.2f} (end of backtest)")
            
            # Calculate final equity
            equity = balance
            
            # Store final results
            self.results['balance'].append(balance)
            self.results['equity'].append(equity)
            self.results['positions'].append(0)
            self.results['trades'] = trades
            
            # Calculate returns
            self.results['returns'] = [0]
            for i in range(1, len(self.results['equity'])):
                self.results['returns'].append(
                    (self.results['equity'][i] - self.results['equity'][i-1]) / self.results['equity'][i-1]
                )
            
            # Calculate drawdowns
            self.results['drawdowns'] = [0]
            peak = self.results['equity'][0]
            for i in range(1, len(self.results['equity'])):
                if self.results['equity'][i] > peak:
                    peak = self.results['equity'][i]
                    self.results['drawdowns'].append(0)
                else:
                    self.results['drawdowns'].append((peak - self.results['equity'][i]) / peak)
            
            # Calculate statistics
            stats = self.calculate_statistics()
            
            logger.info(f"Backtest completed for {self.symbol} ({self.timeframe})")
            logger.info(f"Initial balance: ${self.initial_balance:.2f}")
            logger.info(f"Final balance: ${balance:.2f}")
            logger.info(f"Total profit: ${balance - self.initial_balance:.2f}")
            logger.info(f"Return: {(balance / self.initial_balance - 1) * 100:.2f}%")
            logger.info(f"Number of trades: {len(trades)}")
            
            return {
                'results': self.results,
                'stats': stats,
                'data': self.data
            }
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    def calculate_statistics(self):
        # Implementation of calculate_statistics method
        pass


if __name__ == "__main__":
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest()
    backtester.plot_results(results)
