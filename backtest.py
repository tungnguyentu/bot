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
                try:
                    end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
                except ValueError:
                    logger.error(f"Invalid end date format: {self.end_date}. Expected format: YYYY-MM-DD")
                    return None
            else:
                end_date = datetime.now()
            
            if self.start_date:
                try:
                    start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
                except ValueError:
                    logger.error(f"Invalid start date format: {self.start_date}. Expected format: YYYY-MM-DD")
                    return None
            else:
                # Default to 30 days before end date
                start_date = end_date - timedelta(days=30)
            
            # Validate date range
            if start_date >= end_date:
                logger.error(f"Start date ({start_date.strftime('%Y-%m-%d')}) must be before end date ({end_date.strftime('%Y-%m-%d')})")
                return None
                
            # Check if start date is in the future
            if start_date > datetime.now():
                logger.error(f"Start date ({start_date.strftime('%Y-%m-%d')}) cannot be in the future")
                return None
            
            # Convert to milliseconds timestamp
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            logger.info(f"Fetching historical data for {self.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            try:
                # Fetch historical data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=start_timestamp,
                    limit=1000
                )
                
                # Check if data was returned
                if not ohlcv or len(ohlcv) == 0:
                    logger.error(f"No data returned for {self.symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    return None
                
                # Log raw data for debugging
                logger.info(f"Raw data sample: {ohlcv[:1]}")
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Log DataFrame info before filtering
                logger.info(f"DataFrame info before filtering: {df.shape}")
                logger.info(f"DataFrame date range: {df.index.min()} to {df.index.max()}")
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Log filtered DataFrame info
                logger.info(f"DataFrame info after filtering: {df.shape}")
                
                # Check if filtered data is empty
                if df.empty:
                    logger.error(f"No data available for {self.symbol} in the specified date range")
                    return None
                
                logger.info(f"Fetched {len(df)} candles for {self.symbol}")
                
                return df
            except ccxt.NetworkError as e:
                logger.error(f"Network error fetching historical data: {e}")
                return None
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching historical data: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching historical data: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def prepare_data(self, data):
        """
        Prepare data for backtesting.
        
        Args:
            data (pd.DataFrame): Historical data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        try:
            # Check if data is empty
            if data is None or data.empty:
                logger.error("No historical data available for backtesting")
                return None
                
            # Check if required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Log data info for debugging
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Data types: {data.dtypes.to_dict()}")
            logger.info(f"Data sample: {data.head(1).to_dict()}")
            
            # Add indicators
            df = data.copy()
            
            try:
                # Add RSI
                logger.info("Calculating RSI...")
                df['rsi'] = calculate_rsi(df['close'])
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
                return None
            
            try:
                # Add VWAP
                logger.info("Calculating VWAP...")
                df['vwap'] = calculate_vwap(df)
            except Exception as e:
                logger.error(f"Error calculating VWAP: {e}")
                return None
            
            try:
                # Add ATR
                logger.info("Calculating ATR...")
                df['atr'] = calculate_atr(df)
            except Exception as e:
                logger.error(f"Error calculating ATR: {e}")
                return None
            
            try:
                # Add volume spike
                logger.info("Detecting volume spikes...")
                df['volume_spike'] = detect_volume_spike(df['volume'])
            except Exception as e:
                logger.error(f"Error detecting volume spikes: {e}")
                return None
            
            try:
                # Add Bollinger Bands
                logger.info("Calculating Bollinger Bands...")
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")
                return None
            
            try:
                # Add MACD
                logger.info("Calculating MACD...")
                df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")
                return None
            
            try:
                # Add returns
                logger.info("Calculating returns...")
                df['return'] = df['close'].pct_change()
            except Exception as e:
                logger.error(f"Error calculating returns: {e}")
                return None
            
            try:
                # Add target for AI models
                logger.info("Calculating target variables...")
                df['return_1'] = df['close'].pct_change(1).shift(-1)  # Next candle's return
                df['return_5'] = df['close'].pct_change(5).shift(-5)  # 5 candles ahead return
                df['return_10'] = df['close'].pct_change(10).shift(-10)  # 10 candles ahead return
            except Exception as e:
                logger.error(f"Error calculating target variables: {e}")
                return None
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            logger.info(f"Prepared data shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def run_backtest(self):
        """
        Run backtest.
        
        Returns:
            dict: Backtest results
        """
        try:
            # Fetch historical data
            data = self.fetch_historical_data()
            
            # Check if data is valid
            if data is None or data.empty:
                logger.error("No historical data available for backtesting")
                return None
                
            # Prepare data
            self.data = self.prepare_data(data)
            
            # Check if prepared data is valid
            if self.data is None or self.data.empty:
                logger.error("Failed to prepare data for backtesting")
                return None
            
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
                if analysis and (analysis.get('long_signal', False) or analysis.get('short_signal', False)):
                    # Check if we can open new positions
                    if len(positions) < self.strategy.max_active_positions:
                        # Open position
                        position_id = f"position_{len(trades)}"
                        position_type = 'long' if analysis.get('long_signal', False) else 'short'
                        entry_price = current_price
                        
                        # Calculate position size
                        position_size = self.initial_balance * self.strategy.position_size / 100
                        
                        # Calculate quantity
                        quantity = position_size / entry_price
                        
                        # Calculate stop loss and take profit
                        if self.strategy.use_atr_for_sl and 'atr' in analysis:
                            stop_loss = entry_price - (analysis['atr'] * self.strategy.atr_multiplier) if position_type == 'long' else entry_price + (analysis['atr'] * self.strategy.atr_multiplier)
                        else:
                            stop_loss = entry_price * (1 - self.strategy.stop_loss_percent) if position_type == 'long' else entry_price * (1 + self.strategy.stop_loss_percent)
                        
                        take_profit = entry_price * (1 + self.strategy.take_profit_percent) if position_type == 'long' else entry_price * (1 - self.strategy.take_profit_percent)
                        
                        # Create position
                        position = {
                            'id': position_id,
                            'type': position_type,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': current_time,
                            'exit_time': None,
                            'exit_price': None,
                            'profit': None,
                            'status': 'open',
                            'cost': position_size
                        }
                        
                        # Add position
                        positions[position_id] = position
                        
                        # Add trade
                        trades.append(position)
                        
                        logger.info(f"Opened {position_type} position at {entry_price} with stop loss at {stop_loss} and take profit at {take_profit}")
                
                # Manage positions
                for position_id, position in list(positions.items()):
                    # Check if position should be closed
                    if position['status'] == 'open':
                        # Check stop loss
                        if (position['type'] == 'long' and current_price <= position['stop_loss']) or (position['type'] == 'short' and current_price >= position['stop_loss']):
                            # Close position at stop loss
                            position['exit_price'] = position['stop_loss']
                            position['exit_time'] = current_time
                            position['status'] = 'closed'
                            
                            # Calculate profit
                            if position['type'] == 'long':
                                position['profit'] = position['quantity'] * (position['exit_price'] - position['entry_price'])
                            else:  # short
                                position['profit'] = position['quantity'] * (position['entry_price'] - position['exit_price'])
                            
                            # Update balance
                            balance += position['profit']
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed {position['type']} position at stop loss ({position['exit_price']}) with profit {position['profit']:.2f}")
                        
                        # Check take profit
                        elif (position['type'] == 'long' and current_price >= position['take_profit']) or (position['type'] == 'short' and current_price <= position['take_profit']):
                            # Close position at take profit
                            position['exit_price'] = position['take_profit']
                            position['exit_time'] = current_time
                            position['status'] = 'closed'
                            
                            # Calculate profit
                            if position['type'] == 'long':
                                position['profit'] = position['quantity'] * (position['exit_price'] - position['entry_price'])
                            else:  # short
                                position['profit'] = position['quantity'] * (position['entry_price'] - position['exit_price'])
                            
                            # Update balance
                            balance += position['profit']
                            
                            # Remove position
                            del positions[position_id]
                            
                            logger.info(f"Closed {position['type']} position at take profit ({position['exit_price']}) with profit {position['profit']:.2f}")
                        
                        # Check trailing stop
                        elif self.strategy.use_trailing_stop:
                            # Calculate price movement
                            if position['type'] == 'long':
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
                            else:  # short
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
                
                # Calculate daily return
                if len(self.results['equity']) > 0:
                    daily_return = (equity / self.results['equity'][-1]) - 1
                else:
                    daily_return = 0
                
                # Store results
                self.results['balance'].append(balance)
                self.results['equity'].append(equity)
                self.results['positions'].append(len(positions))
                self.results['returns'].append(daily_return)
            
            # Close any remaining positions at the last price
            last_price = self.data.iloc[-1]['close']
            last_time = self.data.index[-1]
            
            for position_id, position in list(positions.items()):
                # Close position at current price
                position['exit_price'] = last_price
                position['exit_time'] = last_time
                position['status'] = 'closed'
                
                # Calculate profit
                if position['type'] == 'long':
                    position['profit'] = position['quantity'] * (position['exit_price'] - position['entry_price'])
                else:  # short
                    position['profit'] = position['quantity'] * (position['entry_price'] - position['exit_price'])
                
                # Update balance
                balance += position['profit']
                
                logger.info(f"Closed {position['type']} position at end of backtest ({position['exit_price']}) with profit {position['profit']:.2f}")
            
            # Calculate final equity
            equity = balance
            
            # Store final results
            self.results['balance'].append(balance)
            self.results['equity'].append(equity)
            self.results['positions'].append(0)
            
            # Calculate drawdowns
            self.results['drawdowns'] = []
            peak = self.results['equity'][0]
            for i in range(len(self.results['equity'])):
                if self.results['equity'][i] > peak:
                    peak = self.results['equity'][i]
                    self.results['drawdowns'].append(0)
                else:
                    self.results['drawdowns'].append((peak - self.results['equity'][i]) / peak)
            
            # Calculate statistics
            stats = self.calculate_statistics()
            
            # Store trades in results
            self.results['trades'] = trades
            
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def calculate_statistics(self):
        """
        Calculate statistics from backtest results.
        
        Returns:
            dict: Statistics
        """
        try:
            # Check if we have enough data
            if not self.results['equity'] or len(self.results['equity']) < 2:
                logger.error("Not enough data to calculate statistics")
                return {
                    'total_return': 0,
                    'annual_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_trades': 0
                }
            
            # Calculate returns
            initial_equity = self.results['equity'][0]
            final_equity = self.results['equity'][-1]
            total_return = (final_equity / initial_equity - 1) * 100
            
            # Calculate annual return
            days = (self.data.index[-1] - self.data.index[0]).days
            if days > 0:
                annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
            else:
                annual_return = 0
            
            # Calculate Sharpe ratio
            if len(self.results['returns']) > 1:
                returns_mean = np.mean(self.results['returns'])
                returns_std = np.std(self.results['returns'])
                sharpe_ratio = returns_mean / returns_std * np.sqrt(252) if returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            max_drawdown = max(self.results['drawdowns']) * 100 if self.results['drawdowns'] else 0
            
            # Calculate win rate and profit factor
            winning_trades = [t for t in self.results['trades'] if t.get('profit', 0) > 0]
            losing_trades = [t for t in self.results['trades'] if t.get('profit', 0) <= 0]
            
            total_trades = len(self.results['trades'])
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            total_profit = sum(t.get('profit', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades
            }
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }

    def plot_results(self, results):
        """
        Plot backtest results.
        
        Args:
            results (dict): Backtest results
        """
        try:
            # Check if results is valid
            if not results or 'results' not in results:
                logger.error("No results to plot")
                return
                
            # Extract data from results
            backtest_results = results['results']
            data = results['data']
            
            # Check if we have enough data
            if not backtest_results['equity'] or len(backtest_results['equity']) < 2:
                logger.error("Not enough data to plot")
                return
            
            # Create directory for plots if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Get dates and prices
            dates = data.index
            prices = data['close']
            
            # Plot price
            ax1.plot(dates, prices, label='Price', color='blue')
            ax1.set_ylabel('Price')
            ax1.set_title(f'Backtest Results for {self.symbol} ({self.timeframe})')
            ax1.grid(True)
            
            # Plot trades
            for trade in backtest_results['trades']:
                if trade.get('type') == 'long':
                    ax1.scatter(trade.get('entry_time'), trade.get('entry_price'), color='green', marker='^', s=100)
                    if trade.get('exit_time'):
                        ax1.scatter(trade.get('exit_time'), trade.get('exit_price'), color='red', marker='v', s=100)
                        ax1.plot([trade.get('entry_time'), trade.get('exit_time')], 
                                [trade.get('entry_price'), trade.get('exit_price')], 
                                color='gray', linestyle='--', alpha=0.5)
                else:  # short
                    ax1.scatter(trade.get('entry_time'), trade.get('entry_price'), color='red', marker='v', s=100)
                    if trade.get('exit_time'):
                        ax1.scatter(trade.get('exit_time'), trade.get('exit_price'), color='green', marker='^', s=100)
                        ax1.plot([trade.get('entry_time'), trade.get('exit_time')], 
                                [trade.get('entry_price'), trade.get('exit_price')], 
                                color='gray', linestyle='--', alpha=0.5)
            
            # Plot equity
            ax2.plot(dates[:len(backtest_results['equity'])], backtest_results['equity'], label='Equity', color='green')
            ax2.set_ylabel('Equity')
            ax2.grid(True)
            
            # Plot drawdown
            if backtest_results['drawdowns']:
                ax3.fill_between(dates[:len(backtest_results['drawdowns'])], 
                                [d * 100 for d in backtest_results['drawdowns']], 
                                0, color='red', alpha=0.3)
                ax3.set_ylabel('Drawdown (%)')
                ax3.set_xlabel('Date')
                ax3.grid(True)
            
            # Save plot
            plt.tight_layout()
            filename = f"plots/backtest_{self.symbol}_{self.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            logger.info(f"Backtest plot saved to {filename}")
            
            # Close plot
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest()
    backtester.plot_results(results)
