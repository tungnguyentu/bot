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
    detect_volume_spike
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
    
    def __init__(self, symbol=None, timeframe=None, start_date=None, end_date=None):
        """
        Initialize the backtester.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.start_date = start_date or config.BACKTEST_START_DATE
        self.end_date = end_date or config.BACKTEST_END_DATE
        
        # Strategy parameters
        self.rsi_period = config.RSI_PERIOD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.rsi_oversold = config.RSI_OVERSOLD
        self.vwap_period = config.VWAP_PERIOD
        self.atr_period = config.ATR_PERIOD
        self.volume_threshold = config.VOLUME_THRESHOLD
        
        # Risk management
        self.take_profit_percent = config.TAKE_PROFIT_PERCENT / 100
        self.stop_loss_percent = config.STOP_LOSS_PERCENT / 100
        self.use_atr_for_sl = config.USE_ATR_FOR_SL
        self.atr_multiplier = config.ATR_MULTIPLIER
        self.use_trailing_stop = config.USE_TRAILING_STOP
        self.trailing_stop_activation = config.TRAILING_STOP_ACTIVATION / 100
        self.trailing_stop_callback = config.TRAILING_STOP_CALLBACK / 100
        
        # Position management
        self.max_active_positions = config.MAX_ACTIVE_POSITIONS
        self.position_size = config.POSITION_SIZE
        self.leverage = config.LEVERAGE
        
        # Initialize exchange for historical data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Initialize results
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.active_positions = {}
        
        logger.info(f"Backtester initialized for {self.symbol} ({self.timeframe}) from {self.start_date} to {self.end_date}.")
    
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
            # Calculate indicators
            self.data['rsi'] = calculate_rsi(self.data, period=self.rsi_period)
            self.data['vwap'] = calculate_vwap(self.data, period=self.vwap_period)
            self.data['atr'] = calculate_atr(self.data, period=self.atr_period)
            self.data['volume_spike'] = detect_volume_spike(self.data, threshold=self.volume_threshold)
            
            # Determine trend based on VWAP
            self.data['trend'] = np.where(self.data['close'] > self.data['vwap'], 'bullish', 'bearish')
            
            # Generate signals - removed volume spike requirement
            self.data['long_signal'] = (
                (self.data['trend'] == 'bullish') &
                (self.data['rsi'] < self.rsi_oversold)
                # Volume spike requirement removed
            )
            
            self.data['short_signal'] = (
                (self.data['trend'] == 'bearish') &
                (self.data['rsi'] > self.rsi_overbought)
                # Volume spike requirement removed
            )
            
            # Count signals
            long_signals = self.data['long_signal'].sum()
            short_signals = self.data['short_signal'].sum()
            
            # Count individual conditions
            bullish_trend = (self.data['trend'] == 'bullish').sum()
            bearish_trend = (self.data['trend'] == 'bearish').sum()
            rsi_oversold = (self.data['rsi'] < self.rsi_oversold).sum()
            rsi_overbought = (self.data['rsi'] > self.rsi_overbought).sum()
            volume_spikes = self.data['volume_spike'].sum()
            
            # Count combined conditions
            bullish_and_oversold = ((self.data['trend'] == 'bullish') & (self.data['rsi'] < self.rsi_oversold)).sum()
            bearish_and_overbought = ((self.data['trend'] == 'bearish') & (self.data['rsi'] > self.rsi_overbought)).sum()
            
            logger.info(f"Data prepared for backtesting.")
            logger.info(f"Signal counts - Long: {long_signals}, Short: {short_signals}")
            logger.info(f"Condition counts:")
            logger.info(f"  Bullish trend: {bullish_trend}/{len(self.data)} ({bullish_trend/len(self.data)*100:.2f}%)")
            logger.info(f"  Bearish trend: {bearish_trend}/{len(self.data)} ({bearish_trend/len(self.data)*100:.2f}%)")
            logger.info(f"  RSI oversold: {rsi_oversold}/{len(self.data)} ({rsi_oversold/len(self.data)*100:.2f}%)")
            logger.info(f"  RSI overbought: {rsi_overbought}/{len(self.data)} ({rsi_overbought/len(self.data)*100:.2f}%)")
            logger.info(f"  Volume spikes: {volume_spikes}/{len(self.data)} ({volume_spikes/len(self.data)*100:.2f}%)")
            logger.info(f"  Bullish & Oversold: {bullish_and_oversold}/{len(self.data)} ({bullish_and_oversold/len(self.data)*100:.2f}%)")
            logger.info(f"  Bearish & Overbought: {bearish_and_overbought}/{len(self.data)} ({bearish_and_overbought/len(self.data)*100:.2f}%)")
            
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
            
            # Iterate through data
            for i in range(1, len(self.data)):
                current_row = self.data.iloc[i]
                previous_row = self.data.iloc[i-1]
                
                # Check for entry signals
                if len(self.active_positions) < self.max_active_positions:
                    # Check for long signal
                    if previous_row['long_signal']:
                        self._open_long_position(current_row, balance)
                    
                    # Check for short signal
                    elif previous_row['short_signal']:
                        self._open_short_position(current_row, balance)
                
                # Manage open positions
                self._manage_positions(current_row)
                
                # Update equity curve
                total_equity = balance
                for position_id, position in self.active_positions.items():
                    if position['type'] == 'long':
                        unrealized_pnl = (current_row['close'] - position['entry_price']) * position['quantity'] * self.leverage
                    else:  # short
                        unrealized_pnl = (position['entry_price'] - current_row['close']) * position['quantity'] * self.leverage
                    
                    total_equity += unrealized_pnl
                
                self.equity_curve.append({
                    'timestamp': current_row.name,
                    'equity': total_equity
                })
            
            # Close any remaining positions
            for position_id in list(self.active_positions.keys()):
                position = self.active_positions[position_id]
                last_price = self.data.iloc[-1]['close']
                self._close_position(position_id, 'end_of_backtest', last_price)
            
            # Calculate backtest results
            results = self._calculate_results(initial_balance)
            
            logger.info("Backtest completed.")
            logger.info(f"Total trades: {results['total_trades']}")
            logger.info(f"Win rate: {results['win_rate']:.2f}%")
            logger.info(f"Profit factor: {results['profit_factor']:.2f}")
            logger.info(f"Final balance: ${results['final_balance']:.2f}")
            logger.info(f"Return: {results['return']:.2f}%")
            logger.info(f"Max drawdown: {results['max_drawdown']:.2f}%")
            
            return results
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _open_long_position(self, row, balance):
        """
        Open a long position.
        
        Args:
            row (pd.Series): Current data row
            balance (float): Current balance
        """
        # Calculate position size
        position_amount = balance * self.position_size
        entry_price = row['close']
        
        # Calculate stop loss price
        if self.use_atr_for_sl:
            stop_loss_price = calculate_atr_stop_loss(
                entry_price=entry_price,
                atr_value=row['atr'],
                atr_multiplier=self.atr_multiplier,
                position_type='long'
            )
        else:
            stop_loss_price = calculate_stop_loss_price(
                entry_price=entry_price,
                stop_loss_percent=self.stop_loss_percent,
                position_type='long'
            )
        
        # Calculate take profit price
        take_profit_price = calculate_take_profit_price(
            entry_price=entry_price,
            take_profit_percent=self.take_profit_percent,
            position_type='long'
        )
        
        # Calculate quantity
        quantity = position_amount / entry_price
        
        # Store position details
        position_id = f"long_{len(self.trades)}"
        position = {
            'id': position_id,
            'symbol': self.symbol,
            'type': 'long',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': row.name,
            'status': 'open',
            'trailing_stop_activated': False,
            'trailing_stop_price': None
        }
        
        self.active_positions[position_id] = position
        
        # Record trade
        trade = {
            'id': position_id,
            'symbol': self.symbol,
            'type': 'long',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': row.name,
            'exit_price': None,
            'exit_time': None,
            'profit_loss': None,
            'profit_loss_percent': None,
            'close_reason': None,
            'status': 'open'
        }
        
        self.trades.append(trade)
    
    def _open_short_position(self, row, balance):
        """
        Open a short position.
        
        Args:
            row (pd.Series): Current data row
            balance (float): Current balance
        """
        # Calculate position size
        position_amount = balance * self.position_size
        entry_price = row['close']
        
        # Calculate stop loss price
        if self.use_atr_for_sl:
            stop_loss_price = calculate_atr_stop_loss(
                entry_price=entry_price,
                atr_value=row['atr'],
                atr_multiplier=self.atr_multiplier,
                position_type='short'
            )
        else:
            stop_loss_price = calculate_stop_loss_price(
                entry_price=entry_price,
                stop_loss_percent=self.stop_loss_percent,
                position_type='short'
            )
        
        # Calculate take profit price
        take_profit_price = calculate_take_profit_price(
            entry_price=entry_price,
            take_profit_percent=self.take_profit_percent,
            position_type='short'
        )
        
        # Calculate quantity
        quantity = position_amount / entry_price
        
        # Store position details
        position_id = f"short_{len(self.trades)}"
        position = {
            'id': position_id,
            'symbol': self.symbol,
            'type': 'short',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': row.name,
            'status': 'open',
            'trailing_stop_activated': False,
            'trailing_stop_price': None
        }
        
        self.active_positions[position_id] = position
        
        # Record trade
        trade = {
            'id': position_id,
            'symbol': self.symbol,
            'type': 'short',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'entry_time': row.name,
            'exit_price': None,
            'exit_time': None,
            'profit_loss': None,
            'profit_loss_percent': None,
            'close_reason': None,
            'status': 'open'
        }
        
        self.trades.append(trade)
    
    def _manage_positions(self, row):
        """
        Manage open positions.
        
        Args:
            row (pd.Series): Current data row
        """
        if not self.active_positions:
            return
        
        current_price = row['close']
        positions_to_remove = []
        
        for position_id, position in self.active_positions.items():
            if position['status'] != 'open':
                continue
            
            position_type = position['type']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Check if position should be closed
            if position_type == 'long':
                # Check for stop loss
                if current_price <= stop_loss:
                    self._close_position(position_id, 'stop_loss', current_price)
                    positions_to_remove.append(position_id)
                    continue
                
                # Check for take profit
                if current_price >= take_profit:
                    self._close_position(position_id, 'take_profit', current_price)
                    positions_to_remove.append(position_id)
                    continue
                
                # Check for trailing stop
                if self.use_trailing_stop:
                    # Calculate profit percentage
                    profit_percent = (current_price - entry_price) / entry_price
                    
                    # Check if trailing stop should be activated
                    if profit_percent >= self.trailing_stop_activation:
                        if not position['trailing_stop_activated']:
                            # Activate trailing stop
                            position['trailing_stop_activated'] = True
                            position['trailing_stop_price'] = current_price * (1 - self.trailing_stop_callback)
                        else:
                            # Update trailing stop if price moves up
                            new_trailing_stop = current_price * (1 - self.trailing_stop_callback)
                            if new_trailing_stop > position['trailing_stop_price']:
                                position['trailing_stop_price'] = new_trailing_stop
                        
                        # Check if price hits trailing stop
                        if position['trailing_stop_activated'] and current_price <= position['trailing_stop_price']:
                            self._close_position(position_id, 'trailing_stop', current_price)
                            positions_to_remove.append(position_id)
            
            elif position_type == 'short':
                # Check for stop loss
                if current_price >= stop_loss:
                    self._close_position(position_id, 'stop_loss', current_price)
                    positions_to_remove.append(position_id)
                    continue
                
                # Check for take profit
                if current_price <= take_profit:
                    self._close_position(position_id, 'take_profit', current_price)
                    positions_to_remove.append(position_id)
                    continue
                
                # Check for trailing stop
                if self.use_trailing_stop:
                    # Calculate profit percentage
                    profit_percent = (entry_price - current_price) / entry_price
                    
                    # Check if trailing stop should be activated
                    if profit_percent >= self.trailing_stop_activation:
                        if not position['trailing_stop_activated']:
                            # Activate trailing stop
                            position['trailing_stop_activated'] = True
                            position['trailing_stop_price'] = current_price * (1 + self.trailing_stop_callback)
                        else:
                            # Update trailing stop if price moves down
                            new_trailing_stop = current_price * (1 + self.trailing_stop_callback)
                            if new_trailing_stop < position['trailing_stop_price']:
                                position['trailing_stop_price'] = new_trailing_stop
                        
                        # Check if price hits trailing stop
                        if position['trailing_stop_activated'] and current_price >= position['trailing_stop_price']:
                            self._close_position(position_id, 'trailing_stop', current_price)
                            positions_to_remove.append(position_id)
        
        # Remove closed positions
        for position_id in positions_to_remove:
            del self.active_positions[position_id]
    
    def _close_position(self, position_id, reason, exit_price):
        """
        Close a position.
        
        Args:
            position_id (str): Position ID
            reason (str): Reason for closing position
            exit_price (float): Exit price
        """
        position = self.active_positions.get(position_id)
        
        if not position:
            logger.warning(f"Position {position_id} not found.")
            return
        
        # Calculate profit/loss
        if position['type'] == 'long':
            profit_loss = (exit_price - position['entry_price']) * position['quantity'] * self.leverage
            profit_loss_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100 * self.leverage
        else:  # short
            profit_loss = (position['entry_price'] - exit_price) * position['quantity'] * self.leverage
            profit_loss_percent = (position['entry_price'] - exit_price) / position['entry_price'] * 100 * self.leverage
        
        # Update position status
        position['status'] = 'closed'
        position['exit_price'] = exit_price
        position['exit_time'] = self.data.index[self.data.index.get_loc(position['entry_time']) + 1]
        position['profit_loss'] = profit_loss
        position['profit_loss_percent'] = profit_loss_percent
        position['close_reason'] = reason
        
        # Update trade record
        for trade in self.trades:
            if trade['id'] == position_id:
                trade['exit_price'] = exit_price
                trade['exit_time'] = position['exit_time']
                trade['profit_loss'] = profit_loss
                trade['profit_loss_percent'] = profit_loss_percent
                trade['close_reason'] = reason
                trade['status'] = 'closed'
                break
    
    def _calculate_results(self, initial_balance):
        """
        Calculate backtest results.
        
        Args:
            initial_balance (float): Initial balance
            
        Returns:
            dict: Backtest results
        """
        # Calculate trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit_loss'] > 0])
        losing_trades = len([t for t in self.trades if t['profit_loss'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = sum([t['profit_loss'] for t in self.trades if t['profit_loss'] > 0])
        gross_loss = sum([t['profit_loss'] for t in self.trades if t['profit_loss'] < 0])
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        net_profit = gross_profit + gross_loss
        final_balance = initial_balance + net_profit
        
        # Calculate return
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        # Calculate drawdown
        equity_curve_df = pd.DataFrame(self.equity_curve)
        equity_curve_df.set_index('timestamp', inplace=True)
        
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['peak']) / equity_curve_df['peak'] * 100
        
        max_drawdown = abs(equity_curve_df['drawdown'].min())
        
        # Calculate average trade
        avg_profit = net_profit / total_trades if total_trades > 0 else 0
        avg_profit_percent = sum([t['profit_loss_percent'] for t in self.trades]) / total_trades if total_trades > 0 else 0
        
        # Calculate average winning and losing trade
        avg_winning_trade = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_losing_trade = gross_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate average trade duration
        trade_durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in self.trades if t['exit_time'] is not None]
        avg_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
        
        # Calculate Sharpe ratio
        if len(equity_curve_df) > 1:
            equity_curve_df['returns'] = equity_curve_df['equity'].pct_change()
            sharpe_ratio = np.sqrt(252) * equity_curve_df['returns'].mean() / equity_curve_df['returns'].std() if equity_curve_df['returns'].std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Return results
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'return': total_return,
            'max_drawdown': max_drawdown,
            'avg_profit': avg_profit,
            'avg_profit_percent': avg_profit_percent,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'avg_trade_duration': avg_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
    
    def plot_results(self, results=None):
        """
        Plot backtest results.
        
        Args:
            results (dict): Backtest results
        """
        if results is None:
            results = self._calculate_results(10000)
        
        # Create directory for plots
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        equity_curve_df = pd.DataFrame(results['equity_curve'])
        equity_curve_df.set_index('timestamp', inplace=True)
        equity_curve_df['equity'].plot()
        plt.title(f'Equity Curve - {self.symbol} ({self.timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig(f'plots/equity_curve_{self.symbol}_{self.timeframe}.png')
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['peak']) / equity_curve_df['peak'] * 100
        equity_curve_df['drawdown'].plot()
        plt.title(f'Drawdown - {self.symbol} ({self.timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(f'plots/drawdown_{self.symbol}_{self.timeframe}.png')
        
        # Plot trade results
        plt.figure(figsize=(12, 6))
        trade_results = [t['profit_loss'] for t in results['trades']]
        plt.bar(range(len(trade_results)), trade_results)
        plt.title(f'Trade Results - {self.symbol} ({self.timeframe})')
        plt.xlabel('Trade #')
        plt.ylabel('Profit/Loss ($)')
        plt.grid(True)
        plt.savefig(f'plots/trade_results_{self.symbol}_{self.timeframe}.png')
        
        # Plot trade distribution
        plt.figure(figsize=(12, 6))
        plt.hist(trade_results, bins=20)
        plt.title(f'Trade Distribution - {self.symbol} ({self.timeframe})')
        plt.xlabel('Profit/Loss ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'plots/trade_distribution_{self.symbol}_{self.timeframe}.png')
        
        # Plot monthly returns
        plt.figure(figsize=(12, 6))
        equity_curve_df['month'] = equity_curve_df.index.to_period('M')
        monthly_returns = equity_curve_df.groupby('month')['equity'].last().pct_change() * 100
        monthly_returns.plot(kind='bar')
        plt.title(f'Monthly Returns - {self.symbol} ({self.timeframe})')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.savefig(f'plots/monthly_returns_{self.symbol}_{self.timeframe}.png')
        
        logger.info("Plots saved to 'plots' directory.")


if __name__ == '__main__':
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(initial_balance=10000)
    backtester.plot_results(results) 