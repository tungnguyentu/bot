import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import config
from binance_client import BinanceClient
from strategies.scalping import ScalpingStrategy
from strategies.swing import SwingStrategy
from utils.logger import setup_logger

class Backtester:
    def __init__(self, strategy_type, symbol, start_date, end_date=None, initial_balance=10000):
        self.strategy_type = strategy_type  # 'scalping' or 'swing'
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.initial_balance = initial_balance
        
        self.logger = setup_logger('backtester', 'logs/backtesting.log')
        self.client = BinanceClient()
        
        # Create the appropriate strategy
        if strategy_type == 'scalping':
            self.strategy = ScalpingStrategy(self.client)
            self.timeframe = config.TIMEFRAMES['scalping']
        else:
            self.strategy = SwingStrategy(self.client)
            self.timeframe = config.TIMEFRAMES['swing']
            
        self.logger.info(
            f"Initialized backtester for {symbol} using {strategy_type} strategy "
            f"from {start_date} to {end_date}"
        )
        
    def run(self):
        """Run the backtesting simulation."""
        try:
            self.logger.info(f"Starting backtest for {self.symbol}")
            
            # Fetch historical data
            historical_data = self.fetch_historical_data()
            
            if historical_data.empty:
                self.logger.error("No historical data available for backtesting")
                return None
                
            # Prepare for simulation
            balance = self.initial_balance
            position = None
            trades = []
            equity_curve = [balance]
            
            # Run through each candle
            for i in range(100, len(historical_data)):
                current_date = historical_data.index[i]
                
                # Skip until start date
                if current_date < self.start_date:
                    continue
                    
                # Stop at end date
                if current_date > self.end_date:
                    break
                    
                # Get current price
                current_price = historical_data.iloc[i]['close']
                
                # Check if we need to close a position
                if position:
                    # Check if stop loss hit
                    if (position['type'] == 'long' and current_price <= position['stop_loss']) or \
                       (position['type'] == 'short' and current_price >= position['stop_loss']):
                        # Calculate PnL
                        pnl = self.calculate_pnl(position, current_price, 'stop_loss')
                        balance += pnl
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'type': position['type'],
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'pnl': pnl,
                            'exit_type': 'stop_loss'
                        })
                        
                        self.logger.info(
                            f"Stop Loss hit: {position['type']} position closed at {current_price} "
                            f"with PnL {pnl:.2f}"
                        )
                        position = None
                        
                    # Check if take profit hit
                    elif (position['type'] == 'long' and current_price >= position['take_profit']) or \
                         (position['type'] == 'short' and current_price <= position['take_profit']):
                        # Calculate PnL
                        pnl = self.calculate_pnl(position, current_price, 'take_profit')
                        balance += pnl
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'type': position['type'],
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'pnl': pnl,
                            'exit_type': 'take_profit'
                        })
                        
                        self.logger.info(
                            f"Take Profit hit: {position['type']} position closed at {current_price} "
                            f"with PnL {pnl:.2f}"
                        )
                        position = None
                
                # If no position, check for new signals
                if not position:
                    # Prepare data for signal generation
                    historical_subset = historical_data.iloc[i-100:i+1].copy()
                    
                    # Generate signal based on historical data
                    signal = self.generate_signal_from_data(historical_subset)
                    
                    if signal and signal['action'] in ['BUY', 'SELL']:
                        # Calculate position size (assuming 2% risk per trade)
                        risk_amount = balance * config.RISK_PER_TRADE
                        price_diff = abs(signal['price'] - signal['stop_loss'])
                        position_size = risk_amount / price_diff
                        
                        # Open new position
                        position = {
                            'type': 'long' if signal['action'] == 'BUY' else 'short',
                            'entry_price': signal['price'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'size': position_size,
                            'entry_date': current_date
                        }
                        
                        self.logger.info(
                            f"New {position['type']} position opened at {position['entry_price']} "
                            f"(SL: {position['stop_loss']}, TP: {position['take_profit']})"
                        )
                
                # Track equity
                equity_curve.append(balance + (self.calculate_unrealized_pnl(position, current_price) if position else 0))
            
            # Close any open position at the end
            if position:
                final_price = historical_data.iloc[-1]['close']
                pnl = self.calculate_pnl(position, final_price, 'end_of_backtest')
                balance += pnl
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': historical_data.index[-1],
                    'type': position['type'],
                    'entry': position['entry_price'],
                    'exit': final_price,
                    'pnl': pnl,
                    'exit_type': 'end_of_backtest'
                })
                
                self.logger.info(
                    f"End of backtest: {position['type']} position closed at {final_price} "
                    f"with PnL {pnl:.2f}"
                )
            
            # Calculate performance metrics
            results = self.calculate_performance(trades, equity_curve)
            
            # Log results
            self.logger.info(f"Backtest completed for {self.symbol}. Final balance: {balance:.2f}")
            for metric, value in results.items():
                self.logger.info(f"{metric}: {value}")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            return None
            
    def fetch_historical_data(self):
        """Fetch historical data for the trading pair."""
        try:
            self.logger.info(f"Fetching historical data for {self.symbol}")
            
            # Calculate date range
            end_date_str = self.end_date.strftime('%d %b %Y %H:%M:%S')
            
            # For simplicity, we're getting a fixed number of candles
            # In a real implementation, you might want to fetch based on date range
            data = self.client.get_historical_klines(
                self.symbol,
                self.timeframe,
                limit=1000  # Adjust based on your needs
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
            
    def generate_signal_from_data(self, df):
        """Generate trading signals from historical data."""
        try:
            if self.strategy_type == 'scalping':
                # Inject data into scalping strategy
                result = self.strategy.calculate_indicators(df)
                
                # Check the last row for signals
                last_row = result.iloc[-1]
                prev_row = result.iloc[-2]
                
                # Use the same signal generation logic as in the strategy
                if (last_row['rsi'] < self.strategy.params['rsi_oversold'] and 
                    last_row['close'] < last_row['lower_band'] and
                    last_row['close'] > last_row['ma_fast'] and
                    last_row['ma_fast'] > prev_row['ma_fast']):
                    
                    entry_price = last_row['close']
                    stop_loss = entry_price - (2 * (last_row['atr']))
                    take_profit = entry_price + (3 * (last_row['atr']))
                    
                    return {
                        'action': 'BUY',
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                elif (last_row['rsi'] > self.strategy.params['rsi_overbought'] and 
                      last_row['close'] > last_row['upper_band'] and
                      last_row['close'] < last_row['ma_fast'] and
                      last_row['ma_fast'] < prev_row['ma_fast']):
                    
                    entry_price = last_row['close']
                    stop_loss = entry_price + (2 * (last_row['atr']))
                    take_profit = entry_price - (3 * (last_row['atr']))
                    
                    return {
                        'action': 'SELL',
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
            else:  # swing
                # Inject data into swing strategy
                result = self.strategy.calculate_indicators(df)
                
                # Check the last row for signals
                last_row = result.iloc[-1]
                prev_row = result.iloc[-2]
                
                # Use the same signal generation logic as in the strategy
                if (last_row['close'] > last_row['senkou_span_a'] and
                    last_row['close'] > last_row['senkou_span_b'] and
                    last_row['macd'] > last_row['macd_signal'] and
                    prev_row['macd'] <= prev_row['macd_signal'] and
                    last_row['volume'] > last_row['volume_ma']):
                    
                    entry_price = last_row['close']
                    stop_loss = min(last_row['kijun_sen'], last_row['senkou_span_b']) * 0.99
                    take_profit = entry_price + (1.5 * (entry_price - stop_loss))
                    
                    return {
                        'action': 'BUY',
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                elif (last_row['close'] < last_row['senkou_span_a'] and
                      last_row['close'] < last_row['senkou_span_b'] and
                      last_row['macd'] < last_row['macd_signal'] and
                      prev_row['macd'] >= prev_row['macd_signal'] and
                      last_row['volume'] > last_row['volume_ma']):
                    
                    entry_price = last_row['close']
                    stop_loss = max(last_row['kijun_sen'], last_row['senkou_span_a']) * 1.01
                    take_profit = entry_price - (1.5 * (stop_loss - entry_price))
                    
                    return {
                        'action': 'SELL',
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            
            return None  # No signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal from historical data: {str(e)}")
            return None
            
    def calculate_pnl(self, position, current_price, exit_type):
        """Calculate PnL for a position."""
        try:
            if position['type'] == 'long':
                pnl = position['size'] * (current_price - position['entry_price'])
            else:  # short
                pnl = position['size'] * (position['entry_price'] - current_price)
                
            # Simulate trading fees (0.1% for entry and exit)
            fees = position['size'] * position['entry_price'] * 0.001 * 2
            
            return pnl - fees
            
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {str(e)}")
            return 0
            
    def calculate_unrealized_pnl(self, position, current_price):
        """Calculate unrealized PnL for an open position."""
        if not position:
            return 0
            
        try:
            if position['type'] == 'long':
                return position['size'] * (current_price - position['entry_price'])
            else:  # short
                return position['size'] * (position['entry_price'] - current_price)
                
        except Exception as e:
            self.logger.error(f"Error calculating unrealized PnL: {str(e)}")
            return 0
            
    def calculate_performance(self, trades, equity_curve):
        """Calculate performance metrics."""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0.0,  # Added missing key
                    'gross_profit': 0.0,  # Added missing key
                    'gross_loss': 0.0     # Added missing key
                }
                
            # Convert trades to DataFrame for analysis
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0.0
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 0.0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk metrics
            equity_curve = np.array(equity_curve)
            
            # Calculate total return
            total_return = 0.0
            if len(equity_curve) > 1:
                total_return = (equity_curve[-1] / equity_curve[0]) - 1
            
            # Calculate daily returns (assuming equity_curve is daily)
            returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])
            
            # Sharpe ratio (annualized)
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Max drawdown
            max_drawdown = 0.0
            if len(equity_curve) > 1:
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak
                max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0.0,  # Added missing key
                'gross_profit': 0.0,  # Added missing key
                'gross_loss': 0.0,    # Added missing key
                'error': str(e)
            }
            
    def plot_results(self, trades, equity_curve):
        """Plot backtest results."""
        try:
            # Create figure and grid
            fig = plt.figure(figsize=(15, 10))
            
            # Equity curve
            plt.subplot(2, 1, 1)
            plt.plot(equity_curve)
            plt.title(f'Equity Curve - {self.symbol} ({self.strategy_type})')
            plt.ylabel('Account Value')
            plt.grid(True)
            
            # Trade distribution
            plt.subplot(2, 2, 3)
            trades_df = pd.DataFrame(trades)
            trades_df['pnl'].hist(bins=20)
            plt.title('PnL Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
            
            # Drawdown
            plt.subplot(2, 2, 4)
            equity_array = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (peak - equity_array) / peak * 100
            plt.plot(drawdown)
            plt.title('Drawdown (%)')
            plt.xlabel('Trade #')
            plt.ylabel('Drawdown %')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'backtest_{self.symbol}_{self.strategy_type}.png')
            plt.close()
            
            self.logger.info(f"Plots saved to backtest_{self.symbol}_{self.strategy_type}.png")
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
