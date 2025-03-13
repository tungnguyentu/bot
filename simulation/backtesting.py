import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from binance.client import Client
import os
import pickle
import time
from tqdm import tqdm

logger = logging.getLogger("BinanceBot.BacktestEngine")

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies with historical data
    """
    def __init__(self, starting_balance=1000.0, start_date=None, end_date=None):
        """
        Initialize the backtesting engine
        
        Args:
            starting_balance (float): Initial balance for backtesting
            start_date (str): Start date for backtesting (YYYY-MM-DD)
            end_date (str): End date for backtesting (YYYY-MM-DD)
        """
        self.balance = starting_balance
        self.initial_balance = starting_balance
        self.positions = {}
        self.orders = {}
        self.historical_data = {}
        self.equity_curve = []
        self.trades = []
        
        # Set date range
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        logger.info(f"Backtesting engine initialized with {starting_balance} USDT")
        if self.start_date and self.end_date:
            logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
    
    def load_historical_data(self, symbol, timeframe):
        """
        Load historical data for the specified symbol and timeframe
        First checks if data is cached, otherwise fetches from Binance API
        """
        cache_file = f"data/{symbol}_{timeframe}_historical.pkl"
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded historical data for {symbol} from cache")
                self.historical_data[symbol] = data
                return data
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}")
        
        # Fetch from Binance API if no cache
        logger.info(f"Fetching historical data for {symbol} from Binance API")
        
        # Initialize Binance client
        client = Client()
        
        # Map timeframe to Binance kline interval
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }
        
        # Convert dates to millisecond timestamps for Binance API
        start_ts = int(self.start_date.timestamp() * 1000) if self.start_date else None
        end_ts = int(self.end_date.timestamp() * 1000) if self.end_date else None
        
        # Fetch data in chunks to handle large datasets
        all_klines = []
        
        # If no start date is specified, get last 500 candles
        if not start_ts:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval_map[timeframe],
                limit=500
            )
            all_klines.extend(klines)
        else:
            # Fetch data in chunks of 1000 (Binance API limit)
            temp_start = start_ts
            while True:
                klines = client.futures_klines(
                    symbol=symbol,
                    interval=interval_map[timeframe],
                    startTime=temp_start,
                    endTime=end_ts,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # If we received less than 1000 candles, we've reached the end
                if len(klines) < 1000:
                    break
                
                # Update start time for next chunk
                temp_start = klines[-1][0] + 1  # Add 1ms to avoid duplicate candle
        
        # Convert to DataFrame
        data = []
        for k in all_klines:
            data.append({
                'timestamp': k[0],
                'datetime': datetime.fromtimestamp(k[0] / 1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            })
        
        # Sort by timestamp
        data = sorted(data, key=lambda x: x['timestamp'])
        
        # Store data
        self.historical_data[symbol] = data
        
        # Cache the data for future use
        try:
            os.makedirs("data", exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Cached historical data for {symbol}")
        except Exception as e:
            logger.warning(f"Error caching data: {e}")
        
        logger.info(f"Loaded {len(data)} candles for {symbol} from {data[0]['datetime']} to {data[-1]['datetime']}")
        
        return data
    
    def run_backtest(self, symbol, strategy, risk_manager, config):
        """
        Run backtest for the specified symbol using the provided strategy
        """
        # Check if we have historical data
        if symbol not in self.historical_data:
            logger.error(f"No historical data loaded for {symbol}")
            return None
        
        data = self.historical_data[symbol]
        
        # Reset backtest state
        self.balance = self.initial_balance
        self.positions = {}
        self.orders = {}
        self.equity_curve = []
        self.trades = []
        
        # Initialize position
        self.positions[symbol] = {
            'symbol': symbol,
            'size': 0,
            'entry_price': 0,
            'mark_price': 0,
            'unrealized_pnl': 0,
            'liquidation_price': None
        }
        
        # Save initial equity point
        self.equity_curve.append({
            'timestamp': data[0]['timestamp'],
            'datetime': data[0]['datetime'],
            'equity': self.balance,
            'drawdown_pct': 0
        })
        
        # Set up lookahead prevention (we don't use future data)
        lookback = max(
            config.BB_PERIOD,
            config.RSI_PERIOD,
            config.MACD_SLOW + config.MACD_SIGNAL,
            config.ATR_PERIOD
        ) + 10  # Add buffer
        
        # Process each candle
        peak_equity = self.balance
        max_drawdown_pct = 0
        
        for i in tqdm(range(lookback, len(data)), desc=f"Backtesting {symbol}", ncols=100):
            # Get historical data up to current candle
            historical_slice = data[i-lookback:i]
            current_candle = data[i]
            
            # Update current price
            current_price = current_candle['close']
            self.positions[symbol]['mark_price'] = current_price
            
            # Update unrealized P&L
            if self.positions[symbol]['size'] != 0:
                entry_price = self.positions[symbol]['entry_price']
                size = self.positions[symbol]['size']
                
                if size > 0:  # Long position
                    self.positions[symbol]['unrealized_pnl'] = (current_price - entry_price) * size
                else:  # Short position
                    self.positions[symbol]['unrealized_pnl'] = (entry_price - current_price) * abs(size)
            
            # Generate trading signal
            signal = strategy.generate_signal(symbol, historical_slice)
            
            # Execute trading logic
            if signal:
                direction, reasoning = signal
                position = self.positions[symbol]
                
                # Check if we already have an open position
                if position['size'] == 0:
                    # Enter new position
                    self._execute_backtest_trade(symbol, direction, current_price, historical_slice, risk_manager, config)
                else:
                    # Check if we need to close the position
                    current_direction = "LONG" if position['size'] > 0 else "SHORT"
                    if direction != current_direction:
                        self._close_backtest_trade(symbol, current_direction, current_price, reasoning)
            
            # Check for stop loss and take profit triggers
            self._check_sl_tp_triggers(symbol, current_price)
            
            # Update equity curve
            current_equity = self.balance
            if self.positions[symbol]['size'] != 0:
                current_equity += self.positions[symbol]['unrealized_pnl']
            
            # Update peak equity and drawdown
            peak_equity = max(peak_equity, current_equity)
            drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            
            self.equity_curve.append({
                'timestamp': current_candle['timestamp'],
                'datetime': current_candle['datetime'],
                'equity': current_equity,
                'drawdown_pct': drawdown_pct
            })
        
        # Calculate backtest metrics
        metrics = self._calculate_metrics(max_drawdown_pct)
        
        return metrics
    
    def _execute_backtest_trade(self, symbol, direction, current_price, historical_data, risk_manager, config):
        """Execute a trade in backtest mode"""
        # Calculate position size based on risk management
        position_size = self._calculate_backtest_position_size(symbol, direction, current_price, historical_data, config)
        
        # Get ATR for stop loss and take profit
        df = pd.DataFrame(historical_data)
        from indicators.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        df = ti.add_atr(df, config.ATR_PERIOD)
        atr = df['atr'].iloc[-1]
        
        # Calculate entry price with slippage
        slippage = current_price * 0.0005  # 0.05% slippage
        if direction == "LONG":
            entry_price = current_price + slippage
            stop_loss = entry_price - (atr * config.SL_ATR_MULTIPLIER)
            tp1 = entry_price + (atr * config.PARTIAL_TP_ATR_MULTIPLIER)
            tp2 = entry_price + (atr * config.FULL_TP_ATR_MULTIPLIER)
        else:  # SHORT
            entry_price = current_price - slippage
            stop_loss = entry_price + (atr * config.SL_ATR_MULTIPLIER)
            tp1 = entry_price - (atr * config.PARTIAL_TP_ATR_MULTIPLIER)
            tp2 = entry_price - (atr * config.FULL_TP_ATR_MULTIPLIER)
        
        # Update position
        self.positions[symbol]['size'] = position_size if direction == "LONG" else -position_size
        self.positions[symbol]['entry_price'] = entry_price
        
        # Record orders
        self.orders[symbol] = {
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp1_size': position_size * config.PARTIAL_TP_SIZE,
            'tp2_size': position_size * (1 - config.PARTIAL_TP_SIZE)
        }
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_time': historical_data[-1]['datetime'],
            'entry_price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
        }
        self.trades.append(trade)
        
        logger.debug(f"Backtest: {direction} {position_size} {symbol} at {entry_price}")
    
    def _close_backtest_trade(self, symbol, direction, current_price, reasoning):
        """Close a position in backtest mode"""
        position = self.positions[symbol]
        size = abs(position['size'])
        entry_price = position['entry_price']
        
        # Calculate exit price with slippage
        slippage = current_price * 0.0005  # 0.05% slippage
        if direction == "LONG":
            exit_price = current_price - slippage
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            exit_price = current_price + slippage
            pnl = (entry_price - exit_price) * size
        
        # Update balance
        self.balance += pnl
        
        # Update position
        self.positions[symbol]['size'] = 0
        self.positions[symbol]['entry_price'] = 0
        self.positions[symbol]['unrealized_pnl'] = 0
        
        # Clean up orders
        if symbol in self.orders:
            del self.orders[symbol]
        
        # Complete the trade record
        for trade in reversed(self.trades):
            if trade['symbol'] == symbol and 'exit_price' not in trade:
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = exit_price
                trade['pnl'] = pnl
                trade['pnl_pct'] = (pnl / (entry_price * size)) * 100
                trade['reason'] = 'Signal reversal'
                break
        
        logger.debug(f"Backtest: Closed {direction} position on {symbol} at {exit_price}, PnL: {pnl}")
    
    def _check_sl_tp_triggers(self, symbol, current_price):
        """Check if stop loss or take profit levels are triggered"""
        if symbol not in self.orders or self.positions[symbol]['size'] == 0:
            return
        
        position = self.positions[symbol]
        orders = self.orders[symbol]
        
        # Check stop loss
        if position['size'] > 0 and current_price <= orders['stop_loss']:  # Long SL
            self._execute_sl_tp(symbol, "LONG", current_price, "Stop Loss", position['size'])
        elif position['size'] < 0 and current_price >= orders['stop_loss']:  # Short SL
            self._execute_sl_tp(symbol, "SHORT", current_price, "Stop Loss", abs(position['size']))
        
        # Check take profit 1
        elif position['size'] > 0 and current_price >= orders['tp1']:  # Long TP1
            partial_size = orders['tp1_size']
            self._execute_sl_tp(symbol, "LONG", current_price, "Take Profit 1", partial_size)
        elif position['size'] < 0 and current_price <= orders['tp1']:  # Short TP1
            partial_size = orders['tp1_size']
            self._execute_sl_tp(symbol, "SHORT", current_price, "Take Profit 1", partial_size)
        
        # Check take profit 2
        elif position['size'] > 0 and current_price >= orders['tp2']:  # Long TP2
            remaining_size = abs(position['size'])
            self._execute_sl_tp(symbol, "LONG", current_price, "Take Profit 2", remaining_size)
        elif position['size'] < 0 and current_price <= orders['tp2']:  # Short TP2
            remaining_size = abs(position['size'])
            self._execute_sl_tp(symbol, "SHORT", current_price, "Take Profit 2", remaining_size)
    
    def _execute_sl_tp(self, symbol, direction, current_price, reason, size):
        """Execute a stop loss or take profit in backtest"""
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Calculate exit price with slippage
        slippage = current_price * 0.001  # Increased slippage for SL/TP (0.1%)
        if direction == "LONG":
            exit_price = current_price - slippage
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            exit_price = current_price + slippage
            pnl = (entry_price - exit_price) * size
        
        # Update balance
        self.balance += pnl
        
        # Update position
        remaining_size = abs(position['size']) - size
        if remaining_size <= 0.0001:  # Close position if remaining size is negligible
            self.positions[symbol]['size'] = 0
            self.positions[symbol]['entry_price'] = 0
            if symbol in self.orders:
                del self.orders[symbol]
        else:
            # Update position size
            self.positions[symbol]['size'] = remaining_size if position['size'] > 0 else -remaining_size
        
        # Complete the trade record for partial/full close
        trade_found = False
        for trade in reversed(self.trades):
            if trade['symbol'] == symbol and 'exit_price' not in trade:
                trade_found = True
                # For full close
                if remaining_size <= 0.0001:
                    trade['exit_time'] = datetime.now()
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['pnl_pct'] = (pnl / (entry_price * size)) * 100
                    trade['reason'] = reason
                # For partial close, create a new trade record for the closed portion
                else:
                    closed_trade = trade.copy()
                    closed_trade['exit_time'] = datetime.now()
                    closed_trade['exit_price'] = exit_price
                    closed_trade['size'] = size
                    closed_trade['pnl'] = pnl
                    closed_trade['pnl_pct'] = (pnl / (entry_price * size)) * 100
                    closed_trade['reason'] = reason
                    self.trades.append(closed_trade)
                break
        
        if not trade_found:
            logger.warning(f"No matching trade found for SL/TP execution: {symbol} {direction} {reason}")
        
        logger.debug(f"Backtest: {reason} executed for {symbol} {direction} at {exit_price}, PnL: {pnl}")
    
    def _calculate_backtest_position_size(self, symbol, direction, current_price, historical_data, config):
        """Calculate position size based on risk management for backtest"""
        # Calculate ATR for stop loss
        df = pd.DataFrame(historical_data)
        from indicators.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        df = ti.add_atr(df, config.ATR_PERIOD)
        atr = df['atr'].iloc[-1]
        
        # Calculate stop loss distance
        sl_distance = atr * config.SL_ATR_MULTIPLIER
        
        # Calculate risk per trade in USD
        risk_amount = self.balance * config.ACCOUNT_RISK_PER_TRADE
        
        # Calculate position size
        position_size = (risk_amount * self.balance) / sl_distance
        
        # Apply leverage
        position_size = position_size * config.LEVERAGE
        
        return position_size
    
    def _calculate_metrics(self, max_drawdown_pct):
        """Calculate performance metrics from backtest results"""
        # Filter completed trades
        completed_trades = [t for t in self.trades if 'exit_price' in t]
        
        # Calculate basic metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profit = sum([t.get('pnl', 0) for t in completed_trades if t.get('pnl', 0) > 0])
        gross_loss = sum([t.get('pnl', 0) for t in completed_trades if t.get('pnl', 0) <= 0])
        
        avg_profit = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Calculate return metrics
        final_balance = self.balance
        total_return = ((final_balance / self.initial_balance) - 1) * 100
        
        # Calculate Sharpe ratio (approximation)
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Return all metrics
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': final_balance,
            'total_return': total_return
        }
    
    def plot_results(self, symbol):
        """Generate equity curve and drawdown plots"""
        if not self.equity_curve:
            logger.warning("No equity curve data available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Convert data to DataFrame
            df = pd.DataFrame(self.equity_curve)
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(df['datetime'], df['equity'], label='Equity', color='blue')
            ax1.set_title(f'Backtest Results: {symbol}')
            ax1.set_ylabel('Equity (USDT)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Mark trade entries and exits
            for trade in self.trades:
                if 'exit_time' in trade:
                    color = 'green' if trade.get('pnl', 0) > 0 else 'red'
                    ax1.plot([trade['entry_time'], trade['exit_time']], 
                             [self.initial_balance, self.initial_balance], 
                             marker='o', color=color, alpha=0.5, linewidth=0)
            
            # Plot drawdown
            ax2.fill_between(df['datetime'], df['drawdown_pct'], 0, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Format dates
            date_format = DateFormatter('%Y-%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()
            
            # Add summary stats as text
            metrics = self._calculate_metrics(max(df['drawdown_pct']))
            stats_text = (
                f"Total Return: {metrics['total_return']:.2f}%\n"
                f"Win Rate: {metrics['win_rate']:.2f}%\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
            )
            ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save the plot
            os.makedirs("results", exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"results/{symbol}_backtest.png")
            plt.close()
            
            logger.info(f"Backtest plot saved to results/{symbol}_backtest.png")
            
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")
