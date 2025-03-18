import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

sys.path.append('/Users/tungnt/Downloads/game')
import config
from data.market_data import MarketData
from indicators.technical_indicators import TechnicalIndicators
from strategies.scalping import ScalpingStrategy
from strategies.swing import SwingStrategy
from strategies.strategy_switcher import StrategySwitcher

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, client, symbols=None, start_date=None, end_date=None, initial_balance=10000):
        """
        Initialize the Backtester.
        
        Args:
            client: Binance API client instance
            symbols: List of trading symbols (defaults to config)
            start_date: Start date for backtesting (defaults to config)
            end_date: End date for backtesting (defaults to config)
            initial_balance: Initial account balance for backtesting
        """
        self.client = client
        self.market_data = MarketData(client)
        self.symbols = symbols if symbols else config.TRADING_SYMBOLS
        self.start_date = start_date if start_date else config.BACKTEST_START_DATE
        self.end_date = end_date if end_date else config.BACKTEST_END_DATE
        self.initial_balance = initial_balance
        
        self.scalping_strategy = ScalpingStrategy()
        self.swing_strategy = SwingStrategy()
        
        # Initialize results storage
        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }
        
        # Performance metrics
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.max_drawdown = 0
        self.open_positions = {}
        
    def run_backtest(self, use_strategy_switcher=True):
        """
        Run the backtest across all symbols.
        
        Args:
            use_strategy_switcher: If True, dynamically switch between strategies
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Initialize equity curve with starting balance
        self.results["equity_curve"] = [{
            "timestamp": self.start_date,
            "balance": self.initial_balance
        }]
        
        # Load and prepare data for all symbols in parallel
        with ThreadPoolExecutor(max_workers=min(len(self.symbols), 5)) as executor:
            future_to_symbol = {
                executor.submit(self._load_data, symbol): symbol for symbol in self.symbols
            }
            
            symbol_data = {}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        symbol_data[symbol] = data
                        logger.info(f"Loaded {len(data)} candles for {symbol}")
                    else:
                        logger.warning(f"No data loaded for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
        
        if not symbol_data:
            logger.error("No data loaded for any symbol. Aborting backtest.")
            return self.results
            
        # Determine the common date range across all symbols
        start_dates = [df.index[0] for df in symbol_data.values()]
        end_dates = [df.index[-1] for df in symbol_data.values()]
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        logger.info(f"Common date range: {common_start} to {common_end}")
        
        # Run simulation day by day
        current_date = common_start
        day_increment = timedelta(days=1)
        
        while current_date <= common_end:
            # Process each symbol
            for symbol, df in symbol_data.items():
                # Get data up to current date
                current_data = df[df.index <= current_date]
                if len(current_data) < 100:  # Minimum data for indicators
                    continue
                
                # Choose strategy
                if use_strategy_switcher:
                    # Calculate recent volatility for strategy selection
                    recent_data = current_data.iloc[-30:]  # Last 30 data points
                    if not recent_data.empty:
                        volatility = recent_data['close'].pct_change().std()
                        strategy = self.scalping_strategy if volatility > config.SWITCH_STRATEGY_VOLATILITY else self.swing_strategy
                else:
                    # Use a fixed strategy
                    strategy = self.scalping_strategy
                
                # Apply strategy to current data
                df_with_indicators = TechnicalIndicators.add_all_indicators(
                    current_data,
                    "scalping" if isinstance(strategy, ScalpingStrategy) else "swing"
                )
                
                # Get the latest full candle
                if len(df_with_indicators) > 1:
                    latest_candle = df_with_indicators.iloc[-1]
                    previous_candle = df_with_indicators.iloc[-2]
                    
                    # Check for open positions for this symbol
                    if symbol in self.open_positions:
                        self._process_open_position(symbol, latest_candle, strategy)
                    else:
                        # Look for new entry signals
                        signals = strategy.analyze(df_with_indicators)
                        self._process_new_signals(symbol, signals, latest_candle, strategy)
            
            # Move to next day
            current_date += day_increment
            
            # Update equity curve once per day
            self.results["equity_curve"].append({
                "timestamp": current_date.strftime("%Y-%m-%d"),
                "balance": self.balance
            })
            
            # Print progress every 30 days
            if current_date.day == 1 or current_date == common_end:
                logger.info(f"Backtesting progress: {current_date.strftime('%Y-%m-%d')} | Balance: ${self.balance:.2f}")
                
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed. Final balance: ${self.balance:.2f}")
        logger.info(f"Total trades: {len(self.results['trades'])}")
        
        return self.results
        
    def _load_data(self, symbol):
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame: OHLCV data for the symbol
        """
        # Determine the most granular timeframe needed
        if config.TRADING_TIMEFRAMES["scalping"] < config.TRADING_TIMEFRAMES["swing"]:
            timeframe = config.TRADING_TIMEFRAMES["scalping"]
        else:
            timeframe = config.TRADING_TIMEFRAMES["swing"]
            
        try:
            # Load historical data
            df = self.market_data.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=self.start_date,
                end_str=self.end_date
            )
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
            
    def _process_new_signals(self, symbol, signals, candle, strategy):
        """Process new trading signals."""
        signal = signals.get("signal", "neutral")
        strength = signals.get("strength", 0)
        reasoning = signals.get("reasoning", "")
        
        # Only take trades with sufficient strength
        min_strength_threshold = 0.4
        
        if signal in ["buy", "sell"] and strength >= min_strength_threshold:
            # Determine entry price and calculate position size
            entry_price = candle["close"]
            
            # Calculate stop loss and take profit based on strategy
            if signal == "buy":
                stop_loss = strategy.get_stop_loss_price(entry_price, "buy", candle.get("atr"))
                take_profit = strategy.get_take_profit_price(entry_price, "buy", candle.get("atr"))
            else:  # sell
                stop_loss = strategy.get_stop_loss_price(entry_price, "sell", candle.get("atr"))
                take_profit = strategy.get_take_profit_price(entry_price, "sell", candle.get("atr"))
                
            # Calculate risk (1% of account balance)
            risk_amount = self.balance * config.RISK_PER_TRADE
            
            # Calculate position size based on risk
            price_distance = abs(entry_price - stop_loss)
            position_size = risk_amount / price_distance
            
            # Apply leverage
            position_size = position_size * config.TRADING_LEVERAGE
            
            # Adjust for position cost
            position_cost = position_size * entry_price / config.TRADING_LEVERAGE
            
            if position_cost > self.balance:
                # Not enough balance for this position
                return
                
            # Record the position
            self.open_positions[symbol] = {
                "entry_price": entry_price,
                "position_size": position_size,
                "side": signal,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": candle.name,  # Use the candle timestamp as entry time
                "max_price": entry_price if signal == "buy" else None,
                "min_price": entry_price if signal == "sell" else None
            }
            
            # Record the trade entry
            trade = {
                "symbol": symbol,
                "side": signal,
                "entry_price": entry_price,
                "position_size": position_size,
                "entry_time": candle.name,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy": strategy.name,
                "reasoning": reasoning
            }
            
            self.results["trades"].append(trade)
            
            logger.info(f"New {signal.upper()} position for {symbol} at {entry_price}")
            
    def _process_open_position(self, symbol, candle, strategy):
        """Process an open position for exit conditions."""
        position = self.open_positions[symbol]
        current_price = candle["close"]
        position_size = position["position_size"]
        entry_price = position["entry_price"]
        side = position["side"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Update max/min price for trailing stop calculation
        if side == "buy" and current_price > position.get("max_price", 0):
            position["max_price"] = current_price
        elif side == "sell" and (position.get("min_price") is None or current_price < position["min_price"]):
            position["min_price"] = current_price
            
        # Check for exit conditions
        exit_reason = None
        exit_price = None
        
        # 1. Stop Loss
        if (side == "buy" and current_price <= stop_loss) or (side == "sell" and current_price >= stop_loss):
            exit_reason = "Stop Loss"
            exit_price = stop_loss  # Assume we get the exact stop loss price
            
        # 2. Take Profit
        elif (side == "buy" and current_price >= take_profit) or (side == "sell" and current_price <= take_profit):
            exit_reason = "Take Profit"
            exit_price = take_profit  # Assume we get the exact take profit price
            
        # 3. Trailing Stop (if activated)
        elif side == "buy" and position.get("max_price"):
            # For long positions
            price_movement = (position["max_price"] - entry_price) / entry_price
            if price_movement > config.TRAILING_STOP_ACTIVATION:
                # Calculate trailing stop level
                trailing_stop = position["max_price"] * (1 - config.TRAILING_STOP_ACTIVATION/2)
                if trailing_stop > stop_loss and current_price <= trailing_stop:
                    exit_reason = "Trailing Stop"
                    exit_price = trailing_stop
                    
        elif side == "sell" and position.get("min_price"):
            # For short positions
            price_movement = (entry_price - position["min_price"]) / entry_price
            if price_movement > config.TRAILING_STOP_ACTIVATION:
                # Calculate trailing stop level
                trailing_stop = position["min_price"] * (1 + config.TRAILING_STOP_ACTIVATION/2)
                if trailing_stop < stop_loss and current_price >= trailing_stop:
                    exit_reason = "Trailing Stop"
                    exit_price = trailing_stop
        
        # Process exit if conditions are met
        if exit_reason and exit_price:
            # Calculate P/L
            if side == "buy":
                pnl = (exit_price - entry_price) * position_size
            else:  # sell
                pnl = (entry_price - exit_price) * position_size
                
            # Apply leverage to P/L calculation
            pnl = pnl * config.TRADING_LEVERAGE
            
            # Update account balance
            self.balance += pnl
            
            # Update max balance and drawdown
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            else:
                current_drawdown = (self.max_balance - self.balance) / self.max_balance
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
            
            # Complete the trade record
            for trade in self.results["trades"]:
                if (trade["symbol"] == symbol and 
                    trade["side"] == side and 
                    trade.get("exit_time") is None):
                    
                    trade["exit_time"] = candle.name
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = exit_reason
                    trade["pnl"] = pnl
                    trade["roi"] = pnl / (entry_price * position_size / config.TRADING_LEVERAGE)
                    break
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            logger.info(f"Closed {side} position for {symbol} at {exit_price} | Reason: {exit_reason} | P/L: ${pnl:.2f}")
    
    def _calculate_performance_metrics(self):
        """Calculate and store performance metrics."""
        trades = self.results["trades"]
        
        if not trades:
            self.results["metrics"] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit": 0,
                "total_loss": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "avg_profit_per_trade": 0,
                "avg_win": 0,
                "avg_loss": 0,
            }
            return
        
        # Count completed trades
        completed_trades = [t for t in trades if t.get("exit_time")]
        
        if not completed_trades:
            return
            
        # Calculate basic metrics
        winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in completed_trades if t.get("pnl", 0) <= 0]
        
        total_profit = sum([t.get("pnl", 0) for t in winning_trades])
        total_loss = abs(sum([t.get("pnl", 0) for t in losing_trades]))
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate returns for Sharpe ratio
        if len(self.results["equity_curve"]) > 1:
            # Convert equity curve to dataframe
            equity_df = pd.DataFrame(self.results["equity_curve"])
            equity_df["balance"] = equity_df["balance"].astype(float)
            
            # Calculate daily returns
            equity_df["returns"] = equity_df["balance"].pct_change()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            avg_return = equity_df["returns"].mean()
            std_return = equity_df["returns"].std()
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Store metrics
        self.results["metrics"] = {
            "total_trades": len(completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_profit_per_trade": sum([t.get("pnl", 0) for t in completed_trades]) / len(completed_trades) if completed_trades else 0,
            "avg_win": sum([t.get("pnl", 0) for t in winning_trades]) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum([t.get("pnl", 0) for t in losing_trades]) / len(losing_trades) if losing_trades else 0,
        }
        
        logger.info(f"Performance metrics: Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}, Max drawdown: {self.max_drawdown:.2%}")
        
    def plot_equity_curve(self, save_path=None):
        """
        Plot the equity curve.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        if not self.results["equity_curve"]:
            logger.warning("No equity curve data to plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract data
        dates = [datetime.strptime(ec["timestamp"], "%Y-%m-%d") for ec in self.results["equity_curve"]]
        balances = [ec["balance"] for ec in self.results["equity_curve"]]
        
        # Plot equity curve
        plt.plot(dates, balances, 'b-', label='Account Balance')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Balance (USD)')
        plt.title('Backtesting Equity Curve')
        plt.grid(True)
        
        # Add starting balance line for reference
        plt.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
        
        # Add legend
        plt.legend()
        
        # Rotate x-axis dates for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self, report_path=None):
        """
        Generate a detailed backtest report.
        
        Args:
            report_path: Path to save the report (optional)
        """
        metrics = self.results["metrics"]
        trades = self.results["trades"]
        
        report = []
        report.append("=" * 60)
        report.append("BACKTESTING REPORT")
        report.append("=" * 60)
        report.append(f"Start Date: {self.start_date}")
        report.append(f"End Date: {self.end_date}")
        report.append(f"Symbols: {', '.join(self.symbols)}")
        report.append(f"Initial Balance: ${self.initial_balance:.2f}")
        report.append(f"Final Balance: ${self.balance:.2f}")
        report.append(f"Total Return: {(self.balance - self.initial_balance) / self.initial_balance:.2%}")
        report.append("-" * 60)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 60)
        report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Average Profit per Trade: ${metrics.get('avg_profit_per_trade', 0):.2f}")
        report.append(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        report.append(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        
        # Trading breakdown by symbol
        report.append("-" * 60)
        report.append("TRADING BREAKDOWN BY SYMBOL")
        report.append("-" * 60)
        
        symbol_stats = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
                
            if "pnl" in trade:
                symbol_stats[symbol]["trades"] += 1
                symbol_stats[symbol]["pnl"] += trade["pnl"]
                if trade["pnl"] > 0:
                    symbol_stats[symbol]["wins"] += 1
                else:
                    symbol_stats[symbol]["losses"] += 1
        
        for symbol, stats in symbol_stats.items():
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            report.append(f"{symbol}: {stats['trades']} trades, {win_rate:.2%} win rate, ${stats['pnl']:.2f} P/L")
        
        # Trading breakdown by strategy
        report.append("-" * 60)
        report.append("TRADING BREAKDOWN BY STRATEGY")
        report.append("-" * 60)
        
        strategy_stats = {}
        for trade in trades:
            strategy = trade.get("strategy", "Unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
                
            if "pnl" in trade:
                strategy_stats[strategy]["trades"] += 1
                strategy_stats[strategy]["pnl"] += trade["pnl"]
                if trade["pnl"] > 0:
                    strategy_stats[strategy]["wins"] += 1
                else:
                    strategy_stats[strategy]["losses"] += 1
        
        for strategy, stats in strategy_stats.items():
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            report.append(f"{strategy}: {stats['trades']} trades, {win_rate:.2%} win rate, ${stats['pnl']:.2f} P/L")
            
        report_str = "\n".join(report)
        
        if report_path:
            try:
                with open(report_path, 'w') as f:
                    f.write(report_str)
                logger.info(f"Backtest report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
                
        return report_str
