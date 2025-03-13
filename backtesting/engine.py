import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.initial_balance = config.initial_balance
        self.leverage = config.initial_leverage
        self.commission = 0.0004  # 0.04% per trade
        
        # Backtesting results
        self.trades = []
        self.equity_curve = []
        self.balance_history = []
        
    def run(self, data):
        """Run backtest on historical data"""
        logger.info(f"Running backtest on {len(data)} candles of data")
        
        # Copy data to avoid modifying the original
        df = data.copy()
        
        # Initialize balance and position
        balance = self.initial_balance
        position = 0  # 0: no position, 1: long, -1: short
        position_price = 0
        position_size = 0
        entry_index = 0
        
        # Prepare for tracking results
        equity = []
        balance_history = []
        trades = []
        
        # Iterate through data
        for i in range(100, len(df)):  # Start with enough lookback for indicators
            current_row = df.iloc[i]
            current_index = df.index[i]
            current_price = current_row['close']
            
            # Calculate unrealized PnL if we have a position
            unrealized_pnl = 0
            if position != 0:
                if position == 1:  # Long position
                    unrealized_pnl = (current_price - position_price) / position_price * position_size * self.leverage
                else:  # Short position
                    unrealized_pnl = (position_price - current_price) / position_price * position_size * self.leverage
            
            # Current equity is balance + unrealized PnL
            current_equity = balance + unrealized_pnl
            equity.append(current_equity)
            balance_history.append(balance)
            
            # Get subset of data up to current point for feature generation
            data_subset = df.iloc[:i+1].copy()
            
            # Create account state for the model
            account_state = {
                'balance': balance,
                'position': position,
                'position_price': position_price,
                'position_size_usd': position_size
            }
            
            # Get action from model
            # Here we map the model output to actions
            # 0: do nothing, 1: close position, 2: open long, 3: open short
            model_prediction = self.model_manager.get_trading_action(data_subset, account_state)
            
            # Process the action
            if model_prediction == 1 and position != 0:
                # Close position
                if position == 1:  # Close long
                    pnl = (current_price - position_price) / position_price * position_size * self.leverage
                else:  # Close short
                    pnl = (position_price - current_price) / position_price * position_size * self.leverage
                
                # Subtract commission
                commission = position_size * self.commission
                pnl -= commission
                
                # Update balance
                balance += pnl
                
                # Record trade
                trades.append({
                    'entry_time': df.index[entry_index],
                    'exit_time': current_index,
                    'entry_price': position_price,
                    'exit_price': current_price,
                    'position': position,
                    'size': position_size,
                    'pnl': pnl,
                    'balance_after': balance
                })
                
                # Reset position
                position = 0
                position_price = 0
                position_size = 0
                
            elif model_prediction == 2 and position == 0:
                # Open long position
                # Calculate position size
                position_size = balance * self.config.position_size_percent
                
                # Set position data
                position = 1
                position_price = current_price
                entry_index = i
                
                # Subtract commission
                commission = position_size * self.commission
                balance -= commission
                
            elif model_prediction == 3 and position == 0:
                # Open short position
                # Calculate position size
                position_size = balance * self.config.position_size_percent
                
                # Set position data
                position = -1
                position_price = current_price
                entry_index = i
                
                # Subtract commission
                commission = position_size * self.commission
                balance -= commission
        
        # Close position at the end if still open
        if position != 0:
            current_price = df['close'].iloc[-1]
            
            if position == 1:  # Close long
                pnl = (current_price - position_price) / position_price * position_size * self.leverage
            else:  # Close short
                pnl = (position_price - current_price) / position_price * position_size * self.leverage
            
            # Subtract commission
            commission = position_size * self.commission
            pnl -= commission
            
            # Update balance
            balance += pnl
            
            # Record trade
            trades.append({
                'entry_time': df.index[entry_index],
                'exit_time': df.index[-1],
                'entry_price': position_price,
                'exit_price': current_price,
                'position': position,
                'size': position_size,
                'pnl': pnl,
                'balance_after': balance
            })
        
        # Store results
        self.trades = trades
        self.equity_curve = equity
        self.balance_history = balance_history
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(df.index[-len(equity):], equity, trades)
        
        # Generate backtest report
        self._generate_report(df.index[-len(equity):], equity, balance_history, trades, metrics)
        
        return metrics
        
    def _calculate_metrics(self, dates, equity, trades):
        """Calculate performance metrics from backtest results"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'total_return_pct': 0,
                'expectancy': 0
            }
        
        # Convert equity to numpy array for calculations
        equity_arr = np.array(equity)
        
        # Calculate returns
        returns = np.diff(equity_arr) / equity_arr[:-1]
        
        # Calculate Sharpe ratio (annualized)
        risk_free_rate = 0  # Assuming zero risk-free rate
        sharpe_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)  # 252 trading days
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate total return
        total_return = (equity_arr[-1] - equity_arr[0]) / equity_arr[0]
        
        # Calculate trade metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate expectancy
        expectancy = np.mean([t['pnl'] for t in trades]) if trades else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate * 100,  # as percentage
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,  # as percentage
            'total_return_pct': total_return * 100,  # as percentage
            'expectancy': expectancy
        }
        
    def _generate_report(self, dates, equity, balance_history, trades, metrics):
        """Generate backtest report with charts"""
        # Create directory for reports if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/backtest_report_{timestamp}.html"
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(dates, equity, label='Equity', color='blue')
        ax1.plot(dates, balance_history, label='Balance', color='green', linestyle='--')
        
        # Mark trades on the equity curve
        for trade in trades:
            if trade['position'] == 1:  # Long trade
                color = 'green' if trade['pnl'] > 0 else 'red'
                ax1.scatter(trade['entry_time'], balance_history[dates.get_loc(trade['entry_time'])], 
                          marker='^', color=color, s=100)
                ax1.scatter(trade['exit_time'], balance_history[dates.get_loc(trade['exit_time'])] + trade['pnl'], 
                          marker='v', color=color, s=100)
            else:  # Short trade
                color = 'green' if trade['pnl'] > 0 else 'red'
                ax1.scatter(trade['entry_time'], balance_history[dates.get_loc(trade['entry_time'])], 
                          marker='v', color=color, s=100)
                ax1.scatter(trade['exit_time'], balance_history[dates.get_loc(trade['exit_time'])] + trade['pnl'], 
                          marker='^', color=color, s=100)
        
        # Format x-axis dates
        date_format = DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_format)
        
        # Set axis labels and title
        ax1.set_title('Backtest Results')
        ax1.set_ylabel('Account Value (USDT)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100  # as percentage
        
        ax2.fill_between(dates, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Add metrics text
        metrics_text = (
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Total Return: {metrics['total_return_pct']:.2f}%\n"
            f"Expectancy: {metrics['expectancy']:.2f}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plot_path = f"reports/backtest_plot_{timestamp}.png"
        plt.savefig(plot_path)
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Backtest Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .metrics {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <ul>
                    <li>Total Trades: {metrics['total_trades']}</li>
                    <li>Win Rate: {metrics['win_rate']:.2f}%</li>
                    <li>Profit Factor: {metrics['profit_factor']:.2f}</li>
                    <li>Sharpe Ratio: {metrics['sharpe_ratio']:.2f}</li>
                    <li>Max Drawdown: {metrics['max_drawdown_pct']:.2f}%</li>
                    <li>Total Return: {metrics['total_return_pct']:.2f}%</li>
                    <li>Expectancy: {metrics['expectancy']:.2f} USDT per trade</li>
                </ul>
            </div>
            
            <h2>Equity Curve</h2>
            <img src="{os.path.basename(plot_path)}" alt="Equity Curve" style="width: 100%;">
            
            <h2>Trade List</h2>
            <table>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Position</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Size</th>
                    <th>PnL</th>
                    <th>Balance After</th>
                </tr>
        """
        
        # Add trade rows
        for trade in trades:
            pnl_class = "positive" if trade['pnl'] > 0 else "negative"
            position_type = "LONG" if trade['position'] == 1 else "SHORT"
            
            html_content += f"""
                <tr>
                    <td>{trade['entry_time'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{trade['exit_time'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{position_type}</td>
                    <td>{trade['entry_price']:.2f}</td>
                    <td>{trade['exit_price']:.2f}</td>
                    <td>{trade['size']:.6f}</td>
                    <td class="{pnl_class}">{trade['pnl']:.2f}</td>
                    <td>{trade['balance_after']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(report_filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report generated: {report_filename}")
        return report_filename
