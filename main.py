import logging
import sys
import os
import time
import schedule
import signal
import argparse
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import json
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import setup_logger
from data.market_data import MarketData
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager
from notifications.telegram_bot import TelegramNotifier
from strategies.strategy_switcher import StrategySwitcher
from backtesting.backtester import Backtester

# Global variables
running = True
client = None
risk_manager = None
order_manager = None
market_data = None
telegram = None
strategy_switcher = None

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down the bot."""
    global running
    logger = logging.getLogger(__name__)
    logger.info("Shutdown signal received. Closing gracefully...")
    running = False

def initialize_bot():
    """Initialize the trading bot components."""
    global client, risk_manager, order_manager, market_data, telegram, strategy_switcher
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing trading bot in {config.TRADING_MODE} mode")
    
    # Initialize Binance client
    try:
        # Use testnet for paper trading
        if config.TRADING_MODE == "paper_trading":
            logger.info("Connecting to Binance Futures testnet for paper trading")
            client = Client(
                config.BINANCE_API_KEY, 
                config.BINANCE_API_SECRET,
                testnet=True  # Use testnet
            )
        else:
            client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        
        # Test API connection
        server_time = client.get_server_time()
        logger.info(f"Connected to Binance API. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # For paper trading, log the test account balance
        if config.TRADING_MODE == "paper_trading":
            try:
                account_info = client.futures_account_balance()
                usdt_balance = next((item for item in account_info if item['asset'] == 'USDT'), {}).get('balance', 0)
                logger.info(f"Testnet account USDT balance: {usdt_balance}")
            except Exception as e:
                logger.error(f"Error fetching testnet account balance: {e}")
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error connecting to Binance: {e}")
        sys.exit(1)
        
    # Initialize components
    market_data = MarketData(client)
    risk_manager = RiskManager(client)
    order_manager = OrderManager(client, risk_manager)
    telegram = TelegramNotifier()
    strategy_switcher = StrategySwitcher(market_data)
    
    # Send initialization message
    api_type = "Testnet" if config.TRADING_MODE == "paper_trading" else "Production"
    telegram.send_system_status(f"Trading bot initialized successfully. Using Binance {api_type} API.")
    
    logger.info("All components initialized successfully")
    return True

def check_market_conditions(symbol):
    """Check market conditions and determine strategy for a symbol."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get current strategy for market conditions
        strategy = strategy_switcher.determine_best_strategy(symbol)
        strategy_params = strategy_switcher.get_strategy_params()
        
        logger.info(f"Selected {strategy_params['name']} strategy for {symbol}")
        
        # Fetch appropriate timeframe based on strategy
        timeframe = strategy_params["timeframe"]
        
        # Get historical data for analysis
        df = market_data.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            limit=100  # Get enough data for indicators
        )
        
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return None, None
            
        # Analyze market data
        from indicators.technical_indicators import TechnicalIndicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(
            df,
            "scalping" if strategy_params["name"] == "Scalping" else "swing"
        )
        
        # Get trading signals
        signals = strategy.analyze(df_with_indicators)
        
        logger.info(f"Analysis for {symbol}: {signals['signal']} with strength {signals['strength']}")
        
        return signals, strategy
        
    except Exception as e:
        logger.error(f"Error checking market conditions for {symbol}: {e}")
        return None, None

def process_trading_signals(symbol, signals, strategy):
    """Process trading signals and execute trades if appropriate."""
    logger = logging.getLogger(__name__)
    
    if not signals:
        return False
        
    signal = signals.get("signal", "neutral")
    strength = signals.get("strength", 0)
    reasoning = signals.get("reasoning", "No detailed analysis available")
    
    # Check if we have a valid signal with sufficient strength
    if signal == "neutral" or strength < 0.4:
        logger.info(f"No actionable signal for {symbol} (Signal: {signal}, Strength: {strength:.2f})")
        return False
    
    try:
        # Check for existing position
        if symbol in risk_manager.open_positions:
            # We already have a position, update trailing stop if needed
            current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
            order_manager.update_trailing_stop(symbol, current_price)
            return False
            
        # Get current price
        ticker = client.futures_ticker(symbol=symbol)
        price = float(ticker['lastPrice'])
        
        # Calculate stop loss and take profit levels
        if signal == "buy":
            stop_loss = strategy.get_stop_loss_price(price, "buy", None)  # ATR not available here
            take_profit = strategy.get_take_profit_price(price, "buy", None)
        else:  # sell
            stop_loss = strategy.get_stop_loss_price(price, "sell", None)
            take_profit = strategy.get_take_profit_price(price, "sell", None)
        
        # Execute the entry order
        order_side = "BUY" if signal == "buy" else "SELL"
        order = order_manager.execute_entry(
            symbol=symbol,
            side=order_side,
            price=None,  # Use market price
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order:
            # Send notification
            telegram.send_trade_notification(
                action="Entry",
                symbol=symbol,
                side=order_side,
                quantity=order.get("quantity", 0),
                price=price,
                reasoning=reasoning
            )
            
            logger.info(f"Executed {order_side} order for {symbol} at {price}")
            return True
        else:
            logger.warning(f"Failed to execute {order_side} order for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing trading signal for {symbol}: {e}")
        telegram.send_error_alert(f"Error processing signal for {symbol}: {str(e)}")
        return False

def check_open_positions():
    """Check and manage open positions."""
    logger = logging.getLogger(__name__)
    
    for symbol in list(risk_manager.open_positions.keys()):
        try:
            # Get current price
            ticker = client.futures_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            
            # Update trailing stop
            order_manager.update_trailing_stop(symbol, current_price)
            
            logger.debug(f"Updated position for {symbol}, current price: {current_price}")
            
        except Exception as e:
            logger.error(f"Error checking position for {symbol}: {e}")

def trading_cycle():
    """Run one complete trading cycle."""
    logger = logging.getLogger(__name__)
    logger.info("Starting trading cycle")
    
    try:
        # Check open positions first
        check_open_positions()
        
        # Process each trading symbol
        for symbol in config.TRADING_SYMBOLS:
            # Skip if we've hit max open trades
            if len(risk_manager.open_positions) >= config.MAX_OPEN_TRADES:
                logger.info(f"Maximum number of open trades reached ({config.MAX_OPEN_TRADES})")
                break
                
            # Skip if we already have a position for this symbol
            if symbol in risk_manager.open_positions:
                continue
                
            # Check market conditions and get trading signals
            signals, strategy = check_market_conditions(symbol)
            
            # Process signals if available
            if signals and strategy:
                process_trading_signals(symbol, signals, strategy)
                
            # Sleep briefly to avoid API rate limits
            time.sleep(1)
            
        # Send performance metrics every 24 hours
        if datetime.now().hour == 0 and datetime.now().minute < 5:
            metrics = risk_manager.get_performance_metrics()
            telegram.send_performance_update(metrics)
            
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")
        telegram.send_error_alert(f"Trading cycle error: {str(e)}")

def run_trading_bot():
    """Run the main trading bot loop."""
    logger = logging.getLogger(__name__)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize bot components
    if not initialize_bot():
        logger.error("Failed to initialize trading bot")
        return
    
    # Schedule regular trading cycles
    if config.TRADING_MODE == "live_trading":
        # More frequent checks for live trading
        schedule.every(5).minutes.do(trading_cycle)
    else:
        # Less frequent for paper trading to reduce API calls
        schedule.every(15).minutes.do(trading_cycle)
    
    # Initial trading cycle
    trading_cycle()
    
    logger.info(f"Trading bot started in {config.TRADING_MODE} mode")
    telegram.send_system_status(f"Trading bot running in {config.TRADING_MODE} mode")
    
    # Main loop
    while running:
        schedule.run_pending()
        time.sleep(1)
    
    logger.info("Trading bot stopped")
    telegram.send_system_status("Trading bot stopped")

def run_backtesting():
    """Run backtesting process."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize Binance client for historical data
        client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        
        # Create backtester instance
        backtester = Backtester(client)
        
        # Run backtest
        logger.info("Starting backtesting process...")
        results = backtester.run_backtest(use_strategy_switcher=True)
        
        # Generate report
        report_dir = "backtest_results"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"backtest_report_{timestamp}.txt")
        chart_file = os.path.join(report_dir, f"equity_curve_{timestamp}.png")
        
        # Save report and charts
        report = backtester.generate_report(report_file)
        backtester.plot_equity_curve(chart_file)
        
        logger.info(f"Backtesting completed. Results saved to {report_dir}")
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        traceback.print_exc()

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'], 
                       help='Trading mode (overrides config setting)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level=log_level)
    
    # Override trading mode if specified
    if args.mode:
        if args.mode == 'backtest':
            config.TRADING_MODE = 'backtesting'
        elif args.mode == 'paper':
            config.TRADING_MODE = 'paper_trading'
        elif args.mode == 'live':
            config.TRADING_MODE = 'live_trading'
    
    # Run in appropriate mode
    if config.TRADING_MODE == 'backtesting':
        run_backtesting()
    else:
        run_trading_bot()

if __name__ == "__main__":
    main()
