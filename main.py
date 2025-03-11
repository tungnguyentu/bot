"""
Main module for the AI Trading Bot.
"""

import os
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

import config
from utils import setup_logger
from binance_client import BinanceClient
from telegram_notifier import TelegramNotifier
from strategy import ScalpingStrategy
from backtest import Backtester

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(config.LOG_LEVEL)


def run_bot(test_mode=False):
    """
    Run the trading bot.
    
    Args:
        test_mode (bool): Run in test mode (no real trades)
    """
    try:
        logger.info("Starting AI Trading Bot...")
        
        # Initialize Binance client
        binance_client = BinanceClient(testnet=test_mode)
        
        # Initialize Telegram notifier
        telegram_notifier = TelegramNotifier() if config.ENABLE_TELEGRAM else None
        
        # Initialize strategy
        strategy = ScalpingStrategy(binance_client, telegram_notifier)
        
        # Set leverage
        binance_client.set_leverage(config.SYMBOL, config.LEVERAGE)
        
        # Send startup notification
        if telegram_notifier:
            telegram_notifier.notify_system_status(
                f"AI Trading Bot started for {config.SYMBOL} ({config.TIMEFRAME}).\n"
                f"Mode: {'TEST' if test_mode else 'LIVE'}"
            )
        
        logger.info(f"Bot initialized for {config.SYMBOL} ({config.TIMEFRAME}).")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        
        # Main loop
        while True:
            try:
                # Analyze market
                analysis = strategy.analyze_market()
                
                # Execute signals
                execution_result = strategy.execute_signals(analysis)
                
                # Manage positions
                strategy.manage_positions()
                
                # Sleep until next candle
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if telegram_notifier:
                    telegram_notifier.notify_error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        if telegram_notifier:
            telegram_notifier.notify_system_status("Bot stopped by user.")
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        if telegram_notifier:
            telegram_notifier.notify_error(f"Critical error: {e}")


def run_backtest(symbol=None, timeframe=None, start_date=None, end_date=None, initial_balance=10000):
    """
    Run backtest.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        initial_balance (float): Initial balance
    """
    try:
        logger.info("Starting backtest...")
        
        # Initialize backtester
        backtester = Backtester(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Run backtest
        results = backtester.run_backtest(initial_balance=initial_balance)
        
        # Plot results
        backtester.plot_results(results)
        
        logger.info("Backtest completed.")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no real trades)")
    parser.add_argument("--symbol", type=str, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, help="Timeframe")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Initial balance for backtest")
    
    args = parser.parse_args()
    
    if args.backtest:
        # Run backtest
        run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.initial_balance
        )
    else:
        # Run bot
        run_bot(test_mode=args.test) 