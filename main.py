"""
Main module for the AI Trading Bot.
"""

import os
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv
import requests

import config
from utils import setup_logger
from binance_client import BinanceClient
from telegram_notifier import TelegramNotifier
from strategy import ScalpingStrategy, SwingStrategy, BreakoutStrategy
from backtest import Backtester

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(config.LOG_LEVEL)


def get_strategy_class(strategy_name):
    """
    Get the strategy class based on the strategy name.
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        class: Strategy class
    """
    strategy_map = {
        'scalping': (ScalpingStrategy, config.SCALPING_TIMEFRAME),
        'swing': (SwingStrategy, config.SWING_TIMEFRAME),
        'breakout': (BreakoutStrategy, config.BREAKOUT_TIMEFRAME)
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Invalid strategy: {strategy_name}. Available strategies: {', '.join(strategy_map.keys())}")
    
    return strategy_map[strategy_name]


def run_bot(test_mode=False, api_key=None, api_secret=None, symbol=None, leverage=None, timeframe=None, strategy_name='scalping'):
    """
    Run the trading bot.
    
    Args:
        test_mode (bool): Run in test mode (no real trades)
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        symbol (str): Trading symbol
        leverage (int): Trading leverage
        timeframe (str): Trading timeframe
        strategy_name (str): Name of the strategy to use
    """
    telegram_notifier = None  # Initialize to None to avoid UnboundLocalError
    
    try:
        logger.info("Starting AI Trading Bot...")
        
        # Get strategy class and default timeframe
        strategy_class, default_timeframe = get_strategy_class(strategy_name.lower())
        
        # Use provided timeframe or default for the strategy
        timeframe = timeframe or default_timeframe
        
        # Initialize Binance client
        try:
            binance_client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=test_mode
            )
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            if "Invalid API" in str(e):
                logger.error("Invalid Binance API credentials. Please check your .env file.")
                logger.error("For testnet, you need to generate API keys from https://testnet.binancefuture.com/")
            elif "Connection" in str(e):
                logger.error("Connection error. Please check your internet connection.")
            return
        
        # Initialize Telegram notifier
        try:
            telegram_notifier = TelegramNotifier() if config.ENABLE_TELEGRAM else None
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            telegram_notifier = None
        
        # Initialize strategy
        strategy = strategy_class(binance_client, telegram_notifier, symbol, timeframe, leverage)
        
        # Set leverage
        try:
            binance_client.set_leverage(symbol, leverage)
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            if "Invalid API" in str(e):
                logger.error("Invalid Binance API credentials. Please check your .env file.")
                logger.error("For testnet, you need to generate API keys from https://testnet.binancefuture.com/")
                return
            else:
                logger.warning(f"Continuing with default leverage. Some features may be limited.")
        
        # Send startup notification
        if telegram_notifier:
            try:
                telegram_notifier.notify_system_status(
                    f"AI Trading Bot started for {symbol} ({timeframe}).\n"
                    f"Strategy: {strategy_name.upper()}\n"
                    f"Mode: {'TEST' if test_mode else 'LIVE'}"
                )
            except Exception as e:
                logger.error(f"Error sending Telegram notification: {e}")
        
        logger.info(f"Bot initialized for {symbol} ({timeframe}).")
        logger.info(f"Strategy: {strategy_name.upper()}")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        
        # Main loop
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                # Analyze market
                analysis = strategy.analyze_market()
                
                # Execute signals
                execution_result = strategy.execute_signals(analysis)
                
                # Manage positions
                strategy.manage_positions()
                
                # Reset retry count on successful iteration
                retry_count = 0
                
                # Sleep until next candle
                time.sleep(60)  # Check every minute
            
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.error(f"Connection error in main loop (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"Maximum retries ({max_retries}) reached. Exiting.")
                    if telegram_notifier:
                        try:
                            telegram_notifier.notify_error(f"Bot stopped after {max_retries} failed connection attempts.")
                        except Exception as notify_error:
                            logger.error(f"Error sending Telegram notification: {notify_error}")
                    break
                
                # Wait before retrying with exponential backoff
                wait_time = 60 * (2 ** (retry_count - 1))  # 60s, 120s, 240s, ...
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if telegram_notifier:
                    try:
                        telegram_notifier.notify_error(f"Error in main loop: {e}")
                    except Exception as notify_error:
                        logger.error(f"Error sending Telegram notification: {notify_error}")
                time.sleep(60)  # Wait before retrying
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        if telegram_notifier:
            try:
                telegram_notifier.notify_system_status("Bot stopped by user.")
            except Exception as e:
                logger.error(f"Error sending Telegram notification: {e}")
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        if telegram_notifier:
            try:
                telegram_notifier.notify_error(f"Critical error: {e}")
            except Exception as notify_error:
                logger.error(f"Error sending Telegram notification: {notify_error}")


def run_backtest(symbol=None, timeframe=None, start_date=None, end_date=None, initial_balance=10000, strategy_name='scalping'):
    """
    Run backtest.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        initial_balance (float): Initial balance
        strategy_name (str): Name of the strategy to use
    """
    try:
        logger.info("Starting backtest...")
        logger.info(f"Strategy: {strategy_name.upper()}")
        
        # Get strategy class and default timeframe
        strategy_class, default_timeframe = get_strategy_class(strategy_name.lower())
        
        # Use provided timeframe or default for the strategy
        timeframe = timeframe or default_timeframe
        
        # Initialize backtester
        backtester = Backtester(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_class=strategy_class
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
    parser.add_argument("--leverage", type=int, help="Leverage")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['scalping', 'swing', 'breakout'],
        default='scalping',
        help="Trading strategy to use"
    )

    args = parser.parse_args()
    if args.backtest:
        # Run backtest
        run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.initial_balance,
            strategy_name=args.strategy
        )
    else:
        # Run bot
        run_bot(
            test_mode=args.test,
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            symbol=args.symbol,
            leverage=args.leverage,
            timeframe=args.timeframe,
            strategy_name=args.strategy
        )
