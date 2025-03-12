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
from ai_strategy import AIStrategy, AIScalpingStrategy, AISwingStrategy, AIBreakoutStrategy
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
        'breakout': (BreakoutStrategy, config.BREAKOUT_TIMEFRAME),
        'ai': (AIStrategy, config.SCALPING_TIMEFRAME),
        'ai_scalping': (AIScalpingStrategy, config.SCALPING_TIMEFRAME),
        'ai_swing': (AISwingStrategy, config.SWING_TIMEFRAME),
        'ai_breakout': (AIBreakoutStrategy, config.BREAKOUT_TIMEFRAME)
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Invalid strategy: {strategy_name}. Available strategies: {', '.join(strategy_map.keys())}")
    
    return strategy_map[strategy_name]


def run_bot(test_mode=False, api_key=None, api_secret=None, symbol=None, leverage=None, timeframe=None, strategy_name='scalping', train_ai=False, start_date=None, end_date=None):
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
        train_ai (bool): Whether to train AI models before running
        start_date (str): Start date for AI training (YYYY-MM-DD)
        end_date (str): End date for AI training (YYYY-MM-DD)
    """
    telegram_notifier = None  # Initialize to None to avoid UnboundLocalError
    
    try:
        # Set default values from config if not provided
        symbol = symbol or config.SYMBOL
        leverage = leverage or config.LEVERAGE
        
        # Check if it's an AI strategy
        is_ai_strategy = strategy_name.startswith('ai')
        
        # Log startup information
        if test_mode:
            logger.info(f"Starting AI Trading Bot in TEST MODE (no real trades)...")
            if is_ai_strategy:
                logger.info(f"Using AI strategy: {strategy_name.upper()}")
                if train_ai:
                    logger.info(f"AI models will be trained before running")
        else:
            logger.info(f"Starting AI Trading Bot in LIVE MODE...")
        
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
        
        # Train AI models if requested and if it's an AI strategy
        if train_ai and strategy_name.startswith('ai'):
            logger.info(f"Training AI models for {strategy_name.upper()} strategy...")
            if hasattr(strategy, 'train_models'):
                # Log training parameters
                logger.info(f"Training period: {start_date or 'default start'} to {end_date or 'default end'}")
                
                # Train models
                success = strategy.train_models(start_date=start_date, end_date=end_date)
                
                if success:
                    logger.info("AI models trained successfully.")
                    if test_mode:
                        logger.info("Continuing to run in TEST MODE with trained models.")
                    else:
                        logger.info("Continuing to run in LIVE MODE with trained models.")
                else:
                    logger.error("Failed to train AI models.")
                    if telegram_notifier:
                        telegram_notifier.notify_error("Failed to train AI models.")
                    return  # Exit if training failed
            else:
                logger.warning(f"Strategy {strategy_name} does not support training.")
        
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
                # Create a more detailed status message
                mode_str = 'TEST' if test_mode else 'LIVE'
                ai_str = ' (AI-powered)' if strategy_name.startswith('ai') else ''
                
                status_message = (
                    f"AI Trading Bot started for {symbol} ({timeframe}).\n"
                    f"Strategy: {strategy_name.upper()}{ai_str}\n"
                    f"Mode: {mode_str}\n"
                )
                
                # Add training info if applicable
                if train_ai and strategy_name.startswith('ai'):
                    status_message += f"AI models: Trained\n"
                
                # Add leverage info
                status_message += f"Leverage: {leverage}x"
                
                telegram_notifier.notify_system_status(status_message)
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


def run_backtest(symbol=None, timeframe=None, days=30, start_date=None, end_date=None, strategy_name='scalping'):
    """
    Run backtest.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Trading timeframe
        days (int): Number of days to backtest
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        strategy_name (str): Name of the strategy to use
    """
    try:
        logger.info("Starting backtest...")
        
        # Get strategy class and default timeframe
        strategy_class, default_timeframe = get_strategy_class(strategy_name.lower())
        
        # Use provided timeframe or default for the strategy
        timeframe = timeframe or default_timeframe
        
        # Check if it's an AI strategy
        is_ai_strategy = strategy_name.startswith('ai')
        
        # Initialize backtester
        backtester = Backtester(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_class=strategy_class,
            is_ai_strategy=is_ai_strategy
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        if results and 'stats' in results and 'results' in results:
            # Plot results
            backtester.plot_results(results)
            
            # Print summary
            stats = results['stats']
            logger.info("Backtest Summary:")
            logger.info(f"Total Return: {stats['total_return']:.2f}%")
            logger.info(f"Annual Return: {stats['annual_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
            logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
            logger.info(f"Total Trades: {stats['total_trades']}")
            
            return results
        else:
            logger.error("Backtest did not produce valid results")
            return None
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    
    parser.add_argument("--test", action="store_true", help="Run in test mode (no real trades)")
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--leverage", type=int, help="Trading leverage")
    parser.add_argument("--timeframe", type=str, help="Trading timeframe (e.g., 1m, 5m, 15m, 1h)")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['scalping', 'swing', 'breakout', 'ai', 'ai_scalping', 'ai_swing', 'ai_breakout'],
        default='scalping',
        help="Trading strategy to use"
    )
    parser.add_argument("--train-ai", action="store_true", help="Train AI models before running")
    parser.add_argument("--start-date", type=str, help="Start date for AI training (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for AI training (YYYY-MM-DD)")
    
    # Backtest arguments
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--backtest-days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--backtest-start", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--backtest-end", type=str, help="End date for backtest (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Check if we need to run a backtest
    if args.backtest:
        run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.backtest_days,
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            strategy_name=args.strategy
        )
    # Check if we need to train AI models and then run in test mode
    elif args.train_ai and args.test:
        logger.info("Training AI models and then running in test mode...")
        run_bot(
            test_mode=True,  # Always use test mode
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            symbol=args.symbol,
            leverage=args.leverage,
            timeframe=args.timeframe,
            strategy_name=args.strategy,
            train_ai=True,  # Train AI models
            start_date=args.start_date,
            end_date=args.end_date
        )
    # Otherwise, run the bot normally
    else:
        run_bot(
            test_mode=args.test,
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            symbol=args.symbol,
            leverage=args.leverage,
            timeframe=args.timeframe,
            strategy_name=args.strategy,
            train_ai=args.train_ai,
            start_date=args.start_date,
            end_date=args.end_date
        )
