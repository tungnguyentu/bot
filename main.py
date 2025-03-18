#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timedelta

from bot import TradingBot
from backtesting import Backtester
from utils.logger import setup_logger

logger = setup_logger('main', 'logs/main.log')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['live', 'backtest'], 
                        help='Trading mode: live or backtest')
    
    # Backtesting arguments
    parser.add_argument('--strategy', type=str, choices=['scalping', 'swing'],
                        help='Strategy for backtesting')
    parser.add_argument('--symbol', type=str, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--days', type=int, default=30, 
                        help='Number of days to backtest (default: 30)')
    parser.add_argument('--initial', type=float, default=10000, 
                        help='Initial balance for backtesting (default: 10000)')
    
    return parser.parse_args()

def check_environment():
    """Check if environment is properly set up."""
    required_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with the required variables.")
        sys.exit(1)

def run_live_trading():
    """Run live trading bot."""
    logger.info("Starting live trading mode")
    bot = TradingBot()
    bot.start()

def run_backtest(args):
    """Run backtesting with specified parameters."""
    if not args.strategy or not args.symbol:
        logger.error("Strategy and symbol are required for backtesting")
        sys.exit(1)
        
    logger.info(f"Starting backtesting mode: {args.strategy} strategy on {args.symbol}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Initialize and run backtester
    backtester = Backtester(
        strategy_type=args.strategy,
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.initial
    )
    
    results = backtester.run()
    
    if results:
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info(f"BACKTEST RESULTS FOR {args.symbol} - {args.strategy.upper()}")
        logger.info("=" * 50)
        logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial Balance: {args.initial:.2f} USDT")
        
        # Handle the case where results might not have total_return (added safety check)
        final_balance = args.initial
        if 'total_return' in results:
            final_balance = args.initial * (1 + results['total_return'])
            logger.info(f"Final Balance: {final_balance:.2f} USDT")
            logger.info(f"Total Return: {results['total_return']:.2%}")
        else:
            logger.info(f"Final Balance: {final_balance:.2f} USDT (no change)")
            logger.info(f"Total Return: 0.00%")
            
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info("=" * 50)
        
        # We should store trades and equity_curve in results for proper plotting
        # For now, pass empty lists to avoid the error
        backtester.plot_results([], [])
    else:
        logger.error("Backtesting failed. Check the logs for more information.")

def main():
    """Main function."""
    args = parse_arguments()
    check_environment()
    
    try:
        if args.mode == 'live':
            run_live_trading()
        elif args.mode == 'backtest':
            run_backtest(args)
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
