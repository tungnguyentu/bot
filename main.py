import argparse
import logging
import os
from dotenv import load_dotenv

from bot.trading_bot import TradingBot
from bot.config import BotConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("backtest_results", exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Trading Bot for Binance Futures')
    
    parser.add_argument('--symbol', type=str, default='SOLUSDT', 
                        help='Trading symbol (default: SOLUSDT)')
    parser.add_argument('--leverage', type=int, default=20, 
                        help='Leverage to use (default: 20)')
    parser.add_argument('--mode', type=str, choices=['test', 'live', 'backtest'], 
                        default='test', help='Trading mode (default: test)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe for data (default: 1h)')
    parser.add_argument('--quantity', type=float, default=0.1,
                        help='Trading quantity in base asset (default: 0.1)')
    parser.add_argument('--sl_atr_multiplier', type=float, default=1.5,
                        help='Stop Loss ATR multiplier (default: 1.5)')
    parser.add_argument('--tp_atr_multiplier', type=float, default=2.0,
                        help='Take Profit ATR multiplier (default: 2.0)')
    parser.add_argument('--train', action='store_true',
                        help='Train the model before trading')
    parser.add_argument('--invest', type=float, default=100.0,
                        help='Amount to invest in trading (default: 100.0)')
    parser.add_argument('--quick', action='store_true',
                        help='Force immediate position opening for quick testing')
    
    return parser.parse_args()

def main():
    """Main function to run the trading bot."""
    # Create directories
    create_directories()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create bot configuration
    config = BotConfig(
        symbol=args.symbol,
        leverage=args.leverage,
        mode=args.mode,
        interval=args.interval,
        quantity=args.quantity,
        sl_atr_multiplier=args.sl_atr_multiplier,
        tp_atr_multiplier=args.tp_atr_multiplier,
        train=args.train,
        invest=args.invest,
        quick=args.quick
    )
    
    logger.info(f"Starting trading bot with config: {config}")
    
    # Initialize and run the trading bot
    bot = TradingBot(config)
    bot.run()

if __name__ == "__main__":
    main()
