import os
import logging
import pandas as pd
from datetime import datetime

from ai_strategy import AIScalpingStrategy
from binance_client import BinanceClient
from telegram_notifier import TelegramNotifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_ai_strategy")

def test_ai_strategy():
    """Test the AI strategy."""
    try:
        # Initialize Binance client in test mode
        binance_client = BinanceClient(testnet=True)
        
        # Initialize Telegram notifier (optional)
        telegram_notifier = None
        
        # Initialize AI strategy
        strategy = AIScalpingStrategy(
            binance_client=binance_client,
            telegram_notifier=telegram_notifier,
            symbol="BTCUSDT",
            timeframe="5m",
            leverage=10
        )
        
        # Train models with a small date range
        start_date = "2025-03-01"
        end_date = "2025-03-05"
        
        logger.info(f"Training AI models from {start_date} to {end_date}...")
        success = strategy.train_models(start_date=start_date, end_date=end_date)
        
        if success:
            logger.info("AI models trained successfully.")
            
            # Test market analysis
            logger.info("Testing market analysis...")
            analysis = strategy.analyze_market()
            
            if analysis:
                logger.info(f"Market analysis successful: {analysis}")
                return True
            else:
                logger.error("Market analysis failed.")
                return False
        else:
            logger.error("AI model training failed.")
            return False
        
    except Exception as e:
        logger.error(f"Error testing AI strategy: {e}")
        return False

if __name__ == "__main__":
    test_ai_strategy() 