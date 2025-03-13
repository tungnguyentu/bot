import argparse
import logging
import traceback
from config import Config
from data.collector import BinanceDataCollector
from models.model_manager import ModelManager
from trading.bot import TradingBot
from utils.telegram_notifier import TelegramNotifier
from backtesting.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot for Binance Futures")
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "test", "live"],
        default="test",
        help="Bot operation mode",
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading pair symbol"
    )
    parser.add_argument(
        "--interval", type=str, default="1h", help="Candlestick interval"
    )
    args = parser.parse_args()

    config = Config()
    telegram = TelegramNotifier(config.telegram_token, config.telegram_chat_id)

    try:
        data_collector = BinanceDataCollector(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            symbol=args.symbol,
            interval=args.interval,
        )

        model_manager = ModelManager(config)

        if args.mode == "train":
            logger.info("Starting model training...")
            telegram.send_message("🧠 Bot training started")
            historical_data = data_collector.get_historical_data()
            
            if historical_data is None or historical_data.empty:
                error_msg = "Failed to retrieve historical data"
                logger.error(error_msg)
                telegram.send_message(f"❌ Training failed: {error_msg}")
                return
                
            # Check for NaN values
            nan_count = historical_data.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Data contains {nan_count} NaN values that will be handled")
            
            try:
                model_manager.train(historical_data)
                telegram.send_message("✅ Bot training completed")
            except Exception as e:
                error_msg = f"Training failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                telegram.send_message(f"❌ {error_msg}")

        elif args.mode == "backtest":
            logger.info("Starting backtesting...")
            telegram.send_message("📊 Backtesting started")
            historical_data = data_collector.get_historical_data()
            
            if historical_data is None or historical_data.empty:
                error_msg = "Failed to retrieve historical data for backtesting"
                logger.error(error_msg)
                telegram.send_message(f"❌ Backtesting failed: {error_msg}")
                return
                
            try:
                backtest_engine = BacktestEngine(config, model_manager)
                metrics = backtest_engine.run(historical_data)
                metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                telegram.send_message(f"📈 Backtest Results:\n{metrics_str}")
            except Exception as e:
                error_msg = f"Backtesting failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                telegram.send_message(f"❌ {error_msg}")

        elif args.mode in ["test", "live"]:
            logger.info(f"Starting bot in {args.mode} mode")
            telegram.send_message(f"🚀 Bot started in {args.mode.upper()} mode")
            
            try:
                bot = TradingBot(
                    config=config,
                    data_collector=data_collector,
                    model_manager=model_manager,
                    telegram=telegram,
                    is_test_mode=(args.mode == "test"),
                )
                
                bot.run()
            except Exception as e:
                error_msg = f"Bot execution failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                telegram.send_message(f"❌ {error_msg}")
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        telegram.send_message(f"❌ {error_msg}")


if __name__ == "__main__":
    main()
