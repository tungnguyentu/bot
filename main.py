import argparse
import logging
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
        model_manager.train(historical_data)
        telegram.send_message("✅ Bot training completed")

    elif args.mode == "backtest":
        logger.info("Starting backtesting...")
        telegram.send_message("📊 Backtesting started")
        backtest_engine = BacktestEngine(config, model_manager)
        historical_data = data_collector.get_historical_data()
        metrics = backtest_engine.run(historical_data)
        telegram.send_message(f"📈 Backtest Results:\n{metrics}")

    elif args.mode in ["test", "live"]:
        logger.info(f"Starting bot in {args.mode} mode")
        telegram.send_message(f"🚀 Bot started in {args.mode.upper()} mode")

        bot = TradingBot(
            config=config,
            data_collector=data_collector,
            model_manager=model_manager,
            telegram=telegram,
            is_test_mode=(args.mode == "test"),
        )

        bot.run()


if __name__ == "__main__":
    main()
