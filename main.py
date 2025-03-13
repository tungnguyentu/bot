import logging
from datetime import datetime
import time
import argparse
import os
import pandas as pd

from config import Config
from exchange.binance_futures import BinanceFutures
from strategy.mean_reversion import MeanReversionStrategy
from risk_management.risk_manager import RiskManager
from notifications.telegram_notifier import TelegramNotifier
from data.market_data import MarketData
from simulation.test_mode import TestModeExchange
from simulation.backtesting import BacktestEngine
from ai.reinforcement_learning import RLTrader
from tqdm import trange

def countdown_timer(seconds):
    for i in trange(seconds, desc="Waiting...", ncols=80):
        time.sleep(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("BinanceBot")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Binance Futures Trading Bot")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Trading symbols (comma-separated, e.g., BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        help="Leverage to use for trading (e.g., 20)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (no real trades executed)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode to evaluate strategy on historical data",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run in training mode to optimize strategy with reinforcement learning",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backtesting (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backtesting (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-balance",
        type=float,
        default=1000.0,
        help="Starting balance for test/backtest mode (default: 1000 USDT)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for RL training (default: 100)",
    )
    return parser.parse_args()


class TradingBot:
    def __init__(self, cli_args=None):
        self.config = Config(cli_args)
        self.cli_args = cli_args
        
        # Create appropriate exchange interface based on mode
        if cli_args:
            if cli_args.train:
                logger.info("Starting in TRAINING MODE with reinforcement learning")
                self.mode = "training"
                self.exchange = TestModeExchange(starting_balance=cli_args.test_balance)
                self.rl_trainer = RLTrader(
                    state_size=5,  # Base features: BB, RSI, MACD, VWAP, ATR
                    action_size=3,  # No action, Long, Short
                    batch_size=64,
                    episodes=cli_args.episodes
                )
            elif cli_args.backtest:
                logger.info("Starting in BACKTEST MODE to evaluate strategy performance")
                self.mode = "backtest"
                self.exchange = BacktestEngine(
                    starting_balance=cli_args.test_balance,
                    start_date=cli_args.start_date if cli_args.start_date else None,
                    end_date=cli_args.end_date if cli_args.end_date else None
                )
            elif cli_args.test:
                logger.info(f"Starting in TEST MODE with {cli_args.test_balance} USDT balance")
                self.mode = "test"
                self.exchange = TestModeExchange(starting_balance=cli_args.test_balance)
            else:
                self.mode = "live"
                self.exchange = BinanceFutures(
                    api_key=self.config.BINANCE_API_KEY,
                    api_secret=self.config.BINANCE_API_SECRET,
                )
        else:
            self.mode = "live"
            self.exchange = BinanceFutures(
                api_key=self.config.BINANCE_API_KEY,
                api_secret=self.config.BINANCE_API_SECRET,
            )

        # Initialize components
        self.market_data = MarketData(self.exchange)
        self.risk_manager = RiskManager(self.exchange, self.config)
        self.strategy = MeanReversionStrategy(self.market_data, self.config)
        self.notifier = TelegramNotifier(
            token=self.config.TELEGRAM_BOT_TOKEN, chat_id=self.config.TELEGRAM_CHAT_ID
        )

    def run(self):
        # Different run modes based on the selected mode
        if self.mode == "backtest":
            self._run_backtest()
        elif self.mode == "training":
            self._run_training()
        else:
            self._run_live()

    def _run_backtest(self):
        """Run backtesting mode to evaluate strategy on historical data"""
        logger.info(f"Starting backtest on symbols: {', '.join(self.config.TRADING_SYMBOLS)}")
        
        # For each symbol, get historical data and backtest
        for symbol in self.config.TRADING_SYMBOLS:
            logger.info(f"Backtesting {symbol}...")
            
            # Load historical data
            self.exchange.load_historical_data(symbol, self.config.TIMEFRAME)
            
            # Run the backtest
            results = self.exchange.run_backtest(
                symbol,
                self.strategy,
                self.risk_manager,
                self.config
            )
            
            # Output the results
            logger.info(f"Backtest results for {symbol}:")
            logger.info(f"Total trades: {results['total_trades']}")
            logger.info(f"Winning trades: {results['winning_trades']}")
            logger.info(f"Losing trades: {results['losing_trades']}")
            logger.info(f"Win rate: {results['win_rate']:.2f}%")
            logger.info(f"Average profit: {results['avg_profit']:.2f}%")
            logger.info(f"Average loss: {results['avg_loss']:.2f}%")
            logger.info(f"Profit factor: {results['profit_factor']:.2f}")
            logger.info(f"Maximum drawdown: {results['max_drawdown']:.2f}%")
            logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"Final balance: {results['final_balance']:.2f} USDT")
            logger.info(f"Return: {results['total_return']:.2f}%")
            
            # Plot the equity curve
            self.exchange.plot_results(symbol)
    
    def _run_training(self):
        """Run training mode with reinforcement learning"""
        logger.info(f"Starting RL training on symbols: {', '.join(self.config.TRADING_SYMBOLS)}")
        
        for symbol in self.config.TRADING_SYMBOLS:
            logger.info(f"Training on {symbol}...")
            
            # Load training data
            historical_data = self.market_data.get_historical_data(
                symbol, self.config.TIMEFRAME, 5000  # Use larger dataset for training
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Calculate indicators
            df = self.strategy.calculate_indicators(df)
            
            # Train the RL agent
            self.rl_trainer.train(
                symbol=symbol,
                data=df,
                config=self.config
            )
            
            # Save the trained model
            model_path = f"models/rl_model_{symbol}.h5"
            self.rl_trainer.save_model(model_path)
            logger.info(f"Training completed for {symbol}. Model saved to {model_path}")
            
            # Evaluate the trained model
            evaluation = self.rl_trainer.evaluate(
                symbol=symbol,
                data=df.iloc[-1000:],  # Use last part of data for evaluation
                config=self.config
            )
            
            logger.info(f"Model evaluation for {symbol}:")
            logger.info(f"Total trades: {evaluation['total_trades']}")
            logger.info(f"Win rate: {evaluation['win_rate']:.2f}%")
            logger.info(f"Total return: {evaluation['total_return']:.2f}%")
            logger.info(f"Sharpe ratio: {evaluation['sharpe_ratio']:.2f}")

    def _run_live(self):
        """Run live trading or test mode"""
        mode_str = "TEST MODE" if self.mode == "test" else "LIVE MODE"
        logger.info(f"Starting Binance Futures Trading Bot in {mode_str}")
        logger.info(f"Trading symbols: {', '.join(self.config.TRADING_SYMBOLS)}")
        logger.info(f"Leverage: {self.config.LEVERAGE}x")

        self.notifier.send_message(
            f"🤖 Trading Bot Started in {mode_str}\n"
            f"Symbols: {', '.join(self.config.TRADING_SYMBOLS)}\n"
            f"Leverage: {self.config.LEVERAGE}x"
        )

        # Initialize RL model if available
        if os.path.exists(f"models/rl_model_{self.config.TRADING_SYMBOLS[0]}.h5"):
            logger.info("Loading RL model to enhance trading decisions")
            self.rl_model = RLTrader(state_size=5, action_size=3)
            self.rl_model.load_model(f"models/rl_model_{self.config.TRADING_SYMBOLS[0]}.h5")
            use_rl = True
        else:
            use_rl = False
            logger.info("No RL model found, using standard strategy")

        while True:
            try:
                for symbol in self.config.TRADING_SYMBOLS:
                    # Check for high-impact news events
                    if self.market_data.check_high_impact_news():
                        logger.info("High impact news detected, skipping trading")
                        continue

                    # Fetch latest market data
                    historical_data = self.market_data.get_historical_data(
                        symbol, self.config.TIMEFRAME, self.config.LOOKBACK_PERIOD
                    )

                    # Generate trading signals
                    signal = self.strategy.generate_signal(symbol, historical_data)
                    
                    # If RL model is available, enhance decision with AI
                    if use_rl and signal:
                        # Prepare state for RL model
                        df = pd.DataFrame(historical_data)
                        df = self.strategy.calculate_indicators(df)
                        state = self.strategy.prepare_state_for_rl(df.iloc[-1])
                        
                        # Get RL model's action recommendation
                        action = self.rl_model.predict_action(state)
                        
                        # Override signal if RL model strongly disagrees
                        if action == 0:  # No action
                            signal = None
                            logger.info(f"RL model recommended no action for {symbol}, signal canceled")
                        elif action == 1 and signal[0] == "SHORT":  # RL recommends LONG but signal is SHORT
                            logger.info(f"RL model and strategy conflict for {symbol}, following strategy")
                        elif action == 2 and signal[0] == "LONG":  # RL recommends SHORT but signal is LONG
                            logger.info(f"RL model and strategy conflict for {symbol}, following strategy")

                    # Execute trades based on signals
                    logger.info(f"Signal for {symbol}: {signal}")
                    if signal:
                        trade_direction, reasoning = signal
                        position = self.exchange.get_position(symbol)

                        # Check if we already have an open position
                        if position["size"] == 0:
                            # No open position, we can enter a new trade
                            if self.risk_manager.can_open_trade(symbol):
                                self._execute_trade(symbol, trade_direction, reasoning)
                        else:
                            # We have an open position, check if we need to close it
                            current_direction = (
                                "LONG" if position["size"] > 0 else "SHORT"
                            )
                            if trade_direction != current_direction:
                                self._close_trade(symbol, current_direction, reasoning)

                    # Update stop-loss and take-profit levels for existing positions
                    position = self.exchange.get_position(symbol)
                    if position["size"] != 0:
                        self.risk_manager.update_risk_levels(
                            symbol, position, historical_data
                        )

                # If in test mode, update simulated prices and check for triggered orders
                if self.mode == "test":
                    self.exchange.update_simulated_prices()

                # Sleep to avoid API rate limits
                countdown_timer(self.config.SCAN_INTERVAL)

            except Exception as e:
                error_msg = f"Error in main loop: {str(e)}"
                logger.error(error_msg)
                self.notifier.send_message(f"⚠️ Error: {error_msg}")
                time.sleep(60)  # Wait a bit longer if there's an error
    
    def _execute_trade(self, symbol, direction, reasoning):
        try:
            # Calculate position size based on risk management
            amount = self.risk_manager.calculate_position_size(symbol)

            # Execute the order
            if direction == "LONG":
                order = self.exchange.create_market_buy_order(symbol, amount)
                order_type = "Long"
            else:  # SHORT
                order = self.exchange.create_market_sell_order(symbol, amount)
                order_type = "Short"

            # Get current price as entry price
            entry_price = self.exchange.get_current_price(symbol)

            # Calculate SL and TP levels based on ATR
            historical_data = self.market_data.get_historical_data(
                symbol, self.config.TIMEFRAME, self.config.LOOKBACK_PERIOD
            )
            df = pd.DataFrame(historical_data)
            df = self.risk_manager.indicators.add_atr(df, self.config.ATR_PERIOD)
            atr = df["atr"].iloc[-1]

            if direction == "LONG":
                stop_loss = entry_price - (atr * self.config.SL_ATR_MULTIPLIER)
                partial_tp = entry_price + (atr * self.config.PARTIAL_TP_ATR_MULTIPLIER)
                full_tp = entry_price + (atr * self.config.FULL_TP_ATR_MULTIPLIER)
            else:  # SHORT
                stop_loss = entry_price + (atr * self.config.SL_ATR_MULTIPLIER)
                partial_tp = entry_price - (atr * self.config.PARTIAL_TP_ATR_MULTIPLIER)
                full_tp = entry_price - (atr * self.config.FULL_TP_ATR_MULTIPLIER)

            # Set initial SL and TP orders
            self.risk_manager.set_stop_loss_take_profit(symbol, direction, order)

            # Format price values for display
            entry_price_str = (
                f"{entry_price:.2f}" if entry_price >= 10 else f"{entry_price:.4f}"
            )
            stop_loss_str = (
                f"{stop_loss:.2f}" if stop_loss >= 10 else f"{stop_loss:.4f}"
            )
            partial_tp_str = (
                f"{partial_tp:.2f}" if partial_tp >= 10 else f"{partial_tp:.4f}"
            )
            full_tp_str = f"{full_tp:.2f}" if full_tp >= 10 else f"{full_tp:.4f}"

            # Send notification
            msg = (
                f"🚀 New {order_type} position opened on {symbol}\n"
                f"Entry: {entry_price_str} USDT\n"
                f"Stop Loss: {stop_loss_str} USDT\n"
                f"Take Profit 1: {partial_tp_str} USDT\n"
                f"Take Profit 2: {full_tp_str} USDT\n"
                f"Reason: {reasoning}"
            )
            self.notifier.send_message(msg)
            logger.info(f"Executed {direction} trade for {symbol}: {reasoning}")

        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)
            self.notifier.send_message(f"⚠️ {error_msg}")

    def _close_trade(self, symbol, current_direction, reasoning):
        try:
            position = self.exchange.get_position(symbol)
            current_price = self.exchange.get_current_price(symbol)

            # Calculate PnL before closing
            entry_price = position["entry_price"]
            if current_direction == "LONG":
                pnl_pct = ((current_price / entry_price) - 1) * 100
                order = self.exchange.create_market_sell_order(
                    symbol, abs(position["size"])
                )
                order_type = "Long"
            else:  # SHORT
                pnl_pct = ((entry_price / current_price) - 1) * 100
                order = self.exchange.create_market_buy_order(
                    symbol, abs(position["size"])
                )
                order_type = "Short"

            # Format price for display
            price_str = (
                f"{current_price:.2f}"
                if current_price >= 10
                else f"{current_price:.4f}"
            )
            entry_str = (
                f"{entry_price:.2f}" if entry_price >= 10 else f"{entry_price:.4f}"
            )

            # Send notification
            msg = (
                f"🔴 Closed {order_type} position on {symbol}\n"
                f"Entry: {entry_str} USDT\n"
                f"Exit: {price_str} USDT\n"
                f"P&L: {pnl_pct:.2f}%\n"
                f"Reason: {reasoning}"
            )
            self.notifier.send_message(msg)
            logger.info(f"Closed {current_direction} trade for {symbol}: {reasoning}")

        except Exception as e:
            error_msg = f"Error closing trade: {str(e)}"
            logger.error(error_msg)
            self.notifier.send_message(f"⚠️ {error_msg}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Parse command line arguments
    args = parse_arguments()

    bot = TradingBot(cli_args=args)
    bot.run()
