import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# For paper trading, use testnet credentials if provided
BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", BINANCE_API_KEY)
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", BINANCE_API_SECRET)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Trading Parameters
TRADING_SYMBOLS = ["SOLUSDT"]  # Trading pairs
TRADING_TIMEFRAMES = {
    "scalping": "5m",           # 5-minute candles for scalping
    "swing": "4h"               # 4-hour candles for swing trading
}
TRADING_LEVERAGE = 20            # Leverage (adjust based on risk tolerance)

# Strategy Parameters
# Scalping Strategy
SCALPING_RSI_PERIOD = 14
SCALPING_RSI_OVERBOUGHT = 70
SCALPING_RSI_OVERSOLD = 30
SCALPING_BB_PERIOD = 20
SCALPING_BB_STD = 2
SCALPING_MA_PERIOD_SHORT = 9
SCALPING_MA_PERIOD_LONG = 21

# Swing Trading Strategy
SWING_MACD_FAST = 12
SWING_MACD_SLOW = 26
SWING_MACD_SIGNAL = 9
ICHIMOKU_CONVERSION_LINE_PERIOD = 9
ICHIMOKU_BASE_LINE_PERIOD = 26
ICHIMOKU_LEADING_SPAN_B_PERIOD = 52
ICHIMOKU_DISPLACEMENT = 26

# Risk Management
RISK_PER_TRADE = 0.02          # 2% of account per trade
MAX_DRAWDOWN_PERCENTAGE = 0.20  # 20% maximum drawdown limit
MAX_OPEN_TRADES = 3             # Maximum concurrent open trades
TRAILING_STOP_ACTIVATION = 0.02 # Activate trailing stop after 2% profit

# Backtesting
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"

# Mode
TRADING_MODE = "paper_trading"    # Options: "backtesting", "paper_trading", "live_trading"

# Decision making
SWITCH_STRATEGY_VOLATILITY = 0.03  # Volatility threshold for switching strategies (3%)
