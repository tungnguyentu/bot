"""
Configuration settings for the AI Trading Bot.
"""

# Trading settings
SYMBOL = "BTCUSDT"  # Trading pair
TIMEFRAME = "5m"    # 5-minute timeframe
LEVERAGE = 20       # x20 leverage
POSITION_SIZE = 0.05  # 5% of available balance per trade

# Strategy parameters
RSI_PERIOD = 7      # RSI period
RSI_OVERBOUGHT = 60  # RSI overbought threshold (lowered from 70)
RSI_OVERSOLD = 40    # RSI oversold threshold (increased from 30)
VWAP_PERIOD = 14     # VWAP period
ATR_PERIOD = 10      # ATR period for stop loss calculation
VOLUME_THRESHOLD = 1.2  # Volume spike threshold (lowered from 1.5)

# Risk management
TAKE_PROFIT_PERCENT = 0.4  # 0.4% take profit
STOP_LOSS_PERCENT = 0.2    # 0.2% stop loss
USE_ATR_FOR_SL = True      # Use ATR for stop loss calculation
ATR_MULTIPLIER = 1.5       # ATR multiplier for stop loss
USE_TRAILING_STOP = True   # Use trailing stop
TRAILING_STOP_ACTIVATION = 0.2  # Activate trailing stop after 0.2% price move
TRAILING_STOP_CALLBACK = 0.1    # Trailing stop callback rate

# Execution settings
USE_MARKET_ORDERS = True   # Use market orders for entry and exit
MAX_ACTIVE_POSITIONS = 1   # Maximum number of active positions

# Notification settings
ENABLE_TELEGRAM = True     # Enable Telegram notifications
NOTIFY_ON_TRADE_OPEN = True
NOTIFY_ON_TRADE_CLOSE = True
NOTIFY_ON_ERROR = True

# Backtesting settings
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"

# API rate limits
MAX_REQUESTS_PER_MINUTE = 50
RATE_LIMIT_BUFFER = 0.8  # Use only 80% of available rate limit

# Logging settings
LOG_LEVEL = "INFO"
SAVE_TRADE_HISTORY = True

# Backtest settings
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-06-01"
DEFAULT_INITIAL_BALANCE = 10000 