"""
Configuration settings for the AI Trading Bot.
"""

# Trading settings
SYMBOL = "BTCUSDT"  # Trading pair
TIMEFRAME = "5m"    # 5-minute timeframe
LEVERAGE = 20       # x20 leverage
POSITION_SIZE = 0.1  # 10% of available balance

# Strategy parameters - Scalping
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VWAP_PERIOD = 20
ATR_PERIOD = 14
VOLUME_THRESHOLD = 2.0

# Strategy parameters - Swing Trading
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BB_PERIOD = 20
BB_STD = 2.0

# Strategy parameters - Breakout
BREAKOUT_PERIOD = 20

# Risk management
TAKE_PROFIT_PERCENT = 1.0  # 1%
STOP_LOSS_PERCENT = 0.5    # 0.5%
USE_ATR_FOR_SL = True
ATR_MULTIPLIER = 2.0
USE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION = 0.5  # 0.5%
TRAILING_STOP_CALLBACK = 0.2    # 0.2%

# Execution settings
USE_MARKET_ORDERS = True
MAX_ACTIVE_POSITIONS = 3

# Notification settings
ENABLE_TELEGRAM = True     # Enable Telegram notifications
NOTIFY_ON_TRADE_OPEN = True
NOTIFY_ON_TRADE_CLOSE = True
NOTIFY_ON_ERROR = True

# Backtesting settings
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = "2023-12-31"

# API rate limits
MAX_REQUESTS_PER_MINUTE = 1200
MAX_ORDERS_PER_SECOND = 10
RATE_LIMIT_BUFFER = 0.8  # Use only 80% of available rate limit

# Logging settings
LOG_LEVEL = "INFO"
SAVE_TRADE_HISTORY = True
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Backtest settings
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-06-01"
DEFAULT_INITIAL_BALANCE = 10000

# Strategy timeframes
SCALPING_TIMEFRAME = '5m'
SWING_TIMEFRAME = '1h'
BREAKOUT_TIMEFRAME = '15m'

# Strategy-specific risk settings
SCALPING_TAKE_PROFIT = 0.5      # 0.5%
SCALPING_STOP_LOSS = 0.3        # 0.3%
SWING_TAKE_PROFIT = 2.0         # 2.0%
SWING_STOP_LOSS = 1.0           # 1.0%
BREAKOUT_TAKE_PROFIT = 1.5      # 1.5%
BREAKOUT_STOP_LOSS = 0.8        # 0.8% 