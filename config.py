import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading Parameters
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Trading pairs
TIMEFRAMES = {
    'scalping': '5m',    # 5-minute candles for scalping
    'swing': '4h'        # 4-hour candles for swing trading
}

# Strategy Parameters
STRATEGY_PARAMS = {
    'scalping': {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'bb_period': 20,
        'bb_std': 2.0,
        'ma_fast': 9,
        'ma_slow': 21
    },
    'swing': {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ichimoku_tenkan': 9,
        'ichimoku_kijun': 26,
        'ichimoku_senkou_span_b': 52,
        'volume_ma': 20
    }
}

# Risk Management
RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_DRAWDOWN = 0.20    # 20% max drawdown
MAX_OPEN_TRADES = 3    # Maximum concurrent open trades
DEFAULT_LEVERAGE = 5   # Default leverage

# Performance Evaluation
METRICS = ['sharpe_ratio', 'profit_factor', 'max_drawdown', 'win_rate']

# Execution Settings
EXECUTION_RETRY_ATTEMPTS = 3
EXECUTION_RETRY_DELAY = 2  # seconds
