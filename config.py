import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    def __init__(self, cli_args=None):
        # API Keys
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
        self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

        # Telegram Settings
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

        # Trading Parameters - Override with CLI arguments if provided
        if cli_args and cli_args.symbols:
            self.TRADING_SYMBOLS = cli_args.symbols.split(",")
        else:
            symbols_env = os.getenv("TRADING_SYMBOLS")
            self.TRADING_SYMBOLS = (
                symbols_env.split(",")
                if symbols_env
                else ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            )

        if cli_args and cli_args.leverage:
            self.LEVERAGE = cli_args.leverage
        else:
            self.LEVERAGE = int(os.getenv("LEVERAGE", 20))  # Default to 20x leverage

        self.TIMEFRAME = os.getenv("TIMEFRAME", "15m")  # Candlestick timeframe
        self.LOOKBACK_PERIOD = int(
            os.getenv("LOOKBACK_PERIOD", 100)
        )  # Number of candles to fetch
        self.SCAN_INTERVAL = int(
            os.getenv("SCAN_INTERVAL", 30)
        )  # Time between scans in seconds

        # Risk Management
        self.ACCOUNT_RISK_PER_TRADE = float(
            os.getenv("ACCOUNT_RISK_PER_TRADE", 0.01)
        )  # 1% of account balance per trade
        self.MAX_DRAWDOWN_PER_TRADE = float(
            os.getenv("MAX_DRAWDOWN_PER_TRADE", 0.02)
        )  # 2% of account balance maximum drawdown

        # Technical Indicators
        self.BB_PERIOD = int(os.getenv("BB_PERIOD", 14))  # Bollinger Bands period
        self.BB_STD_DEV = int(
            os.getenv("BB_STD_DEV", 2)
        )  # Bollinger Bands standard deviation
        self.RSI_PERIOD = int(os.getenv("RSI_PERIOD", 6))  # RSI period
        self.RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", 25))  # RSI oversold threshold
        self.RSI_OVERBOUGHT = int(
            os.getenv("RSI_OVERBOUGHT", 75)
        )  # RSI overbought threshold
        self.MACD_FAST = int(os.getenv("MACD_FAST", 5))  # MACD fast period
        self.MACD_SLOW = int(os.getenv("MACD_SLOW", 13))  # MACD slow period
        self.MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 1))  # MACD signal period
        self.ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))  # ATR period

        # Take Profit and Stop Loss
        self.SL_ATR_MULTIPLIER = float(
            os.getenv("SL_ATR_MULTIPLIER", 1.5)
        )  # Stop loss = ATR * multiplier
        self.PARTIAL_TP_ATR_MULTIPLIER = float(
            os.getenv("PARTIAL_TP_ATR_MULTIPLIER", 1.0)
        )  # First take profit
        self.FULL_TP_ATR_MULTIPLIER = float(
            os.getenv("FULL_TP_ATR_MULTIPLIER", 2.0)
        )  # Second take profit
        self.PARTIAL_TP_SIZE = float(
            os.getenv("PARTIAL_TP_SIZE", 0.5)
        )  # % of position to close at first TP
