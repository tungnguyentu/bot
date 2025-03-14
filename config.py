import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()

        # API credentials for production
        self.binance_api_key = os.getenv("BINANCE_API_KEY", "")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")
        
        # API credentials for testnet
        self.binance_testnet_api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        self.binance_testnet_api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        
        # Telegram credentials
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        # Trading parameters
        self.max_open_positions = 3
        self.initial_leverage = 20
        self.max_daily_drawdown = 0.08  # 8%

        # Risk management
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        self.position_size_percent = 0.02  # 2% of balance per trade
        
        # Fixed take profit and stop loss percentages (for faster trading)
        self.take_profit_pct = 0.015  # 1.5% take profit
        self.stop_loss_pct = 0.01     # 1% stop loss

        # Model parameters
        self.model_dir = "models"
        self.rl_model_name = "ppo_trading_model"
        self.prediction_model_name = "xgb_prediction_model"
        
        # Trading frequency enhancement
        self.trading_sensitivity = 0.7  # Higher value = more trades (0.0 to 1.0)

        # Features for the model
        self.features = [
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
            "atr",
            "vwap",
            "adx",
            "obv",
        ]

        # Backtesting parameters
        self.initial_balance = 10000  # USDT

        # Advanced settings
        self.lookback_window = 100  # Number of candles to consider for features
        self.retraining_interval = 168  # Retrain model every 168 hours (1 week)
