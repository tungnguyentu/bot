# Binance Futures AI Trading Bot

## AI-Powered Mean Reversion Strategy

This bot implements a fully automated mean reversion trading strategy for Binance Futures, using technical indicators and risk management rules.

### Features

- **Mean Reversion Strategy**: Trading based on price deviations from the mean
- **Technical Indicators**: Bollinger Bands, RSI, MACD, VWAP
- **Risk Management**: Dynamic stop-loss and take-profit based on ATR
- **Telegram Notifications**: Real-time alerts for all trading activities
- **AI Decision Making**: Signal generation and optimization based on market conditions
- **Reinforcement Learning**: Train and optimize trading decisions through AI
- **Multiple Operation Modes**: Live trading, test simulation, backtesting, and training

### Requirements

- Python 3.8+
- TA-Lib (technical analysis library)
- Binance API key with Futures trading permissions
- Telegram Bot for notifications
- TensorFlow (optional, for reinforcement learning features)

### Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure environment variables**

Copy the example environment file and edit it with your own API keys:

```bash
cp .env.example .env
nano .env  # or use any text editor
```

Fill in your:
- Binance API Key and Secret (with Futures trading permissions)
- Telegram Bot Token and Chat ID
- Optional: Customize trading parameters

3. **Run the bot**

Basic usage (live trading):
```bash
python main.py
```

With command line arguments:
```bash
# Specify trading symbols
python main.py --symbols BTCUSDT,ETHUSDT

# Specify leverage
python main.py --leverage 10

# Specify both
python main.py --symbols SOLUSDT,BNBUSDT --leverage 15
```

### Operation Modes

#### Test Mode
Run the bot in test mode to simulate trading without executing real trades:

```bash
# Basic test mode with default 1000 USDT balance
python main.py --test

# Test with specific symbols and starting balance
python main.py --test --symbols BTCUSDT,ETHUSDT --test-balance 5000
```

#### Backtesting Mode
Evaluate strategy performance with historical data:

```bash
# Basic backtest with default settings
python main.py --backtest --symbols BTCUSDT

# Backtest with specific date range
python main.py --backtest --symbols BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31
```

#### Training Mode
Train the reinforcement learning model to optimize trading decisions:

```bash
# Train the RL model on BTCUSDT data
python main.py --train --symbols BTCUSDT

# Train with more episodes
python main.py --train --symbols ETHUSDT --episodes 200
```

### Trading Strategy

The bot uses a mean reversion strategy with the following rules:

- **Buy Signal**: Price below lower Bollinger Band + RSI < 25 + MACD histogram turning positive
- **Sell Signal**: Price above upper Bollinger Band + RSI > 75 + MACD histogram turning negative

### Risk Management

- Configurable leverage (default: 20x)
- Dynamic stop-loss: 1.5x ATR
- Take-profit strategy: Partial (1x ATR) and full (2x ATR)
- Maximum risk per trade: 1% of account balance

### Configuration Options

The bot can be customized through environment variables or command line arguments:

**Command Line Arguments**:
- `--symbols`: Comma-separated list of trading pairs (e.g., BTCUSDT,ETHUSDT)
- `--leverage`: Trading leverage (e.g., 20)
- `--test`: Run in test mode without making real trades
- `--backtest`: Run in backtest mode with historical data
- `--train`: Run in training mode with reinforcement learning
- `--test-balance`: Starting balance for test mode (default: 1000 USDT)
- `--start-date`: Start date for backtesting (format: YYYY-MM-DD)
- `--end-date`: End date for backtesting (format: YYYY-MM-DD)
- `--episodes`: Number of episodes for RL training (default: 100)

**Environment Variables**:
See `.env.example` for all available options.

### Disclaimer

Trading cryptocurrencies involves significant risk and can lead to loss of capital. This trading bot is provided for educational purposes only. Use it at your own risk.
