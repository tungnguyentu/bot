# AI Trading Bot for Binance Futures

An AI-powered trading bot that executes mean reversion strategies, adapts to market conditions, and optimizes trades using machine learning.

## Features

- Automated trading on Binance Futures using XGBoost and Reinforcement Learning
- Real-time trading signals with mean reversion and trend-following strategies
- Comprehensive backtesting framework with performance metrics
- Risk management with dynamic ATR-based stop-loss and take-profit
- Telegram notifications for trades and account status

## Setup

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory with your API keys:
   ```
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## Usage

### Train the AI model
```
python main.py --mode train --symbol BTCUSDT --interval 1h
```

### Run backtest
```
python main.py --mode backtest --symbol BTCUSDT --interval 1h
```

### Test mode (simulated trading with real market data)
```
python main.py --mode test --symbol BTCUSDT --interval 1h
```

### Live trading mode
```
python main.py --mode live --symbol BTCUSDT --interval 1h
```

## Performance Metrics

The bot evaluates performance based on:
- Sharpe Ratio (risk-adjusted return)
- Max Drawdown (risk exposure)
- Win Rate (trade accuracy)
- Profit Factor (profitability measure)
- Expectancy (average profit per trade)

## Risk Management

- Max Daily Drawdown: Stop trading if daily losses exceed configured threshold
- Position Sizing: Risk a small percentage of account per trade
- ATR-based Stop Loss: Dynamic placement based on market volatility

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and you can lose substantial funds. Always test thoroughly before trading with real money.
