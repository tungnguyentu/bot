# AI-Powered Trading Bot for Binance Futures

This trading bot uses machine learning and technical indicators to generate trading signals for Binance Futures. It can operate in backtest, test (using Binance Testnet), or live trading modes.

## Features

- **AI-Powered Predictions**: Uses XGBoost to predict price movements
- **Technical Indicators**: Combines multiple indicators for signal generation
  - Bollinger Bands (14, 2)
  - RSI (6)
  - MACD (5, 13, 1)
  - VWAP
  - ATR (14)
- **Risk Management**: Dynamic stop-loss and take-profit levels based on ATR
- **Multiple Modes**: Backtest, test (Binance Testnet), and live trading
- **Telegram Notifications**: Trade signals, executions, and performance reports
- **Investment Management**: Control your trading capital with the invest parameter

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy the example environment file and update it with your API keys:
```bash
cp .env.example .env
```
4. Edit the `.env` file with your Binance API keys and Telegram credentials

## Usage

### Training the Model

```bash
python main.py --symbol SOLUSDT --mode backtest --train
```

### Backtesting

```bash
python main.py --symbol SOLUSDT --mode backtest --invest 500
```

### Test Trading (Binance Testnet)

```bash
python main.py --symbol SOLUSDT --mode test --leverage 20 --invest 100
```

### Live Trading

```bash
python main.py --symbol SOLUSDT --mode live --leverage 20 --invest 100
```

## Command Line Arguments

- `--symbol`: Trading pair (default: SOLUSDT)
- `--leverage`: Leverage to use (default: 20)
- `--mode`: Trading mode - 'backtest', 'test', or 'live' (default: test)
- `--interval`: Timeframe for data (default: 1h)
- `--quantity`: Trading quantity in base asset (default: 0.1)
- `--sl_atr_multiplier`: Stop Loss ATR multiplier (default: 1.5)
- `--tp_atr_multiplier`: Take Profit ATR multiplier (default: 2.0)
- `--train`: Train the model before trading
- `--invest`: Amount to invest in trading (default: 100.0)

## Trading Strategy

The bot follows a Mean Reversion Strategy with AI validation:

1. Identifies price deviations from the mean
2. Confirms reversals with multiple technical indicators
3. Uses AI model to validate trade signals and calculate confidence
4. Implements dynamic risk management based on market volatility

## Risk Management

The bot uses the following risk management principles:
- Maximum 2% risk per trade based on your investment amount
- Position sizing automatically calculated using stop loss and investment value
- Leveraged positions are supported but controlled by your investment parameter

## Telegram Notifications

The bot sends notifications for:
- Trade entry (Buy/Sell, price, reason)
- Trade exit (SL/TP hit, manual exit, reason)
- Performance reports (PnL, win rate, max drawdown)

## Warnings

- Cryptocurrency trading involves significant risk
- Always start with small position sizes
- Test thoroughly before using with real funds
- Past performance does not guarantee future results
