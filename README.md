# AI Trading Bot for Binance Futures

An AI-powered trading bot for Binance Futures using a scalping strategy on the 5-minute timeframe.

## Features

- Real-time market data analysis from Binance API
- Automated trade execution based on predefined conditions
- Technical indicators: VWAP, RSI, ATR
- Order book and volume analysis
- Risk management with stop-loss and take-profit
- Telegram notifications for trade events
- Backtesting capabilities

## Strategy Overview

- **Timeframe:** 5-minute
- **Leverage:** x20
- **Order Type:** Market orders
- **Execution:** Fully automated

### Trade Entry Rules

- **Long Entry:**
  - Price is above VWAP (bullish trend)
  - RSI is below 40 (oversold condition)
  
- **Short Entry:**
  - Price is below VWAP (bearish trend)
  - RSI is above 60 (overbought condition)

### Exit Strategy

- **Take Profit:** 0.3% - 0.5% price move
- **Stop Loss:** ATR-based or fixed at -0.2%
- **Trailing Stop:** Adjusts to lock in profits if the price moves favorably

## Backtest Results

We ran a backtest on BTCUSDT from January 1, 2023, to June 1, 2023, with the following results:

- **Total Trades:** 100
- **Win Rate:** 56.00%
- **Profit Factor:** 1.13
- **Return:** 2.42%
- **Max Drawdown:** 0.98%

For more detailed results, see [RESULTS.md](RESULTS.md).

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Binance API keys and Telegram bot token
4. Run the bot: `python main.py`

## Configuration

Edit the `config.py` file to customize:
- Trading pairs
- Timeframe
- Leverage
- Position size
- Strategy parameters
- Risk management settings

## Documentation

- [Installation Guide](INSTALL.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Backtest Results](RESULTS.md)

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software. 