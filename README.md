# Binance Futures Trading Bot

This bot implements automated trading strategies for Binance Futures, supporting both scalping and swing trading approaches.

## Features

- **Dual trading strategies**: Scalping and Swing Trading
- **Dynamic strategy switching** based on market volatility
- **Risk management** with position sizing and drawdown protection
- **Trailing stop-loss** implementation
- **Telegram notifications** with detailed trade reasoning
- **Backtesting system** with performance metrics
- **Paper trading** using Binance Futures testnet

## Setup Instructions

1. Run the setup script to prepare the environment:
   ```
   ./setup.sh
   ```

2. Edit the `.env` file with your API keys:
   - For live trading: Use your regular Binance Futures API keys
   - For paper trading: Get testnet API keys from https://testnet.binancefuture.com/

3. Start the bot:
   ```
   ./startup.sh
   ```

## Using Binance Futures Testnet

For paper trading, the bot uses the Binance Futures testnet. Follow these steps:

1. Go to https://testnet.binancefuture.com/
2. Register for a testnet account
3. Generate API keys (top-right menu → API Management)
4. Add the testnet API keys to your .env file:
   ```
   BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
   BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here
   ```
5. The testnet provides 10,000 USDT for testing purposes

## Running Modes

- **Backtesting mode**: Test strategies on historical data
  ```
  ./startup.sh --mode backtest
  ```

- **Paper trading mode**: Trade with real-time data but simulated orders (testnet)
  ```
  ./startup.sh --mode paper
  ```

- **Live trading mode**: Trade with real funds (use with caution!)
  ```
  ./startup.sh --mode live
  ```

## Configuration

Edit `config.py` to customize:
- Trading pairs
- Strategy parameters
- Risk management settings
- Timeframes
