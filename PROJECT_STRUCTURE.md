# Project Structure

This document provides an overview of the project structure and the purpose of each file.

## Core Files

- `main.py`: Entry point for the trading bot. Contains the main loop and command-line interface.
- `config.py`: Configuration settings for the trading bot.
- `strategy.py`: Implementation of the scalping strategy.
- `backtest.py`: Backtesting functionality to test the strategy on historical data.

## API Clients

- `binance_client.py`: Client for interacting with the Binance API.
- `telegram_notifier.py`: Client for sending notifications via Telegram.

## Technical Analysis

- `indicators.py`: Implementation of technical indicators (VWAP, RSI, ATR, etc.).

## Utilities

- `utils.py`: Utility functions for the trading bot.

## Configuration Files

- `.env.example`: Template for the environment variables file.
- `requirements.txt`: List of Python dependencies.

## Documentation

- `README.md`: Overview of the project.
- `INSTALL.md`: Installation and usage instructions.
- `PROJECT_STRUCTURE.md`: This file, explaining the project structure.

## Directory Structure

```
ai-trading-bot/
├── main.py                # Entry point
├── config.py              # Configuration settings
├── strategy.py            # Trading strategy
├── backtest.py            # Backtesting functionality
├── binance_client.py      # Binance API client
├── telegram_notifier.py   # Telegram notification service
├── indicators.py          # Technical indicators
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # Project overview
├── INSTALL.md             # Installation instructions
└── PROJECT_STRUCTURE.md   # Project structure documentation
```

## Data Flow

1. `main.py` initializes the components and starts the main loop.
2. The main loop calls `strategy.analyze_market()` to analyze market data.
3. `strategy.py` uses `binance_client.py` to fetch market data.
4. `indicators.py` calculates technical indicators on the market data.
5. `strategy.execute_signals()` executes trading signals based on the analysis.
6. `strategy.manage_positions()` manages open positions.
7. `telegram_notifier.py` sends notifications about trades and errors.
8. `utils.py` provides utility functions used throughout the codebase.

## Backtesting Flow

1. `main.py --backtest` initializes the backtester.
2. `backtest.py` fetches historical data from Binance.
3. The backtester simulates the strategy on historical data.
4. Results are calculated and plotted.

## Adding New Features

- **New Indicators**: Add new technical indicators to `indicators.py`.
- **Strategy Modifications**: Modify the trading logic in `strategy.py`.
- **Risk Management**: Adjust risk management parameters in `config.py`.
- **UI/Visualization**: Add visualization tools for backtesting results. 