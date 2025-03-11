# Installation and Usage Guide

## Prerequisites

- Python 3.8 or higher
- Binance account with API keys
- Telegram bot (optional, for notifications)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-trading-bot
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file from the template:
   ```
   cp .env.example .env
   ```

6. Edit the `.env` file with your Binance API keys and Telegram credentials.

## Configuration

Edit the `config.py` file to customize the trading parameters:

- Trading pair
- Timeframe
- Leverage
- Position size
- Strategy parameters
- Risk management settings

## Usage

### Running the Bot

To run the bot in test mode (no real trades):
```
python main.py --test
```

To run the bot in live mode:
```
python main.py
```

### Running Backtests

To run a backtest with default settings:
```
python main.py --backtest
```

To run a backtest with custom settings:
```
python main.py --backtest --symbol BTCUSDT --timeframe 5m --start-date 2023-01-01 --end-date 2023-12-31 --initial-balance 10000
```

## Telegram Notifications

If you want to receive Telegram notifications:

1. Create a Telegram bot using BotFather
2. Get your chat ID by messaging @userinfobot
3. Add the bot token and chat ID to your `.env` file

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software. 