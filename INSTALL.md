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

## Setting Up API Keys

### For Live Trading (Binance Futures)

1. Log in to your Binance account
2. Go to API Management: https://www.binance.com/en/my/settings/api-management
3. Create a new API key with the following permissions:
   - Enable Reading
   - Enable Futures
   - Enable Spot & Margin Trading
4. Set IP restrictions for security (recommended)
5. Copy the API key and secret to your `.env` file

### For Test Mode (Binance Futures Testnet)

1. Go to Binance Futures Testnet: https://testnet.binancefuture.com/
2. Log in with your Binance account
3. Go to API Management in the testnet
4. Generate new API keys
5. Copy the testnet API key and secret to your `.env` file when running in test mode

**Important**: API keys for the main Binance site will not work on the testnet, and vice versa. Make sure you're using the correct keys for your environment.

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

1. Create a Telegram bot using BotFather (https://t.me/botfather)
2. Get your chat ID by messaging @userinfobot
3. Add the bot token and chat ID to your `.env` file

## Troubleshooting

### API Key Issues

- **"Invalid Api-Key ID"**: Make sure you're using the correct API keys for the environment (main or testnet)
- **Permission Denied**: Check that your API keys have the necessary permissions enabled
- **IP Restriction**: If you've set IP restrictions, make sure your current IP is allowed

### Leverage Setting Issues

- If you encounter issues setting leverage, check that your account has futures trading enabled
- Some symbols may have maximum leverage limits lower than what you're trying to set

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software. 