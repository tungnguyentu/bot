# Binance Futures Trading Bot

A fully automated Binance Futures trading bot that implements both scalping and swing trading strategies. The bot uses technical indicators to make trading decisions, manages risk dynamically, and sends real-time notifications via Telegram.

## Features

- **Dual Trading Strategies**:
  - **Scalping**: Short-term trades using RSI, Bollinger Bands, and Moving Averages
  - **Swing Trading**: Medium-term trades using Ichimoku Cloud, MACD, and volume analysis

- **Smart Strategy Selection**: Automatically selects the optimal strategy based on current market conditions

- **Advanced Order Types**: Supports limit, market, and stop orders with automatic stop-loss and take-profit

- **Dynamic Risk Management**: Position sizing based on account balance and volatility

- **Real-time Monitoring**: Sends detailed trade notifications and status reports via Telegram

- **Comprehensive Backtesting**: Test your strategies with historical data before trading real funds

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/binance-futures-bot.git
   cd binance-futures-bot
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create your configuration file
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your Binance API keys and Telegram credentials

## Usage

### Live Trading

```bash
python main.py --mode live
```

### Backtesting

```bash
python main.py --mode backtest --strategy scalping --symbol BTCUSDT --days 30 --initial 10000
```

#### Backtesting Parameters:
- `--strategy`: Choose between 'scalping' or 'swing'
- `--symbol`: Trading pair to backtest (e.g., BTCUSDT, ETHUSDT)
- `--days`: Number of days to backtest (default: 30)
- `--initial`: Initial balance for backtesting (default: 10000)

## Configuration

Most settings can be configured in the `config.py` file:

- Trading pairs
- Strategy parameters
- Risk management settings
- Timeframes

For API credentials and secrets, use the `.env` file.

## Safety Features

- Maximum drawdown limit (default 20%)
- Maximum number of concurrent trades
- Automatic error recovery and restart mechanism
- Extensive logging and monitoring

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrency futures involves significant risk. Only trade with funds you can afford to lose. The developers of this bot are not responsible for any financial losses incurred through its use.

## License

MIT
