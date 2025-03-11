# AI Trading Bot for Binance Futures

An advanced cryptocurrency trading bot that uses AI-powered technical analysis to trade on Binance Futures. The bot supports multiple trading strategies and includes backtesting capabilities.

## Features

- Multiple trading strategies:
  - **Scalping Strategy (5m)**: Short-term trades based on RSI and VWAP
  - **Swing Strategy (1h)**: Medium-term trades using MACD and Bollinger Bands
  - **Breakout Strategy (15m)**: Trend-following trades based on price breakouts and volume
- Real-time market analysis and automated trading
- Risk management with take profit, stop loss, and trailing stop
- Telegram notifications for trades and system status
- Comprehensive backtesting framework
- Support for multiple trading pairs
- Configurable leverage and position sizing

## Trading Strategies

### 1. Scalping Strategy
- **Timeframe**: 5 minutes
- **Indicators**: RSI and VWAP
- **Entry Rules**:
  - Long: RSI < 30 (oversold) and price above VWAP
  - Short: RSI > 70 (overbought) and price below VWAP
- **Risk Settings**:
  - Take Profit: 0.5%
  - Stop Loss: 0.3%

### 2. Swing Strategy
- **Timeframe**: 1 hour
- **Indicators**: MACD and Bollinger Bands
- **Entry Rules**:
  - Long: MACD crossover (bullish) and price below lower BB
  - Short: MACD crossunder (bearish) and price above upper BB
- **Risk Settings**:
  - Take Profit: 2.0%
  - Stop Loss: 1.0%

### 3. Breakout Strategy
- **Timeframe**: 15 minutes
- **Indicators**: ATR and Volume
- **Entry Rules**:
  - Long: Break above recent high with volume confirmation
  - Short: Break below recent low with volume confirmation
- **Risk Settings**:
  - Take Profit: 1.5%
  - Stop Loss: 0.8%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

5. Edit `.env` with your Binance API credentials and Telegram bot token.

## Configuration

The bot's behavior can be customized through the `config.py` file:

- Trading settings (symbol, leverage, position size)
- Strategy-specific parameters
- Risk management settings
- Notification preferences
- Backtesting parameters

## Usage

### Running the Bot

Run the bot with a specific strategy:

```bash
# Run with Scalping Strategy (default)
python main.py --symbol BTCUSDT --strategy scalping

# Run with Swing Strategy
python main.py --symbol BTCUSDT --strategy swing

# Run with Breakout Strategy
python main.py --symbol BTCUSDT --strategy breakout
```

Additional options:
- `--test`: Run in test mode (no real trades)
- `--timeframe`: Override default strategy timeframe
- `--leverage`: Set custom leverage

### Backtesting

Test strategies on historical data:

```bash
# Backtest Scalping Strategy
python main.py --backtest --symbol BTCUSDT --strategy scalping --start-date 2023-01-01 --end-date 2023-06-01

# Backtest Swing Strategy
python main.py --backtest --symbol BTCUSDT --strategy swing --start-date 2023-01-01 --end-date 2023-06-01

# Backtest Breakout Strategy
python main.py --backtest --symbol BTCUSDT --strategy breakout --start-date 2023-01-01 --end-date 2023-06-01
```

Additional backtest options:
- `--initial-balance`: Set initial balance (default: 10000 USDT)
- `--timeframe`: Override default strategy timeframe

## Backtest Results

The bot generates detailed backtest reports and visualizations:

- Equity curve
- Drawdown analysis
- Trade distribution
- Monthly returns
- Performance metrics:
  - Total trades
  - Win rate
  - Profit factor
  - Maximum drawdown
  - Risk-adjusted return

Results are saved in the `plots/` directory with strategy-specific filenames.

## Risk Warning

This bot is for educational purposes only. Cryptocurrency trading carries significant risks:

- High volatility
- Potential for significant losses
- Technical failures
- Market manipulation

Always start with small amounts and test thoroughly in test mode before live trading.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- New features
- Strategy improvements
- Documentation updates

## License

This project is licensed under the MIT License - see the LICENSE file for details. 