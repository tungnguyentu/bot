# Troubleshooting Guide

This guide covers common issues you might encounter when running the AI Trading Bot and how to resolve them.

## API Key Issues

### Invalid API Key

**Error Message:**
```
Invalid API key. Please check your credentials.
```

**Solution:**
1. Make sure you're using the correct API keys for the environment (main or testnet)
2. For testnet, generate API keys from https://testnet.binancefuture.com/
3. For live trading, generate API keys from https://www.binance.com/en/my/settings/api-management
4. Ensure your API keys have the necessary permissions enabled (Futures trading, reading)
5. Check for any IP restrictions on your API keys

### Connection Issues

**Error Message:**
```
Connection error. Please check your internet connection.
```

**Solution:**
1. Verify your internet connection is stable
2. Check if Binance is accessible from your location (you might need a VPN)
3. Ensure your firewall or security software isn't blocking the connection
4. Try increasing the timeout in the Binance client configuration

## Leverage Setting Issues

**Error Message:**
```
Error setting leverage: Leverage is too high
```

**Solution:**
1. Different symbols have different maximum leverage limits
2. Try setting a lower leverage value in `config.py`
3. The bot will automatically use the maximum allowed leverage if your setting is too high

## No Trading Signals

**Issue:** The bot is running but not generating any trading signals.

**Solution:**
1. Check the strategy parameters in `config.py` and adjust them:
   - Try lowering the RSI thresholds (e.g., RSI_OVERBOUGHT to 60, RSI_OVERSOLD to 40)
   - Reduce the VOLUME_THRESHOLD value
2. Run a backtest to see if your current parameters generate any signals:
   ```
   python main.py --backtest --symbol BTCUSDT --timeframe 5m --start-date 2023-01-01 --end-date 2023-06-01
   ```
3. Review the logs to see if conditions are being met individually but not simultaneously

## Telegram Notification Issues

**Error Message:**
```
Failed to initialize Telegram bot: Invalid token
```

**Solution:**
1. Make sure you've created a Telegram bot using BotFather (https://t.me/botfather)
2. Check that the bot token in your `.env` file is correct
3. Ensure you've added the correct chat ID to your `.env` file
4. Try disabling Telegram notifications by setting `ENABLE_TELEGRAM = False` in `config.py`

## Installation Issues

**Error Message:**
```
ModuleNotFoundError: No module named 'binance'
```

**Solution:**
1. Make sure you've installed all dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Check if you're using the correct Python environment
3. Try creating a new virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Performance Issues

**Issue:** The bot is generating too many or too few signals.

**Solution:**
1. Adjust the strategy parameters in `config.py`:
   - For more signals: Make the conditions less strict (lower RSI thresholds, lower volume threshold)
   - For fewer signals: Make the conditions more strict (higher RSI thresholds, higher volume threshold)
2. Run backtests with different parameters to find the optimal settings
3. Consider adding additional indicators or conditions to filter out poor-quality signals

## Getting Help

If you're still experiencing issues:

1. Check the logs in the `logs/` directory for more detailed error messages
2. Review the documentation in the `README.md` and `INSTALL.md` files
3. Make sure you're using the latest version of the bot
4. Open an issue on the GitHub repository with a detailed description of your problem 