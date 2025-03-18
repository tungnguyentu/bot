#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p backtest_results

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << EOL
# Binance API credentials (Production)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Binance Testnet API credentials (For paper trading)
# Get your testnet API keys from https://testnet.binancefuture.com/
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here

# Telegram bot credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
EOL
    echo ".env template created. Please edit it with your API keys."
fi

# Make scripts executable
chmod +x startup.sh
chmod +x setup.sh

echo "Setup complete. Edit the .env file with your API keys before running the bot."
echo "For paper trading, get testnet API keys from: https://testnet.binancefuture.com/"
echo "To start the bot, run: ./startup.sh"
