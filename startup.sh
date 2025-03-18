#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate virtual environment if it exists
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
    echo "Activated virtual environment"
fi

# Start the trading bot with watchdog
echo "Starting trading bot with watchdog..."
python "$DIR/utils/watchdog.py" "$DIR/main.py" $@

# Deactivate virtual environment if activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
