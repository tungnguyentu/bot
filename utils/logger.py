import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    """Set up and return a logger with the given name and log file."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger(name)
    
    # Check if logger was already configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and set it for handlers
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
