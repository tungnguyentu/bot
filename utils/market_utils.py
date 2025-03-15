import logging
import math

logger = logging.getLogger(__name__)

def round_to_step_size(quantity, step_size):
    """Round quantity to the correct step size for Binance orders"""
    if step_size == 0:
        return quantity
        
    # Convert step size to precision
    precision = 0
    if '.' in str(step_size):
        precision_str = str(step_size).rstrip('0')
        if '.' in precision_str:
            precision = len(precision_str.split('.')[1])
    
    # Round down quantity to the correct step size
    return math.floor(quantity * (10 ** precision)) / (10 ** precision)

def get_quantity_precision(symbol_info):
    """Extract the quantity precision from symbol info"""
    for filter in symbol_info.get('filters', []):
        if filter.get('filterType') == 'LOT_SIZE':
            step_size = float(filter.get('stepSize', '0.00000001'))
            # Calculate the precision from step size
            if step_size == 0:
                return 8  # Default high precision
                
            precision = 0
            if '.' in str(step_size):
                precision_str = str(step_size).rstrip('0')
                if '.' in precision_str:
                    precision = len(precision_str.split('.')[1])
            return precision
    
    # Default if LOT_SIZE filter is not found
    return 8

def validate_and_format_order(symbol_info, side, quantity):
    """Validate and format order quantity according to symbol rules"""
    try:
        # Extract the quantity precision from LOT_SIZE filter
        precision = get_quantity_precision(symbol_info)
        
        # Find the step size for rounding
        step_size = 0.00000001  # Default
        min_qty = 0.00000001    # Default
        for filter in symbol_info.get('filters', []):
            if filter.get('filterType') == 'LOT_SIZE':
                step_size = float(filter.get('stepSize', step_size))
                min_qty = float(filter.get('minQty', min_qty))
        
        # Round to step size
        formatted_qty = round_to_step_size(quantity, step_size)
        
        # Ensure quantity is not less than min quantity
        if formatted_qty < min_qty:
            logger.warning(f"Quantity {formatted_qty} is less than minimum {min_qty}")
            return None, f"Quantity below minimum of {min_qty}"
        
        # Check for minimum notional value if NOTIONAL filter exists
        for filter in symbol_info.get('filters', []):
            if filter.get('filterType') == 'MIN_NOTIONAL':
                min_notional = float(filter.get('minNotional', 0))
                # Get current price to calculate notional value
                # Note: This is a simplified approach - in production, you'd want to use the actual price
                if 'markPrice' in symbol_info:
                    price = float(symbol_info['markPrice'])
                    notional_value = formatted_qty * price
                    if notional_value < min_notional:
                        logger.warning(f"Order notional value {notional_value} is below minimum {min_notional}")
                        return None, f"Order value too small"
        
        # Fix: Format with correct precision and ensure it's a string for Binance API
        formatted_qty_str = f"{{:.{precision}f}}".format(formatted_qty)
        
        # Remove trailing zeros (but keep the required precision)
        if '.' in formatted_qty_str:
            parts = formatted_qty_str.split('.')
            whole_part = parts[0]
            decimal_part = parts[1].ljust(precision, '0')[:precision]  # Ensure exact precision
            formatted_qty_str = f"{whole_part}.{decimal_part}"
            
            # Trim unnecessary trailing zeros only if precision allows
            if decimal_part.rstrip('0') != '':
                formatted_qty_str = f"{whole_part}.{decimal_part.rstrip('0')}"
            elif precision == 0:
                formatted_qty_str = whole_part
        
        logger.info(f"Formatted quantity: {quantity} -> {formatted_qty_str} (precision: {precision})")
        
        # Return the formatted string quantity for direct use in API calls
        return formatted_qty_str, None
        
    except Exception as e:
        logger.error(f"Error formatting order quantity: {e}")
        return None, str(e)
