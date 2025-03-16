from dataclasses import dataclass

@dataclass
class BotConfig:
    """Configuration for the trading bot."""
    symbol: str
    leverage: int
    mode: str
    interval: str
    quantity: float
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    train: bool
    
    def __str__(self) -> str:
        """String representation of the config."""
        return (
            f"BotConfig(symbol={self.symbol}, leverage={self.leverage}, "
            f"mode={self.mode}, interval={self.interval}, quantity={self.quantity}, "
            f"sl_atr_multiplier={self.sl_atr_multiplier}, tp_atr_multiplier={self.tp_atr_multiplier}, "
            f"train={self.train})"
        )
