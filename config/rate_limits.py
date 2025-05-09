# config/rate_limits.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class RateLimitConfig:
    # Délais entre les requêtes (en secondes)
    DEFAULT_DELAY: float = 2.0
    FAST_DELAY: float = 1.0
    SLOW_DELAY: float = 5.0
    
    # Configuration par type d'opération
    OPERATION_DELAYS: Dict[str, float] = None
    
    # Configuration des tentatives
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0
    BACKOFF_FACTOR: float = 2.0
    
    # Cache
    CACHE_TTL: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.OPERATION_DELAYS is None:
            self.OPERATION_DELAYS = {
                'history': self.DEFAULT_DELAY,
                'fundamentals': self.SLOW_DELAY,
                'financials': self.SLOW_DELAY,
                'isin': self.FAST_DELAY,
            }
    
    def get_delay_for_operation(self, operation: str) -> float:
        """Obtenir le délai pour un type d'opération"""
        return self.OPERATION_DELAYS.get(operation, self.DEFAULT_DELAY)

# Instance globale de configuration
rate_limit_config = RateLimitConfig()