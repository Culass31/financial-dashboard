from dataclasses import dataclass

@dataclass
class Config:
    CACHE_TTL: int = 3600
    MAX_STOCKS_PER_BATCH: int = 100
    DEFAULT_PERIOD: str = "10y"
    
    # API configurations
    YAHOO_FINANCE_CONFIG = {
        'retry_attempts': 3,
        'timeout': 30
    }
    
    # UI settings
    UI_CONFIG = {
        'page_title': "Analyse Marchés Mondiaux",
        'page_icon': "📈",
        'layout': "wide"
    }

config = Config()