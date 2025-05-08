import logging
from datetime import datetime

class AppLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.FileHandler(f'logs/{name}_{datetime.now():%Y%m%d}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_analysis(self, ticker, analysis_type):
        self.logger.info(f"Analysis performed: {analysis_type} for {ticker}")