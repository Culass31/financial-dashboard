# services/data_service.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import time
from functools import wraps
from threading import Lock
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import de curl_cffi pour contourner les limitations
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    logger.warning("curl_cffi not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "curl_cffi"])
    from curl_cffi import requests as curl_requests

# Dictionnaire global pour stocker les derniers appels API par ticker
_api_calls = {}
_api_lock = Lock()
_session_cache = None

# Configuration des délais
RATE_LIMIT_DELAY = 2.0  # Délai entre les requêtes en secondes
MAX_RETRIES = 3
RETRY_DELAY = 5

def get_chrome_session():
    """Créer ou récupérer une session Chrome pour contourner les limitations"""
    global _session_cache
    if _session_cache is None:
        _session_cache = curl_requests.Session(impersonate="chrome")
    return _session_cache

# Décorateur pour gérer le rate limiting
def rate_limit(min_interval_seconds=RATE_LIMIT_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(self, ticker, *args, **kwargs):
            with _api_lock:
                now = datetime.now()
                last_call = _api_calls.get(ticker)
                
                if last_call is not None:
                    time_since_last_call = (now - last_call).total_seconds()
                    if time_since_last_call < min_interval_seconds:
                        sleep_time = min_interval_seconds - time_since_last_call
                        logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds for {ticker}")
                        time.sleep(sleep_time)
                
                # Essayer plusieurs fois en cas d'échec
                last_exception = None
                for attempt in range(MAX_RETRIES):
                    try:
                        result = func(self, ticker, *args, **kwargs)
                        _api_calls[ticker] = datetime.now()
                        return result
                    except Exception as e:
                        last_exception = e
                        if attempt < MAX_RETRIES - 1:
                            logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        else:
                            logger.error(f"All attempts failed for {ticker}: {str(e)}")
                            raise last_exception
        return wrapper
    return decorator

@st.cache_data(ttl=1800)  # Cache réduit à 30 minutes pour les données de marché
def cached_get_stock_history_with_session(ticker: str, period: str = '10y', session=None) -> pd.DataFrame:
    """Retrieve historical stock data with custom session"""
    try:
        if session is None:
            session = get_chrome_session()
        
        # Créer l'objet Ticker avec la session personnalisée
        stock = yf.Ticker(ticker, session=session)
        
        # Récupérer l'historique
        df = stock.history(period=period)
        
        if df.empty:
            logger.warning(f"No data available for {ticker}")
            return pd.DataFrame()
        
        df['date_str'] = df.index.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        logger.error(f"Error retrieving history for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def cached_get_fundamental_data_with_session(ticker: str, session=None) -> Optional[Dict]:
    """Retrieve fundamental data with custom session"""
    try:
        if session is None:
            session = get_chrome_session()
            
        stock = yf.Ticker(ticker, session=session)
        info = stock.info
        
        if not info:
            return None
        
        # Récupération des états financiers avec la même session
        income_stmt, balance_sheet, cashflow = cached_get_historical_financials_with_session(ticker, session)
        
        # Construction des données fondamentales
        general_data = {
            "Nom": info.get('longName', None),
            "Secteur": info.get('sector', None),
            "Industrie": info.get('industry', None),
            'Pays': info.get('country', ''),
            "Site web": info.get('website', None),
            "Description": info.get('longBusinessSummary', None)
        }
        
        market_data = {
            "Prix actuel": info.get('currentPrice', info.get('regularMarketPrice', None)),
            "Prix d'ouverture": info.get('open', None),
            "Précédente clôture": info.get('previousClose', None),
            "Prix haut (jour)": info.get('dayHigh', None),
            "Prix bas (jour)": info.get('dayLow', None),
            "Prix cible moyen": info.get('targetMeanPrice', None),
            "Prix cible haut": info.get('targetHighPrice', None),
            "Prix cible bas": info.get('targetLowPrice', None),
            "Recommandation": info.get('recommendationKey', None),
            "52 semaines - Haut": info.get('fiftyTwoWeekHigh', None),
            "52 semaines - Bas": info.get('fiftyTwoWeekLow', None),
            "Rendement du dividende": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else None,
            "Ex-Date dividende": info.get('exDividendDate', None),
            "Actions en circulation": info.get('sharesOutstanding', None),
            "Volume": info.get('volume', None),
            "Volume moyen": info.get('averageVolume', None),
            "Volume moyen (10j)": info.get('averageVolume10days', None),
            "Capitalisation boursière": info.get('marketCap', None),
            "Beta": info.get('beta', None),
        }
        
        fundamental_data = {
            "PER": info.get('trailingPE', None),
            "PER (Forward)": info.get('forwardPE', None),
            "BPA": info.get('trailingEps', None),
            "BPA (Forward)": info.get('forwardEps', None),
            "PEG Ratio": info.get('trailingPegRatio', None),
            "P/B Cours/Valeur Comptable": info.get('priceToBook', None),
            "P/S (Price to Sales)": info.get('priceToSalesTrailing12Months', None),
            "Marge brute (%)": f"{info.get('grossMargins', 0) * 100:.2f}%" if info.get('grossMargins') else None,
            "Marge opérationnelle (%)": f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else None,
            "Marge nette (%)": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
            "ROE (%)": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
            "ROA (%)": info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else None,
            "Ratio d'endettement": info.get('debtToEquity', None),
            "Croissance du BPA (%)": info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None,
            "Croissance du CA (%)": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
            "Dividende": info.get('dividendRate', None),
            "Rendement du dividende (%)": info.get('dividendYield', 0) if info.get('dividendYield') else None,
            "Quick Ratio": info.get('quickRatio', None),
            "Current Ratio": info.get('currentRatio', None),
            "EV/EBITDA": info.get('enterpriseToEbitda', None),
            "EV/Revenue": info.get('enterpriseToRevenue', None),
        }
        
        # Données financières additionnelles
        financial_data = {}
        
        if not income_stmt.empty and len(income_stmt.columns) > 0:
            last_year = income_stmt.columns[0]
            financial_data.update({
                "Chiffre d'affaires": income_stmt.loc['Total Revenue', last_year] if 'Total Revenue' in income_stmt.index else None,
                "Résultat net": income_stmt.loc['Net Income', last_year] if 'Net Income' in income_stmt.index else None,
                "EBITDA": income_stmt.loc['EBITDA', last_year] if 'EBITDA' in income_stmt.index else None
            })
        
        if not balance_sheet.empty and len(balance_sheet.columns) > 0:
            last_period = balance_sheet.columns[0]
            financial_data.update({
                "Total Actif": balance_sheet.loc['Total Assets', last_period] if 'Total Assets' in balance_sheet.index else None,
                "Total Dette": balance_sheet.loc['Total Debt', last_period] if 'Total Debt' in balance_sheet.index else None,
                "Fonds propres": balance_sheet.loc['Total Equity', last_period] if 'Total Equity' in balance_sheet.index else None
            })
        
        if not cashflow.empty and len(cashflow.columns) > 0:
            last_period = cashflow.columns[0]
            financial_data.update({
                "Free Cash Flow": cashflow.loc['Free Cash Flow', last_period] if 'Free Cash Flow' in cashflow.index else None
            })
        
        return {
            "Données générales": general_data,
            "Données de marché": market_data,
            "Données fondamentales": fundamental_data,
            "Données financières": financial_data
        }
    
    except Exception as e:
        logger.error(f"Error retrieving fundamental data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def cached_get_historical_financials_with_session(ticker: str, session=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieve historical financial statements with custom session"""
    try:
        if session is None:
            session = get_chrome_session()
            
        stock = yf.Ticker(ticker, session=session)
        
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        if income_stmt is None or income_stmt.empty:
            income_stmt = pd.DataFrame()
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = pd.DataFrame()
        if cashflow is None or cashflow.empty:
            cashflow = pd.DataFrame()
        
        return income_stmt, balance_sheet, cashflow
        
    except Exception as e:
        logger.error(f"Error retrieving financial data for {ticker}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

class DataService:
    """Service responsible for all data operations with rate limiting"""
    
    def __init__(self):
        self.market_structure = None
        self.session = get_chrome_session()
    
    @rate_limit()
    def get_stock_history(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        """Retrieve historical stock data with rate limiting"""
        return cached_get_stock_history_with_session(ticker, period, self.session)
    
    @rate_limit()
    def get_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Retrieve fundamental data with rate limiting"""
        return cached_get_fundamental_data_with_session(ticker, self.session)
    
    @rate_limit()
    def get_historical_financials(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve historical financial statements with rate limiting"""
        return cached_get_historical_financials_with_session(ticker, self.session)
    
    def get_market_structure(self) -> Dict:
        """Get or refresh market structure"""
        if self.market_structure is None:
            self.market_structure = cached_get_market_structure()
        return self.market_structure
    
    def flatten_market_structure(
        self, 
        filter_type: Optional[str] = None, 
        level1: Optional[str] = None, 
        level2: Optional[str] = None
    ) -> Dict:
        """Flatten market structure based on filters"""
        market_structure = self.get_market_structure()
        
        if not filter_type or not level1:
            return market_structure['all_stocks']
        
        flattened_stocks = {}
        
        if filter_type == 'region':
            if level1 in market_structure['regions']:
                if level2 and level2 in market_structure['regions'][level1]:
                    return market_structure['regions'][level1][level2]
                else:
                    for pays, stocks in market_structure['regions'][level1].items():
                        flattened_stocks.update(stocks)
                    return flattened_stocks
        
        elif filter_type == 'secteur':
            if level1 in market_structure['secteurs']:
                if level2 and level2 in market_structure['secteurs'][level1]:
                    return market_structure['secteurs'][level1][level2]
                else:
                    for industrie, stocks in market_structure['secteurs'][level1].items():
                        flattened_stocks.update(stocks)
                    return flattened_stocks
        
        elif filter_type == 'marche':
            if level1 in market_structure['marches']:
                return market_structure['marches'][level1]
        
        return {}
    
    @rate_limit()
    def get_isin_for_ticker(self, ticker: str) -> str:
        """Get ISIN for a ticker symbol with rate limiting"""
        try:
            stock = yf.Ticker(ticker, session=self.session)
            isin = stock.isin if hasattr(stock, 'isin') else 'N/A'
            return isin
        except:
            return 'N/A'