# services/data_service.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import functools
from threading import Lock
import time

# Dictionnaire global pour stocker les derniers appels API par ticker
_api_calls = {}
_api_lock = Lock()

# Décorateur pour gérer le rate limiting
def rate_limit(min_interval_seconds=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, ticker, *args, **kwargs):
            with _api_lock:
                now = datetime.now()
                last_call = _api_calls.get(ticker)
                
                if last_call is not None:
                    time_since_last_call = (now - last_call).total_seconds()
                    if time_since_last_call < min_interval_seconds:
                        time.sleep(min_interval_seconds - time_since_last_call)
                
                result = func(self, ticker, *args, **kwargs)
                _api_calls[ticker] = datetime.now()
                return result
        return wrapper
    return decorator

# Cache partagé entre tous les appels
@st.cache_data(ttl=3600)
def cached_get_ticker_data(ticker: str) -> yf.Ticker:
    """Get cached ticker object"""
    return yf.Ticker(ticker)

@st.cache_data(ttl=3600)
def cached_get_ticker_info(ticker: str) -> Dict:
    """Get cached ticker info"""
    try:
        ticker_obj = cached_get_ticker_data(ticker)
        return ticker_obj.info
    except Exception as e:
        st.error(f"Error getting info for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def cached_get_stock_history(ticker: str, period: str = '5y') -> pd.DataFrame:
    """Retrieve historical stock data (cached function)"""
    try:
        # Attendez un peu avant l'appel pour respecter le rate limit
        time.sleep(0.5)
        
        ticker_obj = cached_get_ticker_data(ticker)
        df = ticker_obj.history(period=period)
        
        if df.empty:
            st.warning(f"No data available for {ticker}")
            return pd.DataFrame()
        
        df['date_str'] = df.index.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        if "rate limit" in str(e).lower():
            st.error(f"Rate limit reached for {ticker}. Please wait a moment.")
            time.sleep(2)  # Wait 2 seconds on rate limit error
        else:
            st.error(f"Error retrieving history for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cached_get_fundamental_data(ticker: str) -> Optional[Dict]:
    """Retrieve fundamental data for a stock (cached function)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Vérifier si les données sont valides
        if not info or info is None:
            st.warning(f"Aucune donnée disponible pour {ticker}")
            return None
            
        # Vérifier si l'API a retourné une erreur
        if 'error' in info:
            st.error(f"Erreur API pour {ticker}: {info.get('error', 'Erreur inconnue')}")
            return None
            
        # Vérifier si les données minimales sont présentes
        required_fields = ['longName', 'sector', 'currentPrice']
        missing_fields = [field for field in required_fields if field not in info or info[field] is None]
        
        if missing_fields:
            st.warning(f"Données incomplètes pour {ticker}. Champs manquants: {', '.join(missing_fields)}")
            # Continuer avec les données partielles
        
        income_stmt, balance_sheet, cashflow = cached_get_historical_financials(ticker)
        
        general_data = {
            "Nom": info.get('longName', ticker),  # Utiliser le ticker comme fallback
            "Secteur": info.get('sector', 'Non spécifié'),
            "Industrie": info.get('industry', 'Non spécifié'),
            'Pays': info.get('country', 'Non spécifié'),
            "Site web": info.get('website', None),
            "Description": info.get('longBusinessSummary', None)
        }
        
        # Gestion des données de marché avec valeurs par défaut
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
            "Actions ordinaires": balance_sheet.loc['Ordinary Shares Number', income_stmt.columns[0]] if not balance_sheet.empty and 'Ordinary Shares Number' in balance_sheet.index else None,
            "Volume": info.get('volume', None),
            "Volume moyen": info.get('averageVolume', None),
            "Volume moyen (10j)": info.get('averageVolume10days', None),
            "Capitalisation boursière": info.get('marketCap', None),
            "Beta": info.get('beta', None),
        }
        
        # Reste de la fonction...
        return {
            "Données générales": general_data,
            "Données de marché": market_data,
            "Données fondamentales": fundamental_data,
            "Données financières": financial_data
        }
    
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def cached_get_historical_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieve historical financial statements (cached function)"""
    try:
        # Attendez avant l'appel
        time.sleep(0.5)
        
        ticker_obj = cached_get_ticker_data(ticker)
        
        income_stmt = ticker_obj.income_stmt
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow
        
        if income_stmt is None or income_stmt.empty:
            income_stmt = pd.DataFrame()
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = pd.DataFrame()
        if cashflow is None or cashflow.empty:
            cashflow = pd.DataFrame()
        
        return income_stmt, balance_sheet, cashflow
        
    except Exception as e:
        if "rate limit" in str(e).lower():
            st.error(f"Rate limit reached for {ticker}. Please wait a moment.")
            time.sleep(2)
        else:
            st.error(f"Error retrieving financial data for {ticker}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=86400)
def cached_get_market_structure():
    try:
        df = pd.read_csv(
            r"https://raw.githubusercontent.com/Culass31/actions/refs/heads/main/actions.csv", 
            sep=";", 
            encoding='utf-8-sig'
        )
        
        market_structure = {
            'regions': {},
            'secteurs': {},
            'industries': {},
            'marches': {},
            'all_stocks': {}
        }
        
        for _, row in df.iterrows():
            nom = row['name']
            ticker = row['ticker']
            region = row['region']
            pays = row['pays_fr']
            secteur = row['sector_fr']
            industrie = row['industry_fr']
            marche = row['indice']
            
            stock_info = {
                'ticker': ticker,
                'region': region,
                'pays': pays,
                'secteur': secteur,
                'industrie': industrie,
                'marche': marche
            }
            
            market_structure['all_stocks'][nom] = stock_info
            
            if region not in market_structure['regions']:
                market_structure['regions'][region] = {}
            if pays not in market_structure['regions'][region]:
                market_structure['regions'][region][pays] = {}
            market_structure['regions'][region][pays][nom] = stock_info
            
            if secteur not in market_structure['secteurs']:
                market_structure['secteurs'][secteur] = {}
            if industrie not in market_structure['secteurs'][secteur]:
                market_structure['secteurs'][secteur][industrie] = {}
            market_structure['secteurs'][secteur][industrie][nom] = stock_info
            
            if marche not in market_structure['marches']:
                market_structure['marches'][marche] = {}
            market_structure['marches'][marche][nom] = stock_info
        
        return market_structure
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {e}")
        return {}

@st.cache_data(ttl=3600)
def cached_get_isin_for_ticker(ticker: str) -> str:
    """Get ISIN for a ticker symbol (cached function)"""
    try:
        ticker_obj = cached_get_ticker_data(ticker)
        isin = ticker_obj.isin if hasattr(ticker_obj, 'isin') else 'N/A'
        return isin
    except:
        return 'N/A'

class DataService:
    """Service responsible for all data operations"""
    
    def __init__(self):
        self.market_structure = None
    
    @rate_limit(min_interval_seconds=1.0)
    def get_stock_history(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        """Retrieve historical stock data"""
        return cached_get_stock_history(ticker, period)
    
    @rate_limit(min_interval_seconds=1.0)
    def get_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Retrieve fundamental data for a stock"""
        return cached_get_fundamental_data(ticker)
    
    @rate_limit(min_interval_seconds=1.0)
    def get_historical_financials(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve historical financial statements"""
        return cached_get_historical_financials(ticker)
    
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
    
    @rate_limit(min_interval_seconds=1.0)
    def get_isin_for_ticker(self, ticker: str) -> str:
        """Get ISIN for a ticker symbol"""
        return cached_get_isin_for_ticker(ticker)