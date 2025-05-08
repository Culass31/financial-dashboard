# services/data_service.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional, List
import requests
import html

@st.cache_data(ttl=3600)
def get_market_structure():

    try:
        df = pd.read_csv(
            r"https://raw.githubusercontent.com/Culass31/financial-dashboard/refs/heads/main/models/actions.csv", 
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

class DataService:
    """Service responsible for all data operations"""
    
    def __init__(self):
        self.market_structure = None
    
    @st.cache_data(ttl=3600)
    def get_stock_history(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        """Retrieve historical stock data"""
        try:
            stock_data = yf.Ticker(ticker)
            df = stock_data.history(period=period)
            
            if df.empty:
                st.warning(f"No data available for {ticker}")
                return pd.DataFrame()
            
            df['date_str'] = df.index.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            st.error(f"Error retrieving history for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Retrieve fundamental data for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return None
            
            income_stmt, balance_sheet, cashflow = self.get_historical_financials(ticker)
            
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
                "Actions ordinaires": balance_sheet.loc['Ordinary Shares Number', income_stmt.columns[0]] if 'Ordinary Shares Number' in balance_sheet.index and not balance_sheet.empty else None,
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
                "Ratio dette/capitaux propres": info.get('debtToEquity', None),
                "Quick Ratio": info.get('quickRatio', None),
                "Current Ratio": info.get('currentRatio', None),
                "EV/EBITDA": info.get('enterpriseToEbitda', None),
                "EV/Revenue": info.get('enterpriseToRevenue', None),
            }
            
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
            st.error(f"Error retrieving fundamental data for {ticker}: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def get_historical_financials(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve historical financial statements"""
        try:
            stock = yf.Ticker(ticker)
            
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
            st.error(f"Error retrieving financial data for {ticker}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
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
    
    @st.cache_data(ttl=3600)
    def get_isin_for_ticker(self, ticker: str) -> str:
        """Get ISIN for a ticker symbol"""
        try:
            stock = yf.Ticker(ticker)
            isin = stock.isin if hasattr(stock, 'isin') else 'N/A'
            return isin
        except:
            return 'N/A'
