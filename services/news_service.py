# services/news_service.py
import streamlit as st
import pandas as pd
import yfinance as yf
import html
from datetime import datetime
from typing import Dict, List, Optional

from .data_service import DataService


class NewsService:
    """Service responsible for news-related operations"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
    
    @st.cache_data(ttl=3600)
    def get_stock_news(self, ticker: str) -> pd.DataFrame:
        """Retrieve and categorize stock news"""
        try:
            # Utiliser la méthode du data_service au lieu de créer directement un Ticker
            news_items = self.data_service.get_stock_news(ticker)
            
            if not news_items or len(news_items) == 0:
                return pd.DataFrame()
            
            news_data = []
            for item in news_items:
                news_dict = {
                    'id': item.get('id', ''),
                    'title': item.get('content', {}).get('title', ''),
                    'description': item.get('content', {}).get('summary', ''),
                    'link': item.get('content', {}).get('canonicalUrl', {}).get('url', ''),
                    'provider': item.get('content', {}).get('provider', {}).get('displayName', ''),
                    'pubDate': item.get('content', {}).get('pubDate', '') or 
                               item.get('content', {}).get('displayTime', '')
                }
                news_data.append(news_dict)
            
            news_df = pd.DataFrame(news_data)
            
            # Clean and format text
            if 'title' in news_df.columns:
                news_df['title'] = news_df['title'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
            
            if 'description' in news_df.columns:
                news_df['description'] = news_df['description'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
            
            # Convert timestamps
            if 'pubDate' in news_df.columns:
                news_df['published_at'] = pd.to_datetime(news_df['pubDate']).dt.tz_localize(None)
                news_df['date_formatted'] = news_df['published_at'].dt.strftime('%d/%m/%Y %H:%M')
            
            # Get financial data for classification
            # Utiliser get_ticker_object depuis data_service
            from .data_service import get_ticker_object
            ticker_obj = get_ticker_object(ticker)
            dividends = ticker_obj.dividends
            stock_splits = ticker_obj.splits
            
            # Classify news
            news_df = self._classify_news(news_df, ticker, dividends, stock_splits)
            
            # Sort by importance and date
            news_df = news_df.sort_values(by=['importance_score', 'published_at'], ascending=[False, False])
            
            # Ajouter le ticker à chaque actualité pour faciliter l'export
            news_df['ticker'] = ticker
            
            return news_df
        
        except Exception as e:
            st.error(f"Error retrieving news for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _classify_news(
        self, 
        news_df: pd.DataFrame, 
        ticker: str, 
        dividends: pd.Series, 
        stock_splits: pd.Series
    ) -> pd.DataFrame:
        """Classify and enrich news articles"""
        if news_df.empty:
            return news_df
        
        # Keyword categories
        fundamental_changes = [
            'acquisition', 'merger', 'takeover', 'bought', 'sell', 'selling',
            'CEO', 'chief executive', 'president', 'chairman', 'CFO', 'chief financial',
            'director', 'resigned', 'appointed', 'nomination', 'board'
        ]
        
        dividend_keywords = [
            'dividend', 'dividende', 'payout', 'yield', 'distribution',
            'shareholders', 'actionnaires', 'return', 'capital'
        ]
        
        buyback_keywords = [
            'buyback', 'buy back', 'repurchase', 'buy-back', 'rachat d\'actions',
            'rachat', 'programme de rachat', 'treasury stock'
        ]
        
        earnings_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'outlook', 'quarterly', 'annual', 'résultats', 'chiffre d\'affaires'
        ]
        
        # Initialize classification columns
        news_df['category'] = 'Autre'
        news_df['importance_score'] = 1
        news_df['is_fundamental_change'] = False
        news_df['is_dividend_related'] = False
        news_df['is_buyback_related'] = False
        news_df['is_earnings_related'] = False
        news_df['impact_on_valuation'] = 'Neutre'
        
        # Classify each news item
        for idx, row in news_df.iterrows():
            title = str(row.get('title', ''))
            description = str(row.get('description', ''))
            full_text = title + " " + description
            
            # Check categories
            is_fundamental = self._check_keywords(full_text, fundamental_changes)
            is_dividend = self._check_keywords(full_text, dividend_keywords)
            is_buyback = self._check_keywords(full_text, buyback_keywords)
            is_earnings = self._check_keywords(full_text, earnings_keywords)
            
            # Assign category and importance
            if is_fundamental:
                news_df.at[idx, 'category'] = 'Changement Fondamental'
                news_df.at[idx, 'importance_score'] = 5
                news_df.at[idx, 'is_fundamental_change'] = True
                news_df.at[idx, 'impact_on_valuation'] = 'Majeur'
            elif is_dividend:
                news_df.at[idx, 'category'] = 'Dividendes'
                news_df.at[idx, 'importance_score'] = 4
                news_df.at[idx, 'is_dividend_related'] = True
                news_df.at[idx, 'impact_on_valuation'] = 'Significatif'
            elif is_buyback:
                news_df.at[idx, 'category'] = 'Rachat d\'Actions'
                news_df.at[idx, 'importance_score'] = 4
                news_df.at[idx, 'is_buyback_related'] = True
                news_df.at[idx, 'impact_on_valuation'] = 'Significatif'
            elif is_earnings:
                news_df.at[idx, 'category'] = 'Résultats Financiers'
                news_df.at[idx, 'importance_score'] = 3
                news_df.at[idx, 'is_earnings_related'] = True
                news_df.at[idx, 'impact_on_valuation'] = 'Modéré'
            else:
                news_df.at[idx, 'category'] = 'Actualité Générale'
                news_df.at[idx, 'importance_score'] = 1
                news_df.at[idx, 'impact_on_valuation'] = 'Faible/Neutre'
            
            # Adjust score based on freshness
            if 'published_at' in news_df.columns:
                days_old = (datetime.now() - row['published_at']).days
                if days_old < 1:
                    news_df.at[idx, 'importance_score'] += 0.5
                elif days_old < 3:
                    news_df.at[idx, 'importance_score'] += 0.2
        
        return news_df
    
    def _check_keywords(self, text: str, keywords_list: List[str]) -> bool:
        """Check if text contains any of the keywords"""
        if not isinstance(text, str):
            return False
        text = text.lower()
        return any(keyword.lower() in text for keyword in keywords_list)
    
    def get_dividend_policy_analysis(self, ticker: str) -> Dict:
        """Analyze dividend policy"""
        try:
            # Utiliser get_ticker_object depuis data_service
            from .data_service import get_ticker_object
            ticker_obj = get_ticker_object(ticker)
            dividends = ticker_obj.dividends
            info = ticker_obj.info
            
            analysis = {
                'has_dividends': not dividends.empty,
                'dividend_trend': 'Stable',
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
                'dividend_growth': 0,
                'latest_dividend': dividends.iloc[-1] if not dividends.empty else 0,
                'dividend_history': dividends.to_dict() if not dividends.empty else {},
                'buyback_trend': 'Indéterminé',
                'total_shareholder_return': 0,
                'peer_comparison': 'Dans la moyenne',
                'sustainability': 'Soutenable',
                'dividend_health': 'Non évalué',
                'dividend_growth_score': 'Non évalué',
                'dividend_aristocrat': False,
                'consecutive_years_increase': 0,
                'dividend_frequency': 0,
                'dividend_consistency': 'Irrégulier'
            }
            
            # Analyze dividend trends
            if not dividends.empty and len(dividends) > 1:
                div_df = pd.DataFrame(dividends)
                div_df.index = pd.to_datetime(div_df.index)
                annual_div = div_df.resample('YE').sum()
                
                # Calculate frequency
                if len(div_df) > 0:
                    years_span = (div_df.index.max() - div_df.index.min()).days / 365.25
                    if years_span > 0:
                        analysis['dividend_frequency'] = round(len(div_df) / years_span, 1)
                
                if len(annual_div) > 1:
                    # Calculate CAGR
                    first_year = annual_div.iloc[0].values[0]
                    last_year = annual_div.iloc[-1].values[0]
                    years = len(annual_div)
                    
                    if first_year > 0 and years > 1:
                        cagr = ((last_year / first_year) ** (1 / (years - 1))) - 1
                        analysis['dividend_growth'] = cagr * 100
                    
                    # Determine trend
                    if cagr > 0.05:
                        analysis['dividend_trend'] = 'En hausse'
                    elif cagr < -0.05:
                        analysis['dividend_trend'] = 'En baisse'
                    
                    # Calculate consecutive years of increase
                    consecutive_years = 0
                    for i in range(len(annual_div)-1, 0, -1):
                        if annual_div.iloc[i].values[0] > annual_div.iloc[i-1].values[0]:
                            consecutive_years += 1
                        else:
                            break
                    
                    analysis['consecutive_years_increase'] = consecutive_years
                    
                    # Check if dividend aristocrat
                    if consecutive_years >= 25:
                        analysis['dividend_aristocrat'] = True
                    
                    # Evaluate consistency
                    if len(annual_div) >= 5:
                        zeros_or_cuts = 0
                        for i in range(1, len(annual_div)):
                            if annual_div.iloc[i].values[0] < annual_div.iloc[i-1].values[0]:
                                zeros_or_cuts += 1
                        
                        if zeros_or_cuts == 0:
                            analysis['dividend_consistency'] = 'Excellent'
                        elif zeros_or_cuts <= 1:
                            analysis['dividend_consistency'] = 'Bon'
                        elif zeros_or_cuts <= 2:
                            analysis['dividend_consistency'] = 'Moyen'
                        else:
                            analysis['dividend_consistency'] = 'Faible'
            
            # Analyze buyback trend and sustainability
            income_stmt, balance_sheet, cashflow = self.data_service.get_historical_financials(ticker)
            
            if 'CommonStock' in balance_sheet.index:
                common_stock = balance_sheet.loc['CommonStock']
                
                if len(common_stock) > 1:
                    oldest = common_stock.iloc[-1]
                    newest = common_stock.iloc[0]
                    
                    if oldest > newest:
                        analysis['buyback_trend'] = 'Actif'
                    elif oldest < newest:
                        analysis['buyback_trend'] = 'Dilution'
                    else:
                        analysis['buyback_trend'] = 'Stable'
            
            # Dividend sustainability analysis
            if analysis['has_dividends'] and 'NetIncome' in income_stmt.index:
                net_income = income_stmt.loc['NetIncome'].iloc[0]
                
                # Calculate FCF payout ratio if possible
                if 'OperatingCashFlow' in cashflow.index and 'CapitalExpenditures' in cashflow.index:
                    operating_cash = cashflow.loc['OperatingCashFlow'].iloc[0]
                    capex = cashflow.loc['CapitalExpenditures'].iloc[0]
                    free_cash_flow = operating_cash - capex
                    
                    if free_cash_flow is not None and free_cash_flow > 0:
                        recent_div_annual = analysis['latest_dividend'] * analysis['dividend_frequency']
                        shares_outstanding = info.get('sharesOutstanding', 0)
                        total_dividend_payout = recent_div_annual * shares_outstanding
                        fcf_payout_ratio = (total_dividend_payout / free_cash_flow) * 100
                        analysis['fcf_payout_ratio'] = fcf_payout_ratio
                
                # Evaluate dividend health
                if net_income > 0 and analysis['payout_ratio'] > 0:
                    if analysis['payout_ratio'] > 80:
                        analysis['sustainability'] = 'Risqué'
                        health_score = 'Faible'
                    elif analysis['payout_ratio'] > 60:
                        analysis['sustainability'] = 'Modéré'
                        health_score = 'Moyen'
                    else:
                        analysis['sustainability'] = 'Soutenable'
                        health_score = 'Bon'
                    
                    # Adjust based on FCF payout ratio
                    if 'fcf_payout_ratio' in analysis:
                        if analysis['fcf_payout_ratio'] > 90:
                            health_score = 'Faible'
                        elif analysis['fcf_payout_ratio'] > 70 and health_score != 'Faible':
                            health_score = 'Moyen'
                        elif analysis['fcf_payout_ratio'] < 50 and health_score == 'Bon':
                            health_score = 'Excellent'
                    
                    analysis['dividend_health'] = health_score
            
            # Calculate dividend growth score
            if analysis['dividend_growth'] >= 15:
                analysis['dividend_growth_score'] = 'Excellent'
            elif analysis['dividend_growth'] >= 8:
                analysis['dividend_growth_score'] = 'Bon'
            elif analysis['dividend_growth'] >= 3:
                analysis['dividend_growth_score'] = 'Moyen'
            else:
                analysis['dividend_growth_score'] = 'Faible'
            
            # Calculate overall dividend quality
            quality_score = 0
            if analysis['dividend_health'] == 'Excellent':
                quality_score += 3
            elif analysis['dividend_health'] == 'Bon':
                quality_score += 2
            elif analysis['dividend_health'] == 'Moyen':
                quality_score += 1
            
            if analysis['dividend_growth_score'] == 'Excellent':
                quality_score += 3
            elif analysis['dividend_growth_score'] == 'Bon':
                quality_score += 2
            elif analysis['dividend_growth_score'] == 'Moyen':
                quality_score += 1
            
            if analysis['dividend_consistency'] == 'Excellent':
                quality_score += 3
            elif analysis['dividend_consistency'] == 'Bon':
                quality_score += 2
            elif analysis['dividend_consistency'] == 'Moyen':
                quality_score += 1
            
            if quality_score >= 7:
                analysis['dividend_quality'] = 'Excellent'
            elif quality_score >= 5:
                analysis['dividend_quality'] = 'Bon'
            elif quality_score >= 3:
                analysis['dividend_quality'] = 'Moyen'
            else:
                analysis['dividend_quality'] = 'Faible'
            
            return analysis
            
        except Exception as e:
            return {
                'has_dividends': False,
                'error': str(e)
            }
    
    def get_news_sentiment_analysis(self, ticker: str) -> Dict:
        """Analyze sentiment from recent news"""
        try:
            news_df = self.get_stock_news(ticker)
            
            if news_df.empty:
                return {
                    'overall_sentiment': 'Neutre',
                    'sentiment_score': 0,
                    'positive_news': 0,
                    'negative_news': 0,
                    'major_events': []
                }
            
            # Simple sentiment analysis based on keywords
            positive_keywords = ['growth', 'profit', 'increase', 'rise', 'gain', 
                               'positive', 'strong', 'beat', 'exceed', 'surge',
                               'croissance', 'hausse', 'augmentation', 'bénéfice']
            
            negative_keywords = ['decline', 'loss', 'decrease', 'fall', 'drop',
                               'negative', 'weak', 'miss', 'below', 'plunge',
                               'baisse', 'chute', 'perte', 'déclin', 'recul']
            
            sentiment_data = {
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'major_events': []
            }
            
            for idx, row in news_df.iterrows():
                title = row.get('title', '').lower()
                description = row.get('description', '').lower()
                full_text = title + ' ' + description
                
                pos_score = sum(1 for word in positive_keywords if word in full_text)
                neg_score = sum(1 for word in negative_keywords if word in full_text)
                
                if pos_score > neg_score:
                    sentiment_data['positive_count'] += 1
                elif neg_score > pos_score:
                    sentiment_data['negative_count'] += 1
                else:
                    sentiment_data['neutral_count'] += 1
                
                # Identify major events
                if row.get('importance_score', 0) >= 4:
                    sentiment_data['major_events'].append({
                        'title': row.get('title', ''),
                        'category': row.get('category', ''),
                        'impact': row.get('impact_on_valuation', '')
                    })
            
            # Calculate overall sentiment
            total_news = len(news_df)
            sentiment_score = ((sentiment_data['positive_count'] - sentiment_data['negative_count']) / 
                               total_news * 100) if total_news > 0 else 0
            
            if sentiment_score > 20:
                overall_sentiment = 'Très Positif'
            elif sentiment_score > 10:
                overall_sentiment = 'Positif'
            elif sentiment_score < -20:
                overall_sentiment = 'Très Négatif'
            elif sentiment_score < -10:
                overall_sentiment = 'Négatif'
            else:
                overall_sentiment = 'Neutre'
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': sentiment_score,
                'positive_news': sentiment_data['positive_count'],
                'negative_news': sentiment_data['negative_count'],
                'neutral_news': sentiment_data['neutral_count'],
                'major_events': sentiment_data['major_events'][:5]  # Top 5 major events
            }
            
        except Exception as e:
            return {
                'overall_sentiment': 'Erreur',
                'sentiment_score': 0,
                'error': str(e)
            }
    
    def get_news_categories_summary(self, ticker: str) -> Dict:
        """Get a summary of news categories"""
        try:
            news_df = self.get_stock_news(ticker)
            
            if news_df.empty:
                return {}
            
            # Count by category
            category_counts = news_df['category'].value_counts().to_dict()
            
            # Count by impact
            impact_counts = news_df['impact_on_valuation'].value_counts().to_dict()
            
            # Recent major news
            major_news = news_df[news_df['importance_score'] >= 4].head(3).to_dict('records')
            
            return {
                'categories': category_counts,
                'impacts': impact_counts,
                'major_news': major_news,
                'total_news': len(news_df),
                'last_update': news_df['published_at'].max() if not news_df.empty else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def format_news_for_export(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Format news dataframe for export"""
        if news_df.empty:
            return news_df
        
        # Select and order columns for export
        export_columns = [
            'date_formatted', 'title', 'category', 'impact_on_valuation', 
            'importance_score', 'description', 'provider', 'link'
        ]
        
        # Ensure all columns exist
        existing_columns = [col for col in export_columns if col in news_df.columns]
        
        export_df = news_df[existing_columns].copy()
        
        # Rename columns for better readability
        column_mapping = {
            'date_formatted': 'Date',
            'title': 'Titre',
            'category': 'Catégorie',
            'impact_on_valuation': 'Impact',
            'importance_score': 'Score d\'importance',
            'description': 'Description',
            'provider': 'Source',
            'link': 'Lien'
        }
        
        export_df = export_df.rename(columns=column_mapping)
        
        return export_df