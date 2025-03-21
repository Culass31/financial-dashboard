import streamlit as st
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import requests
from datetime import datetime
import time
import html
import math
import pytz

pd.set_option('future.no_silent_downcasting', True) #Configuration globale pour adopter le comportement futur de pandas

# Configuration de l'application Streamlit
st.set_page_config(page_title="Analyse Marchés Mondiaux", page_icon="📈", layout="wide")

# Fonction pour formater les grands nombres
def format_number(num):
	if num is None:
		return 'N/A'
	if isinstance(num, str):
		return num
	
	abs_num = abs(num)
	if abs_num >= 1_000_000_000:
					return f"{num/1_000_000_000:.2f}B"
	elif abs_num >= 1_000_000:
					return f"{num/1_000_000:.2f}M"
	elif abs_num >= 1_000:
					return f"{num/1_000:.2f}K"
	else:
					return f"{num:.2f}"
    
def display_price_with_trend(label, current_price, previous_price, format_func=None, background=True, size="medium"):
    """
    Affiche un prix avec son indicateur de tendance (flèche et pourcentage) avec options étendues
    
    :param label: Libellé à afficher (ex: "Prix actuel")
    :param current_price: Prix actuel
    :param previous_price: Prix de référence pour calculer la tendance
    :param format_func: Fonction optionnelle pour formater le prix (par défaut None)
    :param background: Ajouter un fond de couleur pour plus de visibilité (par défaut True)
    :param size: Taille de l'affichage ("small", "medium", "large")
    :return: None, affiche directement avec st.markdown
    """
    if current_price is not None and previous_price is not None:
        # Calculer le pourcentage de variation
        change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # Déterminer la flèche et la couleur en fonction de la tendance
        if change_pct > 0:
            arrow = "↑"
            color = "green"
            bg_color = "rgba(0, 128, 0, 0.1)"
        elif change_pct < 0:
            arrow = "↓"
            color = "red"
            bg_color = "rgba(255, 0, 0, 0.1)"
        else:
            arrow = "→"
            color = "gray"
            bg_color = "rgba(128, 128, 128, 0.1)"
            
        # Définir la taille de la police en fonction du paramètre
        font_sizes = {
            "small": {"label": "0.9rem", "value": "1.1rem", "change": "0.9rem", "previous": "0.8rem"},
            "medium": {"label": "1rem", "value": "1.4rem", "change": "1rem", "previous": "0.9rem"},
            "large": {"label": "1.2rem", "value": "1.8rem", "change": "1.2rem", "previous": "1rem"}
        }
        
        sizes = font_sizes.get(size, font_sizes["medium"])
        
        # Formater la valeur si une fonction de formatage est fournie
        formatted_price = format_func(current_price) if format_func else f"{current_price:.2f}"
        formatted_previous = format_func(previous_price) if format_func else f"{previous_price:.2f}"
        
        # Créer le style pour l'affichage
        if background:
            style = f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {bg_color}; margin-bottom: 10px;'>
                <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: {sizes["change"]}; color: {color};'>
                        {arrow} {abs(change_pct):.2f}%
                    </div>
                    <div style='font-size: {sizes["previous"]}; margin-left: 10px; color: #666;'>
                        Préc: {formatted_previous}
                    </div>
                </div>
            </div>
            """
        else:
            style = f"""
            <div style='margin-bottom: 10px;'>
                <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: {sizes["change"]}; color: {color};'>
                        {arrow} {abs(change_pct):.2f}%
                    </div>
                    <div style='font-size: {sizes["previous"]}; margin-left: 10px; color: #666;'>
                        Préc: {formatted_previous}
                    </div>
                </div>
            </div>
            """
        
        # Afficher avec le style approprié
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.markdown(f"**{label}:** N/A")

# Fonction pour récupérer l'ISIN d'une action à partir de son ticker
def get_isin_for_ticker(ticker):
    """
    Récupère l'ISIN d'une action à partir de son ticker.
    
    :param ticker: Symbole de l'action
    :return: ISIN ou 'N/A' si non disponible
    """
    try:
        stock = yf.Ticker(ticker)
        isin = stock.isin if hasattr(stock, 'isin') else 'N/A'
        return isin
    except:
        return 'N/A'

# Fonction pour créer une jauge d'indicateur
def create_gauge(value, title, min_val, max_val, format_str="{:.2f}", threshold_values=None, threshold_labels=None):
    if value is None:
        value_str = "N/A"
        normalized_value = 0.5
        color = "gray"
    else:
        value = round(value, 2)
        value_str = format_str.format(value)
        
        # Éviter division par zéro
        if max_val == min_val:
            normalized_value = 0.5
        else:
            normalized_value = (value - min_val) / (max_val - min_val)
            
        normalized_value = max(0, min(1, normalized_value))
       
        # Déterminer la couleur en fonction de la valeur normalisée
        if normalized_value < 0.3:
            color = "red"
        elif normalized_value < 0.7:
            color = "orange"
        else:
            color = "green"
   
    # Configuration de base de la jauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value if value is not None else 50,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}, 'align': 'center'},
        gauge={
            'axis': {
                'range': [min_val, max_val], 
                'tickwidth': 1, 
                'tickmode': 'linear',
                'tick0': min_val,
                'dtick': (max_val - min_val) / 5
            },
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, min_val + (max_val - min_val) * 0.3], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [min_val + (max_val - min_val) * 0.3, min_val + (max_val - min_val) * 0.7], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [min_val + (max_val - min_val) * 0.7, max_val], 'color': 'rgba(0, 128, 0, 0.2)'}
            ],
            'shape': "angular"
        },
        number={
            'suffix': "%" if "%" in format_str else "",
            'font': {'size': 16},
            'valueformat': format_str.replace("{:", "").replace("}", "")
        }
    ))
   
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        showlegend=False,
        autosize=True,
        xaxis={'autorange': True, 'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'autorange': True, 'showgrid': False, 'zeroline': False, 'showticklabels': False}
    )
   
    return fig

@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def get_stock_news(ticker):
    """
    Récupère et traite les actualités liées à une action en les classant par importance
    et impact potentiel sur la valeur intrinsèque.
    
    :param ticker: Symbole de l'action
    :return: DataFrame avec les actualités enrichies et classées
    """
    try:
        # Récupération des actualités directement depuis l'objet Ticker
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news
        
        if not news_items or len(news_items) == 0:
            return pd.DataFrame()
        
        # Préparation des données pour le DataFrame
        news_data = []
        for item in news_items:
            # Extraire les informations pertinentes
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
        
        # Convertir en DataFrame
        news_df = pd.DataFrame(news_data)
        
        # Nettoyer et formater le texte
        if 'title' in news_df.columns:
            news_df['title'] = news_df['title'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
        
        if 'description' in news_df.columns:
            news_df['description'] = news_df['description'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
        
        # Convertir les timestamps ISO 8601 en datetime
        if 'pubDate' in news_df.columns:
            news_df['published_at'] = pd.to_datetime(news_df['pubDate']).dt.tz_localize(None)
            # Formater pour affichage
            news_df['date_formatted'] = news_df['published_at'].dt.strftime('%d/%m/%Y %H:%M')
        
        # Récupérer les dividendes et rachats d'actions
        dividends = ticker_obj.dividends
        stock_splits = ticker_obj.splits
        
        # Analyser et classifier les actualités
        news_df = classify_news(news_df, ticker, dividends, stock_splits)
        
        # Trier par importance et date
        news_df = news_df.sort_values(by=['importance_score', 'published_at'], ascending=[False, False])
        
        return news_df
    
    except Exception as e:
        st.error(f"Erreur lors de la récupération des actualités pour {ticker}: {str(e)}")
        return pd.DataFrame()

def classify_news(news_df, ticker, dividends, stock_splits):
    """
    Classifie et enrichit les actualités en fonction de leur importance
    et leur impact potentiel sur la valeur intrinsèque.
    
    :param news_df: DataFrame contenant les actualités
    :param ticker: Symbole de l'action
    :param dividends: Série de dividendes historiques
    :param stock_splits: Série de fractionnements d'actions
    :return: DataFrame enrichi
    """
    if news_df.empty:
        return news_df
    
    # Mots-clés pour différentes catégories d'importance
    fundamental_changes = [
        'acquisition', 'merger', 'takeover', 'bought', 'sell', 'selling', 'acquisition',
        'CEO', 'chief executive', 'president', 'chairman', 'CFO', 'chief financial',
        'director', 'resigned', 'appointed', 'nomination', 'board', 'rachat', 'acquisition',
        'fusion', 'cession', 'vente', 'achat', 'directeur', 'président', 'nomination'
    ]
    
    dividend_keywords = [
        'dividend', 'dividende', 'payout', 'yield', 'distribution', 'versement', 
        'shareholders', 'actionnaires', 'return', 'capital', 'retour'
    ]
    
    buyback_keywords = [
        'buyback', 'buy back', 'repurchase', 'buy-back', 'rachat d\'actions', 
        'rachat', 'programme de rachat', 'treasury stock', 'repurchasing'
    ]
    
    earnings_keywords = [
        'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
        'outlook', 'quarterly', 'annual', 'résultats', 'chiffre d\'affaires',
        'bénéfice', 'perte', 'prévisions', 'perspectives', 'trimestriel', 'annuel'
    ]
    
    # Initialiser les colonnes de classification
    news_df['category'] = 'Autre'
    news_df['importance_score'] = 1  # 1-5, 5 étant le plus important
    news_df['is_fundamental_change'] = False
    news_df['is_dividend_related'] = False
    news_df['is_buyback_related'] = False
    news_df['is_earnings_related'] = False
    news_df['impact_on_valuation'] = 'Neutre'
    
    # Fonction pour vérifier la présence de mots-clés
    def check_keywords(text, keywords_list):
        if not isinstance(text, str):
            return False
        text = text.lower()
        return any(keyword.lower() in text for keyword in keywords_list)
    
    # Classifier chaque actualité
    for idx, row in news_df.iterrows():
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        full_text = title + " " + description
        
        # Vérifier les différentes catégories
        is_fundamental = check_keywords(full_text, fundamental_changes)
        is_dividend = check_keywords(full_text, dividend_keywords)
        is_buyback = check_keywords(full_text, buyback_keywords)
        is_earnings = check_keywords(full_text, earnings_keywords)
        
        # Attribuer une catégorie et un score d'importance
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
            # Actualité générale
            news_df.at[idx, 'category'] = 'Actualité Générale'
            news_df.at[idx, 'importance_score'] = 1
            news_df.at[idx, 'impact_on_valuation'] = 'Faible/Neutre'
        
        # Ajuster le score en fonction de la fraîcheur de l'actualité (bonus pour les actualités récentes)
        if 'published_at' in news_df.columns:
            days_old = (datetime.now() - row['published_at']).days
            if days_old < 1:  # Moins d'un jour
                news_df.at[idx, 'importance_score'] += 0.5
            elif days_old < 3:  # Moins de trois jours
                news_df.at[idx, 'importance_score'] += 0.2            
    
    return news_df

def get_dividend_policy_analysis(ticker):
    """
    Analyse la politique de dividendes et de rachat d'actions d'une entreprise
    
    :param ticker: Symbole de l'action
    :return: Dict avec l'analyse des dividendes et rachats d'actions, incluant les métriques de santé et croissance
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Récupérer les données de dividendes
        dividends = ticker_obj.dividends
        
        # Récupérer les informations de base
        info = ticker_obj.info
        
        # Analyser les dividendes
        analysis = {
            'has_dividends': not dividends.empty,
            'dividend_trend': 'Stable',
            'dividend_yield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
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
        
        # Calculer la tendance et la croissance si des dividendes existent
        if not dividends.empty and len(dividends) > 1:
            # Convertir en DataFrame annuel
            div_df = pd.DataFrame(dividends)
            div_df.index = pd.to_datetime(div_df.index)
            annual_div = div_df.resample('YE').sum()
            
            # Déterminer la fréquence des dividendes
            if len(div_df) > 0:
                # Calculer le nombre moyen de dividendes par an
                years_span = (div_df.index.max() - div_df.index.min()).days / 365.25
                if years_span > 0:
                    analysis['dividend_frequency'] = round(len(div_df) / years_span, 1)
            
            if len(annual_div) > 1:
                # Calculer la croissance annuelle moyenne
                first_year = annual_div.iloc[0].values[0]
                last_year = annual_div.iloc[-1].values[0]
                years = len(annual_div)
                
                if first_year > 0 and years > 1:
                    cagr = ((last_year / first_year) ** (1 / (years - 1))) - 1
                    analysis['dividend_growth'] = cagr * 100
                
                # Déterminer la tendance
                if cagr > 0.05:  # Croissance > 5%
                    analysis['dividend_trend'] = 'En hausse'
                elif cagr < -0.05:  # Décroissance > 5%
                    analysis['dividend_trend'] = 'En baisse'
                    
                # Calculer le nombre d'années consécutives d'augmentation
                consecutive_years = 0
                for i in range(len(annual_div)-1, 0, -1):
                    if annual_div.iloc[i].values[0] > annual_div.iloc[i-1].values[0]:
                        consecutive_years += 1
                    else:
                        break
                
                analysis['consecutive_years_increase'] = consecutive_years
                
                # Déterminer si c'est un aristocrate du dividende (25+ ans d'augmentation)
                if consecutive_years >= 25:
                    analysis['dividend_aristocrat'] = True
                
                # Évaluer la consistance des dividendes
                if len(annual_div) >= 5:  # Au moins 5 ans d'historique
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
        
        # Analyse des rachats d'actions (approximation via le nombre d'actions)
        try:
            income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
            
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
            
            # Analyse de durabilité du dividende
            if analysis['has_dividends'] and 'NetIncome' in income_stmt.index:
                net_income = income_stmt.loc['NetIncome'].iloc[0]
                free_cash_flow = None
                
                # Calculer le Free Cash Flow si possible
                if 'OperatingCashFlow' in cashflow.index and 'CapitalExpenditures' in cashflow.index:
                    operating_cash = cashflow.loc['OperatingCashFlow'].iloc[0]
                    capex = cashflow.loc['CapitalExpenditures'].iloc[0]
                    free_cash_flow = operating_cash - capex
                
                # Calcul du dividende annuel total
                if not dividends.empty:
                    recent_div_annual = analysis['latest_dividend'] * analysis['dividend_frequency']
                    
                    # Calculer le dividend coverage ratio avec FCF si disponible
                    if free_cash_flow is not None and free_cash_flow > 0:
                        fcf_payout_ratio = 0
                        try:
                            shares_outstanding = info.get('sharesOutstanding', 0)
                            total_dividend_payout = recent_div_annual * shares_outstanding
                            fcf_payout_ratio = (total_dividend_payout / free_cash_flow) * 100
                            analysis['fcf_payout_ratio'] = fcf_payout_ratio
                        except:
                            pass
                
                # Évaluation de la santé du dividende
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
                    
                    # Ajuster le score de santé en fonction du FCF si disponible
                    if 'fcf_payout_ratio' in analysis:
                        if analysis['fcf_payout_ratio'] > 90:
                            health_score = 'Faible'
                        elif analysis['fcf_payout_ratio'] > 70 and health_score != 'Faible':
                            health_score = 'Moyen'
                        elif analysis['fcf_payout_ratio'] < 50 and health_score == 'Bon':
                            health_score = 'Excellent'
                    
                    analysis['dividend_health'] = health_score
                
                # Évaluation du score de croissance
                growth_score = 'Faible'
                if analysis['dividend_growth'] > 10:
                    growth_score = 'Excellent'
                elif analysis['dividend_growth'] > 7:
                    growth_score = 'Bon'
                elif analysis['dividend_growth'] > 3:
                    growth_score = 'Moyen'
                
                # Ajuster le score de croissance en fonction de la consistance
                if analysis['dividend_consistency'] == 'Excellent' and growth_score != 'Faible':
                    if growth_score == 'Moyen':
                        growth_score = 'Bon'
                    elif growth_score == 'Bon':
                        growth_score = 'Excellent'
                
                analysis['dividend_growth_score'] = growth_score
                
                # Calculer le rendement total pour l'actionnaire (approximation)
                if analysis['dividend_yield'] > 0 and analysis['dividend_growth'] > 0:
                    analysis['total_shareholder_return'] = analysis['dividend_yield'] + analysis['dividend_growth']
                    
                # Ajouter une évaluation globale de la qualité du dividende
                dividend_quality_factors = {
                    'Excellent': 4,
                    'Bon': 3,
                    'Moyen': 2,
                    'Faible': 1,
                    'Non évalué': 0
                }
                
                health_score_value = dividend_quality_factors.get(analysis['dividend_health'], 0)
                growth_score_value = dividend_quality_factors.get(analysis['dividend_growth_score'], 0)
                consistency_score_value = dividend_quality_factors.get(analysis['dividend_consistency'], 0)
                
                avg_score = (health_score_value + growth_score_value + consistency_score_value) / 3
                
                if avg_score >= 3.5:
                    analysis['dividend_quality'] = 'Excellent'
                elif avg_score >= 2.5:
                    analysis['dividend_quality'] = 'Bon'
                elif avg_score >= 1.5:
                    analysis['dividend_quality'] = 'Moyen'
                else:
                    analysis['dividend_quality'] = 'Faible'
                
        except Exception as e:
            analysis['financial_analysis_error'] = str(e)
            
        return analysis
        
    except Exception as e:
        print(f"Erreur lors de l'analyse des dividendes: {e}")
        return {
            'has_dividends': False,
            'error': str(e)
        }

# Structure de marché hiérarchique organisée par région, pays, secteur, industrie et marché
@st.cache_data(ttl=86400)  # Mise en cache pour 24 heures
def get_market_structure():
    # Lire le fichier CSV avec l'encodage approprié
    try:
        # Essayer différents encodages courants
        encodings = ['utf-8-sig','latin1', 'ISO-8859-1', 'windows-1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(r"https://raw.githubusercontent.com/Culass31/financial-dashboard/refs/heads/main/actions.csv", sep=";", encoding='utf-8-sig')
                print(f"Fichier lu avec succès avec l'encodage {encoding}:")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("Impossible de déterminer l'encodage correct du fichier")
        
        print(f"Nombre d'entrées: {len(df)}")
        print(f"Colonnes: {df.columns.tolist()}")
        print("\nAperçu des données:")
        print(df.head())
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {e}")
        return {}
    
    # Créer une structure hiérarchique multi-niveaux
    market_structure = {
        'regions': {},
        'secteurs': {},
        'industries': {},
        'marches': {},
        'all_stocks': {}
    }
    
    # Parcourir chaque ligne du DataFrame
    for _, row in df.iterrows():
        # Extraire les données pertinentes
        nom = row['name']
        ticker = row['ticker']
        region = row['region']
        pays = row['pays_fr']
        secteur = row['sector_fr']
        industrie = row['industry_fr']
        marche = row['indice']
        
        # Données de l'action
        stock_info = {
            'ticker': ticker,
            'region': region,
            'pays': pays,
            'secteur': secteur,
            'industrie': industrie,
            'marche': marche
        }
							
        # Ajouter l'action à la liste complète
        market_structure['all_stocks'][nom] = stock_info
        
        # Structure par région et pays
        if region not in market_structure['regions']:
            market_structure['regions'][region] = {}
        
        if pays not in market_structure['regions'][region]:
            market_structure['regions'][region][pays] = {}
        
        market_structure['regions'][region][pays][nom] = stock_info
        
        # Structure par secteur et industrie
        if secteur not in market_structure['secteurs']:
            market_structure['secteurs'][secteur] = {}
        
        if industrie not in market_structure['secteurs'][secteur]:
            market_structure['secteurs'][secteur][industrie] = {}
        
        market_structure['secteurs'][secteur][industrie][nom] = stock_info
        
        # Structure par marché
        if marche not in market_structure['marches']:
            market_structure['marches'][marche] = {}
        
        market_structure['marches'][marche][nom] = stock_info
    
    return market_structure

    try:
        ticker_data = yf.Ticker(ticker_symbol)
        isin = ticker_data.isin
        return isin if isin else "Non disponible"
    except Exception as e:
        print(f"Erreur lors de la récupération de l'ISIN pour {ticker_symbol}: {e}")
        return "Non disponible"

@st.cache_data(ttl=3600)
def get_historical_financials(ticker, period='max'):
    """
    Récupère et prépare les données financières historiques pour une action.
    
    :param ticker: Symbol de l'action
    :return: Tuple de DataFrames (income_statement, balance_sheet, cash_flow)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Utiliser les attributs corrects de l'objet Ticker
        # Ces attributs renvoient directement des DataFrames pandas
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Vérification des données
        if income_stmt is None or income_stmt.empty:
            income_stmt = pd.DataFrame()
            
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = pd.DataFrame()
            
        if cashflow is None or cashflow.empty:
            cashflow = pd.DataFrame()
        
        return income_stmt, balance_sheet, cashflow
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données financières pour {ticker}: {str(e)}")
        # Renvoyer des DataFrames vides en cas d'erreur
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_history(ticker, period='5y'):
    """
    Récupère l'historique des cours d'une action.
    
    :param ticker: Symbol de l'action
    :param period: Période d'historique à récupérer
    :return: DataFrame avec l'historique des cours
    """
    try:
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period=period)
        
        if df.empty:
            st.warning(f"Pas de données disponibles pour {ticker}")
            return pd.DataFrame()
        
        # Ajouter une colonne pour la date au format string (pour l'affichage)
        df['date_str'] = df.index.strftime('%Y-%m-%d')
        
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique pour {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_regression(df):
    """
    Calcule la régression linéaire sur les prix et divers indicateurs techniques.
    
    :param df: DataFrame avec l'historique des cours
    :return: DataFrame avec régression, dict d'indicateurs
    """
    if df.empty:
        return df, {}
    
    # Copie du DataFrame
    df_reg = df.copy()
    
    # Créer un index numérique pour la régression
    df_reg['index'] = range(len(df_reg))
    
    # Ajuster un modèle de régression linéaire
    X = df_reg['index'].values.reshape(-1, 1)
    y = df_reg['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculer les valeurs prédites
    df_reg['Regression'] = model.predict(X)
    
    # Calculer l'écart par rapport à la régression
    df_reg['Deviation'] = df_reg['Close'] - df_reg['Regression']
    
    # Calculer l'écart type
    std_dev = df_reg['Deviation'].std()
    
    # Calculer le nombre d'écarts types
    if std_dev > 0:
        df_reg['DeviationScore'] = df_reg['Deviation'] / std_dev
    else:
        df_reg['DeviationScore'] = 0
    
    # Calculer la progression sur différentes périodes
    current_price = df_reg['Close'].iloc[-1]
    
    # Progression sur 1 an (252 jours de trading)
    days_1y = min(252, len(df_reg) - 1)
    price_1y_ago = df_reg['Close'].iloc[-days_1y-1] if days_1y > 0 else current_price
    prog_1y = ((current_price / price_1y_ago) - 1) * 100 if price_1y_ago > 0 else 0
    
    # Progression sur 5 ans (1260 jours de trading)
    days_5y = min(1260, len(df_reg) - 1)
    price_5y_ago = df_reg['Close'].iloc[-days_5y-1] if days_5y > 0 else current_price
    prog_5y = ((current_price / price_5y_ago) - 1) * 100 if price_5y_ago > 0 else 0
    
    # Calculer la croissance du modèle (annualisée)
    if len(df_reg) > 252:  # Au moins un an de données
        initial_regression = df_reg['Regression'].iloc[0]
        final_regression = df_reg['Regression'].iloc[-1]
        years = len(df_reg) / 252  # Approximation du nombre d'années
        
        if initial_regression > 0:
            model_growth = (((final_regression / initial_regression) ** (1/years)) - 1) * 100
        else:
            model_growth = 0
    else:
        model_growth = 0
    
    # Calculer la corrélation
    correlation = np.corrcoef(df_reg['index'], df_reg['Close'])[0, 1]
    
    # Écart actuel par rapport à la regression
    current_deviation = df_reg['DeviationScore'].iloc[-1]
    
    # Résultats
    indicators = {
        'model_growth': model_growth,
        'correlation': correlation,
        'deviation': current_deviation,
        'prog_1y': prog_1y,
        'prog_5y': prog_5y
    }
    
    return df_reg, indicators

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker):
    """
    Récupère les données fondamentales d'une action.
    
    :param ticker: Symbol de l'action
    :return: Dictionnaire avec les données fondamentales
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return None
        
        # Récupérer les données financières
        income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
        
        #Données générales
        general_data = {
                "Nom": info.get('longName', None),
                "Secteur": info.get('sector', None),
                "Industrie": info.get('industry', None),
                'Pays': info.get('country', ''),
                "Site web": info.get('website', None),
                "Description": info.get('longBusinessSummary', None)
            }

        # Données de marché
        market_data = {
            "Prix actuel": info.get('currentPrice', info.get('regularMarketPrice', None)),
            "Prix d'ouverture": info.get('open', None),
            "Précédente clôture": info.get('previousClose', None),
            "Prix haut (jour)": info.get('dayHigh', None),
            "Prix bas (jour)": info.get('dayLow', None),
            "Prix cible moyen": info.get('targetMeanPrice', None),
            "Prix cible haut": info.get('targetHighPrice', None),
            "Prix cible bas": info.get('targetLowPrice', None),
            "Recommandation": info.get('recommendationKey', None).capitalize(),
            "52 semaines - Haut": info.get('fiftyTwoWeekHigh', None),
            "52 semaines - Bas": info.get('fiftyTwoWeekLow', None),
            "Rendement du dividende": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else  None,
            "Ex-Date dividende": info.get('exDividendDate', None),
            "Actions en circulation": info.get('sharesOutstanding', None),
            "Actions ordinaires": balance_sheet.loc['Ordinary Shares Number', income_stmt.columns[0]] if 'Ordinary Shares Number' in balance_sheet.index else None,
            "Volume": info.get('volume', None),
            "Volume moyen": info.get('averageVolume', None),
            "Volume moyen (10j)": info.get('averageVolume10days', None),
            "Capitalisation boursière": info.get('marketCap', None),
            "Beta": info.get('beta', None),
        }
        
        # Données fondamentales
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
        
        # Données financières
        financial_data = {}
        
        # Ajouter les dernières données financières si disponibles
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
        st.error(f"Erreur lors de la récupération des données fondamentales pour {ticker}: {str(e)}")
        return None

def calculate_intrinsic_value(ticker, fundamental_data, historical_financials=None, market_structure=None):
    """
    Calcule la valeur intrinsèque d'une action en utilisant différentes méthodes,
    incluant une analyse DCF avancée, Graham, PER et patrimoniale.
    
    Args:
        ticker: Symbole boursier de l'entreprise
        fundamental_data: Dictionnaire contenant les données fondamentales
        historical_financials: Tuple optionnel (income_stmt, balance_sheet, cashflow)
        market_structure: Structure de marché optionnelle pour une meilleure identification sectorielle
        
    Returns:
        dict: Résultat de l'analyse avec valeur intrinsèque et recommandation
    """
    try:
        # Récupération des données fondamentales si non fournies
        if fundamental_data is None:
            fundamental_data = get_fundamental_data(ticker)
            
        if historical_financials is None:
            income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
        else:
            income_stmt, balance_sheet, cashflow = historical_financials
        
        # Vérification des données nécessaires
        if not fundamental_data or 'Données de marché' not in fundamental_data:
            return {
                'success': False,
                'message': f"Données fondamentales insuffisantes pour {ticker}"
            }
        
        # Récupération du prix actuel
        current_price = fundamental_data['Données de marché'].get('Prix actuel', None)
        
        if not current_price:
            return {
                'success': False,
                'message': f"Prix actuel non disponible pour {ticker}"
            }
        
        sector = fundamental_data['Données générales'].get('Secteur', None)
        
        # Méthode 1: Modèle DCF
        dcf_value = calculate_dcf_value(
            ticker, 
            income_stmt, 
            cashflow, 
            balance_sheet, 
            fundamental_data, 
            current_price, 
            market_structure
        )
        
        # Méthode 2: Modèle de Graham
        graham_value = None
        eps = fundamental_data['Données fondamentales'].get('BPA', None)
        book_value = fundamental_data['Données fondamentales'].get('Valeur comptable par action', None)
        
        if eps is not None and eps > 0:
            # Taux d'obligation à long terme (approximation)
            bond_yield = 0.04  # 4% pour la France
            
            # Estimation de la croissance
            growth_estimate = 0.05  # 5% par défaut
            
            if not income_stmt.empty and len(income_stmt.columns) >= 3 and 'Net Income' in income_stmt.index:
                # Calculer le taux de croissance du résultat net sur 3 ans
                net_incomes = []
                for col in income_stmt.columns[:3]:
                    if 'Net Income' in income_stmt.index:
                        net_incomes.append(income_stmt.loc['Net Income', col])
                
                if len(net_incomes) == 3 and all(ni is not None and ni > 0 for ni in net_incomes):
                    # Croissance annualisée du résultat net
                    cagr = ((net_incomes[0] / net_incomes[2]) ** (1/2)) - 1
                    growth_estimate = min(max(cagr, 0.02), 0.15)  # Limiter entre 2% et 15%
            
            # Utiliser la formule de Graham: V = EPS * (8.5 + 2g) * 4.4 / Y
            graham_value = eps * (8.5 + 2 * growth_estimate * 100) * 4.4 / (bond_yield * 100)
            
            # Calculer également la valeur Net-Net de Graham si possible
            netnet_value = get_graham_netnet_value(balance_sheet)
            
            # Si la valeur Net-Net est significativement inférieure, l'utiliser pour ajuster la valeur Graham
            if netnet_value and netnet_value < graham_value * 0.7:
                # Donner plus de poids à la valeur Net-Net si elle est très basse
                graham_value = (graham_value * 0.7 + netnet_value * 0.3)
        
        # Méthode 3: Valuation basée sur le PER
        per_based_value = None
        per = fundamental_data['Données fondamentales'].get('PER', None)
        eps = fundamental_data['Données fondamentales'].get('BPA', None)
        per_fwd = fundamental_data['Données fondamentales'].get('PER (Forward)', None)
        
        if per is not None and eps is not None and eps > 0:
            # Récupérer le PER sectoriel
            sector_per, sector_name = get_sector_pe_ratio(ticker)
            
            # Utiliser le PER sectoriel pour calculer la valeur
            per_based_value = eps * sector_per
            
            # Ajuster en fonction du PER futur si disponible
            if per_fwd is not None and per_fwd > 0:
                per_fwd_value = eps * per_fwd
                # Moyenne pondérée avec plus de poids au PER futur
                per_based_value = (per_based_value * 0.4 + per_fwd_value * 0.6)
        
        # Méthode 4: Valeur patrimoniale
        asset_based_value = None
        if not balance_sheet.empty and len(balance_sheet.columns) > 0:
            try:
                # Récupérer les données du bilan
                total_assets = balance_sheet.loc['Total Assets', balance_sheet.columns[0]] if 'Total Assets' in balance_sheet.index else None
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', balance_sheet.columns[0]] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                goodwill = balance_sheet.loc['Goodwill And Intangible Assets', balance_sheet.columns[0]] if 'Goodwill And Intangible Assets' in balance_sheet.index else 0
                
                if total_assets is not None and total_liabilities is not None:
                    # Calculer les actifs nets tangibles
                    tangible_net_assets = total_assets - total_liabilities - goodwill
                    
                    # Récupérer le nombre d'actions en circulation
                    shares_outstanding = fundamental_data['Données de marché'].get('Actions ordinaires', None)
                    
                    if shares_outstanding:
                        # Calculer la valeur patrimoniale par action
                        asset_value_per_share = tangible_net_assets / shares_outstanding
                        
                        # Récupérer le secteur
                        sector = fundamental_data['Données générales'].get('Secteur', None)
                        
                        # Ajuster selon le secteur
                        sector_multiplier = {
                            'Financial Services': 1.2,
                            'Real Estate': 1.5,
                            'Utilities': 1.3,
                            'Basic Materials': 1.1,
                            'Energy': 1.1,
                            'Services Financiers': 1.2,
                            'Immobilier': 1.5,
                            'Services Publics': 1.3,
                            'Matériaux': 1.1,
                            'Énergie': 1.1
                        }
                        
                        multiplier = sector_multiplier.get(sector, 1.0)
                        asset_based_value = asset_value_per_share * multiplier

            except Exception as e:
                print(f"Erreur lors du calcul de la valeur patrimoniale: {e}")
                asset_based_value = None
        
        # Combiner les différentes méthodes pour obtenir une valeur intrinsèque finale
        # Utiliser une moyenne pondérée selon la pertinence des méthodes par secteur
        available_values = []
        weights = []
        
        # Définir les poids selon le secteur
        dcf_weight = 0.5  # Poids par défaut pour DCF
        per_weight = 0.3  # Poids par défaut pour PER
        graham_weight = 0.1  # Poids par défaut pour Graham
        asset_weight = 0.1  # Poids par défaut pour valeur patrimoniale
        
        # Ajuster les poids selon le secteur
        if sector:
            if sector in ['Technology', 'Healthcare', 'Communication Services', 'Technologie', 'Santé', 'Communication']:
                # Secteurs de croissance: privilégier DCF
                dcf_weight = 0.6
                per_weight = 0.3
                graham_weight = 0.1
                asset_weight = 0.0
            elif sector in ['Financial Services', 'Real Estate', 'Utilities', 'Services Financiers', 'Immobilier', 'Services Publics']:
                # Secteurs patrimoniaux: privilégier l'actif net
                dcf_weight = 0.3
                per_weight = 0.3
                graham_weight = 0.1
                asset_weight = 0.3
            elif sector in ['Consumer Defensive', 'Utilities', 'Consommation Non-Cyclique', 'Services Publics']:
                # Secteurs défensifs: équilibrer entre DCF et PER
                dcf_weight = 0.4
                per_weight = 0.4
                graham_weight = 0.1
                asset_weight = 0.1
        
        # Ajouter les valeurs disponibles avec leur poids respectif
        if per_based_value is not None and not (isinstance(per_based_value, float) and math.isnan(per_based_value)):
            available_values.append(per_based_value)
            weights.append(per_weight)
        
        if graham_value is not None and not (isinstance(graham_value, float) and math.isnan(graham_value)):
            available_values.append(graham_value)
            weights.append(graham_weight)
        
        if dcf_value is not None and not (isinstance(dcf_value, float) and math.isnan(dcf_value)):
            available_values.append(dcf_value)
            weights.append(dcf_weight)
        
        if asset_based_value is not None and not (isinstance(asset_based_value, float) and math.isnan(asset_based_value)):
            available_values.append(asset_based_value)
            weights.append(asset_weight)
        
        if not available_values:
            return {
                'success': False,
                'message': f"Impossible de calculer la valeur intrinsèque pour {ticker}"
            }
        
        # Normaliser les poids
        sum_weights = sum(weights)
        normalized_weights = [w / sum_weights for w in weights]
        
        # Calculer la valeur intrinsèque comme la moyenne pondérée des méthodes
        intrinsic_value = sum(v * w for v, w in zip(available_values, normalized_weights))
                
        # Calculer l'écart entre le prix actuel et la valeur intrinsèque
        deviation = (current_price / intrinsic_value - 1) * 100
        
        # Définir une marge de sécurité variable selon le secteur
        sector_mos_map = {
            'Technology': 0.25,  # Secteurs plus volatils: MOS plus élevée
            'Healthcare': 0.25,
            'Consumer Cyclical': 0.25,
            'Financial Services': 0.20,
            'Communication Services': 0.25,
            'Industrials': 0.20,
            'Consumer Defensive': 0.15,  # Secteurs plus stables: MOS plus faible
            'Energy': 0.30,  # Secteur très volatile: MOS très élevée
            'Basic Materials': 0.25,
            'Utilities': 0.15,
            'Real Estate': 0.20,
            # Équivalents en français
            'Technologie': 0.25,
            'Santé': 0.25,
            'Consommation Cyclique': 0.25,
            'Services Financiers': 0.20,
            'Communication': 0.25,
            'Industrie': 0.20,
            'Consommation Non-Cyclique': 0.15,
            'Énergie': 0.30,
            'Matériaux': 0.25,
            'Services Publics': 0.15,
            'Immobilier': 0.20
        }
        
        # Marge de sécurité par défaut si secteur non identifié
        default_mos = 0.20
        mos_percentage = sector_mos_map.get(sector, default_mos)
        
        # Valeur avec marge de sécurité (MOS)
        intrinsic_value_with_mos = intrinsic_value * (1 - mos_percentage)
        
        # Recommandation basée sur la marge de sécurité et l'écart de valorisation
        if current_price <= intrinsic_value_with_mos:
            # Forte sous-valorisation
            if current_price < intrinsic_value_with_mos * 0.8:
                recommendation = "ACHAT FORT"
            else:
                recommendation = "ACHAT"
        elif current_price <= intrinsic_value:
            # Légère sous-valorisation
            recommendation = "ACCUMULATION"
        elif current_price <= intrinsic_value * 1.1:
            # Juste valorisation
            recommendation = "CONSERVER"
        elif current_price <= intrinsic_value * 1.3:
            # Légère survalorisation
            recommendation = "ALLÈGEMENT"
        else:
            # Forte survalorisation
            recommendation = "VENTE"
        
        # Construire et retourner le résultat
        result = {
            'success': True,
            'ticker': ticker,
            'current_price': current_price,
            'intrinsic_value': intrinsic_value,
            'intrinsic_value_with_mos': intrinsic_value_with_mos,
            'mos_percentage': mos_percentage * 100,  # En pourcentage
            'deviation': deviation,
            'recommendation': recommendation,
            'methods': {
                'per_based': {
                    'value': per_based_value,
                    'weight': per_weight if per_based_value is not None else 0
                },
                'graham': {
                    'value': graham_value,
                    'weight': graham_weight if graham_value is not None else 0,
                    'netnet_value': netnet_value
                },
                'dcf': {
                    'value': dcf_value,
                    'weight': dcf_weight if dcf_value is not None else 0
                },
                'asset_based': {
                    'value': asset_based_value,
                    'weight': asset_weight if asset_based_value is not None else 1
                }
            }
        }
        
        # Ajouter des informations supplémentaires sur la qualité si disponibles
        if 'quality_factors' in locals():
            result['quality_factors'] = {
                'adjustment': quality_adjustment,
                'moat': moat_indicator,
                'management': management_quality
            }
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'message': f"Erreur lors du calcul de la valeur intrinsèque pour {ticker}: {str(e)}",
            'error_details': traceback.format_exc()
        }

def calculate_dcf_value(ticker, income_stmt, cashflow, balance_sheet, fundamental_data, current_price, market_structure=None):
    """
    Calcule la valeur DCF (Discounted Cash Flow) d'une action en intégrant le PER sectoriel
    pour une valorisation plus précise et une meilleure estimation de la valeur terminale.
    
    Args:
        ticker: Symbole boursier de l'entreprise
        income_stmt: DataFrame du compte de résultat
        cashflow: DataFrame du tableau de flux de trésorerie
        balance_sheet: DataFrame du bilan
        fundamental_data: Dictionnaire de données fondamentales
        current_price: Prix actuel de l'action
        market_structure: Structure de marché optionnelle pour une meilleure identification sectorielle
        
    Returns:
        float: Valeur DCF estimée par action, ou None si impossible à calculer
    """
    dcf_value = None
    try:
        if not income_stmt.empty and len(income_stmt.columns) >= 3 and 'Free Cash Flow' in cashflow.index:
            # Récupérer les 3 derniers FCF
            fcf_values = []
            for col in cashflow.columns[:3]:
                if 'Free Cash Flow' in cashflow.index:
                    fcf_values.append(cashflow.loc['Free Cash Flow', col])
            
            if len(fcf_values) == 3 and all(v is not None for v in fcf_values):
                # Récupérer et calculer des données importantes en amont
                net_income = income_stmt.loc['Net Income', income_stmt.columns[0]] if 'Net Income' in income_stmt.index else None
                market_cap = fundamental_data['Données de marché'].get('Capitalisation boursière', None)
                total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]] if 'Total Debt' in balance_sheet.index else 0
                cash = balance_sheet.loc['Cash And Cash Equivalents', balance_sheet.columns[0]] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                short_term_investments = balance_sheet.loc['Short Term Investments', balance_sheet.columns[0]] if 'Short Term Investments' in balance_sheet.index else 0
                
                # Calculer le taux de croissance moyen du FCF avec lissage
                growth_rates = []
                for i in range(len(fcf_values)-1):
                    if fcf_values[i] > 0 and fcf_values[i+1] > 0:
                        growth_rates.append((fcf_values[i] / fcf_values[i+1]) - 1)
                
                # Utiliser la médiane des taux pour éviter les valeurs aberrantes
                avg_growth_rate = np.median(growth_rates) if growth_rates else 0.05
                
                # Limiter le taux de croissance pour être conservateur (entre 2% et 15%)
                avg_growth_rate = max(min(avg_growth_rate, 0.15), 0.02)
                
                # Récupérer le PER sectoriel via yfinance en utilisant la structure de marché
                sector_pe, sector_name = get_sector_pe_ratio(ticker, market_structure)
                print(f"Secteur identifié: {sector_name}, PER sectoriel: {sector_pe}")
                
                # ===== Calcul amélioré du WACC (Coût Moyen Pondéré du Capital) =====
                equity_value = market_cap if market_cap else 0
                
                # Structure du capital
                total_capital = equity_value + total_debt
                if total_capital > 0:
                    equity_weight = equity_value / total_capital
                    debt_weight = total_debt / total_capital
                else:
                    equity_weight = 0.8
                    debt_weight = 0.2
                
                # Taux d'imposition (t) - Utilisation des données réelles
                tax_rate = 0.25  # Valeur par défaut
                if 'Income Tax Expense' in income_stmt.index and 'Income Before Tax' in income_stmt.index:
                    income_tax = income_stmt.loc['Income Tax Expense', income_stmt.columns[0]]
                    income_before_tax = income_stmt.loc['Income Before Tax', income_stmt.columns[0]]
                    if income_before_tax and income_before_tax != 0:
                        tax_rate = min(max(abs(income_tax / income_before_tax), 0.15), 0.35)  # Limite raisonnable
                
                # Coût de la dette (Rd) - Plus précis
                debt_cost = 0.05  # Valeur par défaut
                if 'Interest Expense' in income_stmt.index and total_debt > 0:
                    interest_expense = abs(income_stmt.loc['Interest Expense', income_stmt.columns[0]])
                    debt_cost = min(interest_expense / total_debt, 0.10)  # Limité à 10%
                
                # Récupérer le taux sans risque - Amélioré pour être plus précis
                try:
                    # Déterminer si c'est un ticker européen
                    is_european = '.' in ticker and ticker.split('.')[-1] in ['PA', 'AS', 'BR', 'DE', 'MI', 'MC', 'SW']
                    
                    if is_european:
                        risk_free_rate = 0.03  # Approximation pour les OAT 10 ans européennes
                    else:
                        risk_free_rate = 0.04  # Approximation pour les T-Bonds américains
                except:
                    risk_free_rate = 0.035  # Valeur par défaut médiane
                
                # Prime de risque du marché adaptative selon le secteur
                sector_risk_mapping = {
                    'Technology': 0.055,
                    'Healthcare': 0.05, 
                    'Financial Services': 0.06,
                    'Consumer Cyclical': 0.055,
                    'Consumer Defensive': 0.045,
                    'Industrials': 0.05,
                    'Energy': 0.065,
                    'Utilities': 0.04,
                    'Communication Services': 0.055,
                    'Real Estate': 0.06,
                    'Basic Materials': 0.055,
                    # Secteurs européens
                    'Technologie': 0.055,
                    'Santé': 0.05,
                    'Services Financiers': 0.06,
                    'Consommation Cyclique': 0.055,
                    'Consommation Non-Cyclique': 0.045,
                    'Industrie': 0.05,
                    'Énergie': 0.065,
                    'Services Publics': 0.04,
                    'Immobilier': 0.06,
                    'Communication': 0.055,
                    'Matériaux': 0.055
                }
                
                market_risk_premium = sector_risk_mapping.get(sector_name, 0.05)
                
                # Bêta de l'entreprise
                beta = fundamental_data['Données de marché'].get('Beta', 1.0)
                if not beta or beta < 0.3:
                    beta = 1.0  # Valeur par défaut ou correction de valeurs aberrantes
                
                # Calcul du coût des capitaux propres avec le modèle CAPM
                equity_cost = risk_free_rate + beta * market_risk_premium
                
                # Calcul du WACC
                wacc = (equity_weight * equity_cost) + (debt_weight * debt_cost * (1 - tax_rate))
                wacc = max(min(wacc, 0.15), 0.06)  # Limite raisonnable entre 6% et 15%
                
                print(f"WACC calculé: {wacc:.2%}")
                
                # ===== Projection des flux futurs avec une croissance dégressive =====
                fcf_current = fcf_values[0]
                future_fcf = []
                future_earnings = []
                
                # Projection sur 5 ans avec taux dégressif plus réaliste
                growth_rates = []
                for i in range(5):
                    if i < 2:
                        growth_rates.append(avg_growth_rate)
                    else:
                        decay_factor = 0.85 - (i-2) * 0.05  # Décroissance progressive plus marquée
                        growth_rates.append(max(avg_growth_rate * decay_factor, 0.02))
                
                # Calculer les FCF et earnings futurs
                cumulative_growth = 1
                for i in range(5):
                    cumulative_growth *= (1 + growth_rates[i])
                    future_fcf_value = fcf_current * cumulative_growth
                    future_fcf.append(future_fcf_value / ((1 + wacc) ** (i + 1)))
                    
                    if net_income:
                        future_earnings.append(net_income * cumulative_growth)
                
                # ===== Calcul de la valeur terminale avec méthodes hybrides =====
                # Taux de croissance terminal adapté au secteur
                sector_terminal_growth = {
                    'Technology': 0.035,
                    'Healthcare': 0.03,
                    'Financial Services': 0.025,
                    'Consumer Cyclical': 0.02,
                    'Consumer Defensive': 0.025,
                    'Industrials': 0.02,
                    'Energy': 0.015,
                    'Utilities': 0.02,
                    'Communication Services': 0.025,
                    'Real Estate': 0.02,
                    'Basic Materials': 0.02
                }
                
                # Utiliser le taux spécifique au secteur ou une valeur par défaut
                terminal_growth = sector_terminal_growth.get(sector_name, 0.02)
                terminal_growth = min(terminal_growth, avg_growth_rate * 0.5)  # Limitation conservatrice
                
                # Méthode 1: Gordon Growth
                terminal_fcf = fcf_current * cumulative_growth * (1 + terminal_growth)
                terminal_value_gordon = terminal_fcf / (wacc - terminal_growth) / ((1 + wacc) ** 5)
                
                # Méthode 2: Multiple de sortie basé sur le PER sectoriel ajusté
                # Ajuster le PER sectoriel selon la croissance estimée vs. la croissance sectorielle moyenne
                adjusted_sector_pe = sector_pe
                if avg_growth_rate > 0.10:  # Si croissance supérieure à la moyenne
                    adjusted_sector_pe = sector_pe * min(1.3, 1 + (avg_growth_rate - 0.05))
                elif avg_growth_rate < 0.05:  # Si croissance inférieure à la moyenne
                    adjusted_sector_pe = sector_pe * max(0.7, 1 - (0.05 - avg_growth_rate))
                
                if net_income and len(future_earnings) == 5:
                    terminal_earnings = future_earnings[4]
                    terminal_value_pe = terminal_earnings * adjusted_sector_pe / ((1 + wacc) ** 5)
                    
                    # Pondération dynamique basée sur la confiance dans les données
                    gordon_weight = 0.6
                    pe_weight = 0.4
                    
                    # Ajuster les pondérations selon la fiabilité des données
                    if abs(terminal_value_gordon / terminal_value_pe - 1) > 0.3:
                        # Si les méthodes donnent des résultats très différents, favoriser Gordon
                        gordon_weight = 0.7
                        pe_weight = 0.3
                    
                    terminal_value = (terminal_value_gordon * gordon_weight) + (terminal_value_pe * pe_weight)
                else:
                    terminal_value = terminal_value_gordon
                
                # ===== Calcul de la valeur totale et par action =====
                enterprise_value = sum(future_fcf) + terminal_value
                
                # Ajuster pour la dette et la trésorerie
                equity_value = enterprise_value + cash + short_term_investments - total_debt
                
                # Calculer la valeur par action
                if market_cap and current_price and current_price > 0:
                    shares_outstanding = market_cap / current_price
                    dcf_value = equity_value / shares_outstanding if shares_outstanding else None
                    
                    # ===== Vérifications de cohérence et ajustements =====
                    # Calcul du PER implicite et comparaison avec le PER sectoriel
                    if net_income and shares_outstanding and shares_outstanding > 0:
                        eps = net_income / shares_outstanding
                        if eps > 0:
                            implicit_pe = dcf_value / eps
                            pe_ratio = implicit_pe / sector_pe
                            
                            print(f"PER implicite: {implicit_pe:.2f}x, ratio vs sectoriel: {pe_ratio:.2f}x")
                            
                            # Ajustement si l'écart avec le PER sectoriel est important
                            if pe_ratio > 2.0:
                                # Formule de correction progressive
                                correction_factor = 1.0 / (1.0 + 0.3 * (pe_ratio - 2.0))
                                dcf_value = dcf_value * correction_factor
                                print(f"Ajustement de la valorisation (PER implicite trop élevé): correction de {correction_factor:.2f}x")
                            elif pe_ratio < 0.5:
                                # Vérifier si la sous-valorisation est justifiée
                                correction_factor = 1.0 + 0.2 * (0.5 - pe_ratio)
                                dcf_value = dcf_value * correction_factor
                                print(f"Ajustement de la valorisation (PER implicite trop bas): correction de {correction_factor:.2f}x")
                    
                    # Vérification finale de cohérence avec le prix actuel
                    price_ratio = dcf_value / current_price if current_price else 1
                    if price_ratio > 3.0:
                        dcf_value = dcf_value * 0.8
                        print(f"Ajustement conservateur: valorisation excessive vs prix actuel ({price_ratio:.2f}x)")
                    
                    print(f"Valeur DCF calculée: {dcf_value:.2f}, Ratio vs prix actuel: {price_ratio:.2f}x")
    
    except Exception as e:
        print(f"Erreur dans le calcul DCF: {e}")
        dcf_value = None
        
    return dcf_value

def get_sector_pe_ratio(ticker, market_structure=None):
    """
    Récupère le PER moyen du secteur pour un ticker donné en utilisant yfinance
    et la structure de marché existante.
    
    Args:
        ticker: Symbole boursier de l'entreprise
        market_structure: Structure de marché issue de get_market_structure()
    
    Returns:
        tuple: (PER sectoriel moyen, nom du secteur)
    """
    try:
        # Récupérer les informations de l'entreprise via yfinance
        company = yf.Ticker(ticker)
        info = company.info
        sector_yf = info.get('sector', None)
        company_name = info.get('shortName', None)
        
        # Variables pour stocker le secteur et les entreprises similaires
        sector = sector_yf
        similar_companies_tickers = []
        
        # Si la structure de marché est disponible, l'utiliser pour trouver des entreprises similaires
        if market_structure and market_structure.get('all_stocks') and company_name:
            # Chercher d'abord si l'entreprise existe dans notre structure
            found_in_structure = False
            company_info = None
            
            # Chercher par nom d'entreprise ou ticker
            for name, info in market_structure['all_stocks'].items():
                if (company_name.lower() in name.lower()) or (ticker == info['ticker']):
                    company_info = info
                    found_in_structure = True
                    break
            
            if found_in_structure and company_info:
                # Utiliser le secteur de notre structure (plus fiable)
                sector = company_info['secteur']
                industrie = company_info['industrie']
                print(f"Entreprise trouvée dans la structure: Secteur={sector}, Industrie={industrie}")
                
                # Trouver toutes les entreprises du même secteur et industrie
                same_industry_companies = []
                same_sector_companies = []
                
                # Parcourir tous les secteurs
                if 'secteurs' in market_structure and sector in market_structure['secteurs']:
                    # Parcourir toutes les industries du secteur
                    for ind, companies in market_structure['secteurs'][sector].items():
                        for comp_name, comp_info in companies.items():
                            # Exclure l'entreprise elle-même
                            if comp_info['ticker'] != ticker:
                                if ind == industrie:
                                    # Même industrie = priorité plus élevée
                                    same_industry_companies.append(comp_info['ticker'])
                                else:
                                    # Même secteur = priorité plus faible
                                    same_sector_companies.append(comp_info['ticker'])
                
                # Prioriser les entreprises de la même industrie
                if same_industry_companies:
                    similar_companies_tickers = same_industry_companies[:50]  # Limiter à 10 entreprises
                    print(f"Utilisation de {len(similar_companies_tickers)} entreprises de la même industrie")
                elif same_sector_companies:
                    similar_companies_tickers = same_sector_companies[:50]  # Limiter à 10 entreprises
                    print(f"Utilisation de {len(similar_companies_tickers)} entreprises du même secteur")
        
        # Si pas de structure ou pas trouvé dans la structure, utiliser les secteurs yfinance
        if not similar_companies_tickers:
            print("Utilisation des secteurs yfinance pour trouver des entreprises similaires")
            # Liste de tickers connus par secteur (à personnaliser)
            sector_tickers = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
                'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
                'Industrials': ['HON', 'UNP', 'UPS', 'CAT', 'GE'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
                'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'VZ'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'SPG', 'EQIX'],
                'Basic Materials': ['LIN', 'SHW', 'APD', 'ECL', 'NEM']
            }
            
            # Pour les valeurs françaises et européennes
            european_sector_tickers = {
                'Technology': ['CAP.PA', 'SOP.PA', 'ATO.PA', 'ERF.PA', 'IFX.DE'],
                'Healthcare': ['SAN.PA', 'ENGI.PA', 'IPH.PA', 'DBV.PA', 'BIO.PA'],
                'Financial Services': ['BNP.PA', 'ACA.PA', 'GLE.PA', 'CS.PA', 'DEXB.BR'],
                'Consumer Cyclical': ['MC.PA', 'OR.PA', 'RMS.PA', 'ML.PA', 'SW.PA'],
                'Consumer Defensive': ['CA.PA', 'BN.PA', 'RI.PA', 'AHO.AS', 'NESN.SW'],
                'Industrials': ['AIR.PA', 'SAF.PA', 'SU.PA', 'EN.PA', 'DG.PA'],
                'Energy': ['TTE.PA', 'ENGI.PA', 'FP.PA', 'GTT.PA', 'REP.MC'],
                'Utilities': ['ENGI.PA', 'VIE.PA', 'EDF.PA', 'RWE.DE', 'ENEL.MI'],
                'Communication Services': ['ORA.PA', 'VIV.PA', 'PUB.PA', 'TFI.PA', 'PROX.BR'],
                'Real Estate': ['URW.AS', 'COV.PA', 'GFCP.PA', 'GFC.PA', 'DLG.DE'],
                'Basic Materials': ['AI.PA', 'MT.AS', 'NK.PA', 'VK.PA', 'SGL.DE']
            }
            
            # Déterminer si c'est un ticker européen
            is_european = '.' in ticker and ticker.split('.')[-1] in ['PA', 'AS', 'BR', 'DE', 'MI', 'MC', 'SW']
            
            # Choisir la liste appropriée
            if is_european and sector in european_sector_tickers:
                similar_companies_tickers = european_sector_tickers[sector]
            elif sector in sector_tickers:
                similar_companies_tickers = sector_tickers[sector]
        
        # Récupérer les PER des entreprises similaires
        similar_companies_pe = []
        for similar_ticker in similar_companies_tickers:
            try:
                similar_company = yf.Ticker(similar_ticker)
                pe = similar_company.info.get('trailingPE', None)
                if pe and pe > 0 and pe < 200:  # Exclure les valeurs aberrantes
                    similar_companies_pe.append(pe)
                    print(f"PER pour {similar_ticker}: {pe}")
            except Exception as e:
                print(f"Erreur pour {similar_ticker}: {e}")
                continue
        
        # Si des PER ont été trouvés, calculer la moyenne
        if similar_companies_pe:
            # Éliminer les valeurs aberrantes (au-delà de 2 écarts-types)
            if len(similar_companies_pe) >= 3:
                mean_pe = np.mean(similar_companies_pe)
                std_pe = np.std(similar_companies_pe)
                filtered_pes = [pe for pe in similar_companies_pe if abs(pe - mean_pe) <= 2 * std_pe]
                sector_pe = np.mean(filtered_pes) if filtered_pes else np.median(similar_companies_pe)
                print(f"PER sectoriel calculé: {sector_pe:.2f} (moyenne filtrée de {len(filtered_pes)} entreprises)")
            else:
                sector_pe = np.mean(similar_companies_pe)
                print(f"PER sectoriel calculé: {sector_pe:.2f} (moyenne simple de {len(similar_companies_pe)} entreprises)")
            
            return (sector_pe, sector)
        
        # Si on n'a pas pu calculer à partir d'entreprises similaires, essayer avec un ETF sectoriel
        sector_mapping = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Financial Services': 'XLF',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC',
            'Basic Materials': 'XLB'
        }
        
        # Pour l'Europe
        european_sector_mapping = {
            'Technology': 'SX8E.SW',  # EURO STOXX® Technology
            'Healthcare': 'SXDE.SW',  # EURO STOXX® Health Care
            'Financial Services': 'SX7E.SW',  # EURO STOXX® Banks
            'Consumer Cyclical': 'SXQE.SW',  # EURO STOXX® Personal & Household Goods
            'Consumer Defensive': 'SX3E.SW',  # EURO STOXX® Food & Beverage
            'Industrials': 'SXNE.SW',  # EURO STOXX® Industrial Goods & Services
            'Energy': 'SXEE.SW',  # EURO STOXX® Energy
            'Utilities': 'SX6E.SW',  # EURO STOXX® Utilities
            'Communication Services': 'SXME.SW',  # EURO STOXX® Media
            'Real Estate': 'SX86E.SW',  # EURO STOXX® Real Estate
            'Basic Materials': 'SXPE.SW'   # EURO STOXX® Chemicals
        }
        
        # Choisir le mapping approprié
        etf_ticker = None
        if is_european and sector in european_sector_mapping:
            etf_ticker = european_sector_mapping.get(sector)
        else:
            etf_ticker = sector_mapping.get(sector)
        
        if etf_ticker:
            try:
                etf = yf.Ticker(etf_ticker)
                etf_pe = etf.info.get('trailingPE', None)
                if etf_pe and etf_pe > 0:
                    print(f"PER sectoriel via ETF {etf_ticker}: {etf_pe}")
                    return (etf_pe, sector)
            except:
                pass
        
        # Valeurs par défaut par secteur si rien d'autre n'a fonctionné
        default_sector_pes = {
            'Technology': 25.3,
            'Healthcare': 20.1,
            'Consumer Cyclical': 22.5,
            'Consumer Defensive': 18.7,
            'Financial Services': 14.2,
            'Industrials': 17.5,
            'Energy': 12.3,
            'Utilities': 16.8,
            'Real Estate': 19.2,
            'Communication Services': 21.4,
            'Basic Materials': 15.9,
            # Secteurs français/européens
            'Technologie': 27.9,
            'Santé': 39.4,
            'Services Financiers': 10.6,
            'Consommation Cyclique': 35,
            'Consommation Non-Cyclique': 31,
            'Industrie': 31.7,
            'Énergie': 9.5,
            'Services Publics': 17.3,
            'Immobilier': 42,
            'Communication': 27.7,
            'Matériaux': 27.2
        }
        
        default_pe = default_sector_pes.get(sector, 15)
        print(f"Utilisation du PER sectoriel par défaut: {default_pe}")
        return (default_pe, sector)
    
    except Exception as e:
        print(f"Erreur générale dans get_sector_pe_ratio: {e}")
        return (15, "Unknown")  # Valeur par défaut en cas d'erreur

def flatten_market_structure(market_structure, filter_type=None, level1=None, level2=None):
    """
    Convertit la structure hiérarchique en dictionnaire plat d'actions selon les filtres
    
    :param market_structure: Structure hiérarchique des marchés
    :param filter_type: Type de filtre ('region', 'secteur', 'marche')
    :param level1: Premier niveau de filtre (région, secteur ou marché)
    :param level2: Deuxième niveau de filtre (pays ou industrie), si applicable
    :return: Dictionnaire plat d'actions
    """
    flattened_stocks = {}
    
    # Si aucun filtre n'est spécifié, retourner toutes les actions
    if not filter_type or not level1:
        return market_structure['all_stocks']
    
    # Filtrage par région et pays
    if filter_type == 'region':
        if level1 in market_structure['regions']:
            # Si un pays est spécifié
            if level2 and level2 in market_structure['regions'][level1]:
                return market_structure['regions'][level1][level2]
            # Sinon, retourner toutes les actions de la région
            else:
                for pays, stocks in market_structure['regions'][level1].items():
                    flattened_stocks.update(stocks)
                return flattened_stocks
    
    # Filtrage par secteur et industrie
    elif filter_type == 'secteur':
        if level1 in market_structure['secteurs']:
            # Si une industrie est spécifiée
            if level2 and level2 in market_structure['secteurs'][level1]:
                return market_structure['secteurs'][level1][level2]
            # Sinon, retourner toutes les actions du secteur
            else:
                for industrie, stocks in market_structure['secteurs'][level1].items():
                    flattened_stocks.update(stocks)
                return flattened_stocks
    
    # Filtrage par marché
    elif filter_type == 'marche':
        if level1 in market_structure['marches']:
            return market_structure['marches'][level1]
    
    # Si aucune correspondance n'est trouvée, retourner un dictionnaire vide
    return {}

def get_owner_earnings(income_stmt, cashflow, balance_sheet):
    """
    Calcule les "bénéfices du propriétaire" selon Buffett
    (1) Résultat net + (2) Amortissements - (3) Dépenses en capital de maintenance
    
    Returns:
        float: Bénéfices du propriétaire, ou None si impossible à calculer
    """
    try:
        if not income_stmt.empty and not cashflow.empty:
            # Résultat net
            net_income = income_stmt.loc['Net Income', income_stmt.columns[0]] if 'Net Income' in income_stmt.index else None
            
            # Amortissements
            depreciation_amortization = cashflow.loc['Depreciation And Amortization', cashflow.columns[0]] if 'Depreciation And Amortization' in cashflow.index else None
            
            # Capex (dépenses en capital)
            capex = cashflow.loc['Capital Expenditure', cashflow.columns[0]] if 'Capital Expenditure' in cashflow.index else None
            
            # Si toutes les données sont disponibles
            if net_income is not None and depreciation_amortization is not None and capex is not None:
                # Dépenses en capital de maintenance (estimation conservatrice à 70% du Capex total)
                maintenance_capex = abs(capex) * 0.70
                
                # Calcul des bénéfices du propriétaire
                owner_earnings = net_income + depreciation_amortization - maintenance_capex
                return owner_earnings
            else:
                missing = []
                if net_income is None: missing.append("Net Income")
                if depreciation_amortization is None: missing.append("Depreciation And Amortization")
                if capex is None: missing.append("Capital Expenditure")
                print(f"Données manquantes pour le calcul des bénéfices du propriétaire: {', '.join(missing)}")
    except Exception as e:
        print(f"Erreur lors du calcul des bénéfices du propriétaire: {e}")
    
    return None

def get_graham_netnet_value(balance_sheet_df):
    """
    Calcule la valeur "Net-Net" de Graham:
    (Actifs courants - Total des passifs) / Nombre d'actions
    
    Args:
        balance_sheet_df (pd.DataFrame): DataFrame contenant le bilan
        
    Returns:
        float: Valeur Net-Net par action, ou None si impossible à calculer
    """
    try:
        if not balance_sheet_df.empty:
            # Utilisation de la première colonne (dernière période)
            column = balance_sheet_df.columns[1]  # La colonne 0 contient les noms des lignes
            
            # Actifs courants
            current_assets = balance_sheet_df.loc['Current Assets', column] if 'Current Assets' in balance_sheet_df.index else None
            
            # Total du passif
            total_liabilities = balance_sheet_df.loc['Total Liabilities Net Minority Interest', column] if 'Total Liabilities Net Minority Interest' in balance_sheet_df.index else None
            
            # Nombre d'actions ordinaires
            shares_outstanding = balance_sheet_df.loc['Ordinary Shares Number', column] if 'Ordinary Shares Number' in balance_sheet_df.index else None
            
            if current_assets is not None and total_liabilities is not None and shares_outstanding is not None:
                # Valeur Net-Net (division par le nombre d'actions)
                net_net_value = (current_assets - total_liabilities) / shares_outstanding
                return net_net_value
            else:
                missing = []
                if current_assets is None: missing.append("Current Assets")
                if total_liabilities is None: missing.append("Total Liabilities Net Minority Interest")
                if shares_outstanding is None: missing.append("Ordinary Shares Number")
                print(f"Données manquantes pour le calcul de Net-Net: {', '.join(missing)}")

    except Exception as e:
        print(f"Erreur lors du calcul de la valeur Net-Net: {e}")
    
    return None

def calculate_roic(income_stmt, balance_sheet):
    """
    Calcule le ROIC (Return on Invested Capital)
    ROIC = NOPAT / (Capital investi)
    
    Returns:
        float: ROIC (en pourcentage), ou None si impossible à calculer
    """
    try:
        if not income_stmt.empty and not balance_sheet.empty:
            # NOPAT (Net Operating Profit After Tax)
            ebit = income_stmt.loc['EBIT', income_stmt.columns[0]] if 'EBIT' in income_stmt.index else None
            
            # Estimer le taux d'imposition
            income_tax = income_stmt.loc['Income Tax Expense', income_stmt.columns[0]] if 'Income Tax Expense' in income_stmt.index else None
            income_before_tax = income_stmt.loc['Income Before Tax', income_stmt.columns[0]] if 'Income Before Tax' in income_stmt.index else None
            
            tax_rate = 0.25  # Taux par défaut
            if income_tax is not None and income_before_tax is not None and income_before_tax != 0:
                tax_rate = min(max(abs(income_tax / income_before_tax), 0.15), 0.35)
            
            # Capital investi
            total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]] if 'Total Debt' in balance_sheet.index else 0
            stockholders_equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]] if 'Stockholders Equity' in balance_sheet.index else None
            
            if ebit is not None and stockholders_equity is not None:
                nopat = ebit * (1 - tax_rate)
                invested_capital = total_debt + stockholders_equity
                
                if invested_capital > 0:
                    roic = (nopat / invested_capital) * 100
                    return roic
    except Exception as e:
        print(f"Erreur lors du calcul du ROIC: {e}")
    
    return None

def analyze_competitive_advantage(income_stmt, balance_sheet, cashflow, ticker_info):
    """
    Analyse l'avantage compétitif de l'entreprise selon plusieurs métriques
    
    Returns:
        dict: Analyse de l'avantage compétitif
    """
    competitive_analysis = {
        'existence': 'Indéterminé',
        'strength': 0,
        'factors': [],
        'explanation': ''
    }
    
    try:
        # 1. Stabilité des marges brutes (si disponible)
        if not income_stmt.empty and len(income_stmt.columns) >= 3:
            margin_stability = 0
            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                margins = []
                for col in income_stmt.columns[:min(5, len(income_stmt.columns))]:
                    gross_profit = income_stmt.loc['Gross Profit', col]
                    total_revenue = income_stmt.loc['Total Revenue', col]
                    if gross_profit is not None and total_revenue is not None and total_revenue != 0:
                        margins.append(gross_profit / total_revenue)
                
                if len(margins) >= 3:
                    std_margin = np.std(margins)
                    if std_margin < 0.03:  # Faible écart-type indique stabilité
                        competitive_analysis['factors'].append('Marges très stables')
                        margin_stability = 2
                    elif std_margin < 0.05:
                        competitive_analysis['factors'].append('Marges assez stables')
                        margin_stability = 1
        
        # 2. ROE élevé et stable
        if not income_stmt.empty and not balance_sheet.empty and len(income_stmt.columns) >= 3:
            roe_quality = 0
            if 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance_sheet.index:
                roes = []
                for i in range(min(5, len(income_stmt.columns), len(balance_sheet.columns))):
                    net_income = income_stmt.loc['Net Income', income_stmt.columns[i]]
                    equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[i]]
                    if net_income is not None and equity is not None and equity != 0:
                        roes.append(net_income / equity)
                
                if len(roes) >= 3:
                    avg_roe = np.mean(roes) * 100
                    if avg_roe > 15:
                        competitive_analysis['factors'].append(f'ROE élevé ({avg_roe:.1f}%)')
                        roe_quality = 2
                    elif avg_roe > 10:
                        competitive_analysis['factors'].append(f'ROE bon ({avg_roe:.1f}%)')
                        roe_quality = 1
        
        # 3. ROIC élevé
        roic = calculate_roic(income_stmt, balance_sheet)
        roic_quality = 0
        if roic is not None:
            if roic > 15:
                competitive_analysis['factors'].append(f'ROIC excellent ({roic:.1f}%)')
                roic_quality = 2
            elif roic > 10:
                competitive_analysis['factors'].append(f'ROIC bon ({roic:.1f}%)')
                roic_quality = 1
        
        # 4. FCF / Ventes élevé (efficacité de conversion en cash)
        if not cashflow.empty and not income_stmt.empty:
            fcf_efficiency = 0
            if 'Free Cash Flow' in cashflow.index and 'Total Revenue' in income_stmt.index:
                fcf = cashflow.loc['Free Cash Flow', cashflow.columns[0]]
                revenue = income_stmt.loc['Total Revenue', income_stmt.columns[0]]
                if fcf is not None and revenue is not None and revenue != 0:
                    fcf_to_sales = fcf / revenue
                    if fcf_to_sales > 0.15:
                        competitive_analysis['factors'].append(f'Forte conversion en FCF ({fcf_to_sales:.1%})')
                        fcf_efficiency = 2
                    elif fcf_to_sales > 0.08:
                        competitive_analysis['factors'].append(f'Bonne conversion en FCF ({fcf_to_sales:.1%})')
                        fcf_efficiency = 1
        
        # 5. Facteurs qualitatifs (si disponibles dans les informations de base)
        qualitative_score = 0
        sector = ticker_info.get('Secteur', '')
        
        # Vérifier les secteurs avec des avantages compétitifs typiques
        sector_advantage = {
            'Technology': 'Technologie propriétaire',
            'Healthcare': 'Brevets et propriété intellectuelle',
            'Consumer Defensive': 'Marque forte',
            'Communication Services': 'Effet de réseau',
            'Utilities': 'Monopole réglementé',
        }
        
        if sector in sector_advantage:
            competitive_analysis['factors'].append(f'Secteur à avantage: {sector_advantage[sector]}')
            qualitative_score = 1
        
        # Calcul du score global
        total_score = margin_stability + roe_quality + roic_quality + fcf_efficiency + qualitative_score
        
        # Détermination de l'existence d'un avantage compétitif
        if total_score >= 5:
            competitive_analysis['existence'] = 'Fort'
            competitive_analysis['strength'] = 3
            competitive_analysis['explanation'] = "L'entreprise présente plusieurs indicateurs d'un avantage compétitif durable (moat)."
        elif total_score >= 3:
            competitive_analysis['existence'] = 'Modéré'
            competitive_analysis['strength'] = 2
            competitive_analysis['explanation'] = "L'entreprise montre certains signes d'un avantage compétitif, mais manque de solidité sur certains aspects."
        elif total_score >= 1:
            competitive_analysis['existence'] = 'Faible'
            competitive_analysis['strength'] = 1
            competitive_analysis['explanation'] = "L'entreprise présente peu d'indicateurs d'avantage compétitif durable."
        else:
            competitive_analysis['existence'] = 'Non détecté'
            competitive_analysis['strength'] = 0
            competitive_analysis['explanation'] = "Aucun signe clair d'avantage compétitif n'a été détecté dans les données financières."
            
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'avantage compétitif: {e}")
        competitive_analysis['explanation'] = f"Une erreur est survenue lors de l'analyse: {str(e)}"
    
    return competitive_analysis

def buffett_graham_stock_screener(stocks_dict,
                                # Critères financiers fondamentaux (Buffett/Graham)
                                min_revenue_growth=5,
                                min_net_profit_margin=10,
                                min_roe=15,
                                max_debt_to_equity=0.5,
                                min_free_cashflow=0,
                                min_operating_margin=15,
                                min_earnings_stability_years=5,
                                
                                # Critères de régularité/consistance
                                min_gross_margin_consistency=40,
                                min_net_income_growth_consistency=8,
                                consistent_low_debt=True,
                                min_operating_cashflow_growth=5,
                                
                                # Critères de valorisation
                                max_pe_ratio=25,
                                max_pe_to_growth=1.0,
                                max_price_to_book=1.5,
                                
                                # Critères d'efficacité opérationnelle
                                max_rd_sales_ratio=0.05,
                                max_capex_net_income_ratio=0.5,
                                max_interest_ebit_ratio=0.20,
                                min_interest_coverage_ratio=5,
                                
                                # Critères additionnels
                                max_debt_ebitda_ratio=5,
                                min_dividend_years=0,
                                min_current_ratio=1.5):
    """
    Screener d'actions avancé combinant les principes de Warren Buffett et Benjamin Graham
    avec une analyse approfondie de la régularité et de la qualité financière.

    :param stocks_dict: Dictionnaire des actions à analyser
    
    # Critères financiers fondamentaux
    :param min_revenue_growth: Croissance des revenus minimale (%)
    :param min_net_profit_margin: Marge nette minimale (%)
    :param min_roe: Rendement des capitaux propres minimal (%)
    :param max_debt_to_equity: Ratio dette/fonds propres maximal
    :param min_free_cashflow: Free cash flow minimal
    :param min_operating_margin: Marge opérationnelle minimale (%)
    :param min_earnings_stability_years: Nombre d'années consécutives avec des bénéfices positifs
    
    # Critères de régularité/consistance
    :param min_gross_margin_consistency: Marge brute minimale et cohérente sur 5 ans (%)
    :param min_net_income_growth_consistency: Croissance du résultat net minimale et cohérente (%)
    :param consistent_low_debt: Exiger une dette régulièrement faible ou inexistante
    :param min_operating_cashflow_growth: Croissance minimale du cash-flow opérationnel (%)
    
    # Critères de valorisation
    :param max_pe_ratio: Ratio cours/bénéfice maximal
    :param max_pe_to_growth: PEG ratio maximal (P/E divisé par le taux de croissance)
    :param max_price_to_book: Ratio cours/valeur comptable maximal (critère Graham)
    
    # Critères d'efficacité opérationnelle
    :param max_rd_sales_ratio: Ratio maximal des dépenses R&D sur le chiffre d'affaires
    :param max_capex_net_income_ratio: Ratio maximal des dépenses d'investissement sur le résultat net
    :param max_interest_ebit_ratio: Ratio maximal des charges d'intérêts sur l'EBIT
    :param min_interest_coverage_ratio: Ratio minimal de couverture des intérêts
    
    # Critères additionnels
    :param max_debt_ebitda_ratio: Ratio maximal dette/EBITDA
    :param min_dividend_years: Nombre minimal d'années consécutives de versement de dividendes
    :param min_current_ratio: Ratio minimal actifs courants/passifs courants
    
    :return: DataFrame des actions éligibles avec détails d'analyse
    """
    eligible_stocks = []
    detailed_analysis = {}
    
    for stock_name, stock_info in stocks_dict.items():
        ticker = stock_info['ticker']
        try:
            # Données de base via Yahoo Finance
            stock_data = yf.Ticker(ticker)
            info = stock_data.info
            time.sleep(0.5) # Pause pour éviter de dépasser les limites de l'API

            # Récupérer l'historique des données financières
            income_history, balance_history, cashflow_history = get_historical_financials(ticker, period='max')
            
            # Analyser les séries temporelles pour la régularité des performances
            financial_metrics = {}
            consistency_metrics = {}
            intrinsic_value_metrics = {}
            
            # 1. Métriques financières fondamentales
            financial_metrics['revenue_growth'] = info.get('earningsGrowth', 0) * 100
            financial_metrics['net_profit_margin'] = info.get('profitMargins', 0) * 100
            financial_metrics['roe'] = info.get('returnOnEquity', 0) * 100
            financial_metrics['operating_margin'] = info.get('operatingMargins', 0) * 100
            
            total_equity = info.get('totalEquity', 1)
            total_debt = info.get('totalDebt', 0)
            financial_metrics['debt_to_equity'] = total_debt / total_equity if total_equity != 0 else float('inf')
            
            financial_metrics['free_cashflow'] = info.get('freeCashflow', 0)
            financial_metrics['current_ratio'] = info.get('currentRatio', 0)
            
            ebit = info.get('ebit', 1)
            interest_expense = info.get('interestExpense', 0)
            financial_metrics['interest_ebit_ratio'] = interest_expense / ebit if ebit != 0 else float('inf')
            financial_metrics['interest_coverage_ratio'] = ebit / interest_expense if interest_expense != 0 else float('inf')
            
            rd_expenses = info.get('researchDevelopment', 0)
            total_revenue = info.get('totalRevenue', 1)
            financial_metrics['rd_sales_ratio'] = rd_expenses / total_revenue if total_revenue != 0 else 0
            
            ebitda = info.get('ebitda', 1)
            financial_metrics['debt_ebitda_ratio'] = total_debt / ebitda if ebitda != 0 else float('inf')
            
            # 2. Analyse des critères de valorisation
            price = info.get('currentPrice', 0)
            earnings_per_share = info.get('trailingEPS', 0)
            book_value_per_share = info.get('bookValue', 0)
            
            intrinsic_value_metrics['pe_ratio'] = price / earnings_per_share if earnings_per_share != 0 else float('inf')
            intrinsic_value_metrics['price_to_book'] = price / book_value_per_share if book_value_per_share != 0 else float('inf')
            
            # Calculer le PEG ratio (P/E divisé par le taux de croissance)
            growth_5y = info.get('earningsGrowth', 0) * 100
            intrinsic_value_metrics['pe_to_growth'] = intrinsic_value_metrics['pe_ratio'] / growth_5y if growth_5y > 0 else float('inf')
            
            # 3. Analyse de la régularité des performances (consistance)
            
            # 3.1 Marge brute consistante sur 5 ans
            consistent_gross_margin, gross_margin_reason = False, "N/A"
            if not income_history.empty:
                if 'Gross Profit' in income_history.index and 'Total Revenue' in income_history.index:
                    gross_margins_history = ((income_history.loc['Gross Profit'] / income_history.loc['Total Revenue']) * 100).dropna()
                    if not gross_margins_history.empty:
                        consistent_gross_margin, gross_margin_reason = analyze_consistency(gross_margins_history, min_value=min_gross_margin_consistency)
                        consistency_metrics['gross_margin_consistency'] = gross_margin_reason
            
            # 3.2 Croissance du résultat net consistante
            consistent_net_income_growth, net_income_growth_reason = False, "N/A"
            if not income_history.empty:
                if 'Net Income' in income_history.index:
                    net_incomes_history = income_history.loc['Net Income'].dropna()
                    if len(net_incomes_history) > 1:
                        net_income_growth_rates = net_incomes_history.pct_change().dropna() * 100
                        consistent_net_income_growth, net_income_growth_reason = analyze_consistency(net_income_growth_rates, min_growth=min_net_income_growth_consistency, min_length=4)
                        consistency_metrics['net_income_growth_consistency'] = net_income_growth_reason
            
            # 3.3 Dette régulièrement faible
            consistent_low_debt_result, debt_consistency_reason = False, "N/A"
            if not balance_history.empty and 'Total Debt' in balance_history.index and 'Total Stockholder Equity' in balance_history.index:
                debt_equity_ratios = (balance_history.loc['Total Debt'] / balance_history.loc['Total Stockholder Equity']).dropna()
                if not debt_equity_ratios.empty:
                    consistent_low_debt_result, debt_consistency_reason = analyze_consistency(debt_equity_ratios, max_value=max_debt_to_equity)
                    consistency_metrics['debt_consistency'] = debt_consistency_reason
            
            # 3.4 Croissance du cash-flow opérationnel
            consistent_operating_cashflow, operating_cashflow_reason = False, "N/A"
            if not cashflow_history.empty and 'Operating Cash Flow' in cashflow_history.index:
                operating_cashflows = cashflow_history.loc['Operating Cash Flow'].dropna()
                if len(operating_cashflows) > 1:
                    operating_cashflow_growth = operating_cashflows.pct_change().dropna() * 100
                    consistent_operating_cashflow, operating_cashflow_reason = analyze_consistency(operating_cashflow_growth, min_growth=min_operating_cashflow_growth)
                    consistency_metrics['operating_cashflow_consistency'] = operating_cashflow_reason
            
            # 3.5 Stabilité des bénéfices (années consécutives avec bénéfices positifs)
            earnings_stability = False
            if not income_history.empty and 'Net Income' in income_history.index:
                net_incomes = income_history.loc['Net Income'].dropna()
                consecutive_positive_years = count_consecutive_positives(net_incomes)
                earnings_stability = consecutive_positive_years >= min_earnings_stability_years
                consistency_metrics['earnings_stability_years'] = consecutive_positive_years
            
            # 3.6 Historique de dividendes
            dividend_history_sufficient = False
            if not cashflow_history.empty and 'Dividends Paid' in cashflow_history.index:
                dividends = cashflow_history.loc['Dividends Paid'].abs().dropna()  # Les dividendes sont généralement négatifs dans les cash flows
                consecutive_dividend_years = count_consecutive_positives(dividends)
                dividend_history_sufficient = consecutive_dividend_years >= min_dividend_years
                consistency_metrics['dividend_years'] = consecutive_dividend_years
            
            # 3.7 Ratio CAPEX/Résultat Net (efficacité des investissements)
            capex_net_income_ratio = float('inf')
            if not cashflow_history.empty and 'Capital Expenditure' in cashflow_history.index and not income_history.empty and 'Net Income' in income_history.index:
                # Aligner les dates entre les deux dataframes
                common_dates = set(cashflow_history.columns).intersection(set(income_history.columns))
                if common_dates:
                    capex = cashflow_history.loc['Capital Expenditure', list(common_dates)].abs()  # CAPEX est généralement négatif
                    net_income = income_history.loc['Net Income', list(common_dates)]
                    capex_net_income_ratio = (capex / net_income).mean()
                    financial_metrics['capex_net_income_ratio'] = capex_net_income_ratio
            
            # 4. Vérifier tous les critères combinés
            all_criteria_met = (
                # Critères financiers fondamentaux
                financial_metrics['revenue_growth'] >= min_revenue_growth and
                financial_metrics['net_profit_margin'] >= min_net_profit_margin and
                financial_metrics['roe'] >= min_roe and
                financial_metrics['debt_to_equity'] <= max_debt_to_equity and
                financial_metrics['free_cashflow'] >= min_free_cashflow and
                financial_metrics['operating_margin'] >= min_operating_margin and
                financial_metrics['current_ratio'] >= min_current_ratio and
                
                # Critères d'efficacité opérationnelle
                financial_metrics['rd_sales_ratio'] <= max_rd_sales_ratio and
                financial_metrics['interest_ebit_ratio'] <= max_interest_ebit_ratio and
                financial_metrics['interest_coverage_ratio'] >= min_interest_coverage_ratio and
                financial_metrics['debt_ebitda_ratio'] <= max_debt_ebitda_ratio and
                
                # Critères de régularité/consistance
                consistent_gross_margin and
                consistent_net_income_growth and
                (not consistent_low_debt or consistent_low_debt_result) and  # Ignorer si l'option n'est pas activée
                consistent_operating_cashflow and
                earnings_stability and
                (min_dividend_years == 0 or dividend_history_sufficient) and  # Ignorer si 0 années requises
                capex_net_income_ratio <= max_capex_net_income_ratio and
                
                # Critères de valorisation
                intrinsic_value_metrics['pe_ratio'] <= max_pe_ratio and
                intrinsic_value_metrics['pe_to_growth'] <= max_pe_to_growth and
                intrinsic_value_metrics['price_to_book'] <= max_price_to_book
            )
            
            if all_criteria_met:
                # Récupération de l'ISIN et informations géographiques
                isin = get_isin_for_ticker(ticker)
                region = stock_info.get('region', 'N/A')
                country = stock_info.get('country', 'N/A')
                
                # Calcul d'une note globale d'attractivité Buffett-Graham (sur 100)
                buffett_graham_score = calculate_buffett_graham_score(
                    financial_metrics, 
                    consistency_metrics, 
                    intrinsic_value_metrics
                )
                
                # Création d'un résumé des forces et faiblesses
                strengths, weaknesses = identify_strengths_weaknesses(
                    financial_metrics, 
                    consistency_metrics, 
                    intrinsic_value_metrics
                )
                
                # Données à afficher dans le tableau de résultats
                stock_details = {
                    'Nom': stock_name,
                    'Ticker': ticker,
                    'ISIN': isin,
                    'Région': region,
                    'Pays': country,
                    'Score Buffett-Graham': f"{buffett_graham_score}/100",
                    'Croissance CA (%)': f"{financial_metrics['revenue_growth']:.2f}",
                    'Marge nette (%)': f"{financial_metrics['net_profit_margin']:.2f}",
                    'ROE (%)': f"{financial_metrics['roe']:.2f}",
                    'Dette/Fonds propres': f"{financial_metrics['debt_to_equity']:.2f}",
                    'P/E': f"{intrinsic_value_metrics['pe_ratio']:.2f}",
                    'PEG': f"{intrinsic_value_metrics['pe_to_growth']:.2f}",
                    'P/B': f"{intrinsic_value_metrics['price_to_book']:.2f}",
                    'Free Cash Flow': format_number(financial_metrics['free_cashflow']),
                    'Années bénéfices positifs': consistency_metrics['earnings_stability_years'],
                    'Marge Brute Consistante': consistency_metrics['gross_margin_consistency'],
                    'Croissance RN Consistante': consistency_metrics['net_income_growth_consistency'],
                    'Croissance FCF': consistency_metrics['operating_cashflow_consistency'],
                    'Forces': strengths,
                    'Faiblesses': weaknesses
                }
                eligible_stocks.append(stock_details)
                
                # Stockage de l'analyse détaillée pour affichage ultérieur
                detailed_analysis[ticker] = {
                    'financial_metrics': financial_metrics,
                    'consistency_metrics': consistency_metrics,
                    'intrinsic_value_metrics': intrinsic_value_metrics,
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'score': buffett_graham_score
                }

        except Exception as e:
            st.warning(f"Erreur pour {stock_name}: {e}")

    return pd.DataFrame(eligible_stocks), detailed_analysis

# Fonction pour calculer un score global Buffett-Graham (sur 100)
def calculate_buffett_graham_score(financial_metrics, consistency_metrics, intrinsic_value_metrics):
    """
    Calcule un score global basé sur les principes de Buffett et Graham.
    
    :param financial_metrics: Dictionnaire des métriques financières
    :param consistency_metrics: Dictionnaire des métriques de consistance
    :param intrinsic_value_metrics: Dictionnaire des métriques de valorisation
    :return: Score sur 100
    """
    score = 0
    
    # Métriques financières (40 points max)
    score += min(financial_metrics['roe'] / 2, 10)  # 10 points max pour ROE
    score += min(financial_metrics['net_profit_margin'] / 2, 10)  # 10 points max pour la marge nette
    score += min(15 - financial_metrics['debt_to_equity'] * 5, 10)  # 10 points max pour un faible endettement
    score += min(financial_metrics['operating_margin'] / 3, 10)  # 10 points max pour la marge opérationnelle
    
    # Métriques de consistance (40 points max)
    if 'earnings_stability_years' in consistency_metrics:
        score += min(consistency_metrics['earnings_stability_years'] * 2, 10)  # 10 points max pour stabilité des bénéfices
    
    if 'gross_margin_consistency' in consistency_metrics and "Consistante" in consistency_metrics['gross_margin_consistency']:
        score += 10  # 10 points pour marge brute consistante
    
    if 'net_income_growth_consistency' in consistency_metrics and "Consistante" in consistency_metrics['net_income_growth_consistency']:
        score += 10  # 10 points pour croissance du résultat net consistante
    
    if 'operating_cashflow_consistency' in consistency_metrics and "Consistant" in consistency_metrics['operating_cashflow_consistency']:
        score += 10  # 10 points pour cash-flow opérationnel consistant
    
    # Métriques de valorisation (20 points max)
    if intrinsic_value_metrics['pe_ratio'] < float('inf'):
        score += min(20 - intrinsic_value_metrics['pe_ratio'] / 2, 7)  # 7 points max pour un PE faible
    
    if intrinsic_value_metrics['pe_to_growth'] < float('inf'):
        score += min(10 - intrinsic_value_metrics['pe_to_growth'] * 7, 7)  # 7 points max pour un PEG faible
    
    if intrinsic_value_metrics['price_to_book'] < float('inf'):
        score += min(10 - intrinsic_value_metrics['price_to_book'] * 3, 6)  # 6 points max pour un P/B faible
    
    # Arrondir et s'assurer que le score est entre 0 et 100
    return max(0, min(round(score), 100))

# Fonction pour identifier les forces et faiblesses
def identify_strengths_weaknesses(financial_metrics, consistency_metrics, intrinsic_value_metrics):
    """
    Identifie les principales forces et faiblesses d'une action.
    
    :param financial_metrics: Dictionnaire des métriques financières
    :param consistency_metrics: Dictionnaire des métriques de consistance
    :param intrinsic_value_metrics: Dictionnaire des métriques de valorisation
    :return: Tuple (forces, faiblesses) sous forme de listes
    """
    strengths = []
    weaknesses = []
    
    # Analyser les forces
    if financial_metrics['roe'] > 20:
        strengths.append("ROE excellent")
    
    if financial_metrics['net_profit_margin'] > 15:
        strengths.append("Marge nette élevée")
    
    if financial_metrics['debt_to_equity'] < 0.3:
        strengths.append("Endettement très faible")
    
    if 'earnings_stability_years' in consistency_metrics and consistency_metrics['earnings_stability_years'] > 8:
        strengths.append("Bénéfices très stables")
    
    if intrinsic_value_metrics['pe_to_growth'] < 0.8:
        strengths.append("PEG ratio attractif")
    
    # Analyser les faiblesses
    if financial_metrics['roe'] < 10:
        weaknesses.append("ROE insuffisant")
    
    if financial_metrics['debt_to_equity'] > 1:
        weaknesses.append("Endettement élevé")
    
    if intrinsic_value_metrics['pe_ratio'] > 30:
        weaknesses.append("Valorisation élevée")
    
    if 'earnings_stability_years' in consistency_metrics and consistency_metrics['earnings_stability_years'] < 3:
        weaknesses.append("Bénéfices instables")
    
    # Limiter à 3 forces et 3 faiblesses maximum
    return ", ".join(strengths[:3]), ", ".join(weaknesses[:3])

# Fonction auxiliaire pour compter les années consécutives positives
def count_consecutive_positives(series):
    """
    Compte le nombre d'années consécutives où une valeur est positive.
    
    :param series: Série pandas à analyser
    :return: Nombre d'années consécutives avec valeurs positives
    """
    if series.empty:
        return 0
    
    # Trier les données par date (plus récentes d'abord)
    sorted_series = series.sort_index(ascending=False)
    
    # Compter les valeurs positives consécutives en partant de la plus récente
    count = 0
    for value in sorted_series:
        if value > 0:
            count += 1
        else:
            break
    
    return count

# Fonction pour analyser la consistance d'une série temporelle
def analyze_consistency(series, min_value=None, max_value=None, min_growth=None, min_length=5):
    """
    Analyse la consistance d'une série temporelle financière.
    
    :param series: Série pandas à analyser
    :param min_value: Valeur minimale requise pour toutes les données
    :param max_value: Valeur maximale autorisée pour toutes les données
    :param min_growth: Croissance minimale requise (moyenne)
    :param min_length: Longueur minimale de la série pour une analyse fiable
    :return: Tuple (booléen de succès, description de l'analyse)
    """
    if series.empty or len(series) < min_length:
        return False, f"Données insuffisantes (min. {min_length} ans requis)"
    
    # Trier par ordre chronologique si nécessaire
    series = series.sort_index()
    
    # Analyse différente selon les critères spécifiés
    if min_value is not None:
        # Vérifier si toutes les valeurs sont supérieures au minimum
        all_above_min = all(value >= min_value for value in series)
        avg_value = series.mean()
        if all_above_min:
            return True, f"Consistante et élevée (moy. {avg_value:.2f}%)"
        else:
            pct_above = sum(1 for v in series if v >= min_value) / len(series) * 100
            return False, f"Inconstante ({pct_above:.1f}% au-dessus du min.)"
    
    if max_value is not None:
        # Vérifier si toutes les valeurs sont inférieures au maximum
        all_below_max = all(value <= max_value for value in series)
        avg_value = series.mean()
        if all_below_max:
            return True, f"Constamment faible (moy. {avg_value:.2f})"
        else:
            pct_below = sum(1 for v in series if v <= max_value) / len(series) * 100
            return False, f"Inconstante ({pct_below:.1f}% en-dessous du max.)"
    
    if min_growth is not None:
        # Calculer la croissance moyenne
        avg_growth = series.mean()
        growth_consistency = series.std() / (abs(avg_growth) + 1e-10)  # Éviter division par zéro
        
        if avg_growth >= min_growth and growth_consistency < 2:  # Seuil arbitraire pour la consistance
            return True, f"Croissance consistante (moy. {avg_growth:.2f}%)"
        elif avg_growth >= min_growth:
            return False, f"Croissance volatile (moy. {avg_growth:.2f}%)"
        else:
            return False, f"Croissance insuffisante (moy. {avg_growth:.2f}%)"
    
    # Par défaut, analyse générale de la consistance
    mean_value = series.mean()
    std_value = series.std()
    cv = std_value / (abs(mean_value) + 1e-10)  # Coefficient de variation
    
    if cv < 0.3:  # Seuil arbitraire pour la consistance
        return True, f"Très consistante (CV={cv:.2f})"
    elif cv < 0.5:
        return True, f"Assez consistante (CV={cv:.2f})"
    else:
        return False, f"Inconstante (CV={cv:.2f})"
    
# Interface utilisateur Streamlit
def main():
    from datetime import datetime, timedelta
    # Configuration du style
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
        font-weight: bold;
    }
    .custom-metric {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
    }
    .neutral {
        color: gray;
        font-weight: bold;
    }
    .warning {
        color: orange;
        font-weight: bold;
    }
                
    .sidebar-title {
        margin-top: 15px;
        margin-bottom: 5px;
        color: #333;
        font-weight: bold;
    }
    .block-container {
        padding-top: 0;
    }
    .news-item {
        background-color: #f9f9f9;
        border-left: 3px solid #4e8cff;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    .news-date {
        color: #666;
        font-size: 0.8rem;
    }
    .news-title {
        font-weight: bold;
        margin: 5px 0;
    }
    .news-summary {
        font-size: 0.9rem;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête de l'application
    st.title("📈 Dashboard d'Analyse des Marchés Financiers")
    st.markdown("#### Analyse technique, fondamentale et screening d'actions")
    
    # Récupération de la structure du marché
    with st.spinner("Chargement des marchés...", show_time=True):
        market_structure = get_market_structure()

        # Initialisation des valeurs par défaut dans session_state si c'est la première exécution
    if 'filter_method' not in st.session_state:
        st.session_state.filter_method = "Géographique"

    # Création de la sidebar pour la sélection d'actions
    with st.sidebar:
        st.sidebar.title("🔍 Sélection du Marché")
        
        # Méthode de filtrage
        st.sidebar.markdown('<p class="sidebar-title">Méthode de filtrage</p>', unsafe_allow_html=True)
        filter_method = st.sidebar.radio(
            "Choisissez votre approche",
            ["Géographique", "Sectorielle", "Place de marché", "Liste complète"],
            index=["Géographique", "Sectorielle", "Place de marché", "Liste complète"].index(st.session_state.filter_method),
            captions=["Par région et pays", "Par secteur et industrie", "Par indice boursier", "Vue de toutes les actions"],
            key="filter_method"
        )
        
        # Variables pour stocker les sélections
        selected_stock = None
        flattened_stocks = {}
        
        # Approche géographique
        if filter_method == "Géographique":
            # Recherche des noms exacts pour Europe et France dans la structure de données
            regions = list(market_structure['regions'].keys())
            
            # Sélection de la région
            selected_region = st.sidebar.selectbox(
                "Région", 
                regions, 
                index=regions.index("Europe") if "Europe" in regions else 0, # Sélection de la région Europe par défaut
                key="selected_region"
            )
            
            if selected_region:
                pays_list = list(market_structure['regions'][selected_region].keys())
                
                # Sélection du pays
                selected_pays = st.sidebar.selectbox(
                    "Pays", 
                    pays_list, 
                    index=pays_list.index("France") if "France" in pays_list else 0, # Sélection de la France par défaut
                    key="selected_pays"
                )
                
                # Récupérer les actions filtrées
                flattened_stocks = flatten_market_structure(
                    market_structure, 
                    filter_type='region', 
                    level1=selected_region, 
                    level2=selected_pays
                )
        
        # Approche sectorielle
        elif filter_method == "Sectorielle":
            # Sélection par secteur et industrie
            secteurs = list(market_structure['secteurs'].keys())
            selected_secteur = st.sidebar.selectbox("Secteur", secteurs)
            
            if selected_secteur:
                industries = list(market_structure['secteurs'][selected_secteur].keys())
                selected_industrie = st.sidebar.selectbox("Industrie", industries)
                
                # Récupérer les actions filtrées
                flattened_stocks = flatten_market_structure(
                    market_structure, 
                    filter_type='secteur', 
                    level1=selected_secteur, 
                    level2=selected_industrie
                )
        
        # Approche par marché
        elif filter_method == "Place de marché":
            # Sélection par marché
            marches = list(market_structure['marches'].keys())
            selected_marche = st.sidebar.selectbox("Place de marché", marches)
            
            if selected_marche:
                # Récupérer les actions filtrées
                flattened_stocks = flatten_market_structure(
                    market_structure, 
                    filter_type='marche', 
                    level1=selected_marche
                )
        
        # Liste complète
        else:
            st.sidebar.write("Affichage de toutes les actions disponibles")
            flattened_stocks = market_structure['all_stocks']
        
        # Affichage du nombre d'actions trouvées
        st.sidebar.write(f"**{len(flattened_stocks)} actions** correspondent à vos critères")
        
        # Sélection de l'action
        if flattened_stocks:
            stock_names = list(flattened_stocks.keys())
            stock_names.sort()
            
            # Recherche avec autocomplétion
            selected_stock_name = st.sidebar.selectbox(
                "Sélectionnez une action",
                stock_names,
                format_func=lambda x: f"{x} ({flattened_stocks[x]['ticker']})"
            )
            
            if selected_stock_name:
                selected_stock = flattened_stocks[selected_stock_name]
                st.session_state['selected_stock'] = selected_stock
                st.session_state['selected_stock_name'] = selected_stock_name
                st.session_state['ticker'] = selected_stock['ticker']
                
                # Afficher les informations minimalistes de l'action sélectionnée
                st.sidebar.markdown('<div class="selected-info">', unsafe_allow_html=True)
                st.sidebar.markdown(f"**{selected_stock_name}** ({selected_stock['ticker']})")
                st.sidebar.markdown(f"**Marché:** {selected_stock['marche']}")
                st.sidebar.markdown(f"**Secteur:** {selected_stock['secteur']}")
                st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Gestion des onglets pour le contenu principal
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analyse Technique", "📑 Analyse Fondamentale", "📰 Actualités récentes", "🔎 Screener", "🧩 Gestionnaire de Portefeuille Intelligent"])

    # Onglet 1: Analyse Technique
    with tab1:
        if st.session_state.get('selected_stock') is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = st.session_state['ticker']

            # Titre avec le nom de l'action et son ticker
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Technique: {selected_stock_name} ({ticker})</h2>", unsafe_allow_html=True)

            # Création du layout
            col_chart, col_indicators = st.columns([3, 1])
            
            with col_chart:
                # Section d'options avancées
                with st.expander("Options avancées", expanded=False):
                    opt_col1, opt_col2 = st.columns(2)
                    with opt_col1:
                        # Options pour les indicateurs techniques
                        show_bollinger = st.checkbox("Bandes de Bollinger", value=True)
                        show_ma = st.checkbox("Moyennes mobiles", value=False)
                        if show_ma:
                            ma_periods = st.multiselect("Périodes", [20, 50, 100, 200], default=[50, 200])
                    
                    with opt_col2:
                        # Options pour visualiser la tendance
                        regression_type = st.radio("Type de régression", ["Linéaire", "Logarithmique"], index=0)
                        show_volume = st.checkbox("Afficher le volume", value=True)
                
                # Sélection de la période
                period_cols = st.columns(10)
                periods = {"5j":"5d","1m":"1mo", "3m":"3mo", "6m":"6mo", "1a":"1y", "2a":"2y", "5a":"5y","10a":"10y","YTD":"ytd","Max":"max"}
                
                # Créer des boutons pour chaque période
                selected_period = None
                for i, (period_name, period_value) in enumerate(periods.items()):
                    with period_cols[i]:
                        if st.button(period_name, key=f"period_{period_name}"):
                            selected_period = period_value
                
                # Si aucun bouton n'est cliqué, utiliser la période par défaut
                if selected_period is None:
                    selected_period = "10y"  # Par défaut: 10 ans
                
                # Récupération et préparation des données
                with st.spinner("Chargement des données historiques..."):
                    df = get_stock_history(ticker, period=selected_period)
                    
                    if not df.empty:
                        # Calcul de la régression et des indicateurs
                        df_reg, indicators = calculate_regression(df)
                        
                        # Calcul des moyennes mobiles si demandé
                        if show_ma and 'ma_periods' in locals():
                            for period in ma_periods:
                                df_reg[f'MA_{period}'] = df_reg['Close'].rolling(window=period).mean()
                        
                        # Création du graphique avec Plotly
                        fig = make_subplots(rows=2 if show_volume else 1, cols=1, 
                                            shared_xaxes=True, 
                                            vertical_spacing=0.1, 
                                            row_heights=[0.7, 0.3] if show_volume else [1])
                        
                        # Tracé du prix en chandelier
                        fig.add_trace(
                            go.Candlestick(x=df_reg.index,
                                            open=df_reg['Open'],
                                            high=df_reg['High'],
                                            low=df_reg['Low'],
                                            close=df_reg['Close'],
                                            name="Prix",
                                            increasing_line_color='green',
                                            decreasing_line_color='red'),
                            row=1, col=1
                        )
                        
                        # Tracé de la régression
                        fig.add_trace(
                            go.Scatter(x=df_reg.index, y=df_reg['Regression'],
                                        mode='lines', name='Tendance',
                                        line=dict(color='orange', width=2, dash='solid')),
                            row=1, col=1
                        )
                        
                        # Ajout des moyennes mobiles si demandé
                        if show_ma and 'ma_periods' in locals():
                            colors = ['blue', 'purple', 'darkgreen', 'brown']
                            for i, period in enumerate(ma_periods):
                                if f'MA_{period}' in df_reg.columns:
                                    fig.add_trace(
                                        go.Scatter(x=df_reg.index, y=df_reg[f'MA_{period}'],
                                                    mode='lines', name=f'MA {period}',
                                                    line=dict(color=colors[i % len(colors)], width=1.5)),
                                        row=1, col=1
                                    )
                        
                        # Calcul et tracé des bandes de Bollinger si demandé
                        if show_bollinger:
                            std_dev = df_reg['Deviation'].std()
                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=df_reg['Regression'] + 2*std_dev,
                                            mode='lines', name='+2σ',
                                            line=dict(color='red', width=1, dash='dash')),
                                row=1, col=1
                            )

                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=df_reg['Regression'] + std_dev,
                                            mode='lines', name='+1σ',
                                            line=dict(color='#ffcccc', width=1, dash='dash')),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=df_reg['Regression'] - std_dev,
                                            mode='lines', name='-1σ',
                                            line=dict(color='#ccffcc', width=1, dash='dash')),
                                row=1, col=1
                            )

                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=df_reg['Regression'] - 2*std_dev,
                                            mode='lines', name='-2σ',
                                            line=dict(color='green', width=1, dash='dash')),
                                row=1, col=1
                            )
                        
                        # Tracé du volume si demandé
                        if show_volume:
                            colors = ['red' if row['Open'] > row['Close'] else 'green' for i, row in df_reg.iterrows()]
                            fig.add_trace(
                                go.Bar(x=df_reg.index, y=df_reg['Volume'],
                                        name='Volume', marker_color=colors,
                                        marker_line_width=0),
                                row=2, col=1
                            )
                            
                            # Ajouter une moyenne mobile au volume pour mieux visualiser la tendance
                            vol_ma = df_reg['Volume'].rolling(window=20).mean()
                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=vol_ma,
                                            mode='lines', name='Vol MA(20)',
                                            line=dict(color='blue', width=1)),
                                row=2, col=1
                            )
                        
                        # Mise en forme du graphique
                        fig.update_layout(
                            title=None,  # Le titre est déjà en haut de la page
                            height=500,
                            xaxis_rangeslider_visible=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor='rgba(245, 245, 245, 0.5)'
                        )
                        
                        # Améliorer l'apparence des axes
                        fig.update_xaxes(
                            showgrid=True,
                            gridwidth=0.5,
                            gridcolor='rgba(211, 211, 211, 0.5)'
                        )
                        
                        fig.update_yaxes(
                            showgrid=True,
                            gridwidth=0.5,
                            gridcolor='rgba(211, 211, 211, 0.5)',
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='rgba(0, 0, 0, 0.2)'
                        )
                        
                        # Affichage du graphique
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Ajouter une section d'analyse de tendance
                        with st.expander("Analyse détaillée de la tendance", expanded=True):
                            trend_cols = st.columns(3)
                            with trend_cols[0]:
                                # Analyser la direction de la tendance récente (30 derniers jours)
                                recent_prices = df_reg['Close'].iloc[-30:] if len(df_reg) >= 30 else df_reg['Close']
                                recent_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                                recent_direction = "Haussière" if recent_slope > 0 else "Baissière"
                                recent_color = "green" if recent_slope > 0 else "red"
                                
                                st.markdown(f"""
                                #### Tendance récente (30 derniers jours)
                                <span style='color:{recent_color}; font-weight:bold'>{recent_direction}</span>
                                """, unsafe_allow_html=True)
                                
                                # Calculer la volatilité (écart-type des rendements quotidiens)
                                if len(df_reg) > 1:
                                    returns = df_reg['Close'].pct_change().dropna()
                                    volatility = returns.std() * 100  # en pourcentage
                                    st.markdown(f"**Volatilité:** {volatility:.2f}% (quotidienne)")
                            
                            with trend_cols[1]:
                                # Analyser les résistances et supports
                                if len(df_reg) > 30:
                                    recent_high = df_reg['High'].iloc[-30:].max()
                                    recent_low = df_reg['Low'].iloc[-30:].min()
                                    current = df_reg['Close'].iloc[-1]
                                    
                                    st.markdown(f"""
                                    #### Supports/Résistances
                                    **Résistance récente:** {recent_high:.2f}
                                    **Support récent:** {recent_low:.2f}
                                    """)
                                
                                    # Calculer la distance aux support/résistance
                                    dist_to_resistance = ((recent_high / current) - 1) * 100
                                    dist_to_support = (1 - (recent_low / current)) * 100
                                    
                                    # Potentiel de retournement
                                    reversal_potential = "Élevé" if abs(indicators['deviation']) >= 2 else "Moyen" if abs(indicators['deviation']) >= 1 else "Faible"
                                    st.markdown(f"**Potentiel de retournement:** {reversal_potential}")
                                    
                            with trend_cols[2]:
                                # Performances comparées
                                st.markdown("#### Performances")
                                
                                if 'prog_1y' in indicators and 'prog_5y' in indicators:
                                    prog_1y = indicators['prog_1y']
                                    prog_5y = indicators['prog_5y']
                                    
                                    # Comparer à un benchmark (exemple: marché global à 8% annualisé)
                                    benchmark_1y = 8.0
                                    benchmark_5y = 8.0 * 5
                                    
                                    st.markdown(f"""
                                    **1 an:** {prog_1y:.2f}% ({prog_1y - benchmark_1y:+.2f}% vs marché)
                                    **5 ans:** {prog_5y:.2f}% ({prog_5y - benchmark_5y:+.2f}% vs marché)
                                    """)
                    else:
                        st.error(f"Aucune donnée historique disponible pour {ticker}")
            
            with col_indicators:
                # Affichage indicateurs techniques
                st.markdown("<div style='background-color: #f8f9fa; padding: 0px; border-radius: 10px; margin-bottom: 20px;'><h3 style='text-align: center; margin-bottom: 15px;'>Indicateurs Techniques</h3></div>", unsafe_allow_html=True)
                
                # Prix actuel avec tendance
                current_price = df['Close'].iloc[-1] if not df.empty else None
                previous_price = df['Close'].iloc[-2] if not df.empty and len(df) > 1 else None
                
                # Utiliser la fonction améliorée pour afficher le prix actuel
                display_price_with_trend(
                    "Prix actuel",
                    current_price,
                    previous_price,
                    format_func=lambda x: f"{x:.2f} €",
                    background=True,
                    size="large"
                )
                
                # Jauges pour les indicateurs clés
                if 'indicators' in locals() and indicators:
                    # Jauge pour la tendance (correlation)
                    correlation = indicators['correlation']
                    fig_trend = create_gauge(
                        value=correlation,
                        title="Tendance",
                        min_val=-1,
                        max_val=1,
                        threshold_values=[-0.7, 0.7],
                        threshold_labels=["Baissière", "Neutre", "Haussière"]
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Jauge pour la croissance du modèle
                    model_growth = indicators['model_growth']
                    fig_growth = create_gauge(
                        value=model_growth,
                        title="Croissance Annuelle (%)",
                        min_val=-20,
                        max_val=20,
                        threshold_values=[0, 5],
                        threshold_labels=["Négative", "Faible", "Forte"]
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)
                    
                    # Jauge pour l'écart par rapport à la régression
                    deviation = indicators['deviation']
                    fig_deviation = create_gauge(
                        value=deviation,
                        title="Écart (σ)",
                        min_val=-3,
                        max_val=3,
                        threshold_values=[-1.5, 1.5],
                        threshold_labels=["Surachat", "Zone neutre", "Survente"]
                    )
                    st.plotly_chart(fig_deviation, use_container_width=True)
                
                # Ajouter un indicateur de force du momentum
                if not df.empty and len(df) > 10:
                    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
                    
                    st.markdown("<h4>Force du Momentum</h4>", unsafe_allow_html=True)
                    
                    # Définir la couleur en fonction de la valeur
                    momentum_color = "green" if momentum > 3 else "red" if momentum < -3 else "orange"
                    momentum_strength = "Fort" if abs(momentum) > 5 else "Modéré" if abs(momentum) > 2 else "Faible"
                    
                    # Créer un bargraph horizontal pour le momentum
                    momentum_chart = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = momentum,
                        number = {'suffix': "%", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [-10, 10], 'tickwidth': 1},
                            'bar': {'color': momentum_color},
                            'steps': [
                                {'range': [-10, -3], 'color': 'rgba(255, 0, 0, 0.2)'},
                                {'range': [-3, 3], 'color': 'rgba(255, 165, 0, 0.2)'},
                                {'range': [3, 10], 'color': 'rgba(0, 128, 0, 0.2)'}
                            ]
                        },
                        title = {'text': f"{momentum_strength}"}
                    ))
                    
                    momentum_chart.update_layout(
                        height=150,
                        margin=dict(l=10, r=10, t=20, b=10)
                    )
                    
                    st.plotly_chart(momentum_chart, use_container_width=True)
                
                # Ajouter un résumé technique
                st.markdown("<h4>Résumé Technique</h4>", unsafe_allow_html=True)
                
                if 'indicators' in locals() and indicators:
                    # Générer un résumé technique basé sur les indicateurs
                    if indicators['correlation'] > 0.7 and indicators['model_growth'] > 5:
                        signal = "Achat"
                        signal_color = "green"
                    elif indicators['correlation'] < -0.7 and indicators['model_growth'] < -5:
                        signal = "Vente"
                        signal_color = "red"
                    else:
                        signal = "Neutre"
                        signal_color = "orange"
                    
                    st.markdown(f"""
                    <div style='background-color: rgba({', '.join(['0, 128, 0, 0.2' if signal_color == 'green' else '255, 0, 0, 0.2' if signal_color == 'red' else '255, 165, 0, 0.2'])}); padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;'>
                        <h3 style='color: {signal_color}; margin: 0;'>{signal}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Liste des signaux clés
                    signals = []
                    
                    if indicators['correlation'] > 0.5:
                        signals.append("✅ Tendance haussière solide")
                    elif indicators['correlation'] < -0.5:
                        signals.append("❌ Tendance baissière solide")
                    
                    if indicators['deviation'] > 1.5:
                        signals.append("⚠️ Surachat potentiel")
                    elif indicators['deviation'] < -1.5:
                        signals.append("⚠️ Survente potentielle")
                    
                    # Vérifier le croisement des moyennes mobiles si disponibles
                    if 'ma_periods' in locals() and show_ma and len(df_reg) > max(ma_periods):
                        short_ma = 50
                        long_ma = 200
                        if f'MA_{short_ma}' in df_reg.columns and f'MA_{long_ma}' in df_reg.columns:
                            # Vérifier s'il y a un croisement récent (dans les 5 derniers jours)
                            last_days = min(5, len(df_reg) - 1)
                            for i in range(1, last_days + 1):
                                current_short = df_reg[f'MA_{short_ma}'].iloc[-i]
                                current_long = df_reg[f'MA_{long_ma}'].iloc[-i]
                                prev_short = df_reg[f'MA_{short_ma}'].iloc[-(i+1)] if i < len(df_reg) - 1 else None
                                prev_long = df_reg[f'MA_{long_ma}'].iloc[-(i+1)] if i < len(df_reg) - 1 else None
                                
                                if prev_short and prev_long:
                                    # Croisement doré (court au-dessus du long)
                                    if prev_short <= prev_long and current_short > current_long:
                                        signals.append("✅ Croisement doré récent (signal d'achat)")
                                        break
                                    # Croisement de la mort (court en-dessous du long)
                                    elif prev_short >= prev_long and current_short < current_long:
                                        signals.append("❌ Croisement de la mort récent (signal de vente)")
                                        break
                    
                    # Vérifier le momentum
                    if 'momentum' in locals():
                        if momentum > 5:
                            signals.append("✅ Momentum fortement positif")
                        elif momentum < -5:
                            signals.append("❌ Momentum fortement négatif")
                    
                    # Vérifier la volatilité
                    if 'volatility' in locals():
                        market_avg_volatility = 1.5  # Volatilité moyenne du marché (exemple)
                        if volatility > market_avg_volatility * 2:
                            signals.append("⚠️ Volatilité très élevée")
                        elif volatility < market_avg_volatility * 0.5:
                            signals.append("ℹ️ Volatilité très faible")
                    
                    # Affichage des signaux
                    if signals:
                        for signal_text in signals:
                            st.markdown(f"<div style='margin: 5px 0;'>{signal_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("Aucun signal technique significatif")
                    
                    # Ajouter un résumé textuel
                    if indicators['correlation'] > 0.7:
                        if indicators['deviation'] > 1.5:
                            summary = "L'action est en tendance haussière forte mais potentiellement surachetée. Une correction technique pourrait survenir à court terme."
                        elif indicators['deviation'] < -1.5:
                            summary = "L'action est en tendance haussière forte et actuellement sous-évaluée. Bon point d'entrée potentiel."
                        else:
                            summary = "L'action est en tendance haussière forte et évolue dans sa zone de valeur normale."
                    elif indicators['correlation'] < -0.7:
                        if indicators['deviation'] > 1.5:
                            summary = "L'action est en tendance baissière forte mais temporairement surachetée. La tendance baissière pourrait reprendre après ce rebond technique."
                        elif indicators['deviation'] < -1.5:
                            summary = "L'action est en tendance baissière forte et actuellement survendue. Un rebond technique pourrait survenir à court terme."
                        else:
                            summary = "L'action est en tendance baissière forte et évolue dans sa zone de valeur normale."
                    else:
                        if indicators['deviation'] > 1.5:
                            summary = "L'action est en phase de consolidation avec un potentiel de correction à la baisse."
                        elif indicators['deviation'] < -1.5:
                            summary = "L'action est en phase de consolidation avec un potentiel de rebond à la hausse."
                        else:
                            summary = "L'action évolue sans tendance claire dans sa zone de valeur normale."
                    
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 15px;'>
                        <p style='margin: 0;'><em>{summary}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Données insuffisantes pour l'analyse technique")

    # Onglet 2: Analyse Fondamentale avec Projection et Recommandation
    with tab2:
        if st.session_state['selected_stock'] is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = selected_stock['ticker']
            sector = selected_stock['secteur']

            st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Fondamentale: {selected_stock_name} ({ticker})</h2>", unsafe_allow_html=True)
            
            # Création des sous-onglets pour l'analyse fondamentale
            fundamental_tab1, fundamental_tab2 = st.tabs(["📋 Profil & Données Financières", "🔮 Évaluation & Recommandation"])
            
            # Récupération des données fondamentales
            with st.spinner("Chargement des données fondamentales..."):
                fundamental_data = get_fundamental_data(ticker)
                income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
            
            # Sous-onglet 1: Profil et Données Financières
            with fundamental_tab1:
                if fundamental_data:
                    # Disposition en colonnes
                    col1, col2 = st.columns([1, 1])
                    
                    # Colonne 1: Informations générales et données de marché
                    with col1:
                        # Informations générales
                        st.subheader("📝 Profil de l'entreprise")
                        general_data = fundamental_data["Données générales"]
                        
                        # Affichage du nom, secteur, industrie et pays
                        st.markdown(f"**{general_data.get('Nom', 'N/A')}**")
                        st.markdown(f"**Secteur:** {general_data.get('Secteur', 'N/A')} | **Industrie:** {general_data.get('Industrie', 'N/A')} | **Pays:** {general_data.get('Pays', 'N/A')}")
                        
                        # Site web
                        if general_data.get('Site web'):
                            st.markdown(f"**Site web:** [{general_data.get('Site web')}]({general_data.get('Site web')})")
                        
                        # Description avec expander pour ne pas prendre trop de place
                        if general_data.get('Description'):
                            with st.expander("Description de l'entreprise"):
                                st.write(general_data.get('Description'))
                        
                        # Données de marché
                        st.subheader("📈 Données de marché")
                        market_data = fundamental_data["Données de marché"]
                        
                        # Prix actuel avec flèche de tendance
                        current_price = market_data.get('Prix actuel')
                        previous_close = market_data.get('Précédente clôture')
                        if current_price and previous_close:
                            change = ((current_price - previous_close) / previous_close) * 100
                            arrow = "↑" if change >= 0 else "↓"
                            color = "green" if change >= 0 else "red"
                            st.markdown(f"**Prix actuel:** {format_number(current_price)} {arrow} <span style='color:{color};'>({change:.2f}%)</span>", unsafe_allow_html=True)
                        
                        # Afficher les principales données de marché en colonnes
                        market_col1, market_col2 = st.columns(2)
                        
                        with market_col1:
                            st.markdown("**Volume:** " + format_number(market_data.get('Volume', 'N/A')))
                            st.markdown("**52 semaines - Haut:** " + format_number(market_data.get('52 semaines - Haut', 'N/A')))
                            st.markdown("**Capitalisation boursière:** " + format_number(market_data.get('Capitalisation boursière', 'N/A')))
                            st.markdown("**Actions en circulation:** " + format_number(market_data.get('Actions ordinaires', 'N/A')))
                        
                        with market_col2:
                            st.markdown("**Volume moyen:** " + format_number(market_data.get('Volume moyen', 'N/A')))
                            st.markdown("**52 semaines - Bas:** " + format_number(market_data.get('52 semaines - Bas', 'N/A')))
                            st.markdown("**Beta:** " + format_number(market_data.get('Beta', 'N/A')))
                            st.markdown("**Recommandation:** " + str(market_data.get('Recommandation', 'N/A')))
                    
                    # Colonne 2: Ratios fondamentaux et données financières
                    with col2:
                        # Ratios fondamentaux
                        st.subheader("📊 Ratios fondamentaux")
                        fundamental_metrics = fundamental_data["Données fondamentales"]
                        
                        # Afficher les principaux ratios en colonnes
                        ratios_col1, ratios_col2 = st.columns(2)
                        
                        with ratios_col1:
                            st.markdown("**PER:** " + format_number(fundamental_metrics.get('PER', 'N/A')))
                            st.markdown("**PEG Ratio:** " + format_number(fundamental_metrics.get('PEG Ratio', 'N/A')))
                            st.markdown("**P/B (Cours/Valeur Comptable):** " + format_number(fundamental_metrics.get('P/B Cours/Valeur Comptable', 'N/A')))
                            st.markdown("**P/S (Price to Sales):** " + format_number(fundamental_metrics.get('P/S (Price to Sales)', 'N/A')))
                            st.markdown("**Marge nette (%):** " + format_number(fundamental_metrics.get('Marge nette (%)', 'N/A')))
                        
                        with ratios_col2:
                            st.markdown("**ROE (%):** " + format_number(fundamental_metrics.get('ROE (%)', 'N/A')))
                            st.markdown("**ROA (%):** " + format_number(fundamental_metrics.get('ROA (%)', 'N/A')))
                            st.markdown("**Ratio d'endettement:** " + format_number(fundamental_metrics.get('Ratio d\'endettement', 'N/A')))
                            st.markdown("**Dividende:** " + format_number(fundamental_metrics.get('Dividende', 'N/A')))
                            st.markdown("**Rendement du dividende (%):** " + format_number(fundamental_metrics.get('Rendement du dividende (%)', 'N/A')))
                        
                        # Données financières
                        st.subheader("💰 Données financières")
                        financial_data = fundamental_data["Données financières"]
                        
                        # Afficher les principales données financières en colonnes
                        financial_col1, financial_col2 = st.columns(2)
                        
                        with financial_col1:
                            st.markdown("**Chiffre d'affaires:** " + format_number(financial_data.get('Chiffre d\'affaires', 'N/A')))
                            st.markdown("**Résultat net:** " + format_number(financial_data.get('Résultat net', 'N/A')))
                            st.markdown("**EBITDA:** " + format_number(financial_data.get('EBITDA', 'N/A')))
                            st.markdown("**BPA:** " + format_number(fundamental_metrics.get('BPA', 'N/A')))
                        with financial_col2:
                            st.markdown("**Total Actif:** " + format_number(financial_data.get('Total Actif', 'N/A')))
                            st.markdown("**Total Dette:** " + format_number(financial_data.get('Total Dette', 'N/A')))
                            st.markdown("**Fonds propres:** " + format_number(financial_data.get('Fonds propres', 'N/A')))
                            st.markdown("**Free Cash Flow:** " + format_number(financial_data.get('Free Cash Flow', 'N/A')))
                    
                    # Section pour les états financiers historiques
                    st.subheader("📜 États financiers historiques")
                    financial_tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie"])
                    
                    with financial_tabs[0]:
                        if not income_stmt.empty:
                            # Convertir les montants en millions pour une meilleure lisibilité
                            income_display = income_stmt.copy() / 1e6
                            st.dataframe(income_display.style.format("{:.2f}"))
                            st.caption("Montants en millions")
                        else:
                            st.info("Données du compte de résultat non disponibles")
                    
                    with financial_tabs[1]:
                        if not balance_sheet.empty:
                            # Convertir les montants en millions pour une meilleure lisibilité
                            balance_display = balance_sheet.copy() / 1e6
                            st.dataframe(balance_display.style.format("{:.2f}"))
                            st.caption("Montants en millions")
                        else:
                            st.info("Données du bilan non disponibles")
                    
                    with financial_tabs[2]:
                        if not cashflow.empty:
                            # Convertir les montants en millions pour une meilleure lisibilité
                            cashflow_display = cashflow.copy() / 1e6
                            st.dataframe(cashflow_display.style.format("{:.2f}"))
                            st.caption("Montants en millions")
                        else:
                            st.info("Données des flux de trésorerie non disponibles")

                else:
                    st.error(f"Impossible de récupérer les données fondamentales pour {ticker}")
            
            # Sous-onglet 2: Évaluation et Recommandation
            with fundamental_tab2:

                # Créer des onglets pour différentes analyses
                valuation_tabs = st.tabs(["Valorisation intrinsèque", "Méthodes de valorisation", "Avantage compétitif", "Dividendes"])
                
                # Onglet 1: Valorisation intrinsèque
                with valuation_tabs[0]:
                    st.write("#### Calcul de la valeur intrinsèque")
                    
                    # Calculer la valeur intrinsèque
                    intrinsic_value_result = calculate_intrinsic_value(
                        ticker, 
                        fundamental_data, 
                        (income_stmt, balance_sheet, cashflow)
                    )
                    
                    if intrinsic_value_result and intrinsic_value_result.get('success', False):
                        # Afficher les résultats principaux
                        iv_col1, iv_col2, iv_col3 = st.columns(3)
                        
                        with iv_col1:
                            intrinsic_value = intrinsic_value_result.get('intrinsic_value', None)
                            if intrinsic_value:
                                value_color = "green" if intrinsic_value > current_price else "red"
                                st.markdown(f"**Valeur intrinsèque:** <span style='color:{value_color};'>{intrinsic_value:.2f} €</span>", unsafe_allow_html=True)
                        
                        with iv_col2:
                            mos_value = intrinsic_value_result.get('intrinsic_value_with_mos', None)
                            if mos_value:
                                mos_color = "green" if mos_value > current_price else "orange"
                                st.markdown(f"**Avec marge de sécurité:** <span style='color:{mos_color};'>{mos_value:.2f} €</span>", unsafe_allow_html=True)
                        
                        with iv_col3:
                            deviation = intrinsic_value_result.get('deviation', None)
                            if deviation is not None:
                                dev_color = "green" if deviation < 0 else "red"
                                st.markdown(f"**Écart de valorisation:** <span style='color:{dev_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
                        
                        # Afficher la recommandation
                        recommendation = intrinsic_value_result.get('recommendation', '')
                        rec_colors = {
                            "ACHAT FORT": "darkgreen",
                            "ACHAT": "green",
                            "ACCUMULATION": "lightgreen",
                            "CONSERVER": "orange",
                            "ALLÈGEMENT": "pink",
                            "VENTE": "red"
                        }
                        
                        st.markdown(f"**Recommandation:** <span style='color:{rec_colors.get(recommendation, 'gray')};font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
                        
                        # Visualisation comparative
                        fig = go.Figure()
                        
                        # Barres pour le prix actuel et les valeurs intrinsèques
                        fig.add_trace(go.Bar(
                            x=['Prix actuel'],
                            y=[current_price],
                            name='Prix actuel',
                            marker_color='blue'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=['Valeur intrinsèque'],
                            y=[intrinsic_value],
                            name='Valeur intrinsèque',
                            marker_color='green' if intrinsic_value > current_price else 'red'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=['Valeur avec MOS'],
                            y=[mos_value],
                            name='Valeur avec marge de sécurité',
                            marker_color='green' if mos_value > current_price else 'orange'
                        ))
                        
                        fig.update_layout(
                            title='Comparaison des valorisations',
                            xaxis_title='Type de valorisation',
                            yaxis_title='Valeur (€)',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Afficher les méthodes utilisées
                        st.write("##### Détail des méthodes de valorisation")
                        methods = intrinsic_value_result.get('methods', {})
                        
                        method_df = pd.DataFrame({
                            'Méthode': ['DCF','Graham', 'PER', 'Patrimoniale'],
                            'Valeur estimée': [
                                format_number(methods.get('dcf', {}).get('value', None)),
                                format_number(methods.get('graham', {}).get('value', None)),
                                format_number(methods.get('per_based', {}).get('value', None)),
                                format_number(methods.get('asset_based', {}).get('value', None))
                            ],
                            'Potentiel': [
                                format_number(((methods.get('dcf', {}).get('value', None) / current_price) - 1) * 100),
                                format_number(((methods.get('graham', {}).get('value', None)/ current_price) - 1) * 100),
                                format_number(((methods.get('per_based', {}).get('value', None)/ current_price) - 1) * 100),
                                format_number(((methods.get('asset_based', {}).get('value', None)/ current_price) - 1)*100)
                            ],
                            'Poids': [
                                methods.get('dcf', {}).get('weight', 0),
                                methods.get('graham', {}).get('weight', 0),
                                methods.get('per_based', {}).get('weight', 0),
                                methods.get('asset_based', {}).get('weight', 0)
                            ]
                        })
                        
                        # Filtrer les méthodes avec des valeurs non nulles
                        method_df = method_df[method_df['Valeur estimée'].notna()]

                        # Formater les colonnes
                        method_df['Valeur estimée'] = method_df['Valeur estimée'].apply(lambda x: f"{x} €")
                        method_df['Potentiel'] = method_df['Potentiel'].apply(lambda x: f"{x}%")
                        method_df['Poids'] = method_df['Poids'].apply(lambda x: f"{x:.1%}")
                        
                        # Fonction pour coloriser les potentiels
                        def color_potential(val):
                            if "%" not in val:
                                return ""
                            val_num = float(val.strip('%'))
                            return f'color: {"green" if val_num > 0 else "red"}'
                        
                        # Appliquer la coloration et afficher
                        styled_methods_df = method_df.style.map(color_potential)
                        st.dataframe(styled_methods_df, use_container_width=True)
                        
                        # Afficher les facteurs qualitatifs
                        quality_factors = intrinsic_value_result.get('quality_factors', {})
                        if quality_factors:
                            st.write("##### Facteurs qualitatifs")
                            
                            qf_col1, qf_col2 = st.columns(2)
                            
                            with qf_col1:
                                moat = quality_factors.get('moat', 'Indéterminé')
                                moat_color = {
                                    'Fort': 'green',
                                    'Moyen': 'orange',
                                    'Faible': 'red',
                                    'Indéterminé': 'gray'
                                }.get(moat, 'gray')
                                
                                st.markdown(f"**Avantage compétitif:** <span style='color:{moat_color};'>{moat}</span>", unsafe_allow_html=True)
                            
                            with qf_col2:
                                management = quality_factors.get('management', 'Indéterminé')
                                mgmt_color = {
                                    'Excellent': 'green',
                                    'Bon': 'lightgreen',
                                    'Moyen': 'orange',
                                    'Médiocre': 'red',
                                    'Indéterminé': 'gray'
                                }.get(management, 'gray')
                                
                                st.markdown(f"**Qualité du management:** <span style='color:{mgmt_color};'>{management}</span>", unsafe_allow_html=True)
                            
                            adjustment = quality_factors.get('adjustment', 1.0)
                            st.write(f"**Ajustement qualitatif appliqué:** {adjustment:.1%}")
                    else:
                        st.error(f"Impossible de calculer la valeur intrinsèque pour {ticker}")
                        if intrinsic_value_result:
                            st.write(intrinsic_value_result.get('message', ''))

                # Onglet 2: Méthodes de valorisation
                with valuation_tabs[1]:
                    st.write("#### Détail des méthodes de valorisation")
                    
                    # Calculer la valeur intrinsèque avec toutes les méthodes
                    valuation_result = calculate_intrinsic_value(
                        ticker, 
                        fundamental_data, 
                        historical_financials=(income_stmt, balance_sheet, cashflow)
                    )
                    
                    if not valuation_result['success']:
                        st.error(f"Erreur de valorisation: {valuation_result['message']}")
                    else:
                        # Créer des onglets pour chaque méthode
                        method_tabs = st.tabs(["DCF", "Graham", "PER", "Valeur patrimoniale"])
                        
                        # Extraire les résultats des différentes méthodes
                        dcf_value = valuation_result['methods']['dcf']['value']
                        graham_value = valuation_result['methods']['graham']['value']
                        netnet_value = valuation_result['methods']['graham'].get('netnet_value')
                        per_based_value = valuation_result['methods']['per_based']['value']
                        asset_based_value = valuation_result['methods']['asset_based']['value']
                        current_price = valuation_result['current_price']
                        
                        # Sous-onglet DCF
                        with method_tabs[0]:
                            st.write("##### Modèle DCF (Discounted Cash Flow)")
                            
                            if dcf_value:
                                dcf_color = "green" if dcf_value > current_price else "red"
                                st.markdown(f"**Valeur DCF calculée:** <span style='color:{dcf_color};font-weight:bold;'>{dcf_value:.2f} €</span>", unsafe_allow_html=True)
                                
                                # Afficher le potentiel d'appréciation/dépréciation
                                potential = ((dcf_value / current_price) - 1) * 100
                                potential_color = "green" if potential > 0 else "red"
                                st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
                                
                                # Afficher les FCF historiques si disponibles
                                if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                                    # Extraction des FCF historiques
                                    fcf_data = cashflow.loc['Free Cash Flow'].to_dict()
                                    
                                    # Création d'un DataFrame pour le graphique
                                    fcf_df = pd.DataFrame({
                                        'Année': [str(year.year) for year in fcf_data.keys()],
                                        'FCF (millions)': [float(value) / 1e6 for value in fcf_data.values()]
                                    })
                                    
                                    # Tri par année
                                    fcf_df = fcf_df.sort_values('Année')
                                    
                                    # Calcul de la croissance annuelle moyenne
                                    cagr_text = ""
                                    if len(fcf_df) > 1:
                                        fcf_values = fcf_df['FCF (millions)'].values
                                        positive_values_only = fcf_values[fcf_values > 0]
                                        
                                        if len(positive_values_only) > 1:
                                            # Calcul du taux de croissance annuel moyen (CAGR)
                                            start_value = positive_values_only[0]
                                            end_value = positive_values_only[-1]
                                            years = len(positive_values_only) - 1
                                            cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
                                            cagr_text = f"Croissance annuelle moyenne: {cagr:.2f}%"
                                    
                                    # Création du graphique
                                    fig = go.Figure()
                                    
                                    # Ajout des barres pour les FCF
                                    fig.add_trace(go.Bar(
                                        x=fcf_df['Année'],
                                        y=fcf_df['FCF (millions)'],
                                        marker_color=['green' if fcf >= 0 else 'red' for fcf in fcf_df['FCF (millions)']],
                                        text=[f"€{fcf:.2f}M" for fcf in fcf_df['FCF (millions)']],
                                        textposition='outside',
                                        name='Free Cash Flow'
                                    ))
                                    
                                    # Ajout d'une ligne de tendance
                                    if len(fcf_df) > 1:
                                        # Régression linéaire simple
                                        x = np.arange(len(fcf_df))
                                        y = fcf_df['FCF (millions)'].values
                                        model = LinearRegression().fit(x.reshape(-1, 1), y)
                                        line_y = model.predict(x.reshape(-1, 1))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=fcf_df['Année'],
                                            y=line_y,
                                            mode='lines',
                                            line=dict(color='blue', width=2, dash='dash'),
                                            name='Tendance'
                                        ))
                                    
                                    # Configuration du graphique
                                    fig.update_layout(
                                        title=f"💸 Évolution des Free Cash Flows<br><sup>{cagr_text}</sup>",
                                        xaxis_title="Année",
                                        yaxis_title="FCF (millions)",
                                        height=400,
                                        margin=dict(l=50, r=50, t=80, b=50),
                                        hovermode="x unified"
                                    )
                                    
                                    # Affichage du graphique
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Impossible de calculer la valeur DCF pour cette entreprise en raison de données insuffisantes.")
                        
                        # Sous-onglet Graham
                        with method_tabs[1]:
                            st.write("##### Méthode de Graham (Valeur Intrinsèque)")
                            
                            eps = fundamental_data['Données fondamentales'].get('BPA', None)
                            
                            if graham_value and eps:
                                # Afficher la valeur de Graham
                                graham_color = "green" if graham_value > current_price else "red"
                                st.markdown(f"**Valeur selon Graham:** <span style='color:{graham_color};font-weight:bold;'>{graham_value:.2f} €</span>", unsafe_allow_html=True)
                                
                                # Écart avec le prix actuel
                                deviation = ((graham_value / current_price) - 1) * 100
                                deviation_color = "green" if deviation > 0 else "red"
                                st.markdown(f"**Potentiel:** <span style='color:{deviation_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
                                
                                # Taux de croissance estimé (récupéré du calcul de valuation_result)
                                # Par défaut 5% si non spécifié
                                growth_estimate = 0.05
                                if not income_stmt.empty and len(income_stmt.columns) >= 3 and 'Net Income' in income_stmt.index:
                                    net_incomes = []
                                    for col in income_stmt.columns[:3]:
                                        if 'Net Income' in income_stmt.index:
                                            net_incomes.append(income_stmt.loc['Net Income', col])
                                    
                                    if len(net_incomes) == 3 and all(ni is not None and ni > 0 for ni in net_incomes):
                                        cagr = ((net_incomes[0] / net_incomes[2]) ** (1/2)) - 1
                                        growth_estimate = min(max(cagr, 0.02), 0.15)
                                
                                # Afficher la formule utilisée
                                bond_yield = 0.04  # 4% pour la France
                                st.write(f"**Formule utilisée:** V = {eps:.2f} € * (8.5 + 2 * {growth_estimate*100:.1f}%) * 4.4 / {bond_yield*100:.1f}%")
                                
                                # Afficher la valeur Net-Net si disponible
                                if netnet_value is not None and current_price > 0:
                                    netnet_color = "green" if netnet_value > current_price else "red"
                                    st.markdown(f"**Valeur Net-Net par action:** <span style='color:{netnet_color};'>{netnet_value:.2f} €</span>", unsafe_allow_html=True)
                                    
                                    # Expliquer ce qu'est la valeur Net-Net
                                    with st.expander("Qu'est-ce que la valeur Net-Net ?"):
                                        st.write("""
                                        La méthode Net-Net de Benjamin Graham évalue une entreprise uniquement sur la base de ses actifs courants moins ses passifs totaux.
                                        C'est une approche très conservatrice qui cherche des entreprises négociées en dessous de leur valeur liquidative.
                                        Une action négociée en dessous de sa valeur Net-Net peut représenter une opportunité intéressante, mais cela peut aussi indiquer
                                        des problèmes fondamentaux dans l'entreprise.
                                        """)
                            else:
                                st.warning("Impossible de calculer la valeur de Graham en raison d'un BPA (EPS) négatif ou non disponible.")
                        
                        # Sous-onglet PER
                        with method_tabs[2]:
                            st.write("##### Valorisation basée sur le PER (Price Earnings Ratio)")
                            
                            per = fundamental_data['Données fondamentales'].get('PER', None)
                            per_fwd = fundamental_data['Données fondamentales'].get('PER (Forward)', None)
                            eps = fundamental_data['Données fondamentales'].get('BPA', None)
                            
                            if per is not None and eps is not None and eps > 0 and per_based_value:
                                # Récupérer le PER sectoriel
                                sector_per, sector = get_sector_pe_ratio(ticker)
                                
                                # Afficher le PER actuel
                                st.markdown(f"**PER actuel:** {per:.2f}x  |  **PER sectoriel estimé:** {sector_per:.1f}x", unsafe_allow_html=True)
                                
                                # Comparer le PER avec celui du secteur
                                per_ratio = per / sector_per
                                per_ratio_color = "green" if per_ratio < 1 else ("orange" if per_ratio < 1.2 else "red")
                                st.markdown(f"**Ratio PER/PER sectoriel:** <span style='color:{per_ratio_color};'>{per_ratio:.2f}x</span>", unsafe_allow_html=True)
                                
                                # Calculer différentes valorisations basées sur des PER cibles
                                st.write("##### Valorisations selon différents PER cibles")
                                
                                per_targets = {
                                    "PER sectoriel": sector_per,
                                    "PER futur estimé": per_fwd if per_fwd else per,
                                    "PER historique bas": max(per * 0.8, sector_per * 0.7),
                                    "PER historique haut": per * 1.2
                                }
                                
                                per_values = []
                                for label, per_target in per_targets.items():
                                    value = eps * per_target
                                    per_values.append({
                                        "Scénario": label,
                                        "PER cible": f"{per_target:.1f}x",
                                        "Valorisation": f"{value:.2f} €",
                                        "Potentiel": f"{((value / current_price) - 1) * 100:.1f}%"
                                    })
                                
                                per_df = pd.DataFrame(per_values)
                                
                                # Coloriser le tableau
                                def color_potential(val):
                                    val_num = float(val.strip('%'))
                                    return f'color: {"green" if val_num > 0 else "red"}'
                                
                                styled_df = per_df.style.map(color_potential, subset=['Potentiel'])
                                st.dataframe(styled_df, use_container_width=True)
                                
                                # Visualisation graphique des différentes valorisations basées sur le PER
                                fig = go.Figure()
                                
                                # Ajouter une ligne pour le prix actuel
                                fig.add_shape(
                                    type="line",
                                    x0=-0.5,
                                    y0=current_price,
                                    x1=3.5,
                                    y1=current_price,
                                    line=dict(color="blue", width=2, dash="dash"),
                                )
                                
                                # Ajouter les barres pour chaque scénario de PER
                                fig.add_trace(go.Bar(
                                    x=[scenario["Scénario"] for scenario in per_values],
                                    y=[float(scenario["Valorisation"].split()[0]) for scenario in per_values],
                                    marker_color=[
                                        'green' if float(scenario["Potentiel"].strip('%')) > 0 else 'red'
                                        for scenario in per_values
                                    ],
                                    text=[scenario["Potentiel"] for scenario in per_values],
                                    textposition='auto'
                                ))
                                
                                fig.update_layout(
                                    title='Valorisations selon différents PER',
                                    xaxis_title='Scénario de PER',
                                    yaxis_title='Valorisation (€)',
                                    annotations=[
                                        dict(
                                            x=-0.2,
                                            y=current_price * 1.02,
                                            text=f"Prix actuel: {current_price:.2f} €",
                                            showarrow=False,
                                            bgcolor="blue",
                                            font=dict(color="white")
                                        )
                                    ]
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Données insuffisantes pour calculer la valorisation basée sur le PER.")
                        
                        # Sous-onglet Valeur patrimoniale
                        with method_tabs[3]:
                            st.write("##### Valorisation basée sur les actifs (Valeur patrimoniale)")
                            
                            if asset_based_value:
                                asset_color = "green" if asset_based_value > current_price else "red"
                                st.markdown(f"**Valeur patrimoniale par action:** <span style='color:{asset_color};font-weight:bold;'>{asset_based_value:.2f} €</span>", unsafe_allow_html=True)
                                
                                # Afficher le potentiel d'appréciation/dépréciation
                                potential = ((asset_based_value / current_price) - 1) * 100
                                potential_color = "green" if potential > 0 else "red"
                                st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
                                
                                # Afficher une explication de la méthode
                                with st.expander("À propos de la valeur patrimoniale"):
                                    st.write("""
                                    La méthode de valorisation patrimoniale évalue une entreprise sur la base de ses actifs nets tangibles.
                                    Elle est particulièrement pertinente pour les entreprises des secteurs financiers, immobiliers ou industriels.
                                    Cette méthode applique un multiplicateur spécifique au secteur pour tenir compte de la valeur de remplacement des actifs.
                                    """)
                                
                                # Afficher les principales composantes du bilan si disponibles
                                if not balance_sheet.empty:
                                    st.write("##### Principales composantes du bilan (dernière année)")
                                    
                                    try:
                                        # Récupérer les données du bilan
                                        balance_data = {}
                                        
                                        # Vérifier la présence des données principales
                                        key_metrics = {
                                            'Total Assets': 'Actifs totaux',
                                            'Total Liabilities Net Minority Interest': 'Passifs totaux',
                                            'Goodwill And Intangible Assets': 'Goodwill et Intangibles',
                                            'Net Tangible Assets': 'Actifs nets tangibles'
                                        }
                                        
                                        for key, label in key_metrics.items():
                                            if key in balance_sheet.index:
                                                balance_data[label] = balance_sheet.loc[key, balance_sheet.columns[0]] / 1e6
                                        
                                        if balance_data:
                                            # Création du graphique
                                            fig = go.Figure()
                                            
                                            # Ajout des barres pour les composantes du bilan
                                            fig.add_trace(go.Bar(
                                                x=list(balance_data.keys()),
                                                y=list(balance_data.values()),
                                                marker_color='blue',
                                                text=[f"€{value:.2f}M" for value in balance_data.values()],
                                                textposition='auto'
                                            ))
                                            
                                            fig.update_layout(
                                                title='Composantes clés du bilan (millions €)',
                                                yaxis_title='Valeur (millions €)',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Erreur lors de l'affichage du bilan: {str(e)}")
                            else:
                                st.warning("Impossible de calculer la valeur patrimoniale pour cette entreprise.")

                # Onglet 3 : Avantage compétitif
                with valuation_tabs[2]:
                    st.write("#### 🛡️ Avantage compétitif")

                    with st.spinner("Analyse de l'avantage compétitif en cours..."):
                        moat_analysis = analyze_competitive_advantage(
                            income_stmt, 
                            balance_sheet, 
                            cashflow, 
                            fundamental_data["Données générales"]
                        )
                    
                    if moat_analysis:
                        # Affichage de l'existence d'un avantage compétitif
                        moat_existence = moat_analysis.get('existence', 'Indéterminé')
                        moat_strength = moat_analysis.get('strength', 0)
                        moat_factors = moat_analysis.get('factors', [])
                        moat_explanation = moat_analysis.get('explanation', '')
                        
                        # Couleur selon la force de l'avantage
                        moat_color = {
                            0: 'gray',   # Indéterminé
                            1: 'orange', # Faible
                            2: 'blue',   # Modéré
                            3: 'green'   # Fort
                        }.get(moat_strength, 'gray')
                        
                        # Affichage de l'avantage compétitif
                        st.markdown(f"""
                        <div style='background-color: {moat_color}; padding: 10px; border-radius: 5px; color: white;'>
                            <h4 style='margin: 0;'>Avantage compétitif: {moat_existence}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Facteurs contribuant à l'avantage compétitif
                        if moat_factors:
                            st.markdown("##### Facteurs contribuant à l'avantage compétitif:")
                            for factor in moat_factors:
                                st.markdown(f"- {factor}")
                        
                        # Explication de l'analyse
                        st.markdown(f"**Analyse:** {moat_explanation}")
                    
                    # Graphique des Free Cash Flows historiques
                    st.subheader("💸 Évolution des Free Cash Flows")
                    
                    if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                        # Extraction des FCF historiques
                        fcf_data = cashflow.loc['Free Cash Flow'].to_dict()
                        
                        # Création d'un DataFrame pour le graphique
                        fcf_df = pd.DataFrame({
                            'Année': [str(year.year) for year in fcf_data.keys()],
                            'FCF (millions)': [float(value) / 1e6 for value in fcf_data.values()]
                        })
                        
                        # Tri par année
                        fcf_df = fcf_df.sort_values('Année')
                        
                        # Calcul de la croissance annuelle moyenne
                        if len(fcf_df) > 1:
                            fcf_values = fcf_df['FCF (millions)'].values
                            positive_values_only = fcf_values[fcf_values > 0]
                            
                            if len(positive_values_only) > 1:
                                # Calcul du taux de croissance annuel moyen (CAGR)
                                start_value = positive_values_only[0]
                                end_value = positive_values_only[-1]
                                years = len(positive_values_only) - 1
                                cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
                                cagr_text = f"Croissance annuelle moyenne: {cagr:.2f}%"
                            else:
                                cagr_text = "Croissance annuelle moyenne: N/A"
                        else:
                            cagr_text = ""
                        
                        # Création du graphique
                        fig = go.Figure()
                        
                        # Ajout des barres pour les FCF
                        fig.add_trace(go.Bar(
                            x=fcf_df['Année'],
                            y=fcf_df['FCF (millions)'],
                            marker_color=['green' if fcf >= 0 else 'red' for fcf in fcf_df['FCF (millions)']],
                            text=[f"€{fcf:.2f}M" for fcf in fcf_df['FCF (millions)']],
                            textposition='outside',
                            name='Free Cash Flow'
                        ))
                        
                        # Ajout d'une ligne de tendance
                        if len(fcf_df) > 1:
                            # Régression linéaire simple
                            x = np.arange(len(fcf_df))
                            y = fcf_df['FCF (millions)'].values
                            model = LinearRegression().fit(x.reshape(-1, 1), y)
                            line_y = model.predict(x.reshape(-1, 1))
                            
                            fig.add_trace(go.Scatter(
                                x=fcf_df['Année'],
                                y=line_y,
                                mode='lines',
                                line=dict(color='blue', width=2, dash='dash'),
                                name='Tendance'
                            ))
                        
                        # Configuration du graphique
                        fig.update_layout(
                            title=f"Évolution des Free Cash Flows<br><sup>{cagr_text}</sup>",
                            xaxis_title="Année",
                            yaxis_title="FCF (millions €)",
                            height=400,
                            margin=dict(l=50, r=50, t=80, b=50),
                            hovermode="x unified"
                        )
                        
                        # Affichage du graphique
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Données de Free Cash Flow non disponibles")
                    
                    # Indicateurs complémentaires
                    st.subheader("🔍 Indicateurs de qualité")
                    
                    # Calcul du ROIC
                    roic = calculate_roic(income_stmt, balance_sheet)
                    
                    # Mise en page en colonnes
                    quality_col1, quality_col2 = st.columns(2)
                    
                    with quality_col1:
                        # Affichage du ROIC
                        if roic is not None:
                            roic_color = "green" if roic > 10 else "orange" if roic > 5 else "red"
                            st.markdown(f"""
                            <div style='background-color: {roic_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;'>
                                <h4 style='margin: 0;'>ROIC: {roic:.2f}%</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Interprétation du ROIC
                            if roic > 15:
                                st.markdown("✅ **ROIC excellent** - L'entreprise génère des rendements très élevés sur son capital investi.")
                            elif roic > 10:
                                st.markdown("✅ **ROIC bon** - L'entreprise génère des rendements satisfaisants sur son capital investi.")
                            elif roic > 5:
                                st.markdown("⚠️ **ROIC moyen** - Les rendements sur le capital investi sont acceptables mais pas exceptionnels.")
                            else:
                                st.markdown("❌ **ROIC faible** - L'entreprise génère des rendements insuffisants sur son capital investi.")
                        else:
                            st.info("ROIC non disponible")
                    
                    with quality_col2:
                        # Valeur Net-Net de Graham
                        graham_netnet = get_graham_netnet_value(balance_sheet)
                        
                        if graham_netnet is not None:
                            # Calcul du ratio Net-Net/Capitalisation
                            market_cap = fundamental_data["Données de marché"].get('Capitalisation boursière', 0)
                            if market_cap > 0:
                                netnet_ratio = (graham_netnet / market_cap) * 100
                                netnet_color = "green" if netnet_ratio > 90 else "orange" if netnet_ratio > 50 else "red"
                                
                                st.markdown(f"""
                                <div style='background-color: {netnet_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;'>
                                    <h4 style='margin: 0;'>Net-Net/Cap: {netnet_ratio:.2f}%</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Interprétation du ratio Net-Net
                                if netnet_ratio > 100:
                                    st.markdown("✅ **Valeur Net-Net > Capitalisation** - L'entreprise se négocie en dessous de sa valeur liquidative.")
                                elif netnet_ratio > 70:
                                    st.markdown("✅ **Valeur Net-Net significative** - L'entreprise a une marge de sécurité importante.")
                                elif netnet_ratio > 40:
                                    st.markdown("⚠️ **Valeur Net-Net moyenne** - Certains actifs courants soutiennent la valorisation.")
                                else:
                                    st.markdown("❌ **Valeur Net-Net faible** - La valeur de l'entreprise repose principalement sur des actifs à long terme.")
                        else:
                            st.info("Valeur Net-Net de Graham non disponible")
                    
                    # Stabilité des marges sur 5 ans (si disponible)
                    if not income_stmt.empty and len(income_stmt.columns) >= 4:
                        try:
                            # Calcul des marges brutes sur les dernières années
                            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                                gross_margins = []
                                years = []
                                
                                for col in income_stmt.columns[:5]:  # Limiter à 5 ans maximum
                                    try:
                                        gross_profit = income_stmt.loc['Gross Profit', col]
                                        total_revenue = income_stmt.loc['Total Revenue', col]
                                        if gross_profit and total_revenue and total_revenue != 0:
                                            margin = gross_profit / total_revenue * 100
                                            gross_margins.append(margin)
                                            years.append(str(col.year))
                                    except:
                                        continue
                                
                                if gross_margins:
                                    st.markdown("##### Stabilité des marges brutes")
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=years, y=gross_margins, name='Marge brute (%)'))
                                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Calcul de la stabilité (écart-type)
                                    stability = np.std(gross_margins)
                                    st.markdown(f"**Stabilité des marges (écart-type):** {stability:.2f}%")
                                    if stability < 3:
                                        st.markdown("🟢 **Marges très stables** (indicateur positif d'avantage compétitif)")
                                    elif stability < 5:
                                        st.markdown("🟡 **Marges relativement stables**")
                                    else:
                                        st.markdown("🔴 **Marges volatiles** (possible absence d'avantage compétitif)")
                        except Exception as e:
                            st.error(f"Erreur lors du calcul de la stabilité des marges: {e}")
                    
                    # Évolution du ROE sur 5 ans
                    if not income_stmt.empty and not balance_sheet.empty and len(income_stmt.columns) >= 4:
                        try:
                            # Calcul du ROE sur les dernières années
                            if 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance_sheet.index:
                                roe_values = []
                                roe_years = []
                                
                                for year in range(min(5, len(income_stmt.columns))):
                                    try:
                                        if year < len(income_stmt.columns) and year < len(balance_sheet.columns):
                                            net_income = income_stmt.loc['Net Income', income_stmt.columns[year]]
                                            equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[year]]
                                            if net_income and equity and equity != 0:
                                                roe = net_income / equity * 100
                                                roe_values.append(roe)
                                                roe_years.append(str(income_stmt.columns[year].year))
                                    except:
                                        continue
                                
                                if roe_values:
                                    # Affichage de l'évolution du ROE
                                    st.markdown("##### Evolution des capitaux propres (ROE)")
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=roe_years, y=roe_values, name='ROE (%)', mode='lines+markers'))
                                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Analyse de la qualité du ROE
                                    avg_roe = np.mean(roe_values)
                                    st.markdown(f"**ROE moyen sur la période:** {avg_roe:.2f}%")
                                    if avg_roe > 15:
                                        st.markdown("🟢 **ROE excellent** (entreprise très rentable)")
                                    elif avg_roe > 10:
                                        st.markdown("🟡 **ROE bon** (entreprise rentable)")
                                    else:
                                        st.markdown("🔴 **ROE faible** (rentabilité à améliorer)")
                        except Exception as e:
                            st.error(f"Erreur lors du calcul de l'évolution du ROE: {e}")
                    
                    # Analyse des ratios d'endettement
                    if not income_stmt.empty and not balance_sheet.empty:
                        try:
                            st.markdown("##### Ratios d'endettement")
                            
                            debt_metrics = {}
                            
                            # Ratio dette à long terme / bénéfice net
                            if 'Long Term Debt' in balance_sheet.index and 'Net Income' in income_stmt.index:
                                long_term_debt = balance_sheet.loc['Long Term Debt', balance_sheet.columns[0]]
                                net_income = income_stmt.loc['Net Income', income_stmt.columns[0]]
                                if long_term_debt is not None and net_income is not None and net_income != 0:
                                    debt_to_income = long_term_debt / net_income
                                    debt_metrics["Dette LT / Bénéfice net"] = f"{debt_to_income:.2f}x"
                                    if debt_to_income < 3:
                                        debt_metrics["Évaluation"] = "🟢 Excellent (< 3x selon Buffett)"
                                    elif debt_to_income < 5:
                                        debt_metrics["Évaluation"] = "🟡 Acceptable (< 5x)"
                                    else:
                                        debt_metrics["Évaluation"] = "🔴 Élevé (> 5x)"
                            
                            # Charges d'intérêts / Résultat brut
                            if 'Interest Expense' in income_stmt.index and 'Gross Profit' in income_stmt.index:
                                interest_expense = abs(income_stmt.loc['Interest Expense', income_stmt.columns[0]]) if 'Interest Expense' in income_stmt.index else 0
                                gross_profit = income_stmt.loc['Gross Profit', income_stmt.columns[0]]
                                if interest_expense is not None and gross_profit is not None and gross_profit != 0:
                                    interest_to_profit = interest_expense / gross_profit
                                    debt_metrics["Charges d'intérêts / Résultat brut"] = f"{interest_to_profit:.2%}"
                                    if interest_to_profit < 0.05:
                                        debt_metrics["Charges/Résultat"] = "🟢 Très faible (< 5%)"
                                    elif interest_to_profit < 0.15:
                                        debt_metrics["Charges/Résultat"] = "🟡 Modéré (5-15%)"
                                    else:
                                        debt_metrics["Charges/Résultat"] = "🔴 Élevé (> 15%)"
                            
                            # Évolution de la dette sur 5 ans (si disponible)
                            if 'Total Debt' in balance_sheet.index and len(balance_sheet.columns) >= 2:
                                debt_years = []
                                debt_values = []
                                
                                for col in balance_sheet.columns[:min(5, len(balance_sheet.columns))]:
                                    try:
                                        debt_value = balance_sheet.loc['Total Debt', col]
                                        if debt_value is not None:
                                            debt_values.append(debt_value / 1e6)  # En millions
                                            debt_years.append(str(col.year))
                                    except:
                                        continue
                                
                                if debt_values:
                                    # Affichage de l'évolution de la dette
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=debt_years, y=debt_values, name='Dette totale (M)'))
                                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Analyse de la tendance
                                    if len(debt_values) > 1:
                                        if debt_values[0] < debt_values[-1]:
                                            st.markdown("🔴 **Dette en augmentation** sur la période")
                                        elif debt_values[0] > debt_values[-1]:
                                            st.markdown("🟢 **Dette en diminution** sur la période")
                                        else:
                                            st.markdown("🟡 **Dette stable** sur la période")
                            
                            # Afficher les métriques d'endettement dans un tableau
                            if debt_metrics:
                                df_debt = pd.DataFrame(debt_metrics.items(), columns=["Métrique", "Valeur"])
                                st.dataframe(df_debt, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.error(f"Erreur lors de l'analyse d'endettement: {e}")

                # Onglet 4 : Dividendes
                with valuation_tabs[3]:
                    st.subheader("💰 Analyse des Dividendes")
                    
                    if st.session_state.get('selected_stock') is not None:
                        selected_stock = st.session_state['selected_stock']
                        selected_stock_name = st.session_state['selected_stock_name']
                        ticker = st.session_state['ticker']
                        
                        # Analyse des dividendes
                        with st.spinner("Analyse de la politique de dividendes en cours..."):
                            dividend_analysis = get_dividend_policy_analysis(ticker)
                            
                            if dividend_analysis.get('error'):
                                st.error(f"Erreur lors de l'analyse des dividendes: {dividend_analysis['error']}")
                            elif not dividend_analysis.get('has_dividends', False):
                                st.info(f"{selected_stock_name} ne verse pas de dividendes actuellement.")
                            else:
                                # Créer une mise en page à deux colonnes
                                col1, col2 = st.columns([1, 1])
                                
                                # Colonne 1: Tableau de bord des métriques clés
                                with col1:                                    
                                    # Créer des métriques avec indicateurs de qualité
                                    st.markdown("#### Indicateurs clés")
                                    
                                    # Fonction d'aide pour afficher un badge coloré
                                    def quality_badge(quality):
                                        colors = {
                                            'Excellent': '#4CAF50',  # Vert
                                            'Bon': '#8BC34A',       # Vert clair
                                            'Moyen': '#FFC107',     # Jaune
                                            'Faible': '#F44336',    # Rouge
                                            'Risqué': '#D32F2F',    # Rouge foncé
                                            'Soutenable': '#4CAF50', # Vert
                                            'Modéré': '#FFC107',    # Jaune
                                            'Non évalué': '#9E9E9E', # Gris
                                            'Indéterminé': '#9E9E9E' # Gris
                                        }
                                        color = colors.get(quality, '#9E9E9E')
                                        return f"<span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;'>{quality}</span>"
                                    
                                    # Créer des lignes de métriques avec badges
                                    metrics_data = [
                                        ("Rendement du dividende", f"{dividend_analysis.get('dividend_yield', 0):.2f}%", ""),
                                        ("Santé du dividende", "", quality_badge(dividend_analysis.get('dividend_health', 'Non évalué'))),
                                        ("Croissance du dividende", f"{dividend_analysis.get('dividend_growth', 0):.2f}%/an", quality_badge(dividend_analysis.get('dividend_growth_score', 'Non évalué'))),
                                        ("Consistance", "", quality_badge(dividend_analysis.get('dividend_consistency', 'Non évalué'))),
                                        ("Qualité globale", "", quality_badge(dividend_analysis.get('dividend_quality', 'Non évalué'))),
                                    ]
                                    
                                    # Afficher les métriques dans un format élégant
                                    for name, value, badge in metrics_data:
                                        if value and badge:
                                            st.markdown(f"**{name}:** {value} {badge}", unsafe_allow_html=True)
                                        elif value:
                                            st.markdown(f"**{name}:** {value}", unsafe_allow_html=True)
                                        elif badge:
                                            st.markdown(f"**{name}:** {badge}", unsafe_allow_html=True)
                                    
                                    # Informations supplémentaires
                                    st.markdown("#### Détails supplémentaires")
                                    
                                    additional_metrics = [
                                        ("Ratio de distribution", f"{dividend_analysis.get('payout_ratio', 0):.2f}%"),
                                        ("Durabilité", dividend_analysis.get('sustainability', 'Non évalué')),
                                        ("Tendance", dividend_analysis.get('dividend_trend')),
                                        ("Années consécutives d'augmentation", dividend_analysis.get('consecutive_years_increase', 0)),
                                        ("Aristocrate du dividende", "Oui" if dividend_analysis.get('dividend_aristocrat', False) else "Non"),
                                        ("Fréquence de versement", f"{dividend_analysis.get('dividend_frequency', 0):.1f} fois/an"),
                                        ("Rachat d'actions", dividend_analysis.get('buyback_trend', 'Indéterminé')),
                                        ("Comparaison sectorielle", dividend_analysis.get('peer_comparison'))
                                    ]
                                    
                                    for name, value in additional_metrics:
                                        st.markdown(f"**{name}:** {value}")
                                    
                                    # Ajouter un indicateur FCF Payout Ratio si disponible
                                    if 'fcf_payout_ratio' in dividend_analysis:
                                        fcf_ratio = dividend_analysis['fcf_payout_ratio']
                                        fcf_color = ""
                                        if fcf_ratio > 90:
                                            fcf_color = "🔴"
                                        elif fcf_ratio > 70:
                                            fcf_color = "🟠"
                                        elif fcf_ratio > 50:
                                            fcf_color = "🟡"
                                        else:
                                            fcf_color = "🟢"
                                        
                                        st.markdown(f"**Ratio FCF/Dividendes:** {fcf_color} {fcf_ratio:.2f}%")
                                
                                # Colonne 2: Graphiques et analyse visuelle
                                with col2:
                                    st.markdown("### 📈 Visualisation des dividendes")
                                    
                                    # Créer le DataFrame des dividendes si disponible
                                    if dividend_analysis.get('dividend_history'):
                                        div_history = dividend_analysis['dividend_history']
                                        dates = list(div_history.keys())
                                        values = list(div_history.values())
                                        
                                        # Convertir en DataFrame pour faciliter le traitement
                                        dividend_df = pd.DataFrame({
                                            'date': pd.to_datetime(dates),
                                            'dividend': values
                                        })
                                        
                                        # Trier par date
                                        dividend_df = dividend_df.sort_values('date')
                                        
                                        # Créer un DataFrame annuel pour l'analyse de croissance
                                        dividend_df['year'] = dividend_df['date'].dt.year
                                        annual_div = dividend_df.groupby('year')['dividend'].sum().reset_index()
                                        
                                        # Graphique 1: Historique des dividendes (trimestriel)
                                        fig1 = go.Figure()
                                        fig1.add_trace(go.Bar(
                                            x=dividend_df['date'],
                                            y=dividend_df['dividend'],
                                            marker_color='#1E88E5',
                                            name='Dividende'
                                        ))
                                        
                                        fig1.update_layout(
                                            title="Historique des versements de dividendes",
                                            xaxis_title="Date",
                                            yaxis_title="Dividende par action",
                                            height=300,
                                            margin=dict(l=0, r=0, t=40, b=0),
                                        )

                                        # Ajout d'une ligne de tendance
                                        if len(dividend_df) > 1:
                                            # Régression linéaire simple
                                            x = np.arange(len(dividend_df))
                                            y = dividend_df['dividend'].values
                                            model = LinearRegression().fit(x.reshape(-1, 1), y)
                                            line_y = model.predict(x.reshape(-1, 1))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=dividend_df['date'],
                                            y=line_y,
                                            mode='lines',
                                            line=dict(color='black', width=2, dash='dash'),
                                            name='Tendance'
                                        ))
                                        
                                        st.plotly_chart(fig1, use_container_width=True)
                                        
                                        # Graphique 2: Croissance annuelle des dividendes
                                        if len(annual_div) > 1:
                                            # Ajouter le taux de croissance annuel
                                            annual_div['growth'] = annual_div['dividend'].pct_change() * 100
                                            
                                            # Créer un graphique combiné (barres + ligne)
                                            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                                            
                                            # Ajouter les barres pour les dividendes annuels
                                            fig2.add_trace(
                                                go.Bar(
                                                    x=annual_div['year'],
                                                    y=annual_div['dividend'],
                                                    name="Dividende annuel",
                                                    marker_color='#43A047'
                                                ),
                                                secondary_y=False
                                            )
                                            
                                            # Ajouter la ligne pour le taux de croissance
                                            fig2.add_trace(
                                                go.Scatter(
                                                    x=annual_div['year'][1:],  # Premier point n'a pas de croissance
                                                    y=annual_div['growth'][1:],
                                                    name="Taux de croissance",
                                                    mode='lines+markers',
                                                    line=dict(color='#E65100', width=3),
                                                    marker=dict(size=8)
                                                ),
                                                secondary_y=True
                                            )
                                            
                                            # Mise à jour des axes
                                            fig2.update_layout(
                                                title="Croissance annuelle des dividendes",
                                                height=300,
                                                margin=dict(l=0, r=0, t=40, b=0),
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="right",
                                                    x=1
                                                )
                                            )
                                            
                                            fig2.update_yaxes(title_text="Dividende par action", secondary_y=False)
                                            fig2.update_yaxes(title_text="Croissance annuelle (%)", secondary_y=True)
                                            
                                            st.plotly_chart(fig2, use_container_width=True)
                                        
                                        # Graphique 3 (Bonus): Santé du dividende
                                        # Créer un graphique radar pour les composantes de la qualité du dividende
                                        
                                        # Définir les scores numériques pour les attributs qualitatifs
                                        quality_scores = {
                                            'Excellent': 100,
                                            'Bon': 75,
                                            'Moyen': 50,
                                            'Faible': 25,
                                            'Non évalué': 0
                                        }
                                        
                                        # Préparer les données du radar
                                        categories = ['Santé', 'Croissance', 'Consistance', 'Durabilité']
                                        values = [
                                            quality_scores.get(dividend_analysis.get('dividend_health', 'Non évalué'), 0),
                                            quality_scores.get(dividend_analysis.get('dividend_growth_score', 'Non évalué'), 0),
                                            quality_scores.get(dividend_analysis.get('dividend_consistency', 'Non évalué'), 0),
                                            75 if dividend_analysis.get('sustainability') == 'Soutenable' else 
                                            50 if dividend_analysis.get('sustainability') == 'Modéré' else 
                                            25 if dividend_analysis.get('sustainability') == 'Risqué' else 0
                                        ]
                                        
                                        fig3 = go.Figure()
                                        
                                        fig3.add_trace(go.Scatterpolar(
                                            r=values,
                                            theta=categories,
                                            fill='toself',
                                            fillcolor='rgba(64, 224, 208, 0.3)',
                                            line=dict(color='rgb(64, 224, 208)', width=2),
                                            name='Qualité du dividende'
                                        ))
                                        
                                        fig3.update_layout(
                                            polar=dict(
                                                radialaxis=dict(
                                                    visible=True,
                                                    range=[0, 100]
                                                )
                                            ),
                                            title="Qualité du dividende",
                                            height=300,
                                            margin=dict(l=40, r=40, t=40, b=0)
                                        )
                                        
                                        st.plotly_chart(fig3, use_container_width=True)
                                
                                # Section inférieure: Analyses et recommandations
                                st.markdown("### 🔍 Analyse et interprétation")
                                
                                # Créer une analyse textuelle basée sur les données
                                analysis_text = []
                                
                                # Synthèse générale
                                quality = dividend_analysis.get('dividend_quality', 'Non évalué')
                                health = dividend_analysis.get('dividend_health', 'Non évalué')
                                growth = dividend_analysis.get('dividend_growth_score', 'Non évalué')
                                consistency = dividend_analysis.get('dividend_consistency', 'Non évalué')
                                
                                # Points forts
                                strengths = []
                                if quality in ['Excellent', 'Bon']:
                                    strengths.append(f"la qualité globale du dividende est **{quality.lower()}**")
                                if health in ['Excellent', 'Bon']:
                                    strengths.append(f"la santé financière supportant le dividende est **{health.lower()}**")
                                if growth in ['Excellent', 'Bon']:
                                    strengths.append(f"la croissance du dividende est **{growth.lower()}**")
                                if consistency in ['Excellent', 'Bon']:
                                    strengths.append(f"la consistance des versements est **{consistency.lower()}**")
                                if dividend_analysis.get('dividend_aristocrat', False):
                                    strengths.append("l'entreprise est un **aristocrate du dividende** avec plus de 25 ans d'augmentations consécutives")
                                if dividend_analysis.get('buyback_trend') == 'Actif':
                                    strengths.append("l'entreprise pratique activement des **rachats d'actions**")
                                
                                # Points faibles
                                weaknesses = []
                                if quality in ['Moyen', 'Faible']:
                                    weaknesses.append(f"la qualité globale du dividende est **{quality.lower()}**")
                                if health in ['Moyen', 'Faible']:
                                    weaknesses.append(f"la santé financière supportant le dividende est **{health.lower()}**")
                                if growth in ['Moyen', 'Faible']:
                                    weaknesses.append(f"la croissance du dividende est **{growth.lower()}**")
                                if consistency in ['Moyen', 'Faible']:
                                    weaknesses.append(f"la consistance des versements est **{consistency.lower()}**")
                                if dividend_analysis.get('sustainability') == 'Risqué':
                                    weaknesses.append("le taux de distribution est **risqué** (>80% des bénéfices)")
                                if dividend_analysis.get('buyback_trend') == 'Dilution':
                                    weaknesses.append("l'entreprise **dilue** ses actionnaires en émettant de nouvelles actions")
                                if 'fcf_payout_ratio' in dividend_analysis and dividend_analysis['fcf_payout_ratio'] > 90:
                                    weaknesses.append("le dividende représente plus de 90% du flux de trésorerie disponible, ce qui est **risqué**")
                                
                                # Afficher l'analyse
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### Points forts")
                                    if strengths:
                                        for strength in strengths:
                                            st.markdown(f"✅ {strength}")
                                    else:
                                        st.markdown("*Aucun point fort significatif identifié pour la politique de dividende.*")
                                
                                with col2:
                                    st.markdown("#### Points à surveiller")
                                    if weaknesses:
                                        for weakness in weaknesses:
                                            st.markdown(f"⚠️ {weakness}")
                                    else:
                                        st.markdown("*Aucun point faible significatif identifié pour la politique de dividende.*")
                                
                                # Recommandation
                                st.markdown("#### Recommandation")
                                
                                # Déterminer la recommandation basée sur les scores
                                if quality == 'Excellent':
                                    recommendation = "Cet investissement présente une **excellente politique de dividende** qui combine santé financière, croissance et consistance. Il convient particulièrement aux investisseurs recherchant un revenu passif fiable et croissant sur le long terme."
                                elif quality == 'Bon':
                                    recommendation = "Cet investissement présente une **bonne politique de dividende** avec une balance appropriée entre distribution et réinvestissement. Le dividende semble soutenable et offre un potentiel de croissance raisonnable pour les investisseurs axés sur le revenu."
                                elif quality == 'Moyen':
                                    recommendation = "Cet investissement présente une politique de dividende **acceptable mais avec des points d'attention**. Bien que le dividende actuel soit versé, des aspects de sa qualité, santé ou croissance méritent d'être surveillés de près."
                                elif quality == 'Faible':
                                    recommendation = "Cet investissement présente une politique de dividende **présentant des risques**. Le dividende pourrait être difficile à maintenir ou manquer de croissance significative. Les investisseurs focalisés sur le revenu devraient envisager des alternatives."
                                else:
                                    recommendation = "L'analyse complète de la politique de dividende n'a pas pu être réalisée avec les données disponibles."
                                
                                st.info(recommendation)
                    else:
                        st.info("Veuillez sélectionner une action dans le panneau latéral pour analyser sa politique de dividendes.")

        else:
            st.info("Veuillez sélectionner une action pour voir l'analyse fondamentale et les projections.")

# Onglet 3: Actualités récentes
    with tab3:     
        if st.session_state.get('selected_stock') is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = st.session_state['ticker']

            # Titre principal
            st.markdown(f"### 📰 Actualités récentes pour {selected_stock_name} ({ticker})")
            
            # Récupérer les actualités
            with st.spinner("Chargement des actualités..."):
                news_df = get_stock_news(ticker)
            
            if news_df.empty:
                st.info(f"Aucune actualité récente trouvée pour {selected_stock_name}.")
            else:
                # Afficher les actualités avec une distinction visuelle basée sur leur importance
                for idx, news in news_df.iterrows():
                    # Déterminer la couleur de fond en fonction de l'importance
                    if news['importance_score'] >= 4:
                        bg_color = "#ffe6e6"  # Rouge clair pour les importantes
                        border_color = "#ffb3b3"
                    elif news['importance_score'] >= 3:
                        bg_color = "#fff2e6"  # Orange clair
                        border_color = "#ffcc99"
                    else:
                        bg_color = "#f2f2f2"  # Gris clair pour les moins importantes
                        border_color = "#e6e6e6"
                    
                    # Créer un conteneur personnalisé pour chaque actualité
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 10px; margin-bottom: 10px; border-left: 5px solid {border_color}; border-radius: 5px;">
                        <h4>{news['title']}</h4>
                        <p><small>Publié le {news['date_formatted']} | Catégorie: <b>{news['category']}</b> | Impact sur la valeur: <b>{news['impact_on_valuation']}</b></small></p>
                        <p>{news['description']}</p>
                        <p><a href="{news['link']}" target="_blank">Lire l'article complet</a></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("Veuillez sélectionner une action dans le menu latéral pour voir les actualités.")

    # Onglet 4: Screener d'Actions
    with tab4:
        st.markdown(f"### 🔎 Screener d'actions Buffett-Graham")

        with st.expander("📊 À propos de ce screener", expanded=True):
            st.markdown("""
            Ce screener d'actions combine les principes d'investissement de Warren Buffett et Benjamin Graham pour identifier des entreprises de qualité à des valorisations raisonnables. Il analyse :
            
            - **Qualité financière** : ROE, marges, stabilité des bénéfices
            - **Santé du bilan** : Niveaux d'endettement, liquidité, capacité à générer du cash
            - **Régularité des performances** : Marges brutes stables, croissance constante
            - **Valorisation** : P/E ratio, PEG ratio, price-to-book
            
            Ajustez les critères ci-dessous pour trouver des actions correspondant à votre stratégie d'investissement.
            """)
        
        # Section des filtres de marché
        st.subheader("Filtrage de l'univers d'investissement")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            filter_type = st.selectbox(
                "Type de filtre",
                ["Toutes les actions", "Par région", "Par secteur", "Par marché"],
                index=0
            )
        
        with filter_col2:
            # Afficher les options de filtre en fonction du type sélectionné
            if filter_type == "Par région":
                # Récupérer la structure de marché
                market_structure = get_market_structure()
                regions = list(market_structure['regions'].keys())
                selected_region = st.selectbox("Sélectionner une région", regions)
                
                # Si une région est sélectionnée, proposer les pays de cette région
                if selected_region:
                    pays_options = ["Tous les pays"] + list(market_structure['regions'][selected_region].keys())
                    selected_pays = st.selectbox("Sélectionner un pays", pays_options)
                    if selected_pays == "Tous les pays":
                        selected_pays = None
                else:
                    selected_pays = None
                
                # Variables pour le filtrage
                level1 = selected_region
                level2 = selected_pays
                
            elif filter_type == "Par secteur":
                # Récupérer la structure de marché
                market_structure = get_market_structure()
                secteurs = list(market_structure['secteurs'].keys())
                selected_secteur = st.selectbox("Sélectionner un secteur", secteurs)
                
                # Si un secteur est sélectionné, proposer les industries de ce secteur
                if selected_secteur:
                    industrie_options = ["Toutes les industries"] + list(market_structure['secteurs'][selected_secteur].keys())
                    selected_industrie = st.selectbox("Sélectionner une industrie", industrie_options)
                    if selected_industrie == "Toutes les industries":
                        selected_industrie = None
                else:
                    selected_industrie = None
                
                # Variables pour le filtrage
                level1 = selected_secteur
                level2 = selected_industrie
                
            elif filter_type == "Par marché":
                # Récupérer la structure de marché
                market_structure = get_market_structure()
                marches = list(market_structure['marches'].keys())
                selected_marche = st.selectbox("Sélectionner un marché", marches)
                
                # Variables pour le filtrage
                level1 = selected_marche
                level2 = None
                
            else:  # "Toutes les actions"
                # Pas de filtres supplémentaires nécessaires
                level1 = None
                level2 = None

        # Colonnes pour organiser les contrôles
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Critères financiers fondamentaux")
            min_revenue_growth = st.slider("Croissance min. des revenus (%)", 0, 30, 5)
            min_net_profit_margin = st.slider("Marge nette min. (%)", 0, 40, 10)
            min_roe = st.slider("ROE min. (%)", 5, 50, 15)
            min_operating_margin = st.slider("Marge opérationnelle min. (%)", 0, 40, 15)
            min_earnings_stability_years = st.slider("Années de bénéfices positifs consécutifs", 0, 10, 5)
        
        with col2:
            st.subheader("Critères d'endettement et valorisation")
            max_debt_to_equity = st.slider("Ratio dette/fonds propres max.", 0.0, 2.0, 0.5, 0.1)
            max_debt_ebitda_ratio = st.slider("Ratio dette/EBITDA max.", 0.0, 10.0, 5.0, 0.5)
            max_pe_ratio = st.slider("P/E ratio max.", 5, 50, 25)
            max_pe_to_growth = st.slider("PEG ratio max.", 0.0, 3.0, 1.0, 0.1)
            max_price_to_book = st.slider("Ratio cours/valeur comptable max.", 0.0, 5.0, 1.5, 0.1)
        
        with col3:
            st.subheader("Critères de consistance")
            min_gross_margin_consistency = st.slider("Marge brute min. consistante (%)", 20, 60, 40)
            min_net_income_growth_consistency = st.slider("Croissance min. du résultat net (%)", 0, 20, 8)
            consistent_low_debt = st.checkbox("Exiger une dette constamment faible", True)
            min_operating_cashflow_growth = st.slider("Croissance min. du cash-flow op. (%)", 0, 20, 5)
            min_dividend_years = st.slider("Années consécutives de dividendes", 0, 20, 0)
        
        # Zone pour les filtres avancés (facultatifs)
        with st.expander("🔍 Filtres avancés"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                min_free_cashflow = st.number_input("Free Cash Flow minimal", 0, 1000000000, 0, 1000000)
                min_current_ratio = st.slider("Ratio de liquidité courante min.", 0.5, 5.0, 1.5, 0.1)
                max_rd_sales_ratio = st.slider("Ratio R&D/Ventes max.", 0.0, 0.20, 0.05, 0.01)
                
            with col_adv2:
                max_capex_net_income_ratio = st.slider("Ratio CAPEX/Résultat Net max.", 0.0, 2.0, 0.5, 0.1)
                max_interest_ebit_ratio = st.slider("Ratio Intérêts/EBIT max.", 0.0, 0.5, 0.20, 0.05)
                min_interest_coverage_ratio = st.slider("Ratio de couverture d'intérêts min.", 1, 20, 5)
        
        # Options d'affichage des résultats
        with st.expander("⚙️ Options d'affichage"):
            display_cols = st.multiselect(
                "Colonnes à afficher dans les résultats",
                ['Nom', 'Ticker', 'ISIN', 'Région', 'Pays', 'Score Buffett-Graham', 
                'Croissance CA (%)', 'Marge nette (%)', 'ROE (%)', 'Dette/Fonds propres', 
                'P/E', 'PEG', 'P/B', 'Free Cash Flow', 'Années bénéfices positifs',
                'Marge Brute Consistante', 'Croissance RN Consistante', 'Croissance FCF',
                'Forces', 'Faiblesses'],
                default=['Nom', 'Ticker', 'Région', 'Score Buffett-Graham', 'ROE (%)', 
                        'Dette/Fonds propres', 'P/E', 'PEG', 'Années bénéfices positifs', 'Forces']
            )
            
            sort_by = st.selectbox(
                "Trier les résultats par",
                ['Score Buffett-Graham', 'ROE (%)', 'Croissance CA (%)', 'Marge nette (%)', 
                'Dette/Fonds propres', 'P/E', 'PEG', 'P/B', 'Années bénéfices positifs'],
                index=0
            )
            
            sort_ascending = st.checkbox("Ordre croissant", False)
        
        # Bouton pour lancer l'analyse
        if st.button("Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                # Récupérer la structure de marché
                market_structure = get_market_structure()
                
                # Convertir le type de filtre de l'interface utilisateur au format attendu par flatten_market_structure
                filter_type_mapping = {
                    "Par région": "region",
                    "Par secteur": "secteur",
                    "Par marché": "marche",
                    "Toutes les actions": None
                }
                
                # Obtenir le dictionnaire plat des actions selon les filtres sélectionnés
                stocks_dict = flatten_market_structure(
                    market_structure,
                    filter_type=filter_type_mapping[filter_type],
                    level1=level1,
                    level2=level2
                )
                
                # Vérifier si des actions ont été trouvées
                if not stocks_dict:
                    st.warning("Aucune action ne correspond aux filtres sélectionnés. Veuillez modifier vos critères.")
                else:
                    st.info(f"{len(stocks_dict)} actions trouvées dans l'univers sélectionné. Application des critères de screening...")

                # Créer une barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Préparation pour le screening par lots
                stock_list = list(stocks_dict.keys())
                total_stocks = len(stock_list)
                
                # Diviser le traitement en lots pour mettre à jour la progression
                batch_size = max(1, total_stocks // 100)  # S'assurer d'au moins 100 mises à jour
                results = []
                detailed_results = {}
                
                for i in range(0, total_stocks, batch_size):
                    # Extraire le lot actuel d'actions
                    batch_stocks = {k: stocks_dict[k] for k in stock_list[i:min(i+batch_size, total_stocks)]}
                    
                    # Mettre à jour la barre de progression et le texte de statut
                    progress = int((i + 1) / total_stocks * 100)
                    progress_bar.progress(progress)
                    current_stock = stock_list[i] if i < total_stocks else ""
                    status_text.text(f"Analyse en cours: {i+1}/{total_stocks} actions - {current_stock}")
                    
                    # Exécuter le screener sur ce lot d'actions
                    batch_results_df, batch_detailed_results = buffett_graham_stock_screener(
                        batch_stocks,
                        # Passer tous vos paramètres ici
                        min_revenue_growth=min_revenue_growth,
                        min_net_profit_margin=min_net_profit_margin,
                        min_roe=min_roe,
                        max_debt_to_equity=max_debt_to_equity,
                        min_free_cashflow=min_free_cashflow,
                        min_operating_margin=min_operating_margin,
                        min_earnings_stability_years=min_earnings_stability_years,
                        min_gross_margin_consistency=min_gross_margin_consistency,
                        min_net_income_growth_consistency=min_net_income_growth_consistency,
                        consistent_low_debt=consistent_low_debt,
                        min_operating_cashflow_growth=min_operating_cashflow_growth,
                        max_pe_ratio=max_pe_ratio,
                        max_pe_to_growth=max_pe_to_growth,
                        max_price_to_book=max_price_to_book,
                        max_rd_sales_ratio=max_rd_sales_ratio,
                        max_capex_net_income_ratio=max_capex_net_income_ratio,
                        max_interest_ebit_ratio=max_interest_ebit_ratio,
                        min_interest_coverage_ratio=min_interest_coverage_ratio,
                        max_debt_ebitda_ratio=max_debt_ebitda_ratio,
                        min_dividend_years=min_dividend_years,
                        min_current_ratio=min_current_ratio
                    )
                    
                    # Ajouter les résultats de ce lot aux résultats globaux
                    if not batch_results_df.empty:
                        results.append(batch_results_df)
                    detailed_results.update(batch_detailed_results)
                
                # Combiner tous les résultats
                if results:
                    results_df = pd.concat(results, ignore_index=True)
                else:
                    results_df = pd.DataFrame()
                
                # Effacer la barre de progression et le texte de statut une fois terminé
                status_text.empty()
                progress_bar.empty()
                    
                if not results_df.empty:
                    # Trier les résultats
                    if 'Score Buffett-Graham' in results_df.columns and sort_by == 'Score Buffett-Graham':
                        # Convertir le score de format "XX/100" à nombre pour le tri
                        results_df['Score Numérique'] = results_df['Score Buffett-Graham'].apply(
                            lambda x: float(x.split('/')[0]) if isinstance(x, str) else 0
                        )
                        results_df = results_df.sort_values('Score Numérique', ascending=sort_ascending)
                        results_df = results_df.drop('Score Numérique', axis=1)
                    else:
                        results_df = results_df.sort_values(sort_by, ascending=sort_ascending)
                    
                    # Afficher les résultats
                    st.subheader(f"Résultats : {len(results_df)} actions correspondent à vos critères")
                    st.dataframe(results_df[display_cols], height=600)
                    
                    # Visualisation des résultats
                    if len(results_df) > 0:
                        with st.expander("📈 Visualisations", expanded=True):
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                # Distribution des scores
                                if 'Score Buffett-Graham' in results_df.columns:
                                    scores = results_df['Score Buffett-Graham'].apply(
                                        lambda x: float(x.split('/')[0]) if isinstance(x, str) else 0
                                    )
                                    fig1 = px.histogram(scores, nbins=10, 
                                                    labels={'value': 'Score', 'count': 'Nombre d\'actions'},
                                                    title='Distribution des scores Buffett-Graham')
                                    st.plotly_chart(fig1, use_container_width=True)
                            
                            with viz_col2:
                                # Relation ROE vs P/E
                                if 'ROE (%)' in results_df.columns and 'P/E' in results_df.columns:
                                    fig2 = px.scatter(results_df, 
                                                    x='ROE (%)', y='P/E',
                                                    hover_name='Nom',
                                                    color='Score Buffett-Graham' if 'Score Buffett-Graham' in results_df.columns else None,
                                                    size='Marge nette (%)' if 'Marge nette (%)' in results_df.columns else None,
                                                    title='ROE vs P/E')
                                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Permettre de voir les détails pour chaque action
                    with st.expander("🔍 Analyse détaillée par action"):
                        selected_ticker = st.selectbox(
                            "Sélectionner une action pour voir l'analyse détaillée",
                            options=results_df['Ticker'].tolist(),
                            index=0
                        )
                        
                        if selected_ticker in detailed_analysis:
                            details = detailed_analysis[selected_ticker]
                            
                            st.subheader(f"Analyse détaillée pour {selected_ticker}")
                            
                            # Afficher les métriques clés en 3 colonnes
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric("Score Buffett-Graham", f"{details['score']}/100")
                                st.metric("ROE", f"{details['financial_metrics']['roe']:.2f}%")
                                st.metric("Marge Nette", f"{details['financial_metrics']['net_profit_margin']:.2f}%")
                            
                            with metric_col2:
                                st.metric("P/E Ratio", f"{details['intrinsic_value_metrics']['pe_ratio']:.2f}")
                                st.metric("PEG Ratio", f"{details['intrinsic_value_metrics']['pe_to_growth']:.2f}")
                                st.metric("Dette/Fonds Propres", f"{details['financial_metrics']['debt_to_equity']:.2f}")
                            
                            with metric_col3:
                                st.metric("FCF", format_number(details['financial_metrics']['free_cashflow']))
                                st.metric("Marge Opérationnelle", f"{details['financial_metrics']['operating_margin']:.2f}%")
                                st.metric("Années Bénéfices +", f"{details['consistency_metrics']['earnings_stability_years']}")
                            
                            # Afficher les forces et faiblesses
                            st.subheader("Forces et Faiblesses")
                            force_col, faiblesse_col = st.columns(2)
                            
                            with force_col:
                                st.markdown("### Forces")
                                st.write(details['strengths'])
                            
                            with faiblesse_col:
                                st.markdown("### Faiblesses")
                                st.write(details['weaknesses'])
                            
                            # Afficher toutes les métriques en détail
                            with st.expander("Toutes les métriques"):
                                # Métriques financières
                                st.markdown("### Métriques Financières")
                                financial_metrics_df = pd.DataFrame({
                                    'Métrique': list(details['financial_metrics'].keys()),
                                    'Valeur': [f"{v:.2f}" if isinstance(v, float) else str(v) 
                                            for v in details['financial_metrics'].values()]
                                })
                                st.dataframe(financial_metrics_df)
                                
                                # Métriques de valorisation
                                st.markdown("### Métriques de Valorisation")
                                value_metrics_df = pd.DataFrame({
                                    'Métrique': list(details['intrinsic_value_metrics'].keys()),
                                    'Valeur': [f"{v:.2f}" if isinstance(v, float) else str(v)
                                            for v in details['intrinsic_value_metrics'].values()]
                                })
                                st.dataframe(value_metrics_df)
                                
                                # Métriques de consistance
                                st.markdown("### Métriques de Consistance")
                                consistency_metrics_df = pd.DataFrame({
                                    'Métrique': list(details['consistency_metrics'].keys()),
                                    'Valeur': list(details['consistency_metrics'].values())
                                })
                                st.dataframe(consistency_metrics_df)
                    
                    # Option pour exporter les résultats
                    with st.expander("💾 Exporter les résultats"):
                        # Convertir en CSV
                        csv = results_df.to_csv(index=False)
                        # Créer un bouton de téléchargement
                        st.download_button(
                            label="Télécharger les résultats (CSV)",
                            data=csv,
                            file_name="buffett_graham_screener_results.csv",
                            mime="text/csv"
                        )
                        
                        # Export Excel plus complet avec onglets
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                            results_df.to_excel(writer, sheet_name="Résultats", index=False)
                            
                            # Créer un onglet pour les détails d'analyse
                            detailed_df = pd.DataFrame()
                            for ticker, details in detailed_analysis.items():
                                ticker_df = pd.DataFrame({
                                    'Ticker': [ticker],
                                    'Score': [details['score']],
                                    **{f"Fin_{k}": v for k, v in details['financial_metrics'].items()},
                                    **{f"Val_{k}": v for k, v in details['intrinsic_value_metrics'].items()},
                                    **{f"Cons_{k}": v for k, v in details['consistency_metrics'].items()}
                                })
                                detailed_df = pd.concat([detailed_df, ticker_df], ignore_index=True)
                            
                            detailed_df.to_excel(writer, sheet_name="Analyse Détaillée", index=False)
                            
                            # Onglet avec les paramètres du screener
                            params_df = pd.DataFrame({
                                'Paramètre': [
                                    'Croissance min. des revenus (%)',
                                    'Marge nette min. (%)',
                                    'ROE min. (%)',
                                    'Dette/Fonds propres max.',
                                    'P/E ratio max.',
                                    'PEG ratio max.',
                                    'P/B max.'
                                ],
                                'Valeur': [
                                    min_revenue_growth,
                                    min_net_profit_margin,
                                    min_roe,
                                    max_debt_to_equity,
                                    max_pe_ratio,
                                    max_pe_to_growth,
                                    max_price_to_book
                                ]
                            })
                            params_df.to_excel(writer, sheet_name="Paramètres", index=False)
                        
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            label="Télécharger rapport complet (Excel)",
                            data=excel_data,
                            file_name="buffett_graham_screener_rapport.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    
                else:
                    st.warning("Aucune action ne correspond à vos critères. Essayez d'assouplir certains paramètres.")
    
    # Onglet 5: Analyse et Optimisation de Portefeuille
    with tab5:
        st.subheader("📊 Analyse et Optimisation de Portefeuille")
        
        def analyze_portfolio(portfolio_df):
            """
            Analyse un portefeuille d'actions et fournit des recommandations
            """
            if portfolio_df.empty:
                return pd.DataFrame(), {}
            
            results = []
            total_portfolio_value = 0
            
            # Analyser chaque position dans le portefeuille
            for index, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                quantity = row['Quantité']
                entry_price = row['PRU']
                
                # Récupérer les données fondamentales et historiques
                fundamental_data = get_fundamental_data(ticker)
                historical_financials = get_historical_financials(ticker)
                stock_history = get_stock_history(ticker)
                
                # Analyse technique
                if not stock_history.empty:
                    df_reg, tech_indicators = calculate_regression(stock_history)
                    tech_recommendation = "NEUTRE"
                    
                    # Déterminer la recommandation technique
                    if tech_indicators['correlation'] > 0.7 and tech_indicators['model_growth'] > 5:
                        if tech_indicators['deviation'] < -1:
                            tech_recommendation = "ACHAT"
                        elif tech_indicators['deviation'] < 0:
                            tech_recommendation = "ACCUMULATION"
                        elif tech_indicators['deviation'] > 1.5:
                            tech_recommendation = "ALLÈGEMENT"
                        else:
                            tech_recommendation = "CONSERVER"
                    elif tech_indicators['correlation'] < -0.7 and tech_indicators['model_growth'] < -5:
                        if tech_indicators['deviation'] < -1.5:
                            tech_recommendation = "CONSERVER"  # Possible rebond technique
                        else:
                            tech_recommendation = "VENTE"
                    else:
                        # Recommandation basée sur l'écart par rapport à la tendance
                        if tech_indicators['deviation'] < -1.5:
                            tech_recommendation = "ACCUMULATION"
                        elif tech_indicators['deviation'] > 1.5:
                            tech_recommendation = "ALLÈGEMENT"
                else:
                    tech_indicators = {}
                    tech_recommendation = "DONNÉES INSUFFISANTES"
                
                # Calcul de la valeur intrinsèque
                intrinsic_value_result = calculate_intrinsic_value(ticker, fundamental_data, historical_financials)
                
                # Prix actuel
                current_price = None
                if fundamental_data and 'Données de marché' in fundamental_data:
                    current_price = fundamental_data['Données de marché'].get('Prix actuel')
                
                if not current_price and not stock_history.empty:
                    current_price = stock_history['Close'].iloc[-1]
                
                if current_price:
                    # Valeur actuelle de la position
                    position_value = quantity * current_price
                    total_portfolio_value += position_value
                    
                    # Performance de la position
                    performance = ((current_price / entry_price) - 1) * 100
                    
                    # Intrinsic value
                    if intrinsic_value_result['success']:
                        intrinsic_value = intrinsic_value_result['intrinsic_value']
                        intrinsic_value_with_mos = intrinsic_value_result['intrinsic_value_with_mos']
                        fundamental_recommendation = intrinsic_value_result['recommendation']
                    else:
                        intrinsic_value = None
                        intrinsic_value_with_mos = None
                        fundamental_recommendation = "DONNÉES INSUFFISANTES"
                    
                    # Potentiel de hausse (basé sur la valeur intrinsèque)
                    upside_potential = None
                    if intrinsic_value:
                        upside_potential = ((intrinsic_value / current_price) - 1) * 100
                    
                    # Combiner les recommandations techniques et fondamentales
                    # Donnons une importance de 40% à l'analyse technique et 60% à l'analyse fondamentale
                    combined_recommendation = "NEUTRE"
                    
                    if tech_recommendation != "DONNÉES INSUFFISANTES" and fundamental_recommendation != "DONNÉES INSUFFISANTES":
                        # Créer un système de points
                        recommendation_points = {
                            "ACHAT": 2,
                            "ACCUMULATION": 1,
                            "CONSERVER": 0,
                            "ALLÈGEMENT": -1,
                            "VENTE": -2
                        }
                        
                        tech_points = recommendation_points.get(tech_recommendation, 0)
                        fundamental_points = recommendation_points.get(fundamental_recommendation, 0)
                        
                        # Pondérer les points
                        weighted_points = (tech_points * 0.4) + (fundamental_points * 0.6)
                        
                        # Convertir les points en recommandation
                        if weighted_points >= 1.5:
                            combined_recommendation = "ACHAT"
                        elif weighted_points >= 0.5:
                            combined_recommendation = "ACCUMULATION"
                        elif weighted_points <= -1.5:
                            combined_recommendation = "VENTE"
                        elif weighted_points <= -0.5:
                            combined_recommendation = "ALLÈGEMENT"
                        else:
                            combined_recommendation = "CONSERVER"
                    elif tech_recommendation != "DONNÉES INSUFFISANTES":
                        combined_recommendation = tech_recommendation
                    elif fundamental_recommendation != "DONNÉES INSUFFISANTES":
                        combined_recommendation = fundamental_recommendation
                    
                    results.append({
                        'Ticker': ticker,
                        'Nom': fundamental_data['Données générales'].get('Nom', ''),
                        'Quantité': quantity,
                        'PRU': entry_price,
                        'Prix actuel': current_price,
                        'Valeur position': position_value,
                        'Performance (%)': performance,
                        'Valeur intrinsèque': intrinsic_value,
                        'VI avec MOS': intrinsic_value_with_mos,
                        'Potentiel (%)': upside_potential,
                        'Rec. Technique': tech_recommendation,
                        'Rec. Fondamentale': fundamental_recommendation,
                        'Recommandation': combined_recommendation
                    })
            
            # Créer un DataFrame avec les résultats
            results_df = pd.DataFrame(results)
            
            # Ajouter des statistiques sur le portefeuille
            portfolio_stats = {
                'total_value': total_portfolio_value,
                'performance': None
            }
            
            # Calculer la performance globale si possible
            if not results_df.empty:
                total_cost = sum(results_df['Quantité'] * results_df['PRU'])
                portfolio_stats['performance'] = ((total_portfolio_value / total_cost) - 1) * 100
            
            return results_df, portfolio_stats
        
        def optimize_portfolio_allocation(portfolio_df, risk_tolerance='moderate'):
            """
            Optimise l'allocation du portefeuille en fonction des recommandations et du profil de risque
            
            :param portfolio_df: DataFrame du portefeuille analysé
            :param risk_tolerance: Profil de risque (conservative, moderate, aggressive)
            :return: DataFrame avec l'allocation optimisée
            """
            if portfolio_df.empty:
                return pd.DataFrame()
            
            # Initialiser la copie pour l'allocation optimisée
            optimized_df = portfolio_df.copy()
            
            # Valeur totale du portefeuille
            total_value = optimized_df['Valeur position'].sum()
            
            # Ajouter une colonne pour le poids actuel
            optimized_df['Poids actuel (%)'] = (optimized_df['Valeur position'] / total_value) * 100
            
            # Définir les poids cibles selon les recommandations et le profil de risque
            recommendation_weights = {
                'conservative': {
                    'ACHAT': 1.5,
                    'ACCUMULATION': 1.2,
                    'CONSERVER': 1.0,
                    'ALLÈGEMENT': 0.7,
                    'VENTE': 0.3,
                    'DONNÉES INSUFFISANTES': 0.8
                },
                'moderate': {
                    'ACHAT': 2.0,
                    'ACCUMULATION': 1.5,
                    'CONSERVER': 1.0,
                    'ALLÈGEMENT': 0.5,
                    'VENTE': 0.2,
                    'DONNÉES INSUFFISANTES': 0.7
                },
                'aggressive': {
                    'ACHAT': 2.5,
                    'ACCUMULATION': 1.8,
                    'CONSERVER': 1.0,
                    'ALLÈGEMENT': 0.3,
                    'VENTE': 0.1,
                    'DONNÉES INSUFFISANTES': 0.6
                }
            }
            
            # Utiliser le profil de risque spécifié ou moderate par défaut
            weights = recommendation_weights.get(risk_tolerance, recommendation_weights['moderate'])
            
            # Calculer les poids bruts initiaux basés sur les recommandations
            optimized_df['Poids brut'] = optimized_df['Recommandation'].map(weights)
            
            # Ajuster par le potentiel de hausse si disponible
            for idx, row in optimized_df.iterrows():
                if pd.notna(row['Potentiel (%)']) and row['Potentiel (%)'] > 0:
                    # Augmenter le poids pour les actions avec potentiel élevé
                    potentiel_factor = min(1 + (row['Potentiel (%)'] / 100), 1.5)
                    optimized_df.at[idx, 'Poids brut'] *= potentiel_factor
            
            # Normaliser les poids pour qu'ils totalisent 100%
            total_weight = optimized_df['Poids brut'].sum()
            optimized_df['Poids cible (%)'] = (optimized_df['Poids brut'] / total_weight) * 100
            
            # Calculer les ajustements nécessaires
            optimized_df['Ajustement (%)'] = optimized_df['Poids cible (%)'] - optimized_df['Poids actuel (%)']
            
            # Calculer les montants à ajuster
            optimized_df['Montant à ajuster'] = (optimized_df['Ajustement (%)'] / 100) * total_value
            
            # Déterminer les actions à acheter/vendre
            optimized_df['Action'] = optimized_df['Ajustement (%)'].apply(
                lambda x: "ACHETER" if x > 1 else "VENDRE" if x < -1 else "MAINTENIR"
            )
            
            # Nombre approximatif d'actions à acheter/vendre
            optimized_df['Qté à ajuster'] = (optimized_df['Montant à ajuster'] / optimized_df['Prix actuel']).round().astype(int)
            
            return optimized_df[['Ticker', 'Poids actuel (%)', 'Poids cible (%)', 'Ajustement (%)', 
                                'Montant à ajuster', 'Action', 'Qté à ajuster', 'Recommandation']]
        
        # Fonction pour créer un diagramme d'allocation par recommandation
        def create_allocation_by_recommendation_chart(portfolio_df):
            if portfolio_df.empty:
                return None
            
            # Regrouper par recommandation et calculer la somme des valeurs de position
            grouped = portfolio_df.groupby('Recommandation')['Valeur position'].sum().reset_index()
            
            # Calculer le pourcentage
            total = grouped['Valeur position'].sum()
            grouped['Pourcentage'] = (grouped['Valeur position'] / total * 100).round(1)
            
            # Créer un graphique camembert
            fig = go.Figure(data=[go.Pie(
                labels=grouped['Recommandation'],
                values=grouped['Pourcentage'],
                hole=.4,
                marker=dict(
                    colors=[
                        '#4CAF50',  # ACHAT (vert)
                        '#8BC34A',  # ACCUMULATION (vert clair)
                        '#FFC107',  # CONSERVER (jaune)
                        '#FF9800',  # ALLÈGEMENT (orange)
                        '#F44336',  # VENTE (rouge)
                        '#9E9E9E'   # DONNÉES INSUFFISANTES (gris)
                    ]
                )
            )])
            
            fig.update_layout(
                title="Allocation par recommandation (%)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
        
        # Fonction pour créer un graphique de performance vs potentiel
        def create_performance_vs_potential_chart(portfolio_df):
            if portfolio_df.empty or 'Potentiel (%)' not in portfolio_df.columns:
                return None
            
            # Filtrer les lignes avec potentiel non-NaN
            filtered_df = portfolio_df[pd.notna(portfolio_df['Potentiel (%)'])]
            
            if filtered_df.empty:
                return None
            
            # Créer un graphique à bulles
            fig = go.Figure()
            
            # Palette de couleurs pour les recommandations
            color_map = {
                'ACHAT': '#4CAF50',
                'ACCUMULATION': '#8BC34A',
                'CONSERVER': '#FFC107',
                'ALLÈGEMENT': '#FF9800',
                'VENTE': '#F44336',
                'DONNÉES INSUFFISANTES': '#9E9E9E'
            }
            
            # Ajouter les bulles pour chaque action
            fig.add_trace(go.Scatter(
                x=filtered_df['Performance (%)'],
                y=filtered_df['Potentiel (%)'],
                mode='markers',
                marker=dict(
                    size=filtered_df['Valeur position'] / filtered_df['Valeur position'].max() * 40 + 10,
                    color=[color_map.get(rec, '#9E9E9E') for rec in filtered_df['Recommandation']],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=filtered_df['Ticker'],
                hovertemplate="<b>%{text}</b><br>Performance: %{x:.1f}%<br>Potentiel: %{y:.1f}%<extra></extra>"
            ))
            
            # Ajouter des lignes de référence
            fig.add_shape(
                type='line',
                x0=min(filtered_df['Performance (%)']) - 5,
                y0=0,
                x1=max(filtered_df['Performance (%)']) + 5,
                y1=0,
                line=dict(color='gray', dash='dash')
            )
            
            fig.add_shape(
                type='line',
                x0=0,
                y0=min(filtered_df['Potentiel (%)']) - 5,
                x1=0,
                y1=max(filtered_df['Potentiel (%)']) + 5,
                line=dict(color='gray', dash='dash')
            )
            
            # Mise en page
            fig.update_layout(
                title="Performance vs Potentiel",
                xaxis_title="Performance (%)",
                yaxis_title="Potentiel de hausse (%)",
                height=500,
                margin=dict(l=50, r=20, t=40, b=50)
            )
            
            return fig
        
        # Définir la clé de cache pour le portefeuille
        PORTFOLIO_CACHE_KEY = "user_portfolio"
        
        # Initialiser le portefeuille dans la session si non existant
        if PORTFOLIO_CACHE_KEY not in st.session_state:
            st.session_state[PORTFOLIO_CACHE_KEY] = pd.DataFrame(
                columns=['Ticker', 'Nom', 'Quantité', 'PRU']
            )
        
        # Interface utilisateur pour l'onglet Portefeuille
        portfolio_tabs = st.tabs(["Mon Portefeuille", "Analyse Détaillée", "Optimisation"])
        
        # Sous-onglet: Mon Portefeuille
        with portfolio_tabs[0]:
            st.markdown("### Mon Portefeuille")
            
            # Colonnes pour l'interface
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Ajouter une action")
                
                # Récupérer la structure du marché pour la sélection d'actions
                market_structure = get_market_structure()
                flattened_stocks = market_structure['all_stocks']
                
                # Créer une liste déroulante pour la sélection d'actions
                stock_options = list(flattened_stocks.keys())
                selected_stock_name = st.selectbox("Sélectionner une action", [""] + stock_options)
                
                # Récupérer le ticker correspondant
                ticker = ""
                if selected_stock_name and selected_stock_name in flattened_stocks:
                    ticker = flattened_stocks[selected_stock_name]['ticker']
                
                # Champs pour la quantité et le prix d'achat
                col_qty, col_price = st.columns(2)
                with col_qty:
                    quantity = st.number_input("Quantité", min_value=1, value=1, step=1)
                with col_price:
                    entry_price = st.number_input("Prix d'achat unitaire", min_value=0.01, value=100.0, step=0.01)
                
                # Bouton pour ajouter l'action au portefeuille
                if st.button("Ajouter au portefeuille"):
                    if selected_stock_name and ticker:
                        # Vérifier si l'action existe déjà dans le portefeuille
                        existing = st.session_state[PORTFOLIO_CACHE_KEY][
                            st.session_state[PORTFOLIO_CACHE_KEY]['Ticker'] == ticker
                        ]
                        
                        if len(existing) > 0:
                            # Mettre à jour la position existante
                            idx = existing.index[0]
                            old_qty = st.session_state[PORTFOLIO_CACHE_KEY].loc[idx, 'Quantité']
                            old_price = st.session_state[PORTFOLIO_CACHE_KEY].loc[idx, 'PRU']
                            
                            # Calculer le nouveau PRU
                            new_qty = old_qty + quantity
                            new_pru = ((old_qty * old_price) + (quantity * entry_price)) / new_qty
                            
                            st.session_state[PORTFOLIO_CACHE_KEY].loc[idx, 'Quantité'] = new_qty
                            st.session_state[PORTFOLIO_CACHE_KEY].loc[idx, 'PRU'] = new_pru
                            
                            st.success(f"Position mise à jour pour {selected_stock_name} ({ticker})")
                        else:
                            # Ajouter une nouvelle position
                            new_row = pd.DataFrame({
                                'Ticker': [ticker],
                                'Nom': [selected_stock_name],
                                'Quantité': [quantity],
                                'PRU': [entry_price]
                            })
                            
                            st.session_state[PORTFOLIO_CACHE_KEY] = pd.concat(
                                [st.session_state[PORTFOLIO_CACHE_KEY], new_row], 
                                ignore_index=True
                            )
                            
                            st.success(f"Action ajoutée au portefeuille: {selected_stock_name} ({ticker})")
                    else:
                        st.error("Veuillez sélectionner une action valide")
            
            with col2:
                st.markdown("#### Importer un portefeuille")
                
                # Option pour importer un fichier CSV
                uploaded_file = st.file_uploader("Importer un fichier CSV", type=['csv'])
                
                if uploaded_file is not None:
                    try:
                        # Déterminer le séparateur
                        first_line = uploaded_file.readline().decode('utf-8-sig')
                        uploaded_file.seek(0)  # Réinitialiser le pointeur de fichier
                        
                        sep = ',' if ',' in first_line else ';'
                        
                        # Lire le fichier CSV
                        import_df = pd.read_csv(uploaded_file, sep=sep)

                        # Vérifier les colonnes requises
                        required_cols = ['Ticker', 'Nom', 'Quantité', 'PRU']
                        
                        # Mapper les noms de colonnes possibles
                        col_mapping = {
                            'ticker': 'Ticker', 'symbol': 'Ticker', 'code': 'Ticker',
                            'nom': 'Nom', 'name': 'Nom', 'designation': 'Nom',
                            'quantité': 'Quantité', 'quantite': 'Quantité', 'quantity': 'Quantité', 'qty': 'Quantité',
                            'pru': 'PRU', 'prix': 'PRU', 'price': 'PRU', 'cout_moyen': 'PRU', 'average_cost': 'PRU'
                        }
                        
                        # Normaliser les noms de colonnes (ignorer la casse)
                        import_df.columns = [col.lower() for col in import_df.columns]
                        
                        # Appliquer le mapping
                        for col in import_df.columns:
                            if col in col_mapping:
                                import_df.rename(columns={col: col_mapping[col]}, inplace=True)
                        
                        # Vérifier que toutes les colonnes requises sont présentes
                        missing_cols = [col for col in required_cols if col not in import_df.columns]
                        
                        if missing_cols:
                            st.error(f"Colonnes manquantes dans le fichier: {', '.join(missing_cols)}")
                        else:
                            # Filtrer les colonnes nécessaires
                            import_df = import_df[required_cols].copy()
                            
                            # Ajouter le nom des actions si manquant
                            if 'Nom' not in import_df.columns:
                                import_df['Nom'] = import_df['Ticker']
                            
                            # Convertir les colonnes numériques
                            import_df['Quantité'] = pd.to_numeric(import_df['Quantité'], errors='coerce')
                            import_df['PRU'] = pd.to_numeric(import_df['PRU'], errors='coerce')
                            
                            # Filtrer les lignes invalides
                            import_df = import_df.dropna()
                            
                            # Remplacer le portefeuille existant
                            st.session_state[PORTFOLIO_CACHE_KEY] = import_df
                            
                            st.success(f"Portefeuille importé avec succès: {len(import_df)} positions")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'importation: {str(e)}")
            
            # Afficher le portefeuille actuel
            st.markdown("### Positions actuelles")
            
            portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
            
            if portfolio_df.empty:
                st.info("Votre portefeuille est vide. Ajoutez des actions pour commencer l'analyse.")
            else:
                # Afficher le tableau des positions
                edited_df = st.data_editor(
                    portfolio_df,
                    hide_index=True,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Nom": st.column_config.TextColumn("Nom"),
                        "Quantité": st.column_config.NumberColumn("Quantité", min_value=1, step=1),
                        "PRU": st.column_config.NumberColumn("PRU (€)", min_value=0.01, format="%.2f €")
                    }
                )
                
                # Mettre à jour le portefeuille si modifié
                if edited_df is not None and not edited_df.equals(portfolio_df):
                    st.session_state[PORTFOLIO_CACHE_KEY] = edited_df
                
                # Option pour supprimer des positions
                st.markdown("#### Supprimer une position")
                
                col_select, col_delete = st.columns([3, 1])
                
                with col_select:
                    positions = [f"{row['Nom']} ({row['Ticker']})" for _, row in portfolio_df.iterrows()]
                    selected_position = st.selectbox("Sélectionner une position", [""] + positions)
                
                with col_delete:
                    st.write("")  # Espace pour aligner le bouton
                    if st.button("Supprimer") and selected_position:
                        # Extraire le ticker de la position sélectionnée
                        selected_ticker = selected_position.split('(')[1].split(')')[0]
                        
                        # Filtrer le DataFrame pour supprimer la position
                        st.session_state[PORTFOLIO_CACHE_KEY] = portfolio_df[
                            portfolio_df['Ticker'] != selected_ticker
                        ]
                        
                        st.success(f"Position supprimée: {selected_position}")
                        st.rerun()
                
                # Option pour exporter le portefeuille
                if st.button("Exporter le portefeuille (CSV)"):
                    csv = portfolio_df.to_csv(index=False)
                    
                    # Créer un lien de téléchargement
                    from datetime import datetime
                    
                    filename = f"portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="Télécharger le fichier CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
        
        # Sous-onglet: Analyse Détaillée
        with portfolio_tabs[1]:
            st.markdown("### Analyse Détaillée du Portefeuille")
            
            portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
            
            if portfolio_df.empty:
                st.info("Veuillez d'abord ajouter des actions à votre portefeuille.")
            else:
                # Bouton pour déclencher l'analyse
                if st.button("Analyser le portefeuille"):
                    with st.spinner("Analyse en cours..."):
                        # Effectuer l'analyse du portefeuille
                        results_df, portfolio_stats = analyze_portfolio(portfolio_df)
                        
                        if not results_df.empty:
                            # Enregistrer les résultats dans la session
                            st.session_state['portfolio_analysis'] = {
                                'results': results_df,
                                'stats': portfolio_stats
                            }
                            
                            st.success("Analyse terminée!")
                
                # Si des résultats d'analyse sont disponibles, les afficher
                if 'portfolio_analysis' in st.session_state:
                    analysis = st.session_state['portfolio_analysis']
                    results_df = analysis['results']
                    portfolio_stats = analysis['stats']
                    
                    # Afficher les statistiques du portefeuille
                    st.markdown("#### Vue d'ensemble")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Valeur totale", 
                            f"{portfolio_stats['total_value']:,.2f} €"
                        )
                    
                    with col2:
                        if portfolio_stats['performance'] is not None:
                            st.metric(
                                "Performance globale", 
                                f"{portfolio_stats['performance']:.2f}%",
                                delta=f"{portfolio_stats['performance']:.2f}%"
                            )
                    
                    with col3:
                        if not results_df.empty and 'Potentiel (%)' in results_df.columns:
                            # Calculer le potentiel moyen pondéré
                            valid_potential = results_df[pd.notna(results_df['Potentiel (%)'])]
                            
                            if not valid_potential.empty:
                                weighted_potential = sum(
                                    valid_potential['Potentiel (%)'] * valid_potential['Valeur position']
                                ) / sum(valid_potential['Valeur position'])
                                
                                st.metric(
                                    "Potentiel moyen",
                                    f"{weighted_potential:.2f}%",
                                    delta=f"{weighted_potential:.2f}%"
                                )
                    
                    # Visualisations
                    st.markdown("#### Visualisations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Graphique d'allocation par recommandation
                        rec_chart = create_allocation_by_recommendation_chart(results_df)
                        if rec_chart:
                            st.plotly_chart(rec_chart, use_container_width=True)
                    
                    with viz_col2:
                        # Graphique performance vs potentiel
                        perf_pot_chart = create_performance_vs_potential_chart(results_df)
                        if perf_pot_chart:
                            st.plotly_chart(perf_pot_chart, use_container_width=True)
                    
                    # Tableau détaillé des positions
                    st.markdown("#### Analyse détaillée des positions")
                    
                    # Formater le tableau pour l'affichage
                    display_df = results_df.copy()
                    
                    # Formater les colonnes numériques
                    numeric_cols = [
                        'Prix actuel', 'Valeur position', 'Performance (%)', 
                        'Valeur intrinsèque', 'VI avec MOS', 'Potentiel (%)'
                    ]
                    
                    for col in numeric_cols:
                        if col in display_df.columns:
                            if 'Prix' in col or 'Valeur' in col:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:,.2f} €" if pd.notna(x) else "N/A"
                                )
                            else:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                                )
                    
                    # Afficher le tableau
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Exporter les résultats
                    if st.button("Exporter l'analyse (CSV)"):
                        csv = results_df.to_csv(index=False)
                        
                        # Créer un lien de téléchargement
                        from datetime import datetime
                        
                        filename = f"analyse_portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="Télécharger l'analyse en CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv"
                        )
        
        # Sous-onglet: Optimisation
        with portfolio_tabs[2]:
            st.markdown("### Optimisation du Portefeuille")
            
            # Vérifier si l'analyse a été effectuée
            if 'portfolio_analysis' not in st.session_state:
                st.info("Veuillez d'abord analyser votre portefeuille dans l'onglet 'Analyse Détaillée'.")
            else:
                # Profil de risque
                risk_profile = st.radio(
                    "Sélectionnez votre profil de risque",
                    ["conservative", "moderate", "aggressive"],
                    captions=["Prudent: privilégie la stabilité", 
                            "Modéré: équilibre risque et rendement", 
                            "Dynamique: maximise le rendement potentiel"],
                    horizontal=True
                )
                
                if st.button("Optimiser l'allocation"):
                    with st.spinner("Optimisation en cours..."):
                        # Récupérer les résultats de l'analyse
                        analysis = st.session_state['portfolio_analysis']
                        results_df = analysis['results']
                        
                        # Optimiser l'allocation
                        optimized_df = optimize_portfolio_allocation(results_df, risk_profile)
                        
                        # Enregistrer les résultats dans la session
                        st.session_state['portfolio_optimization'] = optimized_df
                        
                        st.success("Optimisation terminée!")
                
                # Si des résultats d'optimisation sont disponibles, les afficher
                if 'portfolio_optimization' in st.session_state:
                    optimized_df = st.session_state['portfolio_optimization']
                    
                    # Tableau d'allocation optimisée
                    st.markdown("#### Allocation optimisée")
                    
                    # Formater le tableau pour l'affichage
                    display_df = optimized_df.copy()
                    
                    # Formater les colonnes numériques
                    display_df['Poids actuel (%)'] = display_df['Poids actuel (%)'].apply(lambda x: f"{x:.2f}%")
                    display_df['Poids cible (%)'] = display_df['Poids cible (%)'].apply(lambda x: f"{x:.2f}%")
                    display_df['Ajustement (%)'] = display_df['Ajustement (%)'].apply(lambda x: f"{x:.2f}%")
                    display_df['Montant à ajuster'] = display_df['Montant à ajuster'].apply(lambda x: f"{x:,.2f} €")
                    
                    # Appliquer une mise en forme conditionnelle
                    def highlight_action(val):
                        if val == "ACHETER":
                            return 'background-color: #d4edda; color: #155724'
                        elif val == "VENDRE":
                            return 'background-color: #f8d7da; color: #721c24'
                        return ''
                    
                    # Afficher le tableau
                    st.dataframe(
                        display_df.style.map(highlight_action, subset=['Action']),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualisation de l'allocation
                    st.markdown("#### Visualisation de l'allocation")
                    
                    # Créer un graphique de comparaison
                    allocation_fig = go.Figure()

                    # Ajouter les barres pour le poids actuel
                    allocation_fig.add_trace(go.Bar(
                        x=optimized_df['Ticker'],
                        y=optimized_df['Poids actuel (%)'],#.str.rstrip('%').astype(float),
                        name='Allocation actuelle',
                        marker_color='rgb(158,202,225)'
                    ))
                    
                    # Ajouter les barres pour le poids cible
                    allocation_fig.add_trace(go.Bar(
                        x=optimized_df['Ticker'],
                        y=optimized_df['Poids cible (%)'],#.str.rstrip('%').astype(float),
                        name='Allocation cible',
                        marker_color='rgb(94,158,217)'
                    ))
                    
                    # Mise en page
                    allocation_fig.update_layout(
                        title="Comparaison des allocations",
                        xaxis_title="Ticker",
                        yaxis_title="Allocation (%)",
                        barmode='group',
                        height=500,
                        margin=dict(l=50, r=20, t=40, b=50)
                    )
                    
                    st.plotly_chart(allocation_fig, use_container_width=True)
                    
                    # Exporter les résultats
                    if st.button("Exporter l'allocation optimisée (CSV)"):
                        csv = optimized_df.to_csv(index=False)
                        
                        # Créer un lien de téléchargement
                        from datetime import datetime
                        
                        filename = f"allocation_optimisee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="Télécharger l'allocation en CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv"
                        )  

    # Footer
    st.markdown("---")
    st.markdown("📊 Dashboard développé pour l'analyse des marchés financiers. Données fournies par Yahoo Finance. Ce dashboard est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement.")

# Lancement de l'application
if __name__ == "__main__":
    main()
