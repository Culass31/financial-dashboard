import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import requests
import time
import re
import os

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
    
def display_price_with_trend(label, current_price, previous_price, format_func=None):
    """
    Affiche un prix avec son indicateur de tendance (flèche et pourcentage)
    
    :param label: Libellé à afficher (ex: "Prix actuel")
    :param current_price: Prix actuel
    :param previous_price: Prix de référence pour calculer la tendance
    :param format_func: Fonction optionnelle pour formater le prix (par défaut None)
    :return: None, affiche directement avec st.markdown
    """
    if current_price is not None and previous_price is not None:
        # Calculer le pourcentage de variation
        change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # Déterminer la flèche et la couleur en fonction de la tendance
        if change_pct > 0:
            arrow = "↑"
            color = "green"
        elif change_pct < 0:
            arrow = "↓"
            color = "red"
        else:
            arrow = "→"
            color = "gray"
        
        # Formater la valeur si une fonction de formatage est fournie
        formatted_price = format_func(current_price) if format_func else f"{current_price:.2f}"
        
        # Afficher avec le style approprié
        st.markdown(
            f"**{label}:** {formatted_price} "
            f"<span style='color:{color}'>{arrow} {abs(change_pct):.2f}%</span>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"**{label}:** N/A")

# Structure de marché hiérarchique organisée par région et pays
@st.cache_data(ttl=86400)  # Mise en cache pour 24 heures
def get_market_structure():
    # Lire le fichier CSV avec l'encodage approprié
    try:
        # Essayer différents encodages courants
        encodings = ['latin1', 'ISO-8859-1', 'windows-1252', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv("https://raw.githubusercontent.com/Culass31/financial-dashboard/main/libelles.csv", sep=";", encoding=encoding)
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
    
    # Créer une structure hiérarchique par région et pays
    market_structure = {}
    
    # Parcourir chaque ligne du DataFrame
    for _, row in df.iterrows():
        region = row['Région']
        pays = row['Pays']
        isin = row['ISIN']
        nom = row['nom']
        ticker = row['ticker']
        
        # Ignorer les lignes avec des valeurs manquantes
        if pd.isna(region) or pd.isna(pays) or pd.isna(ticker) or pd.isna(nom):
            continue
            
        # Initialiser la région si elle n'existe pas
        if region not in market_structure:
            market_structure[region] = {}
            
        # Initialiser le pays s'il n'existe pas
        if pays not in market_structure[region]:
            market_structure[region][pays] = {}
            
        # Ajouter l'action
        market_structure[region][pays][nom] = {
            'ticker': ticker,
            'isin': isin
        }
    
    return market_structure

# Fonction pour obtenir le code ISIN
def get_isin_for_ticker(ticker_symbol):
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

# Fonction pour obtenir l'historique des cours d'une action
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

# Fonction pour calculer la régression linéaire et les indicateurs techniques
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

# Fonction pour obtenir les données fondamentales d'une action
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
        
        # Données de marché
        market_data = {
            "Prix actuel": info.get('currentPrice', info.get('regularMarketPrice', None)),
            "Précédente clôture": info.get('previousClose', None),
            "Volume": info.get('volume', None),
            "Volume moyen": info.get('averageVolume', None),
            "Capitalisation boursière": info.get('marketCap', None),
            "Beta": info.get('beta', None),
            "Rendement du dividende (%)": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None
        }
        
        # Données fondamentales
        fundamental_data = {
            "PER": info.get('trailingPE', None),
            "PEG": info.get('pegRatio', None),
            "Cours/Valeur Comptable": info.get('priceToBook', None),
            "Marge bénéficiaire (%)": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
            "ROE (%)": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
            "ROA (%)": info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else None,
            "Ratio d'endettement": info.get('debtToEquity', None),
            "Croissance du BPA (%)": info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None,
            "Croissance du CA (%)": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None
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
            "Données de marché": market_data,
            "Données fondamentales": fundamental_data,
            "Données financières": financial_data
        }
    
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données fondamentales pour {ticker}: {str(e)}")
        return None

def analyze_consistency(series, min_length=5, min_growth=None, min_value=None, max_value=None, within_percentage=None):
    if len(series) < min_length:
        return False, "Données historiques insuffisantes"

    if min_value is not None and any(series < min_value):
        return False, f"Valeur inférieure au minimum ({min_value})"

    if max_value is not None and any(series > max_value):
        return False, f"Valeur supérieure au maximum ({max_value})"

    if min_growth is not None and len(series) > 1:
        growth_rates = series.pct_change().dropna() * 100
        if any(growth_rates < min_growth):
            return False, f"Croissance inférieure au minimum ({min_growth:.2f}%)"

    if within_percentage is not None and len(series) > 1:
        first_value = series.iloc[0]
        lower_bound = first_value * (1 - within_percentage / 100)
        upper_bound = first_value * (1 + within_percentage / 100)
        if any((series < lower_bound) | (series > upper_bound)):
            return False, f"Variation supérieure à {within_percentage:.2f}% par rapport à la première valeur"

    return True, "Consistant"

def buffett_stock_screener(stocks_dict,
                           min_revenue_growth=5,
                           min_net_profit_margin=10,
                           min_roe=15,
                           max_debt_to_equity=1,
                           min_free_cashflow=0,
                           min_gross_margin_consistency=40,
                           max_rd_sales_ratio=0.05,
                           max_interest_ebit_ratio=0.20,
                           min_net_income_growth_consistency=8):
    """
    Screener d'actions basé sur les principes de Warren Buffett avec critères de cohérence.

    :param stocks_dict: Dictionnaire des actions à analyser
    :param min_revenue_growth: Croissance des revenus minimale (ponctuelle)
    :param min_net_profit_margin: Marge nette minimale (ponctuelle)
    :param min_roe: Rendement des capitaux propres minimal (ponctuel)
    :param max_debt_to_equity: Ratio dette/fonds propres maximal (ponctuel)
    :param min_free_cashflow: Free cash flow minimal (ponctuel)
    :param min_gross_margin_consistency: Marge brute minimale et cohérente sur 5 ans (%)
    :param max_rd_sales_ratio: Ratio maximal des dépenses R&D sur le chiffre d'affaires (ponctuel)
    :param max_interest_ebit_ratio: Ratio maximal des charges d'intérêts sur le résultat avant intérêts et impôts (EBIT) (ponctuel)
    :param min_net_income_growth_consistency: Croissance annuelle minimale et cohérente du résultat net sur 5 ans (%)
    :return: DataFrame des actions éligibles
    """
    eligible_stocks = []
    for stock_name, stock_info in stocks_dict.items():
        ticker = stock_info['ticker']
        try:
            stock_data = yf.Ticker(ticker)
            info = stock_data.info

            # Critères ponctuels
            revenue_growth = info.get('earningsGrowth', 0) * 100
            net_profit_margin = info.get('profitMargins', 0) * 100
            roe = info.get('returnOnEquity', 0) * 100
            total_equity = info.get('totalEquity', 1)
            total_debt = info.get('totalDebt', 0)
            debt_to_equity = total_debt / total_equity if total_equity != 0 else float('inf')
            free_cashflow = info.get('freeCashflow', 0)
            rd_expenses = info.get('researchDevelopment', 0)
            total_revenue = info.get('totalRevenue', 1)
            rd_sales_ratio = rd_expenses / total_revenue if total_revenue != 0 else 0
            ebit = info.get('ebit', 1)
            interest_expense = info.get('interestExpense', 0)
            interest_ebit_ratio = interest_expense / ebit if ebit != 0 else 0

            # Récupérer l'historique des données financières
            income_history, balance_history, cashflow_history = get_historical_financials(ticker, period='max')

            consistent_gross_margin, gross_margin_reason = False, "N/A"
            consistent_net_income_growth, net_income_growth_reason = False, "N/A"

            if not income_history.empty:
                if 'Gross Profit' in income_history.index and 'Total Revenue' in income_history.index:
                    gross_margins_history = ((income_history.loc['Gross Profit'] / income_history.loc['Total Revenue']) * 100).dropna()
                    if not gross_margins_history.empty:
                        consistent_gross_margin, gross_margin_reason = analyze_consistency(gross_margins_history, min_value=min_gross_margin_consistency)

                if 'Net Income' in income_history.index:
                    net_incomes_history = income_history.loc['Net Income'].dropna()
                    if len(net_incomes_history) > 1:
                        net_income_growth_rates = net_incomes_history.pct_change().dropna() * 100
                        consistent_net_income_growth, net_income_growth_reason = analyze_consistency(net_income_growth_rates, min_growth=min_net_income_growth_consistency, min_length=4) # Analyse sur les 4 dernières années pour 5 points de données

            # Vérifier tous les critères
            if (revenue_growth >= min_revenue_growth and
                    net_profit_margin >= min_net_profit_margin and
                    roe >= min_roe and
                    debt_to_equity <= max_debt_to_equity and
                    free_cashflow >= min_free_cashflow and
                    rd_sales_ratio <= max_rd_sales_ratio and
                    interest_ebit_ratio <= max_interest_ebit_ratio and
                    consistent_gross_margin and
                    consistent_net_income_growth):
                
                # Ajout du pays et de la région
                region = "N/A"
                country = "N/A"
                
                # Trouver la région et le pays de l'action
                if hasattr(stock_info, 'region') and hasattr(stock_info, 'country'):
                    region = stock_info.region
                    country = stock_info.country
                
                # Récupération de l'ISIN
                isin = get_isin_for_ticker(ticker)
                
                stock_details = {
                    'Nom': stock_name,
                    'Ticker': ticker,
                    'ISIN': isin,
                    'Région': region,
                    'Pays': country,
                    'Croissance des revenus (%)': f"{revenue_growth:.2f}",
                    'Marge nette (%)': f"{net_profit_margin:.2f}",
                    'ROE (%)': f"{roe:.2f}",
                    'Dette/Fonds propres': f"{debt_to_equity:.2f}",
                    'Free Cash Flow': format_number(free_cashflow),
                    'R&D/Ventes (%)': f"{rd_sales_ratio * 100:.2f}",
                    'Intérêts/EBIT (%)': f"{interest_ebit_ratio * 100:.2f}",
                    'Marge Brute Consistante': gross_margin_reason,
                    'Croissance RN Consistante': net_income_growth_reason
                }
                eligible_stocks.append(stock_details)

        except Exception as e:
            st.warning(f"Erreur pour {stock_name}: {e}")

    return pd.DataFrame(eligible_stocks)

# Fonction pour adapter un dictionnaire d'actions à partir de la structure de marché
def flatten_market_structure(market_structure, selected_region=None, selected_country=None):
    """
    Convertit la structure hiérarchique en dictionnaire plat d'actions
    
    :param market_structure: Structure hiérarchique des marchés
    :param selected_region: Région spécifique (optionnel)
    :param selected_country: Pays spécifique (optionnel)
    :return: Dictionnaire plat d'actions
    """
    flattened_stocks = {}
    
    # Si une région et un pays sont spécifiés, ne prendre que ces actions
    if selected_region and selected_country:
        if selected_region in market_structure and selected_country in market_structure[selected_region]:
            for stock_name, stock_info in market_structure[selected_region][selected_country].items():
                stock_info_with_location = stock_info.copy()
                stock_info_with_location['region'] = selected_region
                stock_info_with_location['country'] = selected_country
                flattened_stocks[stock_name] = stock_info_with_location
            return flattened_stocks
        else:
            return {}
    
    # Si seulement la région est spécifiée, prendre toutes les actions de cette région
    elif selected_region:
        if selected_region in market_structure:
            for country, stocks in market_structure[selected_region].items():
                for stock_name, stock_info in stocks.items():
                    # Ajouter des informations sur la région et le pays
                    stock_info_with_location = stock_info.copy()
                    stock_info_with_location['region'] = selected_region
                    stock_info_with_location['country'] = country
                    flattened_stocks[stock_name] = stock_info_with_location
        return flattened_stocks
    
    # Sinon, prendre toutes les actions de toutes les régions
    else:
        for region, countries in market_structure.items():
            for country, stocks in countries.items():
                for stock_name, stock_info in stocks.items():
                    # Ajouter des informations sur la région et le pays
                    stock_info_with_location = stock_info.copy()
                    stock_info_with_location['region'] = region
                    stock_info_with_location['country'] = country
                    flattened_stocks[stock_name] = stock_info_with_location
    
    return flattened_stocks

# Interface utilisateur Streamlit
def main():
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
    .block-container {
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête de l'application
    st.title("📈 Dashboard d'Analyse des Marchés Financiers")
    st.markdown("#### Analyse technique, fondamentale et screening d'actions selon les critères de Warren Buffett")
    
    # Sidebar pour les paramètres
    st.sidebar.title("🔍 Sélection du Marché")
    
    # Récupération de la structure du marché
    with st.spinner("Chargement des marchés...", show_time=True):
        market_structure = get_market_structure()
    
    # Gestion des onglets
    tab1, tab2, tab3 = st.tabs(["📊 Analyse Technique", "📑 Analyse Fondamentale", "🔍 Screener"])
    
    # Sélection de la région et du pays
    regions = list(market_structure.keys())
    selected_region = st.sidebar.selectbox("Région", regions)
    
    if selected_region:
        countries = list(market_structure[selected_region].keys())
        selected_country = st.sidebar.selectbox("Pays", countries)
        
        if selected_country:
            stocks = market_structure[selected_region][selected_country]
            stock_names = list(stocks.keys())
            
            # Ajouter une option de recherche/filtre pour les actions
            search_query = st.sidebar.text_input("Rechercher une action", "")
            if search_query:
                filtered_stock_names = [name for name in stock_names if search_query.lower() in name.lower()]
                if filtered_stock_names:
                    stock_names = filtered_stock_names
                else:
                    st.sidebar.warning(f"Aucune action trouvée pour '{search_query}'")
            
            if stock_names:
                selected_stock = st.sidebar.selectbox("Action", stock_names)
                
                if selected_stock:
                    ticker = stocks[selected_stock]['ticker']
                    
                    # Afficher l'ISIN
                    with st.sidebar.expander("Informations supplémentaires"):
                        isin = get_isin_for_ticker(ticker)
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**ISIN:** {isin}")
                    
                    # Onglet 1: Analyse Technique
                    with tab1:
                        st.subheader(f"Analyse Technique de {selected_stock} ({ticker})")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Sélection de la période
                            periods = {"1 mois": "1mo", "3 mois": "3mo", "6 mois": "6mo", "1 an": "1y", "2 ans": "2y", "5 ans": "5y", "Max": "max"}
                            selected_period_name = st.selectbox("Période", list(periods.keys()))
                            selected_period = periods[selected_period_name]
                            
                            # Récupération et préparation des données
                            with st.spinner("Chargement des données historiques..."):
                                df = get_stock_history(ticker, period=selected_period)
                                
                                if not df.empty:
                                    # Calcul de la régression et des indicateurs
                                    df_reg, indicators = calculate_regression(df)
                                    
                                    # Création du graphique avec Plotly
                                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                        vertical_spacing=0.1, 
                                                        row_heights=[0.7, 0.3])
                                    
                                    # Tracé du prix
                                    fig.add_trace(
                                        go.Candlestick(x=df_reg.index,
                                                        open=df_reg['Open'],
                                                        high=df_reg['High'],
                                                        low=df_reg['Low'],
                                                        close=df_reg['Close'],
                                                        name="Prix"),
                                        row=1, col=1
                                    )
                                    
                                    # Tracé de la régression
                                    fig.add_trace(
                                        go.Scatter(x=df_reg.index, y=df_reg['Regression'],
                                                    mode='lines', name='Tendance',
                                                    line=dict(color='orange', width=2)),
                                        row=1, col=1
                                    )
                                    
                                    # Calcul et tracé des bandes de Bollinger (2 écarts types)
                                    std_dev = df_reg['Deviation'].std()
                                    fig.add_trace(
                                        go.Scatter(x=df_reg.index, y=df_reg['Regression'] + 2*std_dev,
                                                    mode='lines', name='+2σ',
                                                    line=dict(color='red', width=1, dash='dash')),
                                        row=1, col=1
                                    )
                                    
                                    fig.add_trace(
                                        go.Scatter(x=df_reg.index, y=df_reg['Regression'] - 2*std_dev,
                                                    mode='lines', name='-2σ',
                                                    line=dict(color='green', width=1, dash='dash')),
                                        row=1, col=1
                                    )
                                    
                                    # Tracé du volume
                                    colors = ['red' if row['Open'] > row['Close'] else 'green' for i, row in df_reg.iterrows()]
                                    fig.add_trace(
                                        go.Bar(x=df_reg.index, y=df_reg['Volume'],
                                                name='Volume', marker_color=colors,
                                                marker_line_width=0),
                                        row=2, col=1
                                    )
                                    
                                    # Mise en forme du graphique
                                    fig.update_layout(
                                        title=f"{selected_stock} - Analyse de Tendance",
                                        height=600,
                                        xaxis_rangeslider_visible=False,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )
                                    
                                    # Affichage du graphique
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                else:
                                    st.error(f"Aucune donnée historique disponible pour {ticker}")
                        
                        with col2:
                            # Affichage des indicateurs techniques
                            if 'indicators' in locals() and indicators:
                                st.markdown("<p class='medium-font'>Indicateurs Techniques</p>", unsafe_allow_html=True)
                                
                                # Prix actuel
                                st.markdown("<div class='custom-metric'>", unsafe_allow_html=True)
                                current_price = df['Close'].iloc[-1] if not df.empty else None
                                previous_price = df['Close'].iloc[-2] if not df.empty and len(df) > 1 else None
                                
                                if current_price:
                                    # Utiliser la fonction commune pour afficher le prix avec tendance
                                    display_price_with_trend("Prix actuel", current_price, previous_price)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Tendance
                                st.markdown("<div class='custom-metric'>", unsafe_allow_html=True)
                                correlation = indicators['correlation']
                                corr_text = f"{correlation:.2f}"
                                corr_class = "positive" if correlation > 0.7 else "negative" if correlation < -0.7 else "neutral"
                                st.markdown(f"**Corrélation:** <span class='{corr_class}'>{corr_text}</span>", unsafe_allow_html=True)
                                
                                trend_class = "positive" if correlation > 0.7 else "negative" if correlation < -0.7 else "neutral"
                                trend_text = "Haussière" if correlation > 0.7 else "Baissière" if correlation < -0.7 else "Neutre"
                                st.markdown(f"**Tendance:** <span class='{trend_class}'>{trend_text}</span>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Croissance du modèle
                                st.markdown("<div class='custom-metric'>", unsafe_allow_html=True)
                                model_growth = indicators['model_growth']
                                growth_text = f"{model_growth:.2f}% par an"
                                growth_class = "positive" if model_growth > 5 else "negative" if model_growth < 0 else "neutral"
                                st.markdown(f"**Croissance du modèle:** <span class='{growth_class}'>{growth_text}</span>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Écart par rapport à la régression
                                st.markdown("<div class='custom-metric'>", unsafe_allow_html=True)
                                deviation = indicators['deviation']
                                dev_text = f"{deviation:.2f}σ"
                                dev_class = "negative" if deviation > 1.5 else "positive" if deviation < -1.5 else "neutral"
                                st.markdown(f"**Écart actuel:** <span class='{dev_class}'>{dev_text}</span>", unsafe_allow_html=True)
                                
                                signal_text = ""
                                if deviation > 1.5:
                                    signal_text = "Survente potentielle"
                                    signal_class = "negative"
                                elif deviation < -1.5:
                                    signal_text = "Surachat potentiel"
                                    signal_class = "positive"
                                else:
                                    signal_text = "Zone neutre"
                                    signal_class = "neutral"
                                
                                st.markdown(f"**Signal:** <span class='{signal_class}'>{signal_text}</span>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Performances
                                st.markdown("<div class='custom-metric'>", unsafe_allow_html=True)
                                prog_1y = indicators['prog_1y']
                                prog_5y = indicators['prog_5y']
                                
                                prog1y_class = "positive" if prog_1y > 0 else "negative"
                                prog5y_class = "positive" if prog_5y > 0 else "negative"
                                
                                st.markdown(f"**Perf 1 an:** <span class='{prog1y_class}'>{prog_1y:.2f}%</span>", unsafe_allow_html=True)
                                st.markdown(f"**Perf 5 ans:** <span class='{prog5y_class}'>{prog_5y:.2f}%</span>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Onglet 2: Analyse Fondamentale
                    with tab2:
                        st.subheader(f"Analyse Fondamentale de {selected_stock} ({ticker})")
                        
                        with st.spinner("Chargement des données fondamentales..."):
                            fundamental_data = get_fundamental_data(ticker)
                            
                            if fundamental_data:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Données de marché
                                    st.markdown("<p class='medium-font'>Données de Marché</p>", unsafe_allow_html=True)
                                    
                                    # Afficher prix actuel avec tendance
                                    display_price_with_trend(
                                        "Prix actuel",
                                        fundamental_data["Données de marché"]["Prix actuel"],
                                        fundamental_data["Données de marché"]["Précédente clôture"],
                                        format_func=format_number
                                    )
    
                                    # Continuer avec les autres données de marché
                                    for key, value in fundamental_data["Données de marché"].items():
                                        if key not in ["Prix actuel"]:  # Ignorer les éléments déjà affichés
                                            if value is not None:
                                                formatted_value = format_number(value)
                                                st.markdown(f"**{key}:** {formatted_value}")
                                            else:
                                                st.markdown(f"**{key}:** N/A")
                                    
                                    # Continuer avec les autres données de marché
                                    for key, value in fundamental_data["Données de marché"].items():
                                        if key not in ["Prix actuel", "Précédente clôture"]:  # Skip already displayed items
                                            if value is not None:
                                                formatted_value = format_number(value)
                                                st.markdown(f"**{key}:** {formatted_value}")
                                            else:
                                                st.markdown(f"**{key}:** N/A") 

                                with col2:
                                    # Données fondamentales
                                    st.markdown("<p class='medium-font'>Ratios Fondamentaux</p>", unsafe_allow_html=True)
                                    for key, value in fundamental_data["Données fondamentales"].items():
                                        if value is not None:
                                            formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else value
                                            st.markdown(f"**{key}:** {formatted_value}")
                                        else:
                                            st.markdown(f"**{key}:** N/A")
                                
                                # Données financières
                                st.markdown("<p class='medium-font'>Données Financières</p>", unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                i = 0
                                for key, value in fundamental_data["Données financières"].items():
                                    with col1 if i % 2 == 0 else col2:
                                        if value is not None:
                                            formatted_value = format_number(value)
                                            st.markdown(f"**{key}:** {formatted_value}")
                                        else:
                                            st.markdown(f"**{key}:** N/A")
                                    i += 1
                                
                                # Historique financier
                                st.markdown("<p class='medium-font'>Historique Financier</p>", unsafe_allow_html=True)
                                
                                income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
                                
                                if not income_stmt.empty:
                                    with st.expander("Compte de Résultat"):
                                        # Sélection des lignes les plus importantes
                                        important_rows = ['Total Revenue', 'Gross Profit', 'Operating Income', 'EBITDA', 'Net Income']
                                        filtered_df = income_stmt.loc[income_stmt.index.isin(important_rows)]
                                        
                                        # Amélioration de l'affichage
                                        formatted_df = filtered_df.applymap(format_number)
                                        
                                        # Renommage des indices pour une meilleure lisibilité
                                        row_mapping = {
                                            'Total Revenue': "Chiffre d'affaires",
                                            'Gross Profit': 'Marge Brute',
                                            'Operating Income': "Résultat d'exploitation",
                                            'EBITDA': 'EBITDA',
                                            'Net Income': 'Résultat Net'
                                        }
                                        
                                        formatted_df.index = [row_mapping.get(idx, idx) for idx in formatted_df.index]
                                        
                                        st.dataframe(formatted_df)
                                else:
                                    st.info("Aucune donnée de compte de résultat disponible")
                                
                                if not balance_sheet.empty:
                                    with st.expander("Bilan"):
                                        # Sélection des lignes les plus importantes
                                        important_rows = ['Total Assets', 'Total Liabilities', 'Total Equity', 'Total Debt', 'Cash And Cash Equivalents']
                                        filtered_df = balance_sheet.loc[balance_sheet.index.isin(important_rows)]
                                        
                                        # Amélioration de l'affichage
                                        formatted_df = filtered_df.applymap(format_number)
                                        
                                        # Renommage des indices pour une meilleure lisibilité
                                        row_mapping = {
                                            'Total Assets': 'Total Actif',
                                            'Total Liabilities': 'Total Passif',
                                            'Total Equity': 'Fonds Propres',
                                            'Total Debt': 'Dette Totale',
                                            'Cash And Cash Equivalents': 'Trésorerie'
                                        }
                                        
                                        formatted_df.index = [row_mapping.get(idx, idx) for idx in formatted_df.index]
                                        
                                        st.dataframe(formatted_df)
                                else:
                                    st.info("Aucune donnée de bilan disponible")
                                
                                if not cashflow.empty:
                                    with st.expander("Flux de Trésorerie"):
                                        # Sélection des lignes les plus importantes
                                        important_rows = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
                                        filtered_df = cashflow.loc[cashflow.index.isin(important_rows)]
                                        
                                        # Amélioration de l'affichage
                                        formatted_df = filtered_df.applymap(format_number)
                                        
                                        # Renommage des indices pour une meilleure lisibilité
                                        row_mapping = {
                                            'Operating Cash Flow': "Flux d'exploitation",
                                            'Investing Cash Flow': "Flux d'investissement",
                                            'Financing Cash Flow': 'Flux de financement',
                                            'Free Cash Flow': 'Free Cash Flow'
                                        }
                                        
                                        formatted_df.index = [row_mapping.get(idx, idx) for idx in formatted_df.index]
                                        
                                        st.dataframe(formatted_df)
                                else:
                                    st.info("Aucune donnée de flux de trésorerie disponible")
                            
                            else:
                                st.error(f"Aucune donnée fondamentale disponible pour {ticker}")
                    
                    # Onglet 3: Screener de Buffett
                    with tab3:
                        st.subheader("Screener d'Actions selon les Critères de Warren Buffett")
                        
                        # Génération de deux colonnes
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("""
                            Ce screener identifie les actions répondant aux critères de Warren Buffett :
                            
                            1. **Entreprises compréhensibles** avec historique cohérent
                            2. **Avantage compétitif durable** (marge brute stable/élevée)
                            3. **Direction de qualité** (faible R&D/Ventes et forte croissance du résultat net)
                            4. **Retour sur capitaux propres élevé** (ROE > 15%)
                            5. **Faible endettement** (Dette/Fonds propres < 1)
                            6. **Forte rentabilité** (Marge nette > 10%)
                            7. **Génération de cash** (Free Cash Flow positif)
                            """)
                        
                        with col2:
                            st.markdown("<p class='medium-font'>Paramètres de Filtrage</p>", unsafe_allow_html=True)
                            
                            # Paramètres ajustables
                            min_revenue_growth = st.slider("Croissance minimale des revenus (%)", 0, 50, 5)
                            min_net_profit_margin = st.slider("Marge nette minimale (%)", 5, 40, 10)
                            min_roe = st.slider("ROE minimal (%)", 5, 30, 15)
                            max_debt_to_equity = st.slider("Ratio Dette/Fonds propres maximal", 0.0, 2.0, 1.0, 0.1)
                            min_gross_margin_consistency = st.slider("Marge brute minimale consistante (%)", 20, 80, 40)
                            min_net_income_growth = st.slider("Croissance minimale du résultat net (%)", 2, 20, 8)
                        
                        # Sélection de la région pour le screening
                        screen_regions = ["Toutes les régions"] + list(market_structure.keys())
                        selected_screen_region = st.selectbox("Région pour le screening", screen_regions, key="screen_region")
                        
                        screen_countries = []
                        selected_screen_country = None
                        
                        if selected_screen_region != "Toutes les régions":
                            screen_countries = ["Tous les pays"] + list(market_structure[selected_screen_region].keys())
                            selected_screen_country = st.selectbox("Pays pour le screening", screen_countries, key="screen_country")
                        
                        # Construction du dictionnaire d'actions pour le screening
                        if st.button("Lancer le screening"):
                            with st.spinner("Analyse en cours..."):
                                if selected_screen_region == "Toutes les régions":
                                    stocks_to_screen = flatten_market_structure(market_structure)
                                elif selected_screen_country == "Tous les pays":
                                    stocks_to_screen = flatten_market_structure(market_structure, selected_region=selected_screen_region)
                                else:
                                    stocks_to_screen = flatten_market_structure(market_structure, 
                                                                               selected_region=selected_screen_region, 
                                                                               selected_country=selected_screen_country)
                                
                                # Limiter le nombre d'actions pour des raisons de performance
                                max_stocks = 500
                                if len(stocks_to_screen) > max_stocks:
                                    st.warning(f"Le nombre d'actions à analyser ({len(stocks_to_screen)}) est limité à {max_stocks} pour des raisons de performance.")
                                    import random
                                    keys = list(stocks_to_screen.keys())
                                    random_keys = random.sample(keys, max_stocks)
                                    limited_stocks = {k: stocks_to_screen[k] for k in random_keys}
                                    stocks_to_screen = limited_stocks
                                
                                # Appliquer le screener
                                results = buffett_stock_screener(stocks_to_screen,
                                                               min_revenue_growth=min_revenue_growth,
                                                               min_net_profit_margin=min_net_profit_margin,
                                                               min_roe=min_roe,
                                                               max_debt_to_equity=max_debt_to_equity,
                                                               min_gross_margin_consistency=min_gross_margin_consistency,
                                                               min_net_income_growth_consistency=min_net_income_growth)
                                
                                if not results.empty:
                                    st.markdown(f"<p class='medium-font'>Résultats ({len(results)} actions)</p>", unsafe_allow_html=True)
                                    st.dataframe(results)
                                    
                                    # Bouton pour télécharger les résultats
                                    csv = results.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Télécharger les résultats (CSV)",
                                        data=csv,
                                        file_name="buffett_screener_results.csv",
                                        mime="text/csv",
                                    )
                                else:
                                    st.warning("Aucune action ne correspond aux critères sélectionnés.")
                            
            else:
                st.warning(f"Aucune action disponible pour {selected_country}")
    
    # Footer
    st.markdown("---")
    st.markdown("📊 Dashboard développé pour l'analyse des marchés financiers. Les données sont fournies par Yahoo Finance. Ce dashboard est proposé à titre informatif uniquement et ne constitue pas un conseil en investissement.")

# Lancement de l'application
if __name__ == "__main__":
    main()