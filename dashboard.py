import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import requests
from datetime import datetime, timedelta
import time
import html

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
            "small": {"label": "0.9rem", "value": "1.1rem", "change": "0.9rem"},
            "medium": {"label": "1rem", "value": "1.4rem", "change": "1rem"},
            "large": {"label": "1.2rem", "value": "1.8rem", "change": "1.2rem"}
        }
        
        sizes = font_sizes.get(size, font_sizes["medium"])
        
        # Formater la valeur si une fonction de formatage est fournie
        formatted_price = format_func(current_price) if format_func else f"{current_price:.2f}"
        
        # Créer le style pour l'affichage
        if background:
            style = f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {bg_color}; margin-bottom: 10px;'>
                <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                <div style='font-size: {sizes["change"]}; color: {color};'>
                    {arrow} {abs(change_pct):.2f}%
                </div>
            </div>
            """
        else:
            style = f"""
            <div style='margin-bottom: 10px;'>
                <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                <div style='font-size: {sizes["change"]}; color: {color};'>
                    {arrow} {abs(change_pct):.2f}%
                </div>
            </div>
            """
        
        # Afficher avec le style approprié
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.markdown(f"**{label}:** N/A")

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
                df = pd.read_csv(r"C:\Users\culas\OneDrive\Documents\Finances\actions.csv", sep=";", encoding='utf-8-sig')
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
        
        #Données générales
        general_data = {
                "Nom": info.get('longName', None),
                "Secteur": info.get('sector', None),
                "Industrie": info.get('industry', None),
                "Site web": info.get('website', None)
            }

        # Données de marché
        market_data = {
            "Prix actuel": info.get('currentPrice', info.get('regularMarketPrice', None)),
            "Précédente clôture": info.get('previousClose', None),
            "Volume": info.get('volume', None),
            "Volume moyen": info.get('averageVolume', None),
            "Capitalisation boursière": info.get('marketCap', None),
            "Beta": info.get('beta', None),
            "Rendement du dividende (%)": info.get('dividendYield', 0) if info.get('dividendYield') else None
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
            "Données générales": general_data,
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
            time.sleep(0.5) # Pause pour éviter de dépasser les limites de l'API

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

# Fonction pour récupérer les actualités via Alpha Vantage
@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def get_stock_news(ticker, limit=10):
    """
    Récupère les dernières news pour un ticker donné depuis AlphaVantage.
    
    :param ticker: Symbole de l'action
    :param limit: Nombre maximum de news à récupérer
    :return: Liste de dictionnaires contenant les news
    """
    # Clé API Alpha Vantage
    api_key = "7DJB2K5A15PSX63"

    # Convertir le ticker Yahoo Finance au format AlphaVantage si nécessaire
    # Pour les tickers européens, supprimer le suffixe .PA, .MC, etc.
    base_ticker = ticker.split('.')[0] if '.' in ticker else ticker
    
    # URL de l'API AlphaVantage pour les news
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={base_ticker}&apikey={api_key}&limit={limit}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Vérifier si la réponse contient des news
        if "feed" in data:
            return data["feed"]
        else:
            return []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des news: {e}")
        return []

# Fonction pour adapter un dictionnaire d'actions à partir de différents critères
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

    # Création de la sidebar pour la sélection d'actions
    with st.sidebar:
        st.sidebar.title("🔍 Sélection du Marché")
        
        # Méthode de filtrage
        st.sidebar.markdown('<p class="sidebar-title">Méthode de filtrage</p>', unsafe_allow_html=True)
        filter_method = st.sidebar.radio(
            "Choisissez votre approche",
            ["Géographique", "Sectorielle", "Place de marché", "Liste complète"],
            captions=["Par région et pays", "Par secteur et industrie", "Par indice boursier", "Vue de toutes les actions"]
        )
        
        # Variables pour stocker les sélections
        selected_stock = None
        flattened_stocks = {}
        
        # Approche géographique
        if filter_method == "Géographique":
            # Sélection par région et pays
            regions = list(market_structure['regions'].keys())
            selected_region = st.sidebar.selectbox("Région", regions)
            
            if selected_region:
                pays_list = list(market_structure['regions'][selected_region].keys())
                selected_pays = st.sidebar.selectbox("Pays", pays_list)
                
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analyse Technique", "📑 Analyse Fondamentale", "📰 Actualités récentes", "🔮 Projection du cours", "🔎 Screener"])

    # Onglet 1: Analyse Technique
    with tab1:
        if st.session_state.get('selected_stock') is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = st.session_state['ticker']

            # Titre avec le nom de l'action et son ticker
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Technique: {selected_stock_name} ({ticker})</h2>", unsafe_allow_html=True)

            # Création d'un layout plus clair
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
                
                # Sélection de la période avec une interface plus claire
                period_cols = st.columns(7)
                periods = {"1m": "1mo", "3m": "3mo", "6m": "6mo", "1a": "1y", "2a": "2y", "5a": "5y", "Max": "max"}
                
                # Créer des boutons pour chaque période
                selected_period = None
                for i, (period_name, period_value) in enumerate(periods.items()):
                    with period_cols[i]:
                        if st.button(period_name, key=f"period_{period_name}"):
                            selected_period = period_value
                
                # Si aucun bouton n'est cliqué, utiliser la période par défaut
                if selected_period is None:
                    selected_period = "Max"  # Par défaut: 1 an
                
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
                                            line=dict(color='red', width=1, dash='dash')),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(x=df_reg.index, y=df_reg['Regression'] - std_dev,
                                            mode='lines', name='-1σ',
                                            line=dict(color='green', width=1, dash='dash')),
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
                                #### Tendance récente
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

    # Onglet 2: Analyse Fondamentale
    with tab2:

        if st.session_state['selected_stock'] is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = selected_stock['ticker']

            st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Fondamentale: {selected_stock_name} ({ticker})</h2>", unsafe_allow_html=True)
            
            with st.spinner("Chargement des données fondamentales..."):
                fundamental_data = get_fundamental_data(ticker)
                
                if fundamental_data:
                    col1, col2 = st.columns(2)
                    
                    # Données générales
                    st.markdown("<p class='medium-font'>Infomations générales</p>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    i = 0
                    for key, value in fundamental_data["Données générales"].items():
                        with col1 if i % 2 == 0 else col2:
                            if value is not None:
                                formatted_value = format_number(value)
                                st.markdown(f"**{key}:** {formatted_value}")
                            else:
                                st.markdown(f"**{key}:** N/A")
                        i += 1
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
                            formatted_df = filtered_df.map(format_number)
                            
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
                            formatted_df = filtered_df.map(format_number)
                            
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
                            formatted_df = filtered_df.map(format_number)
                            
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
        else:
            st.info("Veuillez d'abord sélectionner une action dans l'onglet 'Sélection'")

    # Onglet 3: Actualités récentes
    with tab3:
        st.subheader("📰 Actualités récentes")
        
        if st.session_state.get('selected_stock') is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = st.session_state['ticker']
            
            with st.spinner("Chargement des actualités..."):
                news_items = get_stock_news(ticker, limit=15)
                
                if news_items:
                    # Afficher un compteur
                    st.caption(f"{len(news_items)} articles trouvés pour {selected_stock_name}")
                    
                    # Créer des colonnes pour les filtres
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Filtre par nombre de jours - Utiliser des chaînes au lieu des entiers
                        days_options = ["1", "3", "7", "30", "Tous"]
                        days_filter = st.radio(
                            "Période",
                            options=days_options,
                            format_func=lambda x: x if x == "Tous" else f"{x} jour{'s' if x != '1' else ''}",
                            index=2
                        )
                    
                    # Filtrer les news par date si nécessaire
                    filtered_news = news_items
                    if days_filter != "Tous":
                        cutoff_date = datetime.now() - timedelta(days=int(days_filter))
                        filtered_news = [
                            item for item in news_items 
                            if datetime.strptime(item.get("time_published", "")[:8], "%Y%m%d") >= cutoff_date
                        ]
                    
                    # Afficher les news filtrées
                    if filtered_news:
                        for item in filtered_news:
                            # Extraire les informations pertinentes
                            title = item.get("title", "Sans titre")
                            source = item.get("source", "Source inconnue")
                            url = item.get("url", "#")
                            summary = item.get("summary", "Pas de résumé disponible.")
                            published_date = item.get("time_published", "")
                            
                            # Formater la date de publication
                            if published_date:
                                try:
                                    date_obj = datetime.strptime(published_date[:8], "%Y%m%d")
                                    time_obj = datetime.strptime(published_date[9:15], "%H%M%S")
                                    formatted_date = f"{date_obj.strftime('%d %b %Y')} à {time_obj.strftime('%H:%M')}"
                                except:
                                    formatted_date = published_date
                            else:
                                formatted_date = "Date inconnue"
                            
                            # Calculer le sentiment si disponible
                            sentiment_score = None
                            sentiment_label = ""
                            if "overall_sentiment_score" in item:
                                try:
                                    sentiment_score = float(item["overall_sentiment_score"])
                                    if sentiment_score > 0.25:
                                        sentiment_label = "🟢 Positif"
                                    elif sentiment_score < -0.25:
                                        sentiment_label = "🔴 Négatif"
                                    else:
                                        sentiment_label = "⚪ Neutre"
                                except:
                                    pass
                            
                            # Créer un conteneur pour chaque news avec un style amélioré
                            with st.container():
                                st.markdown(f"""
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 15px;">
                                    <h3 style="margin-top: 0;">{html.escape(title)}</h3>
                                    <p><strong>Source:</strong> {html.escape(source)} | <strong>Publié le:</strong> {formatted_date} {sentiment_label}</p>
                                    <p>{html.escape(summary)}</p>
                                    <a href="{url}" target="_blank">Lire l'article complet</a>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info(f"Aucun article trouvé dans les {days_filter} derniers jours. Essayez d'étendre la période.")
                else:
                    st.warning(f"Aucune actualité trouvée pour {selected_stock_name}. Vérifiez que l'API AlphaVantage fonctionne correctement.")
        else:
            st.info("Veuillez sélectionner une action pour voir les actualités associées.")

    # Onglet 4: Projection du cours avec marge de sécurité
    with tab4:
        st.subheader("🔮 Projection du cours avec marge de sécurité")
        
        if st.session_state.get('selected_stock') is not None:
            selected_stock = st.session_state['selected_stock']
            selected_stock_name = st.session_state['selected_stock_name']
            ticker = st.session_state['ticker']
            
            # Récupérer le PRU (Prix de Revient Unitaire)
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                user_pru = st.number_input("Entrez votre PRU (€)", min_value=0.01, step=0.01, value=100.0)
            
            with col2:
                safety_margin = st.slider("Marge de sécurité", min_value=0, max_value=50, value=20, 
                                    help="Pourcentage de décote appliqué pour déterminer un prix d'achat sécuritaire")
            
            # Récupérer les données historiques et calculer la régression
            proj_df = get_stock_history(ticker, period="5y")
            
            if not proj_df.empty:
                # Calculer la régression
                _, indicators = calculate_regression(proj_df)
                
                # Récupérer le taux de croissance du modèle
                model_growth = indicators.get('model_growth', 0)
                
                # Créer un message d'information
                with col3:
                    st.info(f"Croissance annuelle du modèle basée sur les 5 dernières années: {model_growth:.2f}% | Mise à jour: {datetime.now().strftime('%B %Y')}")
                
                # Calculer le coefficient de corrélation pour le modèle
                correlation = indicators.get('correlation', 0)
                
                # Calculer la date actuelle
                current_date = datetime.now()
                current_year = current_date.year
                
                # Récupérer le prix actuel
                fund_data = get_fundamental_data(ticker)
                current_price = None
                if fund_data and "Données de marché" in fund_data:
                    current_price = fund_data["Données de marché"].get("Prix actuel")
                
                if current_price is not None:
                    # Récupération de l'écart-type pour les projections sigma
                    std_dev = indicators.get('std_dev', 0)
                    
                    # Calcul du prix avec marge de sécurité
                    safety_price = current_price * (1 - safety_margin/100)
                    
                    # Afficher le prix actuel vs prix avec marge de sécurité
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Prix actuel", f"{current_price:.2f} €")
                    with cols[1]:
                        st.metric("Prix avec marge de sécurité", f"{safety_price:.2f} €", f"-{safety_margin}%")
                    with cols[2]:
                        delta_pru = ((safety_price / user_pru) - 1) * 100
                        st.metric("Comparaison avec PRU", f"{safety_price:.2f} € vs {user_pru:.2f} €", f"{delta_pru:.1f}%")
                    with cols[3]:
                        buy_recommendation = "⭐ ACHAT" if safety_price <= user_pru else "⛔ ATTENDRE"
                        st.metric("Recommandation", buy_recommendation)
                    
                    # Créer un tableau de projections
                    st.subheader("Tableau de projection")
                    
                    # Préparer les données pour le tableau
                    table_data = []
                    
                    # En-têtes
                    headers = ["TICKER", "COURS", "Krach", "PRU", "MOS -20%", "-1 SIGMA", 
                            "Croissance\ndu modèle", "coef\ncorr.",
                            "-1 sigma\ndans 12 mois", "-1 sigma\ndans 24 mois",
                            "-1 sigma\ndans 36 mois", "-1 sigma\ndans 48 mois",
                            "-1 sigma\ndans 60 mois", "-1 sigma\ndans 72 mois",
                            "-1 sigma\ndans 84 mois"]
                    
                    # Calculer la valeur du krach (approximation)
                    krach_value = current_price * 0.85  # 15% de baisse
                    
                    # Calculer la valeur -1 sigma actuelle
                    current_neg_sigma = current_price - std_dev
                    
                    # Calculer les projections -1 sigma
                    years_to_project = 7
                    neg_sigma_values = []
                    
                    for i in range(1, years_to_project + 1):
                        # Calculer pour une période en années
                        months = i * 12
                        # Projection du prix
                        projected_price = current_price * (1 + model_growth/100) ** i

                        # Projection -1 sigma
                        projected_std_dev = std_dev * (projected_price / current_price)
                        neg_sigma = projected_price - projected_std_dev
                        neg_sigma_values.append(f"{neg_sigma:.0f} €")
                    
                    # Créer la ligne de données
                    row = [
                        ticker,
                        f"{current_price:.2f}",
                        f"{krach_value:.0f}",
                        f"{user_pru:.0f}",
                        f"{safety_price:.0f}",
                        f"{current_neg_sigma:.0f}",
                        f"{model_growth:.1f}%",
                        f"{correlation:.2f}",
                    ]
                    
                    # Ajouter les valeurs -1 sigma
                    row.extend(neg_sigma_values)
                    
                    # Ajouter la ligne au tableau
                    table_data.append(row)
                    
                    # Créer un DataFrame
                    df_projection = pd.DataFrame([row], columns=headers)
                    
                    # Appliquer un style conditionnel au tableau
                    def highlight_values(val, column_name):
                        try:
                            val_num = float(val.replace(' €', '').replace('€', ''))
                            
                            # Pour la colonne MOS (Margin of Safety)
                            if column_name == "MOS -20%":
                                if val_num < user_pru:
                                    return 'background-color: #00cc66; color: white'  # Vert foncé
                                else:
                                    return 'background-color: #ff6666; color: white'  # Rouge
                            
                            # Pour les colonnes de projection
                            if column_name.startswith("-1 sigma"):
                                if val_num < user_pru:
                                    return 'background-color: #ffcccc'  # Rouge clair
                                else:
                                    return 'background-color: #ccffcc'  # Vert clair
                        except:
                            return ''
                        
                        return ''
                    
                    # Créer le tableau stylisé avec la fonction highlight_values personnalisée
                    styled_df = df_projection.style.apply(
                        lambda row: [highlight_values(val, col) for val, col in zip(row, df_projection.columns)], 
                        axis=1
                    )
                    
                    # Afficher le tableau
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Explication
                    st.info("""
                    **Comment utiliser ce tableau :**
                    - **MOS -20%** : Prix actuel avec une marge de sécurité de 20% (principe de Buffett)
                    - Les cellules en vert foncé indiquent que le prix avec marge de sécurité est inférieur à votre PRU (bon moment pour acheter)
                    - Les cellules en vert clair indiquent que la projection est supérieure à votre PRU
                    - Les cellules en rouge indiquent que la projection est inférieure à votre PRU
                    - La ligne "-1 sigma" représente le scénario pessimiste (16% des cas)
                    - La valeur "Krach" représente une baisse de 15% du prix actuel
                    """)
                    
                    # Afficher un graphique de projection
                    st.subheader("Graphique de projection")
                    
                    # Préparer les données pour le graphique
                    years = list(range(current_year, current_year + years_to_project + 1))
                    projection_values = [current_price]
                    neg_sigma_values_graph = [current_neg_sigma]
                    
                    # Calculer les valeurs projetées pour chaque année
                    for i in range(1, years_to_project + 1):
                        projected = current_price * (1 + model_growth/100) ** i
                        projection_values.append(projected)
                        
                        projected_std_dev = std_dev * (projected / current_price)
                        neg_sigma = projected - projected_std_dev
                        neg_sigma_values_graph.append(neg_sigma)
                    
                    # Créer des données pour la ligne représentant le PRU
                    pru_values = [user_pru] * (years_to_project + 1)
                    
                    # Créer des données pour la ligne représentant la marge de sécurité
                    safety_values = [safety_price] * (years_to_project + 1)
                    
                    # Créer le graphique
                    fig = go.Figure()
                    
                    # Ajouter la ligne de projection
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=projection_values,
                        mode='lines+markers',
                        name='Projection',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Ajouter la ligne -1 sigma
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=neg_sigma_values_graph,
                        mode='lines',
                        name='-1 Sigma',
                        line=dict(color='purple', width=2, dash='dot')
                    ))
                    
                    # Ajouter la ligne du PRU
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=pru_values,
                        mode='lines',
                        name='Votre PRU',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    
                    # Ajouter la ligne de la marge de sécurité
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=safety_values,
                        mode='lines',
                        name=f'Prix avec {safety_margin}% MOS',
                        line=dict(color='green', width=1, dash='dash')
                    ))
                    
                    # Mise en page du graphique
                    fig.update_layout(
                        title=f"Projection du cours de {selected_stock} sur 7 ans",
                        xaxis_title="Année",
                        yaxis_title="Prix (€)",
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    # Remplir la zone entre la projection et -1 sigma
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=projection_values + neg_sigma_values_graph[::-1],
                        fill='toself',
                        fillcolor='rgba(0,0,255,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Zone de probabilité 68%'
                    ))
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Conclusions et recommandations
                    st.subheader("Recommandations d'investissement")
                    
                    # Déterminer la recommandation finale
                    recommendation = ""
                    if safety_price <= user_pru:
                        recommendation = "✅ **ACHETER** : Le prix actuel avec une marge de sécurité de {}% est inférieur à votre PRU. C'est potentiellement un bon moment pour renforcer votre position.".format(safety_margin)
                        buy_color = "green"
                    else:
                        recommendation = "⛔ **ATTENDRE** : Le prix actuel, même avec une marge de sécurité de {}%, reste supérieur à votre PRU. Il est préférable d'attendre un meilleur point d'entrée.".format(safety_margin)
                        buy_color = "red"
                    
                    # Couleurs pour les métriques
                    growth_color = "green" if model_growth > 0 else "red"
                    corr_color = "green" if correlation > 0.7 else "orange" if correlation > 0.4 else "red"
                    
                    # Afficher les recommandations dans un cadre coloré
                    st.markdown(f"""
                    <div style="border-left: 5px solid {buy_color}; padding: 10px; background-color: rgba(0,0,0,0.05);">
                        <h3 style="color: {buy_color};">{recommendation}</h3>
                        <ul>
                            <li>Croissance du modèle: <span style="color: {growth_color};">{model_growth:.1f}%</span> par an</li>
                            <li>Fiabilité du modèle (corrélation): <span style="color: {corr_color};">{correlation:.2f}</span></li>
                            <li>Prix actuel: {current_price:.2f} €</li>
                            <li>Prix avec marge de sécurité ({safety_margin}%): {safety_price:.2f} €</li>
                            <li>Votre PRU: {user_pru:.2f} €</li>
                        </ul>
                        <p><em>Rappel: Ces projections sont basées sur des modèles statistiques et ne garantissent pas les performances futures.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Impossible de récupérer le prix actuel pour calculer les projections")
            else:
                st.error(f"Impossible de récupérer les données historiques pour {selected_stock}")
        else:
            st.info("Veuillez sélectionner une action pour voir les projections.")

    # Onglet 5: Screener d'Actions
    with tab5:
        st.subheader("Screener d'Actions")
        
        # Génération de deux colonnes
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Options avancées pour le screener
            with st.expander("Paramètres du Screener", expanded=False):
                st.write("Ajustez les critères de sélection selon les principes de Warren Buffett")
                
                # Paramètres pour la qualité de l'entreprise
                st.subheader("Qualité de l'entreprise")
                col_quality1, col_quality2 = st.columns(2)
                with col_quality1:
                    min_revenue_growth = st.slider("Croissance min. des revenus (%)", 0, 30, 5)
                    min_net_profit_margin = st.slider("Marge nette min. (%)", 0, 30, 10)
                    min_roe = st.slider("ROE min. (%)", 0, 50, 15)
                with col_quality2:
                    min_gross_margin_consistency = st.slider("Marge brute min. cohérente (%)", 0, 70, 40)
                    min_net_income_growth_consistency = st.slider("Croissance min. du résultat net (%)", 0, 20, 8)
                
                # Paramètres pour la santé financière
                st.subheader("Santé financière")
                col_health1, col_health2 = st.columns(2)
                with col_health1:
                    max_debt_to_equity = st.slider("Ratio dette/fonds propres max", 0.0, 3.0, 1.0, 0.1)
                    min_free_cashflow = st.slider("Free cash flow min (M$)", -10, 50, 0, 1)
                with col_health2:
                    max_rd_sales_ratio = st.slider("Ratio max R&D/CA", 0.0, 0.20, 0.05, 0.01)
                    max_interest_ebit_ratio = st.slider("Ratio max intérêts/EBIT", 0.0, 0.50, 0.20, 0.01)
            
            # Filtres de marché pour le screener
            filter_cols = st.columns(3)
            with filter_cols[0]:
                region_filter = st.selectbox("Région", ["Toutes les régions", "Amérique du Nord", "Europe", "Asie"])
            with filter_cols[1]:
                secteur_filter = st.selectbox("Secteur", ["Tous les secteurs", "Technology", "Healthcare", "Industrials", "Consumer"])
            with filter_cols[2]:
                marche_filter = st.selectbox("Marché", ["Tous les marchés", "NYSE", "NASDAQ", "Euronext Paris", "Frankfurt"])
            
            # Bouton pour lancer le screening
            if st.button("Lancer le screening", type="primary"):
                with st.spinner("Analyse des actions en cours..."):
                    # Adapter la structure du marché selon les filtres
                    filtered_stocks = flatten_market_structure(
                        market_structure,
                        filter_type="region" if region_filter != "Toutes les régions" else None,
                        level1=region_filter if region_filter != "Toutes les régions" else None
                    )
                    
                    # Filtrer par secteur si applicable
                    if secteur_filter != "Tous les secteurs":
                        filtered_stocks = {k: v for k, v in filtered_stocks.items() 
                                        if v.get('secteur') == secteur_filter}
                    
                    # Filtrer par marché si applicable
                    if marche_filter != "Tous les marchés":
                        filtered_stocks = {k: v for k, v in filtered_stocks.items() 
                                        if v.get('marche') == marche_filter}
                    
                    # Appliquer le screener de Buffett
                    eligible_stocks_df = buffett_stock_screener(
                        filtered_stocks,
                        min_revenue_growth=min_revenue_growth,
                        min_net_profit_margin=min_net_profit_margin,
                        min_roe=min_roe,
                        max_debt_to_equity=max_debt_to_equity,
                        min_free_cashflow=min_free_cashflow,
                        min_gross_margin_consistency=min_gross_margin_consistency,
                        max_rd_sales_ratio=max_rd_sales_ratio,
                        max_interest_ebit_ratio=max_interest_ebit_ratio,
                        min_net_income_growth_consistency=min_net_income_growth_consistency
                    )
                    
                    # Afficher les résultats
                    if not eligible_stocks_df.empty:
                        st.success(f"{len(eligible_stocks_df)} actions correspondent aux critères de Warren Buffett")
                        
                        # Personnaliser l'affichage
                        def highlight_row(row):
                            if row['score_buffett'] > 80:
                                return ['background-color: rgba(0, 128, 0, 0.2)'] * len(row)
                            elif row['score_buffett'] > 60:
                                return ['background-color: rgba(255, 165, 0, 0.2)'] * len(row)
                            return [''] * len(row)
                        
                        # Trier par score décroissant
                        eligible_stocks_df = eligible_stocks_df.sort_values(by='score_buffett', ascending=False)
                        
                        # Afficher le tableau avec mise en forme
                        st.dataframe(
                            eligible_stocks_df.style.apply(highlight_row, axis=1),
                            use_container_width=True,
                            height=400
                        )
                        # Bouton pour télécharger les résultats
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Télécharger les résultats (CSV)",
                            data=csv,
                            file_name="buffett_screener_results.csv",
                            mime="text/csv",
                        )

                    else:
                        st.warning("Aucune action ne correspond aux critères définis")
                        st.info("Essayez d'assouplir les critères pour obtenir plus de résultats")
        
        with col2:
            # Critères de Warren Buffett expliqués
            st.markdown("### Critères de Warren Buffett")
            st.markdown("""
            #### 🧠 Comprendre l'entreprise
            * Entreprise simple et compréhensible
            * Historique opérationnel stable
            * Perspectives à long terme favorables
            
            #### 💰 Qualité financière
            * Rendement des capitaux propres (ROE) > 15%
            * Marge nette élevée et stable
            * Faible niveau d'endettement
            
            #### 🛡️ Avantage concurrentiel
            * "Moat" économique
            * Forte reconnaissance de marque
            * Position dominante sur le marché
            
            #### 💵 Valorisation attractive
            * Prix d'achat raisonnable
            * Marge de sécurité
            * Croissance du bénéfice par action
            """)
            
            # Afficher les entreprises "favorites" de Buffett comme référence
            with st.expander("Entreprises favorites de Buffett"):
                buffett_favs = {
                    "Apple": "Technologie",
                    "Coca-Cola": "Consommation",
                    "Bank of America": "Finance",
                    "American Express": "Finance",
                    "Kraft Heinz": "Consommation"
                }
                
                for company, sector in buffett_favs.items():
                    st.markdown(f"**{company}** - *{sector}*")
    
    # Footer
    st.markdown("---")
    st.markdown("📊 Dashboard développé pour l'analyse des marchés financiers. Données fournies par Yahoo Finance. Ce dashboard est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement.")

# Lancement de l'application
if __name__ == "__main__":
    main()