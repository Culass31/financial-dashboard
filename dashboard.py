import streamlit as st
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# Configuration de l'application Streamlit
st.set_page_config(page_title="Analyse CAC40", page_icon="📈", layout="wide")

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

# Liste des actions du CAC40 avec leurs codes ISIN
CAC40_STOCKS = {
    'Air Liquide': {'ticker': 'AI.PA', 'isin': 'FR0000120073'},
    'Airbus': {'ticker': 'AIR.PA', 'isin': 'NL0000235190'},
    'Alstom': {'ticker': 'ALO.PA', 'isin': 'FR0010220475'},
    'ArcelorMittal': {'ticker': 'MT.AS', 'isin': 'LU1598757687'},
    'AXA': {'ticker': 'CS.PA', 'isin': 'FR0000120628'},
    'BNP Paribas': {'ticker': 'BNP.PA', 'isin': 'FR0000131104'},
    'Bouygues': {'ticker': 'EN.PA', 'isin': 'FR0000120503'},
    'Capgemini': {'ticker': 'CAP.PA', 'isin': 'FR0000125338'},
    'Carrefour': {'ticker': 'CA.PA', 'isin': 'FR0000120172'},
    'Crédit Agricole': {'ticker': 'ACA.PA', 'isin': 'FR0000045072'},
    'Danone': {'ticker': 'BN.PA', 'isin': 'FR0000120644'},
    'Dassault Systèmes': {'ticker': 'DSY.PA', 'isin': 'FR0014003TT8'},
    'Engie': {'ticker': 'ENGI.PA', 'isin': 'FR0010208488'},
    'EssilorLuxottica': {'ticker': 'EL.PA', 'isin': 'FR0000121667'},
    'Hermès': {'ticker': 'RMS.PA', 'isin': 'FR0000052292'},
    'Kering': {'ticker': 'KER.PA', 'isin': 'FR0000121485'},
    'Legrand': {'ticker': 'LR.PA', 'isin': 'FR0010307819'},
    'L\'Oréal': {'ticker': 'OR.PA', 'isin': 'FR0000120321'},
    'LVMH': {'ticker': 'MC.PA', 'isin': 'FR0000121014'},
    'Michelin': {'ticker': 'ML.PA', 'isin': 'FR001400AJ45'},
    'Orange': {'ticker': 'ORA.PA', 'isin': 'FR0000133308'},
    'Pernod Ricard': {'ticker': 'RI.PA', 'isin': 'FR0000120693'},
    'Publicis': {'ticker': 'PUB.PA', 'isin': 'FR0000130577'},
    'Renault': {'ticker': 'RNO.PA', 'isin': 'FR0000131906'},
    'Safran': {'ticker': 'SAF.PA', 'isin': 'FR0000073272'},
    'Saint-Gobain': {'ticker': 'SGO.PA', 'isin': 'FR0000125007'},
    'Sanofi': {'ticker': 'SAN.PA', 'isin': 'FR0000120578'},
    'Schneider Electric': {'ticker': 'SU.PA', 'isin': 'FR0000121972'},
    'Société Générale': {'ticker': 'GLE.PA', 'isin': 'FR0000130809'},
    'Stellantis': {'ticker': 'STLA.PA', 'isin': 'NL00150001Q9'},
    'STMicroelectronics': {'ticker': 'STM.PA', 'isin': 'NL0000226223'},
    'Teleperformance': {'ticker': 'TEP.PA', 'isin': 'FR0000051807'},
    'Thales': {'ticker': 'HO.PA', 'isin': 'FR0000121329'},
    'TotalEnergies': {'ticker': 'TTE.PA', 'isin': 'FR0000120271'},
    'Unibail-Rodamco-Westfield': {'ticker': 'URW.AS', 'isin': 'FR0013326246'},
    'Veolia': {'ticker': 'VIE.PA', 'isin': 'FR0000124141'},
    'Vinci': {'ticker': 'DG.PA', 'isin': 'FR0000125486'},
    'Vivendi': {'ticker': 'VIV.PA', 'isin': 'FR0000127771'},
    'Worldline': {'ticker': 'WLN.PA', 'isin': 'FR0011981968'}
}

# Fonction pour récupérer les actualités via Alpha Vantage
@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def get_news(ticker, api_key, items_limit=5):
    try:
        company_name = ticker.split('.')[0]  # Extraction du symbole sans l'extension de marché
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={company_name}&apikey={api_key}&limit={items_limit}"
        response = requests.get(url)
        data = response.json()
        
        news_items = []
        if "feed" in data:
            for item in data["feed"][:items_limit]:
                news_items.append({
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("url", ""),
                    "time_published": item.get("time_published", ""),
                    "source": item.get("source", ""),
                    "overall_sentiment_score": item.get("overall_sentiment_score", 0),
                    "overall_sentiment_label": item.get("overall_sentiment_label", "")
                })
        
        return news_items
    except Exception as e:
        st.error(f"Erreur lors de la récupération des actualités: {str(e)}")
        return []

# Fonction pour récupérer les données fondamentales
@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def get_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Création d'un dictionnaire structuré avec les données fondamentales
        fundamental_data = {
            "Informations générales": {
                "Nom": info.get('longName', 'N/A'),
                "Secteur": info.get('sector', 'N/A'),
                "Industrie": info.get('industry', 'N/A'),
                "Site web": info.get('website', 'N/A')
            },
            "Données de marché": {
                "Prix actuel": info.get('currentPrice', 'N/A'),
                "Précédente clôture": info.get('previousClose', 'N/A'),
                "Ouverture": info.get('open', 'N/A'),
                "Plage du jour": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
                "Plage sur 52 semaines": f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
                "Volume": info.get('volume', 'N/A'),
                "Volume moyen (3m)": info.get('averageVolume', 'N/A'),
                "Capitalisation boursière": info.get('marketCap', 'N/A'),
                "Variation sur 1 an": f"{info.get('52WeekChange', 0)*100:.2f}%" if info.get('52WeekChange') else 'N/A'
            },
            "Ratios financiers": {
                "P/E Ratio": info.get('trailingPE', 'N/A'),
                "EPS": info.get('trailingEPS', 'N/A'),
                "Prévision croissance EPS": info.get('earningsGrowth', 'N/A'),
                "Rendement du dividende": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                "Prix/Valeur comptable": info.get('priceToBook', 'N/A'),
                "ROA": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                "ROE": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                "Marge bénéficiaire brute": f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else 'N/A',
                "EBITDA": info.get('ebitda', 'N/A'),
                "EV/EBITDA": info.get('enterpriseToEbitda', 'N/A'),
                "Beta": info.get('beta', 'N/A')
            }
        }
        
        return fundamental_data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données fondamentales: {str(e)}")
        return None

# Fonction pour calculer la droite de régression et les bandes sigma
@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def calculate_regression(df, column='Close'):
    try:
        df = df.copy()
        # Création d'un index numérique pour la régression
        df['index'] = np.arange(len(df))
        
        # Préparation des données pour la régression
        X = df['index'].values.reshape(-1, 1)
        y = df[column].values
        
        # Calcul de la régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédiction avec le modèle
        df['regression'] = model.predict(X)
        
        # Calcul des écarts
        df['residuals'] = df[column] - df['regression']
        std_dev = df['residuals'].std()
        
        # Calcul des bandes sigma
        df['upper_1sigma'] = df['regression'] + std_dev
        df['lower_1sigma'] = df['regression'] - std_dev
        df['upper_2sigma'] = df['regression'] + 2 * std_dev
        df['lower_2sigma'] = df['regression'] - 2 * std_dev
        
        # Calcul des indicateurs clés
        coef = model.coef_[0]
        last_price = df[column].iloc[-1]
        regression_value = df['regression'].iloc[-1]
        
        # Calcul de l'écart en pourcentage
        deviation = (last_price - regression_value) / regression_value * 100
        
        # Calcul de la progression sur 1 an et 5 ans (approximative)
        days_in_year = 252  # Nombre approximatif de jours de trading dans une année
        prog_1y = None
        prog_5y = None
        
        if len(df) > days_in_year:
            price_1y_ago = df[column].iloc[-min(days_in_year, len(df))]
            prog_1y = (last_price / price_1y_ago - 1) * 100
        
        if len(df) > 5 * days_in_year:
            price_5y_ago = df[column].iloc[-min(5 * days_in_year, len(df))]
            prog_5y = (last_price / price_5y_ago - 1) * 100
        
        indicators = {
            'coefficient': coef,
            'model_growth': coef / regression_value * 100 * 252,  # Croissance annualisée
            'correlation': np.corrcoef(df['index'], df[column])[0, 1],
            'deviation': deviation,
            'prog_1y': prog_1y,
            'prog_5y': prog_5y
        }
        
        return df, indicators
    except Exception as e:
        st.error(f"Erreur lors du calcul de la régression: {str(e)}")
        return df, {}

# Fonction pour obtenir et traiter l'historique des cours
@st.cache_data(ttl=3600)  # Mise en cache des données pour une heure
def get_stock_history(ticker, period='max'):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calcul des moyennes mobiles
        history['MA20'] = history['Close'].rolling(window=20).mean()
        history['MA50'] = history['Close'].rolling(window=50).mean()
        history['MA100'] = history['Close'].rolling(window=100).mean()
        history['MA200'] = history['Close'].rolling(window=200).mean()
        
        # Réinitialiser l'index pour avoir la date comme colonne
        history = history.reset_index()
        
        return history
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        return pd.DataFrame()

# Fonction pour créer une jauge d'indicateur
def create_gauge(value, title, min_val, max_val, format_str="{:.2f}"):
    if value is None:
        value_str = "N/A"
        normalized_value = 0.5
        color = "gray"
    else:
        value_str = format_str.format(value)
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_value = max(0, min(1, normalized_value))
        
        # Déterminer la couleur en fonction de la valeur normalisée
        if normalized_value < 0.3:
            color = "red"
        elif normalized_value < 0.7:
            color = "orange"
        else:
            color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 50,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}, 'align': 'center'},  # Alignement du titre
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, min_val + (max_val - min_val) * 0.3], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [min_val + (max_val - min_val) * 0.3, min_val + (max_val - min_val) * 0.7], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [min_val + (max_val - min_val) * 0.7, max_val], 'color': 'rgba(0, 128, 0, 0.2)'}
            ]
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
        paper_bgcolor="white"
    )
    
    return fig

# Interface utilisateur Streamlit
def main():
    st.title("📊 Dashboard d'analyse des actions du CAC40")
    
    # Clé API Alpha Vantage
    alpha_vantage_api_key = "K7DJB2K5A15PSX63"
    
    # Sélection de l'action
    stock_name = st.selectbox(
        "Sélectionnez une action du CAC40",
        options=list(CAC40_STOCKS.keys()),
        index=0
    )
    
    ticker = CAC40_STOCKS[stock_name]['ticker']
    isin = CAC40_STOCKS[stock_name]['isin']
    
    st.write(f"Vous avez sélectionné: **{stock_name}** ({ticker}) | ISIN: {isin}")
    
    # Création d'onglets pour organiser l'interface
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyse technique", "🧮 Données fondamentales", "📊 Graphique avec volumes", "📰 News"])
    
    with tab1:
        # Sélection de la période pour l'analyse technique
        period = st.radio(
            "Période d'analyse",
            options=["1 An", "3 Ans", "5 Ans", "Maximum"],
            horizontal=True,
            index=2
        )
        
        period_map = {
            "1 An": "1y",
            "3 Ans": "3y",
            "5 Ans": "5y",
            "Maximum": "max"
        }
        
        # Récupération des données historiques
        df = get_stock_history(ticker, period=period_map[period])
        
        if not df.empty:
            # Récupération des données fondamentales pour le prix actuel
            fund_data = get_fundamental_data(ticker)
            current_price = "N/A"
            prev_close = "N/A"
            price_change = "N/A"
            price_change_pct = "N/A"
            
            if fund_data and "Données de marché" in fund_data:
                current_price = fund_data["Données de marché"].get("Prix actuel", "N/A")
                prev_close = fund_data["Données de marché"].get("Précédente clôture", "N/A")
                
                if current_price != "N/A" and prev_close != "N/A":
                    price_change = current_price - prev_close
                    price_change_pct = (price_change / prev_close) * 100
                    
                    # Formater les valeurs
                    current_price = f"{current_price:.2f} €"
                    price_change_str = f"{price_change:.2f} €"
                    price_change_pct_str = f"{price_change_pct:.2f}%"
                    
                    # Ajouter une couleur en fonction de la direction
                    color = "green" if price_change >= 0 else "red"
                    change_symbol = "+" if price_change >= 0 else ""
                    price_info = f"{current_price} <span style='color:{color}'>{change_symbol}{price_change_str} ({change_symbol}{price_change_pct_str})</span>"
            
            # Calcul de la régression et des indicateurs
            df_with_regression, indicators = calculate_regression(df)
            
            # Affichage des indicateurs clés avec le prix actuel
            if current_price != "N/A" and price_change != "N/A":
                st.markdown(f"### Indicateurs clés | Prix actuel: {price_info}", unsafe_allow_html=True)
            else:
                st.subheader("Indicateurs clés")
            
            # Configuration des jauges
            gauge_config = [
                {"name": "Croiss. modèle", "value": indicators.get('model_growth', 0), "min": -10, "max": 20, "format": "{:.2f}%", "show": True},
                {"name": "Coeff. correl", "value": indicators.get('correlation', 0), "min": -1, "max": 1, "format": "{:.2f}", "show": True},
                {"name": "Prog. 1 an", "value": indicators.get('prog_1y', 0), "min": -30, "max": 50, "format": "{:.2f}%", "show": True},
                # Ne montrer "Prog. 5 ans" que si la période est >= 5 ans
                {"name": "Prog. 5 ans", "value": indicators.get('prog_5y', 0), "min": -50, "max": 200, "format": "{:.2f}%", 
                 "show": period in ["5 Ans", "Maximum"]},
                # Ne pas montrer "Écart type" pour la période Maximum
                {"name": "Écart type", "value": indicators.get('deviation', 0), "min": -30, "max": 30, "format": "{:.2f}%", 
                 "show": period != "Maximum"}
            ]
            
            # Filtrer les jauges à afficher
            visible_gauges = [g for g in gauge_config if g["show"]]
            num_gauges = len(visible_gauges)
            
            # Créer les colonnes en fonction du nombre de jauges à afficher
            cols = st.columns(num_gauges)
            
            # Afficher les jauges
            for i, gauge in enumerate(visible_gauges):
                with cols[i]:
                    st.plotly_chart(create_gauge(
                        gauge["value"], 
                        gauge["name"], 
                        gauge["min"], 
                        gauge["max"], 
                        gauge["format"]
                    ), use_container_width=True)
            
            # Graphique d'analyse technique avec Plotly
            st.subheader("Analyse technique avec régression linéaire")
            
            fig = go.Figure()
            
            # Tracer le prix de clôture
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['Close'],
                mode='lines',
                name='Prix',
                line=dict(color='blue', width=1)
            ))
            
            # Tracer la ligne de régression
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['regression'],
                mode='lines',
                name='Régression',
                line=dict(color='red', width=2)
            ))
            
            # Tracer les bandes sigma
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['upper_1sigma'],
                mode='lines',
                name='+1 Sigma',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['lower_1sigma'],
                mode='lines',
                name='-1 Sigma',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['upper_2sigma'],
                mode='lines',
                name='+2 Sigma',
                line=dict(color='orange', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'],
                y=df_with_regression['lower_2sigma'],
                mode='lines',
                name='-2 Sigma',
                line=dict(color='orange', width=1, dash='dash')
            ))
            
            # Tracer la zone entre les bandes (facultatif car peut rendre le graphique trop chargé)
            fig.add_trace(go.Scatter(
                x=df_with_regression['Date'].tolist() + df_with_regression['Date'].tolist()[::-1],
                y=df_with_regression['upper_1sigma'].tolist() + df_with_regression['lower_1sigma'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,255,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Sigma Zone',
                showlegend=False
            ))
            
            # Mise en page du graphique
            fig.update_layout(
                title=f"Évolution du cours de {stock_name} avec régression linéaire",
                xaxis_title="Date",
                yaxis_title="Prix (€)",
                yaxis_type="log",  # Échelle logarithmique
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Impossible de récupérer les données pour {stock_name}")
    
    with tab2:
        # Récupération et affichage des données fondamentales
        fundamental_data = get_fundamental_data(ticker)
        
        if fundamental_data:
            st.subheader("Données fondamentales")
            
            for section_name, section_data in fundamental_data.items():
                st.markdown(f"### {section_name}")
                
                # Créer un dataframe pour afficher les données de manière propre
                data_list = []
                for key, value in section_data.items():
                    # Formater les grands nombres
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and not isinstance(value, str):
                        if key not in ['P/E Ratio', 'EPS', 'Beta']:
                            formatted_value = format_number(value)
                        else:
                            formatted_value = value
                    else:
                        formatted_value = value
                    
                    data_list.append({"Indicateur": key, "Valeur": formatted_value})
                
                # Afficher les données dans un DataFrame stylé
                df_section = pd.DataFrame(data_list)
                st.dataframe(df_section, use_container_width=True, hide_index=True)
        else:
            st.error(f"Impossible de récupérer les données fondamentales pour {stock_name}")
    
    with tab3:
        # Sélection de la période pour le graphique avec volumes
        chart_period = st.radio(
            "Période",
            options=["1 Mois", "6 Mois", "1 An", "5 Ans", "Maximum"],
            horizontal=True,
            index=2
        )
        
        chart_period_map = {
            "1 Mois": "1mo",
            "6 Mois": "6mo",
            "1 An": "1y",
            "5 Ans": "5y",
            "Maximum": "max"
        }
        
        # Récupération des données historiques
        chart_df = get_stock_history(ticker, period=chart_period_map[chart_period])
        
        if not chart_df.empty:
            # Graphique des prix et volumes avec Plotly
            st.subheader(f"Évolution du cours et des volumes pour {stock_name}")
            
            # Création d'un graphique à deux axes avec plotly
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               subplot_titles=(f"Cours de {stock_name}", "Volume"),
                               row_heights=[0.7, 0.3])
            
            # Tracer le prix de clôture
            fig.add_trace(
                go.Scatter(x=chart_df['Date'], y=chart_df['Close'], name='Prix',
                          line=dict(color='blue', width=1.5)),
                row=1, col=1
            )
            
            # Tracer les moyennes mobiles
            fig.add_trace(
                go.Scatter(x=chart_df['Date'], y=chart_df['MA20'], name='MM20',
                          line=dict(color='red', width=1)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=chart_df['Date'], y=chart_df['MA50'], name='MM50',
                          line=dict(color='green', width=1)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=chart_df['Date'], y=chart_df['MA100'], name='MM100',
                          line=dict(color='purple', width=1)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=chart_df['Date'], y=chart_df['MA200'], name='MM200',
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
            
            # Tracer les volumes
            fig.add_trace(
                go.Bar(x=chart_df['Date'], y=chart_df['Volume'], name='Volume',
                      marker=dict(color='rgba(0, 0, 255, 0.5)')),
                row=2, col=1
            )
            
            # Mise en page du graphique
            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            # Mise à jour des axes Y
            fig.update_yaxes(title_text="Prix (€)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Impossible de récupérer les données pour {stock_name}")

if __name__ == "__main__":
    main()
