"""
Composant pour l'onglet d'analyse technique
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

from services.data_service import get_stock_history, calculate_regression
from services.analysis_service import create_gauge
from utils.ui_styles import create_info_box
from utils.ui_metrics import AppMetrics


def render_technical_analysis_tab(selected_stock):
    """Render l'onglet d'analyse technique"""
    
    if selected_stock is None:
        st.info("Veuillez sélectionner une action dans le menu latéral pour voir l'analyse technique.")
        return
    
    ticker = selected_stock['ticker']
    stock_name = st.session_state.get('selected_stock_name', ticker)
    
    # Titre avec le nom de l'action et son ticker
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Technique: {stock_name} ({ticker})</h2>", unsafe_allow_html=True)
    
    # Layout principal
    col_chart, col_indicators = st.columns([3, 1])
    
    with col_chart:
        render_chart_section(ticker)
    
    with col_indicators:
        render_indicators_section(ticker)


def render_chart_section(ticker):
    """Render la section graphique"""
    
    # Options avancées
    with st.expander("Options avancées", expanded=False):
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            show_bollinger = st.checkbox("Bandes de Bollinger", value=True)
            show_ma = st.checkbox("Moyennes mobiles", value=False)
            if show_ma:
                ma_periods = st.multiselect("Périodes", [20, 50, 100, 200], default=[50, 200])
        
        with opt_col2:
            regression_type = st.radio("Type de régression", ["Linéaire", "Logarithmique"], index=0)
            show_volume = st.checkbox("Afficher le volume", value=True)
    
    # Sélection de la période
    selected_period = render_period_selector()
    
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
            
            # Création du graphique
            fig = create_price_chart(df_reg, show_volume, show_bollinger, show_ma, ma_periods if show_ma else [])
            
            # Affichage du graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de tendance détaillée
            render_trend_analysis(df_reg, indicators)
        else:
            st.error(f"Aucune donnée historique disponible pour {ticker}")


def render_period_selector():
    """Render le sélecteur de période"""
    period_cols = st.columns(10)
    periods = {"5j":"5d","1m":"1mo", "3m":"3mo", "6m":"6mo", "1a":"1y", "2a":"2y", "5a":"5y","10a":"10y","YTD":"ytd","Max":"max"}
    
    # Par défaut: 10 ans
    selected_period = "10y"
    
    for i, (period_name, period_value) in enumerate(periods.items()):
        with period_cols[i]:
            if st.button(period_name, key=f"period_{period_name}"):
                selected_period = period_value
    
    return selected_period


def create_price_chart(df_reg, show_volume, show_bollinger, show_ma, ma_periods):
    """Crée le graphique principal des prix"""
    
    fig = make_subplots(
        rows=2 if show_volume else 1, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3] if show_volume else [1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_reg.index,
            open=df_reg['Open'],
            high=df_reg['High'],
            low=df_reg['Low'],
            close=df_reg['Close'],
            name="Prix",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Ligne de régression
    fig.add_trace(
        go.Scatter(
            x=df_reg.index, 
            y=df_reg['Regression'],
            mode='lines', 
            name='Tendance',
            line=dict(color='orange', width=2, dash='solid')
        ),
        row=1, col=1
    )
    
    # Moyennes mobiles
    if show_ma:
        colors = ['blue', 'purple', 'darkgreen', 'brown']
        for i, period in enumerate(ma_periods):
            if f'MA_{period}' in df_reg.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_reg.index, 
                        y=df_reg[f'MA_{period}'],
                        mode='lines', 
                        name=f'MA {period}',
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )
    
    # Bandes de Bollinger
    if show_bollinger:
        std_dev = df_reg['Deviation'].std()
        fig.add_trace(
            go.Scatter(
                x=df_reg.index, 
                y=df_reg['Regression'] + 2*std_dev,
                mode='lines', 
                name='+2σ',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_reg.index, 
                y=df_reg['Regression'] - 2*std_dev,
                mode='lines', 
                name='-2σ',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Volume
    if show_volume:
        colors = ['red' if row['Open'] > row['Close'] else 'green' for i, row in df_reg.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df_reg.index, 
                y=df_reg['Volume'],
                name='Volume', 
                marker_color=colors,
                marker_line_width=0
            ),
            row=2, col=1
        )
        
        # Moyenne mobile du volume
        vol_ma = df_reg['Volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df_reg.index, 
                y=vol_ma,
                mode='lines', 
                name='Vol MA(20)',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Tendance",
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
    
    return fig


def render_trend_analysis(df_reg, indicators):
    """Render l'analyse détaillée de la tendance"""
    
    with st.expander("Analyse détaillée de la tendance", expanded=True):
        trend_cols = st.columns(3)
        
        with trend_cols[0]:
            # Analyse de la tendance récente
            recent_prices = df_reg['Close'].iloc[-30:] if len(df_reg) >= 30 else df_reg['Close']
            recent_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            recent_direction = "Haussière" if recent_slope > 0 else "Baissière"
            recent_color = "green" if recent_slope > 0 else "red"
            
            st.markdown(f"""
            #### Tendance des 30 derniers jours
            <span style='color:{recent_color}; font-weight:bold'>{recent_direction}</span>
            """, unsafe_allow_html=True)
            
            # Volatilité
            if len(df_reg) > 1:
                returns = df_reg['Close'].pct_change().dropna()
                volatility = returns.std() * 100
                st.markdown(f"**Volatilité:** {volatility:.2f}% (quotidienne)")
        
        with trend_cols[1]:
            # Supports et résistances
            if len(df_reg) > 30:
                recent_high = df_reg['High'].iloc[-30:].max()
                recent_low = df_reg['Low'].iloc[-30:].min()
                current = df_reg['Close'].iloc[-1]
                
                st.markdown(f"#### Supports/Résistances")
                st.markdown(f"**Résistance récente:** {recent_high:.2f}")
                st.markdown(f"**Support récent:** {recent_low:.2f}")
                
                # Potentiel de retournement
                reversal_potential = "Élevé" if abs(indicators['deviation']) >= 2 else "Moyen
