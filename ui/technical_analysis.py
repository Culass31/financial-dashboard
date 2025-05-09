# ui_components/technical_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple

from services import DataService, AnalysisService


class TechnicalAnalysisComponent:
    """Composant d'interface pour l'analyse technique"""
    
    def __init__(self):
        self.data_service = DataService()
        self.analysis_service = AnalysisService(self.data_service)
        self._initialize_session_states()
    
    def _initialize_session_states(self):
        """Initialise les états de session pour l'analyse technique"""
        if 'technical_settings' not in st.session_state:
            st.session_state.technical_settings = {
                'show_bollinger': True,
                'show_ma': False,
                'ma_periods': [50, 200],
                'regression_type': 'Linéaire',
                'show_volume': True,
                'selected_period': '10y',
                'chart_type': 'candlestick'
            }
    
    def render(self, selected_stock: Dict):
        """Render le composant d'analyse technique"""
        if not selected_stock:
            st.info("Veuillez sélectionner une action pour voir l'analyse technique")
            return
        
        ticker = selected_stock['ticker']
        selected_stock_name = selected_stock.get('nom', ticker)
        
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Technique: {selected_stock_name} ({ticker})</h2>", 
                   unsafe_allow_html=True)
        
        # Créer le layout principal
        col_chart, col_indicators = st.columns([3, 1])
        
        # Colonne principale avec le graphique
        with col_chart:
            self._render_chart_section(ticker)
        
        # Colonne des indicateurs
        with col_indicators:
            self._render_indicators_section(ticker)
        
        # Section d'analyse détaillée
        self._render_detailed_analysis_section(ticker)
    
    def _render_chart_section(self, ticker: str):
        """Render la section du graphique principal"""
        # Options avancées
        with st.expander("Options avancées", expanded=False):
            self._render_advanced_options()
        
        # Sélection de la période
        selected_period = self._render_period_selector()
        
        # Mettre à jour la période sélectionnée
        st.session_state.technical_settings['selected_period'] = selected_period
        
        # Récupération et préparation des données
        with st.spinner("Chargement des données historiques..."):
            df = self.data_service.get_stock_history(ticker, period=selected_period)
            
            if not df.empty:
                # Analyse technique
                df_reg, indicators = self.analysis_service.calculate_regression(df)
                
                # Ajouter les moyennes mobiles si demandé
                if st.session_state.technical_settings['show_ma']:
                    self._add_moving_averages(df_reg)
                
                # Créer et afficher le graphique
                fig = self._create_main_chart(df_reg, indicators)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Aucune donnée historique disponible pour {ticker}")
    
    def _render_advanced_options(self):
        """Render les options avancées"""
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            # Type de graphique
            chart_type = st.selectbox(
                "Type de graphique",
                ["Chandeliers", "Ligne", "OHLC"],
                index=0 if st.session_state.technical_settings['chart_type'] == 'candlestick' else 1
            )
            st.session_state.technical_settings['chart_type'] = 'candlestick' if chart_type == "Chandeliers" else 'line'
            
            # Indicateurs techniques
            st.session_state.technical_settings['show_bollinger'] = st.checkbox(
                "Bandes de Bollinger",
                value=st.session_state.technical_settings['show_bollinger']
            )
            
            st.session_state.technical_settings['show_ma'] = st.checkbox(
                "Moyennes mobiles",
                value=st.session_state.technical_settings['show_ma']
            )
            
            if st.session_state.technical_settings['show_ma']:
                st.session_state.technical_settings['ma_periods'] = st.multiselect(
                    "Périodes",
                    [20, 50, 100, 200],
                    default=st.session_state.technical_settings['ma_periods']
                )
        
        with opt_col2:
            # Type de régression
            st.session_state.technical_settings['regression_type'] = st.radio(
                "Type de régression",
                ["Linéaire", "Logarithmique"],
                index=0 if st.session_state.technical_settings['regression_type'] == "Linéaire" else 1
            )
            
            # Affichage du volume
            st.session_state.technical_settings['show_volume'] = st.checkbox(
                "Afficher le volume",
                value=st.session_state.technical_settings['show_volume']
            )
            
            # Indicateurs supplémentaires
            show_rsi = st.checkbox("RSI", value=False)
            if show_rsi:
                st.session_state.technical_settings['show_rsi'] = True
                st.session_state.technical_settings['rsi_period'] = st.slider(
                    "Période RSI",
                    min_value=5,
                    max_value=30,
                    value=14
                )
    
    def _render_period_selector(self) -> str:
        """Render le sélecteur de période"""
        period_cols = st.columns(10)
        periods = {
            "5j": "5d",
            "1m": "1mo",
            "3m": "3mo",
            "6m": "6mo",
            "1a": "1y",
            "2a": "2y",
            "5a": "5y",
            "10a": "10y",
            "YTD": "ytd",
            "Max": "max"
        }
        
        selected_period = st.session_state.technical_settings['selected_period']
        
        for i, (period_name, period_value) in enumerate(periods.items()):
            with period_cols[i]:
                if st.button(
                    period_name,
                    key=f"period_{period_name}",
                    type="primary" if period_value == selected_period else "secondary"
                ):
                    selected_period = period_value
        
        return selected_period
    
    def _add_moving_averages(self, df: pd.DataFrame):
        """Ajoute les moyennes mobiles au DataFrame"""
        for period in st.session_state.technical_settings['ma_periods']:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    
    def _create_main_chart(self, df_reg: pd.DataFrame, indicators: Dict) -> go.Figure:
        """Crée le graphique principal"""
        show_volume = st.session_state.technical_settings['show_volume']
        
        fig = make_subplots(
            rows=2 if show_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3] if show_volume else [1],
            subplot_titles=("Prix", "Volume") if show_volume else ("Prix",)
        )
        
        # Ajouter le graphique de prix
        self._add_price_chart(fig, df_reg)
        
        # Ajouter la régression
        self._add_regression_line(fig, df_reg)
        
        # Ajouter les moyennes mobiles
        if st.session_state.technical_settings['show_ma']:
            self._add_moving_averages_to_chart(fig, df_reg)
        
        # Ajouter les bandes de Bollinger
        if st.session_state.technical_settings['show_bollinger']:
            self._add_bollinger_bands(fig, df_reg)
        
        # Ajouter le volume
        if show_volume:
            self._add_volume_chart(fig, df_reg)
        
        # Mise en forme du graphique
        self._update_chart_layout(fig, show_volume)
        
        return fig
    
    def _add_price_chart(self, fig: go.Figure, df: pd.DataFrame):
        """Ajoute le graphique de prix au figure"""
        if st.session_state.technical_settings['chart_type'] == 'candlestick':
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Prix",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Prix',
                    line=dict(color='#1e88e5', width=2)
                ),
                row=1, col=1
            )
    
    def _add_regression_line(self, fig: go.Figure, df: pd.DataFrame):
        """Ajoute la ligne de régression"""
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Regression'],
                mode='lines',
                name='Tendance',
                line=dict(color='#ff6f00', width=2, dash='solid')
            ),
            row=1, col=1
        )
    
    def _add_moving_averages_to_chart(self, fig: go.Figure, df: pd.DataFrame):
        """Ajoute les moyennes mobiles au graphique"""
        colors = ['#3f51b5', '#9c27b0', '#009688', '#795548']
        
        for i, period in enumerate(st.session_state.technical_settings['ma_periods']):
            if f'MA_{period}' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f'MA_{period}'],
                        mode='lines',
                        name=f'MA {period}',
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )
    
    def _add_bollinger_bands(self, fig: go.Figure, df: pd.DataFrame):
        """Ajoute les bandes de Bollinger"""
        std_dev = df['Deviation'].std()
        
        # Bandes supérieures
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Regression'] + 2*std_dev,
                mode='lines',
                name='+2σ',
                line=dict(color='#e57373', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Regression'] + std_dev,
                mode='lines',
                name='+1σ',
                line=dict(color='#ffcdd2', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Bandes inférieures
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Regression'] - std_dev,
                mode='lines',
                name='-1σ',
                line=dict(color='#c8e6c9', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Regression'] - 2*std_dev,
                mode='lines',
                name='-2σ',
                line=dict(color='#81c784', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    def _add_volume_chart(self, fig: go.Figure, df: pd.DataFrame):
        """Ajoute le graphique de volume"""
        colors = ['#ef5350' if row['Open'] > row['Close'] else '#26a69a' 
                 for i, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                marker_line_width=0
            ),
            row=2, col=1
        )
        
        # Ajouter la moyenne mobile du volume
        vol_ma = df['Volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vol_ma,
                mode='lines',
                name='Vol MA(20)',
                line=dict(color='#1e88e5', width=1)
            ),
            row=2, col=1
        )
    
    def _update_chart_layout(self, fig: go.Figure, show_volume: bool):
        """Met à jour la mise en forme du graphique"""
        fig.update_layout(
            title=None,  # Le titre est affiché séparément
            height=600 if show_volume else 500,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(250, 250, 250, 1)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font=dict(family="Arial, sans-serif", size=12, color="#333333")
        )
        
        # Améliorer l'apparence des axes
        axis_config = {
            'showgrid': True,
            'gridwidth': 0.5,
            'gridcolor': 'rgba(211, 211, 211, 0.5)',
            'showline': True,
            'linewidth': 1,
            'linecolor': '#333333',
            'mirror': True
        }
        
        fig.update_xaxes(**axis_config)
        fig.update_yaxes(**axis_config)
        
        # Configuration spécifique pour l'axe des prix
        fig.update_yaxes(title_text="Prix", row=1, col=1)
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    def _render_indicators_section(self, ticker: str):
        """Render la section des indicateurs techniques"""
        # Prix actuel avec tendance
        self._render_current_price(ticker)
        
        # Récupération des données pour les indicateurs
        df = self.data_service.get_stock_history(ticker, period='1y')
        if df.empty:
            st.warning("Données insuffisantes pour calculer les indicateurs")
            return
        
        df_reg, indicators = self.analysis_service.calculate_regression(df)
        
        # Jauges pour les indicateurs clés
        self._render_indicator_gauges(indicators)
        
        # Force du momentum
        self._render_momentum_indicator(df)
        
        # Objectifs de cours des analystes
        self._render_analyst_targets(ticker)
    
    def _render_current_price(self, ticker: str):
        """Affiche le prix actuel avec indicateur de tendance"""
        df = self.data_service.get_stock_history(ticker, period='5d')
        
        if not df.empty and len(df) > 1:
            current_price = df['Close'].iloc[-1]
            previous_price = df['Close'].iloc[-2]
            
            change_pct = ((current_price - previous_price) / previous_price) * 100
            arrow = "↑" if change_pct >= 0 else "↓"
            color = "green" if change_pct >= 0 else "red"
            
            st.markdown(f"""
            <div style='background-color: rgba(0, 0, 0, 0.05); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <div style='font-size: 1rem; font-weight: bold;'>Prix actuel</div>
                <div style='font-size: 1.4rem; font-weight: bold;'>{current_price:.2f} €</div>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: 1rem; color: {color};'>
                        {arrow} {abs(change_pct):.2f}%
                    </div>
                    <div style='font-size: 0.9rem; margin-left: 10px; color: #666;'>
                        Préc: {previous_price:.2f} €
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_indicator_gauges(self, indicators: Dict):
        """Affiche les jauges pour les indicateurs techniques"""
        if not indicators:
            return
        
        # Tendance (corrélation)
        correlation = indicators.get('correlation', 0)
        fig_trend = self._create_gauge(
            value=correlation,
            title="Tendance",
            min_val=-1,
            max_val=1,
            threshold_values=[-0.7, 0.7],
            threshold_labels=["Baissière", "Neutre", "Haussière"]
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Croissance annualisée
        model_growth = indicators.get('model_growth', 0)
        fig_growth = self._create_gauge(
            value=model_growth,
            title="Croissance Annuelle (%)",
            min_val=-20,
            max_val=20,
            threshold_values=[0, 5],
            threshold_labels=["Négative", "Faible", "Forte"]
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Écart par rapport à la régression
        deviation = indicators.get('deviation', 0)
        fig_deviation = self._create_gauge(
            value=deviation,
            title="Écart (σ)",
            min_val=-3,
            max_val=3,
            threshold_values=[-1.5, 1.5],
            threshold_labels=["Survente", "Zone neutre", "Surachat"]
        )
        st.plotly_chart(fig_deviation, use_container_width=True)
    
    def _create_gauge(
        self,
        value: float,
        title: str,
        min_val: float,
        max_val: float,
        threshold_values: list = None,
        threshold_labels: list = None
    ) -> go.Figure:
        """Crée une jauge pour un indicateur"""
        if value is None:
            value_str = "N/A"
            normalized_value = 0.5
            color = "gray"
        else:
            value = round(value, 2)
            value_str = f"{value:.2f}"
            
            if max_val == min_val:
                normalized_value = 0.5
            else:
                normalized_value = (value - min_val) / (max_val - min_val)
            
            normalized_value = max(0, min(1, normalized_value))
            
            if normalized_value < 0.3:
                color = "#ef5350"
            elif normalized_value < 0.7:
                color = "#ffa726"
            else:
                color = "#66bb6a"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
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
                    {'range': [min_val, min_val + (max_val - min_val) * 0.3], 'color': 'rgba(239, 83, 80, 0.2)'},
                    {'range': [min_val + (max_val - min_val) * 0.3, min_val + (max_val - min_val) * 0.7], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [min_val + (max_val - min_val) * 0.7, max_val], 'color': 'rgba(102, 187, 106, 0.2)'}
                ],
                'shape': "angular"
            },
            number={
                'font': {'size': 16},
                'valueformat': ".2f"
            }
        ))
        
        fig.update_layout(
            height=150,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="white",
            showlegend=False,
            autosize=True
        )
        
        return fig
    
    def _render_momentum_indicator(self, df: pd.DataFrame):
        """Affiche l'indicateur de momentum"""
        if len(df) > 10:
            momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
            
            st.markdown("<h4>Force du Momentum</h4>", unsafe_allow_html=True)
            
            momentum_color = "#66bb6a" if momentum > 3 else "#ef5350" if momentum < -3 else "#ffa726"
            momentum_strength = "Fort" if abs(momentum) > 5 else "Modéré" if abs(momentum) > 2 else "Faible"
            
            momentum_chart = go.Figure(go.Indicator(
                mode="gauge+number",
                value=momentum,
                number={'suffix': "%", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [-10, 10], 'tickwidth': 1},
                    'bar': {'color': momentum_color},
                    'steps': [
                        {'range': [-10, -3], 'color': 'rgba(239, 83, 80, 0.2)'},
                        {'range': [-3, 3], 'color': 'rgba(255, 167, 38, 0.2)'},
                        {'range': [3, 10], 'color': 'rgba(102, 187, 106, 0.2)'}
                    ]
                },
                title={'text': f"{momentum_strength}"}
            ))
            
            momentum_chart.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            
            st.plotly_chart(momentum_chart, use_container_width=True)
    
    def _render_analyst_targets(self, ticker: str):
        """Affiche les objectifs de cours des analystes"""
        # Utiliser get_stock_info et get_stock_analyst_price_targets depuis data_service
        info = self.data_service.get_stock_info(ticker)
        analyst_targets = self.data_service.get_stock_analyst_price_targets(ticker)
        
        if info and analyst_targets:
            st.markdown("<h4>Objectif de cours</h4>", unsafe_allow_html=True)
            
            current_price = info.get('currentPrice', 0)
            mean_target = analyst_targets.get('mean', 0)
            low_target = analyst_targets.get('low', 0)
            high_target = analyst_targets.get('high', 0)
            
            if all([current_price, mean_target, low_target, high_target]):
                # Créer un gauge chart
                fig = go.Figure()
                
                min_display = min(low_target, current_price) * 0.95
                max_display = max(high_target, current_price) * 1.05
                
                fig.add_trace(go.Indicator(
                    mode="number+gauge",
                    value=current_price,
                    gauge={
                        'axis': {'range': [min_display, max_display]},
                        'bar': {'color': "blue"},
                        'steps': [
                            {'range': [min_display, low_target], 'color': "lightgray"},
                            {'range': [low_target, high_target], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "purple", 'width': 4},
                            'thickness': 0.75,
                            'value': mean_target
                        }
                    },
                    number={'suffix': "€", 'font': {'size': 24}}
                ))
                
                fig.add_annotation(
                    x=0.5, y=1.15,
                    text=f"Objectif moyen: {mean_target:.2f}€",
                    showarrow=False,
                    font=dict(size=16, color="purple")
                )
                
                fig.add_annotation(
                    x=0.2, y=1.15,
                    text=f"+ bas: {low_target:.2f}€",
                    showarrow=False,
                    font=dict(size=14, color="lightgray")
                )
                
                fig.add_annotation(
                    x=0.8, y=1.15,
                    text=f"+ haut: {high_target:.2f}€",
                    showarrow=False,
                    font=dict(size=14, color="lightblue")
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_analysis_section(self, ticker: str):
        """Render la section d'analyse détaillée"""
        with st.expander("Analyse détaillée de la tendance", expanded=True):
            # Récupération des données
            df = self.data_service.get_stock_history(ticker, period='1y')
            if df.empty:
                st.warning("Données insuffisantes pour l'analyse détaillée")
                return
            
            df_reg, indicators = self.analysis_service.calculate_regression(df)
            
            trend_cols = st.columns(3)
            
            with trend_cols[0]:
                self._render_short_term_trend(df_reg)
                self._render_volatility(df_reg)
            
            with trend_cols[1]:
                self._render_support_resistance(df_reg)
                self._render_reversal_potential(indicators)
            
            with trend_cols[2]:
                self._render_performance_comparison(indicators)
            
            # Recommandation technique
            self._render_technical_recommendation(indicators, df_reg)
    
    def _render_short_term_trend(self, df: pd.DataFrame):
        """Analyse de la tendance court terme"""
        recent_prices = df['Close'].iloc[-30:] if len(df) >= 30 else df['Close']
        recent_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        recent_direction = "Haussière" if recent_slope > 0 else "Baissière"
        recent_color = "green" if recent_slope > 0 else "red"
        
        st.markdown(f"""
        #### Tendance des 30 derniers jours
        <span style='color:{recent_color}; font-weight:bold'>{recent_direction}</span>
        """, unsafe_allow_html=True)
    
    def _render_volatility(self, df: pd.DataFrame):
        """Calcul et affichage de la volatilité"""
        if len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            volatility_level = "Élevée" if volatility > 2 else "Modérée" if volatility > 1 else "Faible"
            st.markdown(f"**Volatilité:** {volatility:.2f}% ({volatility_level})")
    
    def _render_support_resistance(self, df: pd.DataFrame):
        """Analyse des supports et résistances"""
        if len(df) > 30:
            recent_high = df['High'].iloc[-30:].max()
            recent_low = df['Low'].iloc[-30:].min()
            current = df['Close'].iloc[-1]
            
            st.markdown("#### Supports/Résistances")
            st.markdown(f"**Résistance récente:** {recent_high:.2f}")
            st.markdown(f"**Support récent:** {recent_low:.2f}")
            
            dist_to_resistance = ((recent_high / current) - 1) * 100
            dist_to_support = (1 - (recent_low / current)) * 100
            
            st.markdown(f"Distance résistance: {dist_to_resistance:.1f}%")
            st.markdown(f"Distance support: {dist_to_support:.1f}%")
    
    def _render_reversal_potential(self, indicators: Dict):
        """Évalue le potentiel de retournement"""
        deviation = indicators.get('deviation', 0)
        
        if abs(deviation) >= 2:
            potential = "Élevé"
            color = "#ef5350"
        elif abs(deviation) >= 1:
            potential = "Moyen"
            color = "#ffa726"
        else:
            potential = "Faible"
            color = "#66bb6a"
        
        st.markdown(f"""
        #### Potentiel de retournement
        <span style='color:{color}; font-weight:bold'>{potential}</span>
        """, unsafe_allow_html=True)
    
    def _render_performance_comparison(self, indicators: Dict):
        """Compare les performances à un benchmark"""
        st.markdown("#### Performances")
        
        if 'prog_1y' in indicators and 'prog_5y' in indicators:
            prog_1y = indicators['prog_1y']
            prog_5y = indicators['prog_5y']
            
            # Benchmark: marché global à 8% annualisé
            benchmark_1y = 8.0
            benchmark_5y = 8.0 * 5
            
            diff_1y = prog_1y - benchmark_1y
            diff_5y = prog_5y - benchmark_5y
            
            color_1y = "#66bb6a" if diff_1y > 0 else "#ef5350"
            color_5y = "#66bb6a" if diff_5y > 0 else "#ef5350"
            
            st.markdown(f"""
            **1 an:** {prog_1y:.2f}% 
            <span style='color:{color_1y}'>{diff_1y:+.2f}% vs marché</span>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **5 ans:** {prog_5y:.2f}% 
            <span style='color:{color_5y}'>{diff_5y:+.2f}% vs marché</span>
            """, unsafe_allow_html=True)
    
    def _render_technical_recommendation(self, indicators: Dict, df: pd.DataFrame):
        """Génère et affiche une recommandation technique"""
        st.markdown("<h4>Recommandation technique</h4>", unsafe_allow_html=True)
        
        if not indicators:
            st.warning("Données insuffisantes pour générer une recommandation")
            return
        
        # Générer le signal technique
        signal, signal_color = self._generate_technical_signal(indicators)
        
        # Afficher le signal principal
        st.markdown(f"""
        <div style='background-color: rgba({self._get_rgba_color(signal_color)}); 
                    padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;'>
            <h3 style='color: {signal_color}; margin: 0;'>{signal}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Générer les signaux détaillés
        signals = self._generate_detailed_signals(indicators, df)
        
        # Afficher les signaux
        if signals:
            for signal_text in signals:
                st.markdown(f"<div style='margin: 5px 0;'>{signal_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("Aucun signal technique significatif")
        
        # Résumé textuel
        summary = self._generate_technical_summary(indicators)
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 15px;'>
            <p style='margin: 0;'><em>{summary}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def _generate_technical_signal(self, indicators: Dict) -> Tuple[str, str]:
        """Génère le signal technique principal"""
        correlation = indicators.get('correlation', 0)
        model_growth = indicators.get('model_growth', 0)
        deviation = indicators.get('deviation', 0)
        
        if correlation > 0.7 and model_growth > 5:
            if deviation < -1:
                return "ACHAT", "#66bb6a"
            elif deviation < 0:
                return "RENFORCER", "#81c784"
            elif deviation > 1.5:
                return "ALLEGER", "#ffa726"
            else:
                return "CONSERVER", "#ffeb3b"
        elif correlation < -0.7 and model_growth < -5:
            if deviation < -1.5:
                return "CONSERVER", "#ffeb3b"  # Possible rebond
            else:
                return "VENTE", "#ef5350"
        else:
            if deviation < -1.5:
                return "RENFORCER", "#81c784"
            elif deviation > 1.5:
                return "ALLEGER", "#ffa726"
            else:
                return "NEUTRE", "#9e9e9e"
    
    def _generate_detailed_signals(self, indicators: Dict, df: pd.DataFrame) -> list:
        """Génère les signaux techniques détaillés"""
        signals = []
        
        correlation = indicators.get('correlation', 0)
        deviation = indicators.get('deviation', 0)
        
        # Signaux basés sur la corrélation
        if correlation > 0.5:
            signals.append("✅ Tendance haussière solide")
        elif correlation < -0.5:
            signals.append("❌ Tendance baissière solide")
        
        # Signaux basés sur la déviation
        if deviation > 1.5:
            signals.append("⚠️ Surachat potentiel")
        elif deviation < -1.5:
            signals.append("⚠️ Survente potentielle")
        
        # Vérifier les moyennes mobiles
        if st.session_state.technical_settings['show_ma'] and len(df) > 200:
            ma_signals = self._check_moving_average_crossover(df)
            signals.extend(ma_signals)
        
        # Vérifier le momentum
        if len(df) > 10:
            momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
            if momentum > 5:
                signals.append("✅ Momentum fortement positif")
            elif momentum < -5:
                signals.append("❌ Momentum fortement négatif")
        
        # Vérifier la volatilité
        if len(df) > 20:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            if volatility > 3:
                signals.append("⚠️ Volatilité très élevée")
            elif volatility < 0.5:
                signals.append("ℹ️ Volatilité très faible")
        
        return signals
    
    def _check_moving_average_crossover(self, df: pd.DataFrame) -> list:
        """Vérifie les croisements de moyennes mobiles"""
        signals = []
        
        if 'MA_50' in df.columns and 'MA_200' in df.columns:
            # Vérifier les 5 derniers jours
            last_days = min(5, len(df) - 1)
            
            for i in range(1, last_days + 1):
                current_short = df['MA_50'].iloc[-i]
                current_long = df['MA_200'].iloc[-i]
                
                if i < len(df) - 1:
                    prev_short = df['MA_50'].iloc[-(i+1)]
                    prev_long = df['MA_200'].iloc[-(i+1)]
                    
                    # Croisement doré
                    if prev_short <= prev_long and current_short > current_long:
                        signals.append("✅ Croisement doré récent (signal d'achat)")
                        break
                    # Croisement de la mort
                    elif prev_short >= prev_long and current_short < current_long:
                        signals.append("❌ Croisement de la mort récent (signal de vente)")
                        break
        
        return signals
    
    def _generate_technical_summary(self, indicators: Dict) -> str:
        """Génère un résumé textuel de l'analyse technique"""
        correlation = indicators.get('correlation', 0)
        deviation = indicators.get('deviation', 0)
        
        if correlation > 0.7:
            if deviation > 1.5:
                return "L'action est en tendance haussière forte mais potentiellement surachetée. Une correction technique pourrait survenir à court terme."
            elif deviation < -1.5:
                return "L'action est en tendance haussière forte et actuellement sous-évaluée. Bon point d'entrée potentiel."
            else:
                return "L'action est en tendance haussière forte et évolue dans sa zone de valeur normale."
        elif correlation < -0.7:
            if deviation > 1.5:
                return "L'action est en tendance baissière forte mais temporairement surachetée. La tendance baissière pourrait reprendre."
            elif deviation < -1.5:
                return "L'action est en tendance baissière forte et actuellement survendue. Un rebond technique pourrait survenir."
            else:
                return "L'action est en tendance baissière forte et évolue dans sa zone de valeur normale."
        else:
            if deviation > 1.5:
                return "L'action est en phase de consolidation avec un potentiel de correction à la baisse."
            elif deviation < -1.5:
                return "L'action est en phase de consolidation avec un potentiel de rebond à la hausse."
            else:
                return "L'action évolue sans tendance claire dans sa zone de valeur normale."
    
    def _get_rgba_color(self, color: str) -> str:
        """Convertit une couleur en format RGBA"""
        color_map = {
            "#66bb6a": "102, 187, 106, 0.2",  # Vert
            "#81c784": "129, 199, 132, 0.2",  # Vert clair
            "#ef5350": "239, 83, 80, 0.2",    # Rouge
            "#ffa726": "255, 167, 38, 0.2",   # Orange
            "#ffeb3b": "255, 235, 59, 0.2",   # Jaune
            "#9e9e9e": "158, 158, 158, 0.2"   # Gris
        }
        return color_map.get(color, "158, 158, 158, 0.2")


def render_technical_analysis_tab(selected_stock: Optional[Dict]):
    """Fonction principale pour render l'onglet d'analyse technique"""
    component = TechnicalAnalysisComponent()
    component.render(selected_stock)