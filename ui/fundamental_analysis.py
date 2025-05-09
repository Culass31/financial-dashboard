# ui_components/fundamental_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional
import yfinance as yf
from datetime import datetime

from services import DataService, AnalysisService, NewsService
from utils.common import format_number
from utils.ui_metrics import display_price_with_trend


def render_fundamental_analysis_tab(selected_stock: Optional[Dict]):
    """Render the fundamental analysis tab with enhanced features"""
    
    if not selected_stock:
        st.info("Veuillez sélectionner une action dans le menu latéral pour voir l'analyse fondamentale.")
        return
    
    # Initialize services
    data_service = DataService()
    analysis_service = AnalysisService(data_service)
    news_service = NewsService(data_service)
    
    ticker = selected_stock['ticker']
    stock_name = st.session_state.get('selected_stock_name', '')
    sector = selected_stock['secteur']
    
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Fondamentale: {stock_name} ({ticker})</h2>", unsafe_allow_html=True)
    
    # Create sub-tabs for fundamental analysis
    fundamental_tab1, fundamental_tab2 = st.tabs(["📋 Profil & Données Financières", "🔮 Évaluation & Recommandation"])
    
    # Fetch fundamental data
    with st.spinner("Chargement des données fondamentales..."):
        fundamental_data = data_service.get_fundamental_data(ticker)
        income_stmt, balance_sheet, cashflow = data_service.get_historical_financials(ticker)
    
    # Tab 1: Profile and Financial Data
    with fundamental_tab1:
        if fundamental_data:
            render_company_profile(fundamental_data)
            render_market_data(fundamental_data)
            render_financial_metrics(fundamental_data)
            render_financial_statements(income_stmt, balance_sheet, cashflow)
        else:
            st.error(f"Impossible de récupérer les données fondamentales pour {ticker}")
    
    # Tab 2: Valuation and Recommendation
    with fundamental_tab2:
        render_valuation_analysis(
            ticker, stock_name, fundamental_data, 
            income_stmt, balance_sheet, cashflow,
            analysis_service, news_service
        )


def render_company_profile(fundamental_data: Dict):
    """Render company profile section"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Profil de l'entreprise")
        general_data = fundamental_data["Données générales"]
        
        # Company info
        st.markdown(f"**{general_data.get('Nom', 'N/A')}**")
        st.markdown(f"**Secteur:** {general_data.get('Secteur', 'N/A')} | **Industrie:** {general_data.get('Industrie', 'N/A')} | **Pays:** {general_data.get('Pays', 'N/A')}")
        
        # Website
        if general_data.get('Site web'):
            st.markdown(f"**Site web:** [{general_data.get('Site web')}]({general_data.get('Site web')})")
        
        # Description
        if general_data.get('Description'):
            with st.expander("Description de l'entreprise"):
                st.write(general_data.get('Description'))


def render_market_data(fundamental_data: Dict):
    """Render market data section"""
    st.subheader("📈 Données de marché")
    market_data = fundamental_data["Données de marché"]
    
    # Current price with trend
    current_price = market_data.get('Prix actuel')
    previous_close = market_data.get('Précédente clôture')
    
    if current_price and previous_close:
        display_price_with_trend("Prix actuel", current_price, previous_close)
    
    # Market data in columns
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


def render_financial_metrics(fundamental_data: Dict):
    """Render financial metrics section"""
    st.subheader("📊 Métriques financières clés")
    
    # Create tabs for different metric categories
    metric_tabs = st.tabs(["Ratios de valorisation", "Rentabilité", "Santé financière", "Croissance"])
    
    # Valuation ratios
    with metric_tabs[0]:
        col1, col2, col3 = st.columns(3)
        metrics = fundamental_data["Données fondamentales"]
        
        with col1:
            st.metric("PER", format_number(metrics.get('PER', 'N/A')))
            st.metric("PER Forward", format_number(metrics.get('PER (Forward)', 'N/A')))
        
        with col2:
            st.metric("PEG Ratio", format_number(metrics.get('PEG Ratio', 'N/A')))
            st.metric("P/B", format_number(metrics.get('P/B Cours/Valeur Comptable', 'N/A')))
        
        with col3:
            st.metric("P/S", format_number(metrics.get('P/S (Price to Sales)', 'N/A')))
            st.metric("EV/EBITDA", format_number(metrics.get('EV/EBITDA', 'N/A')))
    
    # Profitability
    with metric_tabs[1]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Marge brute", format_number(metrics.get('Marge brute (%)', 'N/A')))
            st.metric("Marge opérationnelle", format_number(metrics.get('Marge opérationnelle (%)', 'N/A')))
        
        with col2:
            st.metric("Marge nette", format_number(metrics.get('Marge nette (%)', 'N/A')))
            st.metric("ROE", format_number(metrics.get('ROE (%)', 'N/A')))
        
        with col3:
            st.metric("ROA", format_number(metrics.get('ROA (%)', 'N/A')))
            st.metric("ROIC", "À calculer")
    
    # Financial health
    with metric_tabs[2]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ratio d'endettement", format_number(metrics.get('Ratio d\'endettement', 'N/A')))
            st.metric("Quick Ratio", format_number(metrics.get('Quick Ratio', 'N/A')))
        
        with col2:
            st.metric("Current Ratio", format_number(metrics.get('Current Ratio', 'N/A')))
            st.metric("Dette/Capitaux propres", format_number(metrics.get('Ratio dette/capitaux propres', 'N/A')))
        
        with col3:
            debt_quality = evaluate_debt_quality(metrics.get('Ratio d\'endettement', 0))
            st.metric("Qualité de la dette", debt_quality)
    
    # Growth
    with metric_tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Croissance du CA", format_number(metrics.get('Croissance du CA (%)', 'N/A')))
            st.metric("Croissance du BPA", format_number(metrics.get('Croissance du BPA (%)', 'N/A')))
        
        with col2:
            st.metric("Dividende", format_number(metrics.get('Dividende', 'N/A')))
            st.metric("Rendement du dividende", format_number(metrics.get('Rendement du dividende (%)', 'N/A')))


def render_financial_statements(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cashflow: pd.DataFrame):
    """Render historical financial statements"""
    st.subheader("📜 États financiers historiques")
    
    financial_tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie", "Analyse graphique"])
    
    # Income statement
    with financial_tabs[0]:
        if not income_stmt.empty:
            income_display = income_stmt.copy() / 1e6
            st.dataframe(
                income_display.style.format("{:.2f}"),
                use_container_width=True
            )
            st.caption("Montants en millions")
            
            # Key metrics evolution
            if 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                fig = create_financial_evolution_chart(income_stmt, ['Total Revenue', 'Net Income', 'EBITDA'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données du compte de résultat non disponibles")
    
    # Balance sheet
    with financial_tabs[1]:
        if not balance_sheet.empty:
            balance_display = balance_sheet.copy() / 1e6
            st.dataframe(
                balance_display.style.format("{:.2f}"),
                use_container_width=True
            )
            st.caption("Montants en millions")
            
            # Asset composition chart
            if 'Total Assets' in balance_sheet.index:
                fig = create_asset_composition_chart(balance_sheet)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données du bilan non disponibles")
    
    # Cash flow
    with financial_tabs[2]:
        if not cashflow.empty:
            cashflow_display = cashflow.copy() / 1e6
            st.dataframe(
                cashflow_display.style.format("{:.2f}"),
                use_container_width=True
            )
            st.caption("Montants en millions")
            
            # Cash flow evolution
            if 'Operating Cash Flow' in cashflow.index:
                fig = create_cashflow_evolution_chart(cashflow)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données des flux de trésorerie non disponibles")
    
    # Graphical analysis
    with financial_tabs[3]:
        render_financial_analysis_charts(income_stmt, balance_sheet, cashflow)


def render_valuation_analysis(
    ticker: str, 
    stock_name: str,
    fundamental_data: Dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    analysis_service: AnalysisService,
    news_service: NewsService
):
    """Render valuation analysis section"""
    
    # Create tabs for different analyses
    valuation_tabs = st.tabs([
        "Évaluations", 
        "Méthodes de valorisation", 
        "Avantage compétitif", 
        "Dividendes",
        "AI Analysis"  # New tab for AI-powered analysis
    ])
    
    # Main valuation tab
    with valuation_tabs[0]:
        render_main_valuation(ticker, fundamental_data, income_stmt, balance_sheet, cashflow, analysis_service)
    
    # Valuation methods detail
    with valuation_tabs[1]:
        render_valuation_methods(ticker, fundamental_data, income_stmt, balance_sheet, cashflow, analysis_service)
    
    # Competitive advantage analysis
    with valuation_tabs[2]:
        render_competitive_advantage(income_stmt, balance_sheet, cashflow, fundamental_data, analysis_service)
    
    # Dividend analysis
    with valuation_tabs[3]:
        render_dividend_analysis(ticker, stock_name, news_service)
    
    # AI-powered analysis
    with valuation_tabs[4]:
        render_ai_analysis(ticker, fundamental_data, income_stmt, balance_sheet, cashflow)


def render_main_valuation(
    ticker: str,
    fundamental_data: Dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    analysis_service: AnalysisService
):
    """Render main valuation section"""
    
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        st.write("#### Valorisation intrinsèque calculée")
        
        # Calculate intrinsic value
        intrinsic_value_result = analysis_service.calculate_intrinsic_value(
            ticker, 
            fundamental_data, 
            (income_stmt, balance_sheet, cashflow)
        )
        
        if intrinsic_value_result and intrinsic_value_result.get('success', False):
            current_price = fundamental_data['Données de marché'].get('Prix actuel')
            intrinsic_value = intrinsic_value_result.get('intrinsic_value')
            mos_value = intrinsic_value_result.get('intrinsic_value_with_mos')
            deviation = intrinsic_value_result.get('deviation')
            recommendation = intrinsic_value_result.get('recommendation')
            
            # Display key metrics
            iv_col1, iv_col2, iv_col3 = st.columns(3)
            
            with iv_col1:
                if intrinsic_value:
                    value_color = "green" if intrinsic_value > current_price else "red"
                    st.markdown(f"**Valeur intrinsèque:** <span style='color:{value_color};'>{intrinsic_value:.2f} €</span>", unsafe_allow_html=True)
            
            with iv_col2:
                if mos_value:
                    mos_color = "green" if mos_value > current_price else "orange"
                    st.markdown(f"**Avec marge de sécurité:** <span style='color:{mos_color};'>{mos_value:.2f} €</span>", unsafe_allow_html=True)
            
            with iv_col3:
                if deviation is not None:
                    dev_color = "green" if deviation < 0 else "red"
                    st.markdown(f"**Écart de valorisation:** <span style='color:{dev_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
            
            # Recommendation
            rec_colors = {
                "ACHAT FORT": "darkgreen",
                "ACHETER": "green",
                "RENFORCER": "lightgreen",
                "CONSERVER": "gold",
                "ALLEGER": "orange",
                "VENDRE": "red"
            }
            
            st.markdown(f"**Recommandation:** <span style='color:{rec_colors.get(recommendation, 'gray')};font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
            
            # Visualization
            fig = create_valuation_comparison_chart(
                current_price, intrinsic_value, mos_value, 
                fundamental_data, ticker
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Impossible de calculer la valeur intrinsèque")
    
    with main_col2:
        st.write("#### Consensus des analystes")
        render_analyst_consensus(ticker, fundamental_data)


def render_valuation_methods(
    ticker: str,
    fundamental_data: Dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    analysis_service: AnalysisService
):
    """Render detailed valuation methods"""
    
    st.write("#### Détail des méthodes de valorisation")
    
    # Calculate intrinsic value to get method details
    valuation_result = analysis_service.calculate_intrinsic_value(
        ticker, 
        fundamental_data, 
        (income_stmt, balance_sheet, cashflow)
    )
    
    if not valuation_result['success']:
        st.error(f"Erreur de valorisation: {valuation_result['message']}")
        return
    
    # Create tabs for each method
    method_tabs = st.tabs(["DCF", "Graham", "PER", "Valeur patrimoniale"])
    
    current_price = valuation_result['current_price']
    methods = valuation_result['methods']
    
    # DCF Method
    with method_tabs[0]:
        render_dcf_method(methods['dcf'], current_price, cashflow)
    
    # Graham Method
    with method_tabs[1]:
        render_graham_method(methods['graham'], current_price, fundamental_data, income_stmt)
    
    # PER Method
    with method_tabs[2]:
        render_per_method(methods['per_based'], current_price, fundamental_data, ticker)
    
    # Asset-based Method
    with method_tabs[3]:
        render_asset_method(methods['asset_based'], current_price, balance_sheet)


def render_competitive_advantage(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    fundamental_data: Dict,
    analysis_service: AnalysisService
):
    """Render competitive advantage analysis"""
    
    with st.spinner("Analyse de l'avantage compétitif en cours..."):
        moat_analysis = analysis_service.analyze_competitive_advantage(
            income_stmt, 
            balance_sheet, 
            cashflow, 
            fundamental_data["Données générales"]
        )
    
    if moat_analysis:
        # Display moat existence
        moat_existence = moat_analysis.get('existence', 'Indéterminé')
        moat_strength = moat_analysis.get('strength', 0)
        moat_factors = moat_analysis.get('factors', [])
        moat_explanation = moat_analysis.get('explanation', '')
        
        # Color based on strength
        moat_color = {
            0: 'gray',   # Undetermined
            1: 'orange', # Weak
            2: 'blue',   # Moderate
            3: 'green'   # Strong
        }.get(moat_strength, 'gray')
        
        # Display competitive advantage
        st.markdown(f"""
        <div style='background-color: {moat_color}; padding: 10px; border-radius: 5px; color: white;'>
            <h4 style='margin: 0;'>🛡️ Avantage compétitif: {moat_existence}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for moat analysis
        moat_col1, moat_col2 = st.columns(2)
        
        with moat_col1:
            if moat_factors:
                st.markdown("##### Facteurs contribuant à l'avantage compétitif:")
                for factor in moat_factors:
                    st.markdown(f"- {factor}")
            
            # ROIC display
            roic = analysis_service.calculate_roic(income_stmt, balance_sheet)
            if roic is not None:
                roic_color = "green" if roic > 15 else "orange" if roic > 10 else "red"
                st.markdown(f"""
                <div style='background-color: {roic_color}; padding: 5px; border-radius: 5px; color: white; text-align: center;'>
                    <h4 style='margin: 0;'>ROIC: {roic:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
        
        with moat_col2:
            # Create moat visualization
            fig = create_moat_visualization(moat_analysis)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Analyse:** {moat_explanation}")
        
        # Detailed metrics
        render_moat_metrics(income_stmt, balance_sheet, cashflow)


def render_dividend_analysis(ticker: str, stock_name: str, news_service: NewsService):
    """Render dividend analysis section"""
    
    st.subheader("💰 Analyse des Dividendes")
    
    with st.spinner("Analyse de la politique de dividendes en cours..."):
        dividend_analysis = news_service.get_dividend_policy_analysis(ticker)
    
    if dividend_analysis.get('error'):
        st.error(f"Erreur lors de l'analyse des dividendes: {dividend_analysis['error']}")
        return
    
    if not dividend_analysis.get('has_dividends', False):
        st.info(f"{stock_name} ne verse pas de dividendes actuellement.")
        return
    
    # Display dividend metrics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Indicateurs clés")
        
        metrics_data = [
            ("Rendement du dividende", f"{dividend_analysis.get('dividend_yield', 0):.2f}%"),
            ("Santé du dividende", quality_badge(dividend_analysis.get('dividend_health', 'Non évalué'))),
            ("Croissance du dividende", f"{dividend_analysis.get('dividend_growth', 0):.2f}%/an"),
            ("Consistance", quality_badge(dividend_analysis.get('dividend_consistency', 'Non évalué'))),
            ("Qualité globale", quality_badge(dividend_analysis.get('dividend_quality', 'Non évalué'))),
        ]
        
        for name, value in metrics_data:
            st.markdown(f"**{name}:** {value}", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Visualisation des dividendes")
        
        if dividend_analysis.get('dividend_history'):
            fig = create_dividend_charts(dividend_analysis)
            st.plotly_chart(fig, use_container_width=True)
    
    # Dividend quality analysis
    st.markdown("### 🔍 Analyse et interprétation")
    render_dividend_quality_analysis(dividend_analysis)


def render_ai_analysis(
    ticker: str,
    fundamental_data: Dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame
):
    """Render AI-powered analysis section"""
    
    st.subheader("🤖 Analyse IA")
    
    st.info("""
    Cette section utilisera des agents IA pour fournir une analyse approfondie :
    - Agent d'analyse fondamentale
    - Agent d'analyse technique
    - Agent de sentiment de marché
    - Agent de recommandation
    """)
    
    # Placeholder for AI analysis
    st.markdown("### Analyse par agents IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Agent d'analyse fondamentale")
        st.write("- Évaluation des états financiers")
        st.write("- Analyse des ratios clés")
        st.write("- Détection d'anomalies")
    
    with col2:
        st.markdown("#### Agent de sentiment")
        st.write("- Analyse des actualités")
        st.write("- Sentiment des analystes")
        st.write("- Tendances des réseaux sociaux")
    
    st.markdown("#### Recommandation consolidée IA")
    st.warning("Fonctionnalité à implémenter avec des modèles LLM")


# Helper functions

def evaluate_debt_quality(debt_ratio: float) -> str:
    """Evaluate debt quality based on ratio"""
    if isinstance(debt_ratio, str) or debt_ratio is None:
        return "N/A"
    
    if debt_ratio < 0.3:
        return "Excellent"
    elif debt_ratio < 0.5:
        return "Bon"
    elif debt_ratio < 0.8:
        return "Moyen"
    else:
        return "Élevé"


def quality_badge(quality: str) -> str:
    """Create a colored badge for quality indicators"""
    colors = {
        'Excellent': '#4CAF50',
        'Bon': '#8BC34A',
        'Moyen': '#FFC107',
        'Faible': '#F44336',
        'Risqué': '#D32F2F',
        'Soutenable': '#4CAF50',
        'Modéré': '#FFC107',
        'Non évalué': '#9E9E9E',
        'Indéterminé': '#9E9E9E'
    }
    color = colors.get(quality, '#9E9E9E')
    return f"<span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;'>{quality}</span>"


def create_financial_evolution_chart(income_stmt: pd.DataFrame, metrics: list) -> go.Figure:
    """Create financial metrics evolution chart"""
    fig = go.Figure()
    
    for metric in metrics:
        if metric in income_stmt.index:
            values = income_stmt.loc[metric] / 1e6  # Convert to millions
            years = [col.year for col in income_stmt.columns]
            
            fig.add_trace(go.Bar(
                name=metric,
                x=years,
                y=values,
                text=[f'{v:.2f}M' for v in values],
                textposition='auto',
            ))
    
    fig.update_layout(
        title="Évolution des métriques financières",
        xaxis_title="Année",
        yaxis_title="Montant (millions)",
        barmode='group',
        height=400
    )
    
    return fig


def create_asset_composition_chart(balance_sheet: pd.DataFrame) -> go.Figure:
    """Create asset composition pie chart"""
    latest_period = balance_sheet.columns[0]
    
    categories = []
    values = []
    
    asset_categories = {
        'Actifs courants': ['Current Assets'],
        'Actifs long terme': ['Long Term Assets'],
        'Immobilisations': ['Property Plant Equipment Net'],
        'Goodwill': ['Goodwill And Intangible Assets']
    }
    
    for category, items in asset_categories.items():
        for item in items:
            if item in balance_sheet.index:
                value = balance_sheet.loc[item, latest_period]
                if value and value > 0:
                    categories.append(category)
                    values.append(value)
                    break
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=.3
    )])
    
    fig.update_layout(
        title=f"Composition des actifs ({latest_period.year})",
        height=400
    )
    
    return fig


def create_cashflow_evolution_chart(cashflow: pd.DataFrame) -> go.Figure:
    """Create cash flow evolution chart"""
    fig = go.Figure()
    
    metrics = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
    colors = ['blue', 'red', 'green', 'purple']
    
    for metric, color in zip(metrics, colors):
        if metric in cashflow.index:
            values = cashflow.loc[metric] / 1e6  # Convert to millions
            years = [col.year for col in cashflow.columns]
            
            fig.add_trace(go.Scatter(
                name=metric,
                x=years,
                y=values,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title="Évolution des flux de trésorerie",
        xaxis_title="Année",
        yaxis_title="Montant (millions)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_valuation_comparison_chart(
    current_price: float,
    intrinsic_value: float,
    mos_value: float,
    fundamental_data: Dict,
    ticker: str
) -> go.Figure:
    """Create valuation comparison chart"""
    
    price_data = []
    
    # Add current price
    price_data.append({
        'type': 'Prix actuel',
        'valeur': current_price,
        'couleur': 'blue'
    })
    
    # Add intrinsic value
    price_data.append({
        'type': 'Valeur intrinsèque',
        'valeur': intrinsic_value,
        'couleur': 'green' if intrinsic_value > current_price else 'red'
    })
    
    # Add value with margin of safety
    price_data.append({
        'type': 'Valeur avec MOS',
        'valeur': mos_value,
        'couleur': 'green' if mos_value > current_price else 'orange'
    })
    
    # Add analyst target if available
    ticker_obj = yf.Ticker(ticker)
    if hasattr(ticker_obj, 'analyst_price_targets') and ticker_obj.analyst_price_targets:
        target_data = ticker_obj.analyst_price_targets
        mean_target = target_data.get('mean', None)
        if mean_target:
            price_data.append({
                'type': 'Objectif analystes',
                'valeur': mean_target,
                'couleur': 'purple' if mean_target > current_price else 'pink'
            })
    
    # Create figure
    fig = go.Figure()
    
    for item in price_data:
        fig.add_trace(go.Bar(
            x=[item['type']],
            y=[item['valeur']],
            name=item['type'],
            marker_color=item['couleur'],
            text=[f"{item['valeur']:.2f} €"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Comparaison des différentes valorisations',
        xaxis_title='Source de valorisation',
        yaxis_title='Valeur (€)',
        showlegend=False,
        height=400
    )
    
    # Add current price line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=current_price,
        x1=len(price_data)-0.5,
        y1=current_price,
        line=dict(
            color="blue",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=len(price_data)-0.5,
        y=current_price,
        text=f"Prix actuel: {current_price:.2f} €",
        showarrow=False,
        yshift=10,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    return fig


def render_analyst_consensus(ticker: str, fundamental_data: Dict):
    """Render analyst consensus section"""
    
    ticker_obj = yf.Ticker(ticker)
    current_price = fundamental_data['Données de marché'].get('Prix actuel')
    
    # Price targets
    if hasattr(ticker_obj, 'analyst_price_targets') and ticker_obj.analyst_price_targets:
        target_data = ticker_obj.analyst_price_targets
        
        pt_col1, pt_col2, pt_col3 = st.columns(3)
        
        with pt_col1:
            mean_target = target_data.get('mean', None)
            if mean_target:
                mean_color = "green" if mean_target > current_price else "red"
                st.markdown(f"**Objectif moyen:** <span style='color:{mean_color};'>{mean_target:.2f} €</span>", unsafe_allow_html=True)
                potential = ((mean_target / current_price) - 1) * 100
                st.markdown(f"**Potentiel:** <span style='color:{mean_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
        
        with pt_col2:
            high_target = target_data.get('high', None)
            low_target = target_data.get('low', None)
            if high_target and low_target:
                st.markdown(f"**Plus haut:** <span style='color:green;'>{high_target:.2f} €</span>", unsafe_allow_html=True)
                st.markdown(f"**Plus bas:** <span style='color:gray;'>{low_target:.2f} €</span>", unsafe_allow_html=True)
        
        with pt_col3:
            if high_target and low_target and mean_target:
                dispersion = (high_target - low_target) / mean_target * 100
                disp_color = "green" if dispersion < 20 else "orange" if dispersion < 40 else "red"
                st.markdown(f"**Dispersion:** <span style='color:{disp_color};'>{dispersion:.1f}%</span>", unsafe_allow_html=True)
                
                disp_text = "Faible" if dispersion < 20 else "Moyenne" if dispersion < 40 else "Élevée"
                st.markdown(f"**Consensus:** <span style='color:{disp_color};'>{disp_text}</span>", unsafe_allow_html=True)
    else:
        st.info("Aucune donnée d'objectif de cours disponible")
    
    # Recommendations
    if hasattr(ticker_obj, 'recommendations_summary') and not ticker_obj.recommendations_summary.empty:
        latest_rec = ticker_obj.recommendations_summary.iloc[0]
        
        # Create recommendations visualization
        rec_fig = go.Figure()
        
        categories = ['Achat fort', 'Achat', 'Conserver', 'Vendre', 'Vente forte']
        values = [latest_rec['strongBuy'], latest_rec['buy'], latest_rec['hold'], 
                 latest_rec['sell'], latest_rec['strongSell']]
        colors = ['darkgreen', 'green', 'gold', 'orange', 'red']
        
        total_recs = sum(values)
        
        for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
            rec_fig.add_trace(go.Bar(
                y=[f"Mois {latest_rec['period']}"],
                x=[val],
                name=cat,
                orientation='h',
                marker=dict(color=color),
                text=f"{val} ({val/total_recs*100:.0f}%)" if val > 0 else "",
                textposition="inside",
                hoverinfo="text",
                hovertext=f"{cat}: {val} ({val/total_recs*100:.1f}%)"
            ))
        
        rec_fig.update_layout(
            title='Répartition des recommandations',
            barmode='stack',
            height=120,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            xaxis=dict(title="Nombre d'analystes"),
            yaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(rec_fig, use_container_width=True)
        
        # Consensus score
        consensus_score = (latest_rec['strongBuy'] * 2 + latest_rec['buy'] * 1 + 
                          latest_rec['hold'] * 0 + latest_rec['sell'] * -1 + 
                          latest_rec['strongSell'] * -2) / total_recs
        
        if consensus_score > 1:
            cons_color = "darkgreen"
            cons_text = "FORT ACHAT"
        elif consensus_score > 0.5:
            cons_color = "green"
            cons_text = "ACHAT"
        elif consensus_score > -0.5:
            cons_color = "gold"
            cons_text = "NEUTRE"
        elif consensus_score > -1:
            cons_color = "orange"
            cons_text = "VENTE"
        else:
            cons_color = "red"
            cons_text = "FORT VENTE"
        
        st.markdown(f"**Consensus des analystes:** <span style='color:{cons_color};font-weight:bold;'>{cons_text}</span> (Score: {consensus_score:.2f})", unsafe_allow_html=True)
        
        # Historical recommendations
        if len(ticker_obj.recommendations_summary) > 1:
            rec_history_fig = create_recommendations_history_chart(ticker_obj.recommendations_summary)
            st.plotly_chart(rec_history_fig, use_container_width=True)
    else:
        st.info("Aucune recommandation d'analyste disponible")


def render_dcf_method(dcf_data: Dict, current_price: float, cashflow: pd.DataFrame):
    """Render DCF method details"""
    
    st.write("##### Modèle DCF (Discounted Cash Flow)")
    
    dcf_value = dcf_data.get('value')
    if dcf_value:
        dcf_color = "green" if dcf_value > current_price else "red"
        st.markdown(f"**Valeur DCF calculée:** <span style='color:{dcf_color};font-weight:bold;'>{dcf_value:.2f} €</span>", unsafe_allow_html=True)
        
        potential = ((dcf_value / current_price) - 1) * 100
        potential_color = "green" if potential > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
        
        # Display FCF history
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fig = create_fcf_evolution_chart(cashflow)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Impossible de calculer la valeur DCF pour cette entreprise en raison de données insuffisantes.")


def render_graham_method(graham_data: Dict, current_price: float, fundamental_data: Dict, income_stmt: pd.DataFrame):
    """Render Graham method details"""
    
    st.write("##### Méthode de Graham (Valeur Intrinsèque)")
    
    graham_value = graham_data.get('value')
    netnet_value = graham_data.get('netnet_value')
    eps = fundamental_data['Données fondamentales'].get('BPA', None)
    
    if graham_value and eps:
        graham_color = "green" if graham_value > current_price else "red"
        st.markdown(f"**Valeur selon Graham:** <span style='color:{graham_color};font-weight:bold;'>{graham_value:.2f} €</span>", unsafe_allow_html=True)
        
        deviation = ((graham_value / current_price) - 1) * 100
        deviation_color = "green" if deviation > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{deviation_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
        
        # Display formula used
        st.write(f"**Formule utilisée:** V = {eps:.2f} € * (8.5 + 2 * taux de croissance) * 4.4 / rendement obligataire")
        
        # Display Net-Net value if available
        if netnet_value is not None:
            netnet_color = "green" if netnet_value > current_price else "red"
            st.markdown(f"**Valeur Net-Net par action:** <span style='color:{netnet_color};'>{netnet_value:.2f} €</span>", unsafe_allow_html=True)
            
            with st.expander("Qu'est-ce que la valeur Net-Net ?"):
                st.write("""
                La méthode Net-Net de Benjamin Graham évalue une entreprise uniquement sur la base de ses actifs courants moins ses passifs totaux.
                C'est une approche très conservatrice qui cherche des entreprises négociées en dessous de leur valeur liquidative.
                """)
    else:
        st.warning("Impossible de calculer la valeur de Graham en raison d'un BPA négatif ou non disponible.")


def render_per_method(per_data: Dict, current_price: float, fundamental_data: Dict, ticker: str):
    """Render PER method details"""
    
    st.write("##### Valorisation basée sur le PER (Price Earnings Ratio)")
    
    per_value = per_data.get('value')
    per = fundamental_data['Données fondamentales'].get('PER', None)
    eps = fundamental_data['Données fondamentales'].get('BPA', None)
    
    if per is not None and eps is not None and eps > 0 and per_value:
        # Display PER comparison
        st.markdown(f"**PER actuel:** {per:.2f}x")
        
        # Valuation table
        per_col1, per_col2 = st.columns(2)
        
        with per_col2:
            per_scenarios = create_per_scenarios(eps, per, current_price)
            per_df = pd.DataFrame(per_scenarios)
            
            # Style the dataframe
            def color_potential(val):
                val_num = float(val.strip('%'))
                return f'color: {"green" if val_num > 0 else "red"}'
            
            styled_df = per_df.style.applymap(color_potential, subset=['Potentiel'])
            st.dataframe(styled_df, use_container_width=True)
        
        with per_col1:
            # Visualization of PER scenarios
            fig = create_per_scenarios_chart(per_scenarios, current_price)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Données insuffisantes pour calculer la valorisation basée sur le PER.")


def render_asset_method(asset_data: Dict, current_price: float, balance_sheet: pd.DataFrame):
    """Render asset-based valuation method"""
    
    st.write("##### Valorisation basée sur les actifs (Valeur patrimoniale)")
    
    asset_value = asset_data.get('value')
    if asset_value:
        asset_color = "green" if asset_value > current_price else "red"
        st.markdown(f"**Valeur patrimoniale par action:** <span style='color:{asset_color};font-weight:bold;'>{asset_value:.2f} €</span>", unsafe_allow_html=True)
        
        potential = ((asset_value / current_price) - 1) * 100
        potential_color = "green" if potential > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
        
        with st.expander("À propos de la valeur patrimoniale"):
            st.write("""
            La méthode de valorisation patrimoniale évalue une entreprise sur la base de ses actifs nets tangibles.
            Elle est particulièrement pertinente pour les entreprises des secteurs financiers, immobiliers ou industriels.
            """)
        
        # Display balance sheet composition
        if not balance_sheet.empty:
            st.write("##### Principales composantes du bilan (dernière année)")
            fig = create_balance_sheet_composition_chart(balance_sheet)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Impossible de calculer la valeur patrimoniale pour cette entreprise.")


def render_financial_analysis_charts(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame
):
    """Render comprehensive financial analysis charts"""
    
    chart_type = st.selectbox(
        "Sélectionnez le type d'analyse",
        ["Évolution des marges", "Structure financière", "Efficacité opérationnelle", "Rentabilité"]
    )
    
    if chart_type == "Évolution des marges":
        if not income_stmt.empty:
            fig = create_margins_evolution_chart(income_stmt)
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Structure financière":
        if not balance_sheet.empty:
            fig = create_financial_structure_chart(balance_sheet)
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Efficacité opérationnelle":
        if not income_stmt.empty and not balance_sheet.empty:
            fig = create_efficiency_metrics_chart(income_stmt, balance_sheet)
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Rentabilité":
        if not income_stmt.empty and not balance_sheet.empty:
            fig = create_profitability_metrics_chart(income_stmt, balance_sheet)
            st.plotly_chart(fig, use_container_width=True)


def render_moat_metrics(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame
):
    """Render detailed moat metrics"""
    
    # Create columns for main metrics
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        # FCF historical chart
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fig = create_fcf_evolution_chart(cashflow)
            st.plotly_chart(fig, use_container_width=True)
        
        # Margin stability
        if not income_stmt.empty and len(income_stmt.columns) >= 4:
            fig = create_margin_stability_chart(income_stmt)
            st.plotly_chart(fig, use_container_width=True)
    
    with main_col2:
        # ROE evolution
        if not income_stmt.empty and not balance_sheet.empty:
            fig = create_roe_evolution_chart(income_stmt, balance_sheet)
            st.plotly_chart(fig, use_container_width=True)
        
        # Debt analysis
        fig = create_debt_analysis_chart(income_stmt, balance_sheet)
        st.plotly_chart(fig, use_container_width=True)


def render_dividend_quality_analysis(dividend_analysis: Dict):
    """Render dividend quality analysis"""
    
    quality = dividend_analysis.get('dividend_quality', 'Non évalué')
    health = dividend_analysis.get('dividend_health', 'Non évalué')
    growth = dividend_analysis.get('dividend_growth_score', 'Non évalué')
    consistency = dividend_analysis.get('dividend_consistency', 'Non évalué')
    
    # Identify strengths and weaknesses
    strengths = []
    weaknesses = []
    
    if quality in ['Excellent', 'Bon']:
        strengths.append(f"la qualité globale du dividende est **{quality.lower()}**")
    if health in ['Excellent', 'Bon']:
        strengths.append(f"la santé financière supportant le dividende est **{health.lower()}**")
    if growth in ['Excellent', 'Bon']:
        strengths.append(f"la croissance du dividende est **{growth.lower()}**")
    if consistency in ['Excellent', 'Bon']:
        strengths.append(f"la consistance des versements est **{consistency.lower()}**")
    if dividend_analysis.get('dividend_aristocrat', False):
        strengths.append("l'entreprise est un **aristocrate du dividende**")
    
    if quality in ['Moyen', 'Faible']:
        weaknesses.append(f"la qualité globale du dividende est **{quality.lower()}**")
    if health in ['Moyen', 'Faible']:
        weaknesses.append(f"la santé financière supportant le dividende est **{health.lower()}**")
    if dividend_analysis.get('sustainability') == 'Risqué':
        weaknesses.append("le taux de distribution est **risqué**")
    
    # Display analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Points forts")
        if strengths:
            for strength in strengths:
                st.markdown(f"✅ {strength}")
        else:
            st.markdown("*Aucun point fort significatif identifié*")
    
    with col2:
        st.markdown("#### Points à surveiller")
        if weaknesses:
            for weakness in weaknesses:
                st.markdown(f"⚠️ {weakness}")
        else:
            st.markdown("*Aucun point faible significatif identifié*")
    
    # Recommendation
    st.markdown("#### Recommandation")
    
    if quality == 'Excellent':
        recommendation = "Cet investissement présente une **excellente politique de dividende** qui combine santé financière, croissance et consistance."
    elif quality == 'Bon':
        recommendation = "Cet investissement présente une **bonne politique de dividende** avec une balance appropriée entre distribution et réinvestissement."
    elif quality == 'Moyen':
        recommendation = "Cet investissement présente une politique de dividende **acceptable mais avec des points d'attention**."
    elif quality == 'Faible':
        recommendation = "Cet investissement présente une politique de dividende **présentant des risques**."
    else:
        recommendation = "L'analyse complète de la politique de dividende n'a pas pu être réalisée avec les données disponibles."
    
    st.info(recommendation)


# Chart creation functions (implementations)

def create_fcf_evolution_chart(cashflow: pd.DataFrame) -> go.Figure:
    """Create Free Cash Flow evolution chart"""
    
    fcf_data = cashflow.loc['Free Cash Flow'].to_dict()
    
    fcf_df = pd.DataFrame({
        'Année': [str(year.year) for year in fcf_data.keys()],
        'FCF (millions)': [float(value) / 1e6 for value in fcf_data.values()]
    })
    
    fcf_df = fcf_df.sort_values('Année')
    
    # Calculate CAGR
    cagr_text = ""
    if len(fcf_df) > 1:
        fcf_values = fcf_df['FCF (millions)'].values
        positive_values = fcf_values[fcf_values > 0]
        
        if len(positive_values) > 1:
            start_value = positive_values[0]
            end_value = positive_values[-1]
            years = len(positive_values) - 1
            cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
            cagr_text = f"Croissance annuelle moyenne: {cagr:.2f}%"
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=fcf_df['Année'],
        y=fcf_df['FCF (millions)'],
        marker_color=['green' if fcf >= 0 else 'red' for fcf in fcf_df['FCF (millions)']],
        text=[f"€{fcf:.2f}M" for fcf in fcf_df['FCF (millions)']],
        textposition='outside',
        name='Free Cash Flow'
    ))
    
    fig.update_layout(
        title=f"Évolution des Free Cash Flows<br><sup>{cagr_text}</sup>",
        xaxis_title="Année",
        yaxis_title="FCF (millions €)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_recommendations_history_chart(recommendations: pd.DataFrame) -> go.Figure:
    """Create recommendations history chart"""
    
    fig = go.Figure()
    
    # Convert periods to month names
    def get_month_name(period_str):
        months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
        now = datetime.now()
        current_index = now.month - 1
        
        if period_str == "0m":
            return months[current_index]
        
        if period_str.startswith("-") and period_str.endswith("m"):
            months_back = int(period_str[1:-1])
            target_index = (current_index - months_back) % 12
            return months[target_index]
        
        return period_str
    
    # Prepare data
    rec_history = recommendations.iloc[:4].copy()
    periods = [get_month_name(p) for p in rec_history['period']]
    
    # Add traces for each recommendation type
    categories = ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']
    names = ['Achat fort', 'Achat', 'Conserver', 'Vendre', 'Vente forte']
    colors = ['darkgreen', 'green', 'gold', 'orange', 'red']
    
    for cat, name, color in zip(categories, names, colors):
        fig.add_trace(go.Bar(
            x=periods,
            y=rec_history[cat],
            name=name,
            marker_color=color,
            text=rec_history[cat],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Évolution des recommandations des analystes',
        xaxis_title='Période',
        yaxis_title='Nombre d\'analystes',
        barmode='stack',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_per_scenarios(eps: float, current_per: float, current_price: float) -> list:
    """Create PER valuation scenarios"""
    
    scenarios = []
    
    # Different PER scenarios
    per_targets = {
        "PER moyen historique": current_per * 0.9,
        "PER sectoriel": current_per,  # Simplified
        "PER conservateur": current_per * 0.7,
        "PER optimiste": current_per * 1.2
    }
    
    for scenario, per_target in per_targets.items():
        value = eps * per_target
        scenarios.append({
            "Scénario": scenario,
            "PER cible": f"{per_target:.1f}x",
            "Valorisation": f"{value:.2f} €",
            "Potentiel": f"{((value / current_price) - 1) * 100:.1f}%"
        })
    
    return scenarios


def create_per_scenarios_chart(scenarios: list, current_price: float) -> go.Figure:
    """Create PER scenarios visualization"""
    
    fig = go.Figure()
    
    # Add current price line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=current_price,
        x1=3.5,
        y1=current_price,
        line=dict(color="blue", width=2, dash="dash"),
    )
    
    # Add bars for each scenario
    fig.add_trace(go.Bar(
        x=[s["Scénario"] for s in scenarios],
        y=[float(s["Valorisation"].split()[0]) for s in scenarios],
        marker_color=[
            'green' if float(s["Potentiel"].strip('%')) > 0 else 'red'
            for s in scenarios
        ],
        text=[s["Potentiel"] for s in scenarios],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Valorisations selon différents PER',
        xaxis_title='Scénario de PER',
        yaxis_title='Valorisation (€)',
        height=400,
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
    
    return fig


def create_balance_sheet_composition_chart(balance_sheet: pd.DataFrame) -> go.Figure:
    """Create balance sheet composition chart"""
    
    latest_period = balance_sheet.columns[0]
    
    # Get balance sheet data
    balance_data = {}
    
    key_metrics = {
        'Total Assets': 'Actifs totaux',
        'Total Liabilities Net Minority Interest': 'Passifs totaux',
        'Goodwill And Intangible Assets': 'Goodwill et Intangibles',
        'Net Tangible Assets': 'Actifs nets tangibles'
    }
    
    for key, label in key_metrics.items():
        if key in balance_sheet.index:
            balance_data[label] = balance_sheet.loc[key, latest_period] / 1e6
    
    if balance_data:
        fig = go.Figure()
        
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
        
        return fig
    
    return go.Figure()  # Empty figure if no data


def create_margins_evolution_chart(income_stmt: pd.DataFrame) -> go.Figure:
    """Create margins evolution chart"""
    
    fig = go.Figure()
    
    # Calculate margins
    if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
        gross_margins = (income_stmt.loc['Gross Profit'] / income_stmt.loc['Total Revenue'] * 100)
        years = [col.year for col in income_stmt.columns]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=gross_margins,
            name='Marge brute',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
    
    if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
        operating_margins = (income_stmt.loc['Operating Income'] / income_stmt.loc['Total Revenue'] * 100)
        
        fig.add_trace(go.Scatter(
            x=years,
            y=operating_margins,
            name='Marge opérationnelle',
            mode='lines+markers',
            line=dict(color='green', width=2)
        ))
    
    if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
        net_margins = (income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue'] * 100)
        
        fig.add_trace(go.Scatter(
            x=years,
            y=net_margins,
            name='Marge nette',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title='Évolution des marges',
        xaxis_title='Année',
        yaxis_title='Marge (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_financial_structure_chart(balance_sheet: pd.DataFrame) -> go.Figure:
    """Create financial structure chart"""
    
    fig = go.Figure()
    
    years = [col.year for col in balance_sheet.columns]
    
    # Key balance sheet items
    metrics = {
        'Total Assets': 'Actifs totaux',
        'Total Liabilities Net Minority Interest': 'Passifs totaux',
        'Stockholders Equity': 'Capitaux propres',
        'Total Debt': 'Dette totale'
    }
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for (metric, label), color in zip(metrics.items(), colors):
        if metric in balance_sheet.index:
            values = balance_sheet.loc[metric] / 1e6  # Convert to millions
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                name=label,
                mode='lines+markers',
                line=dict(color=color, width=2)
            ))
    
    fig.update_layout(
        title='Structure financière',
        xaxis_title='Année',
        yaxis_title='Montant (millions €)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_efficiency_metrics_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> go.Figure:
    """Create efficiency metrics chart"""
    
    fig = go.Figure()
    
    years = [col.year for col in income_stmt.columns]
    metrics = []
    
    # Asset turnover
    if 'Total Revenue' in income_stmt.index and 'Total Assets' in balance_sheet.index:
        asset_turnover = []
        for i in range(min(len(income_stmt.columns), len(balance_sheet.columns))):
            revenue = income_stmt.loc['Total Revenue', income_stmt.columns[i]]
            assets = balance_sheet.loc['Total Assets', balance_sheet.columns[i]]
            if revenue and assets and assets != 0:
                asset_turnover.append(revenue / assets)
        
        if asset_turnover:
            fig.add_trace(go.Scatter(
                x=years[:len(asset_turnover)],
                y=asset_turnover,
                name='Rotation des actifs',
                mode='lines+markers',
                line=dict(color='blue', width=2)
            ))
    
    # Receivables turnover
    if 'Total Revenue' in income_stmt.index and 'Receivables' in balance_sheet.index:
        receivables_turnover = []
        for i in range(min(len(income_stmt.columns), len(balance_sheet.columns))):
            revenue = income_stmt.loc['Total Revenue', income_stmt.columns[i]]
            receivables = balance_sheet.loc['Receivables', balance_sheet.columns[i]]
            if revenue and receivables and receivables != 0:
                receivables_turnover.append(revenue / receivables)
        
        if receivables_turnover:
            fig.add_trace(go.Scatter(
                x=years[:len(receivables_turnover)],
                y=receivables_turnover,
                name='Rotation des créances',
                mode='lines+markers',
                line=dict(color='green', width=2)
            ))
    
    fig.update_layout(
        title='Métriques d\'efficacité opérationnelle',
        xaxis_title='Année',
        yaxis_title='Ratio',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_profitability_metrics_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> go.Figure:
    """Create profitability metrics chart"""
    
    fig = go.Figure()
    
    years = []
    roe_values = []
    roa_values = []
    
    # Calculate ROE and ROA
    for i in range(min(len(income_stmt.columns), len(balance_sheet.columns))):
        if 'Net Income' in income_stmt.index:
            net_income = income_stmt.loc['Net Income', income_stmt.columns[i]]
            
            # ROE
            if 'Stockholders Equity' in balance_sheet.index:
                equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[i]]
                if net_income and equity and equity != 0:
                    roe_values.append((net_income / equity) * 100)
                    years.append(income_stmt.columns[i].year)
            
            # ROA
            if 'Total Assets' in balance_sheet.index:
                assets = balance_sheet.loc['Total Assets', balance_sheet.columns[i]]
                if net_income and assets and assets != 0:
                    roa_values.append((net_income / assets) * 100)
    
    if roe_values:
        fig.add_trace(go.Scatter(
            x=years,
            y=roe_values,
            name='ROE (%)',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
    
    if roa_values:
        fig.add_trace(go.Scatter(
            x=years,
            y=roa_values,
            name='ROA (%)',
            mode='lines+markers',
            line=dict(color='green', width=2)
        ))
    
    fig.update_layout(
        title='Métriques de rentabilité',
        xaxis_title='Année',
        yaxis_title='Pourcentage (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_margin_stability_chart(income_stmt: pd.DataFrame) -> go.Figure:
    """Create margin stability chart"""
    
    years = [col.year for col in income_stmt.columns[:5]]
    gross_margins = []
    
    if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
        for col in income_stmt.columns[:5]:
            gross_profit = income_stmt.loc['Gross Profit', col]
            total_revenue = income_stmt.loc['Total Revenue', col]
            if gross_profit and total_revenue and total_revenue != 0:
                margin = gross_profit / total_revenue * 100
                gross_margins.append(margin)
    
    fig = go.Figure()
    
    if gross_margins:
        fig.add_trace(go.Bar(
            x=years[:len(gross_margins)],
            y=gross_margins,
            name='Marge brute (%)',
            marker_color='blue'
        ))
        
        # Add average line
        avg_margin = sum(gross_margins) / len(gross_margins)
        fig.add_hline(
            y=avg_margin,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Moyenne: {avg_margin:.1f}%"
        )
    
    fig.update_layout(
        title='Stabilité des marges brutes',
        xaxis_title='Année',
        yaxis_title='Marge brute (%)',
        height=300
    )
    
    return fig


def create_roe_evolution_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> go.Figure:
    """Create ROE evolution chart"""
    
    roe_values = []
    roe_years = []
    
    for i in range(min(5, len(income_stmt.columns), len(balance_sheet.columns))):
        if 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance_sheet.index:
            net_income = income_stmt.loc['Net Income', income_stmt.columns[i]]
            equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[i]]
            if net_income and equity and equity != 0:
                roe = net_income / equity * 100
                roe_values.append(roe)
                roe_years.append(income_stmt.columns[i].year)
    
    fig = go.Figure()
    
    if roe_values:
        fig.add_trace(go.Scatter(
            x=roe_years,
            y=roe_values,
            name='ROE (%)',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        # Add average line
        avg_roe = sum(roe_values) / len(roe_values)
        fig.add_hline(
            y=avg_roe,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Moyenne: {avg_roe:.1f}%"
        )
    
    fig.update_layout(
        title='Évolution du ROE',
        xaxis_title='Année',
        yaxis_title='ROE (%)',
        height=300
    )
    
    return fig


def create_debt_analysis_chart(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> go.Figure:
    """Create debt analysis chart"""
    
    debt_metrics = {}
    
    # Debt to equity ratio
    if 'Total Debt' in balance_sheet.index and 'Stockholders Equity' in balance_sheet.index:
        debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]]
        equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]]
        if debt is not None and equity is not None and equity != 0:
            debt_metrics["Dette/Capitaux propres"] = debt / equity
    
    # Interest coverage ratio
    if 'EBIT' in income_stmt.index and 'Interest Expense' in income_stmt.index:
        ebit = income_stmt.loc['EBIT', income_stmt.columns[0]]
        interest = abs(income_stmt.loc['Interest Expense', income_stmt.columns[0]])
        if ebit is not None and interest is not None and interest != 0:
            debt_metrics["Couverture des intérêts"] = ebit / interest
    
    # Debt evolution
    if 'Total Debt' in balance_sheet.index:
        debt_years = []
        debt_values = []
        
        for col in balance_sheet.columns[:5]:
            debt_value = balance_sheet.loc['Total Debt', col]
            if debt_value is not None:
                debt_values.append(debt_value / 1e6)
                debt_years.append(col.year)
        
        fig = go.Figure()
        
        if debt_values:
            fig.add_trace(go.Bar(
                x=debt_years,
                y=debt_values,
                name='Dette totale (M€)',
                marker_color='red'
            ))
            
            # Add trend analysis
            if len(debt_values) > 1:
                if debt_values[0] < debt_values[-1]:
                    trend_text = "Dette en augmentation"
                    trend_color = "red"
                else:
                    trend_text = "Dette en diminution"
                    trend_color = "green"
                
                fig.add_annotation(
                    x=debt_years[-1],
                    y=max(debt_values) * 1.1,
                    text=trend_text,
                    showarrow=False,
                    font=dict(color=trend_color, size=12, weight='bold')
                )
        
        fig.update_layout(
            title='Évolution de l\'endettement',
            xaxis_title='Année',
            yaxis_title='Dette (millions €)',
            height=300
        )
        
        return fig
    
    return go.Figure()


def create_dividend_charts(dividend_analysis: Dict) -> go.Figure:
    """Create dividend visualization charts"""
    
    div_history = dividend_analysis['dividend_history']
    dates = list(div_history.keys())
    values = list(div_history.values())
    
    dividend_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'dividend': values
    })
    
    dividend_df = dividend_df.sort_values('date')
    dividend_df['year'] = dividend_df['date'].dt.year
    annual_div = dividend_df.groupby('year')['dividend'].sum().reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Historique des dividendes', 'Dividende annuel et croissance'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5]
    )
    
    # Quarterly dividends
    fig.add_trace(
        go.Bar(
            x=dividend_df['date'],
            y=dividend_df['dividend'],
            marker_color='#1E88E5',
            name='Dividende trimestriel'
        ),
        row=1, col=1
    )
    
    # Annual dividends
    fig.add_trace(
        go.Bar(
            x=annual_div['year'],
            y=annual_div['dividend'],
            name="Dividende annuel",
            marker_color='#43A047'
        ),
        row=2, col=1,
        secondary_y=False
    )
    
    # Growth rate
    if len(annual_div) > 1:
        annual_div['growth'] = annual_div['dividend'].pct_change() * 100
        
        fig.add_trace(
            go.Scatter(
                x=annual_div['year'][1:],
                y=annual_div['growth'][1:],
                name="Taux de croissance",
                mode='lines+markers',
                line=dict(color='#E65100', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1,
            secondary_y=True
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Dividende par action", row=1, col=1)
    fig.update_yaxes(title_text="Dividende par action", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Croissance annuelle (%)", secondary_y=True, row=2, col=1)
    
    return fig


def create_moat_visualization(moat_analysis: Dict) -> go.Figure:
    """Create moat strength visualization"""
    
    categories = ['Marges', 'ROE', 'ROIC', 'FCF', 'Total']
    
    # Define scores based on analysis
    scores = {
        'Marges': 0,
        'ROE': 0,
        'ROIC': 0,
        'FCF': 0,
        'Total': moat_analysis.get('strength', 0) * 33.33
    }
    
    # Assign scores based on factors
    for factor in moat_analysis.get('factors', []):
        if 'Marges' in factor:
            scores['Marges'] = 50 if 'très stables' in factor else 30
        elif 'ROE' in factor:
            scores['ROE'] = 50 if 'élevé' in factor else 30
        elif 'ROIC' in factor:
            scores['ROIC'] = 50 if 'excellent' in factor else 30
        elif 'FCF' in factor:
            scores['FCF'] = 50 if 'Forte' in factor else 30
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        fillcolor='rgba(64, 224, 208, 0.3)',
        line=dict(color='rgb(64, 224, 208)', width=2),
        name='Force des avantages'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        title="Analyse de l'avantage compétitif",
        height=400,
        showlegend=False
    )
    
    return fig