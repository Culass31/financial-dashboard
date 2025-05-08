"""
Interface principale de l'application Analyse Marchés Mondiaux
"""
import streamlit as st
from ui_components.sidebar import render_sidebar
from ui_components.technical_analysis import render_technical_analysis_tab
from ui_components.fundamental_analysis import render_fundamental_analysis_tab
from ui_components.news import render_news_tab
from ui_components.screener import render_screener_tab
from ui_components.portfolio import render_portfolio_tab
from ui_components.styles import apply_custom_styles
from core.market_data import get_market_structure

# Configuration de l'application
st.set_page_config(
    page_title="Analyse Marchés Mondiaux", 
    page_icon="📈", 
    layout="wide"
)

# Appliquer les styles personnalisés
apply_custom_styles()

# En-tête de l'application
st.title("📈 Dashboard d'Analyse des Marchés Financiers")
st.markdown("#### Analyse technique, fondamentale et screening d'actions")

# Charger la structure du marché
@st.cache_data(ttl=86400)  # Cache pour 24 heures
def load_market_structure():
    return get_market_structure()

# Charger les données
with st.spinner("Chargement des marchés...", show_time=True):
    market_structure = load_market_structure()

# Render sidebar et récupérer la sélection
selected_stock = render_sidebar(market_structure)

# Onglets principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Analyse Technique", 
    "📑 Analyse Fondamentale", 
    "📰 Actualités", 
    "🔎 Screener", 
    "🧩 Portefeuille"
])

# Render each tab
with tab1:
    render_technical_analysis_tab(selected_stock)

with tab2:
    render_fundamental_analysis_tab(selected_stock)

with tab3:
    render_news_tab(selected_stock)

with tab4:
    render_screener_tab(market_structure)

with tab5:
    render_portfolio_tab(market_structure)

# Footer
st.markdown("---")
st.markdown("📊 Dashboard développé pour l'analyse des marchés financiers. Données fournies par Yahoo Finance.")
st.markdown("⚠️ Ce dashboard est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement.")
