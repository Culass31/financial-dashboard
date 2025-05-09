"""
Interface principale de l'application Analyse Marchés Mondiaux
"""
import streamlit as st
from ui.sidebar import render_sidebar
from ui.technical_analysis import render_technical_analysis_tab
from ui.fundamental_analysis import render_fundamental_analysis_tab
from ui.news import render_news_tab
from ui.screener import render_screener_tab
from ui.portfolio import render_portfolio_tab
from utils.ui_styles import apply_custom_styles
from services import DataService

# Configuration de l'application
st.set_page_config(
    page_title="Analyse Marchés Mondiaux", 
    page_icon="📈", 
    layout="wide"
)
# Instance globale du data service (singleton pattern)
@st.cache_resource
def get_data_service():
    return DataService()

data_service = get_data_service()

# Initialiser la session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_stock_name' not in st.session_state:
    st.session_state.selected_stock_name = ""
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Appliquer les styles personnalisés
apply_custom_styles()

# En-tête de l'application
st.title("📈 Dashboard d'Analyse des Marchés Financiers")
st.markdown("#### Analyse technique, fondamentale et screening d'actions")

# Charger la structure du marché
def load_market_structure():
    return data_service.get_market_structure()

# Charger les données
with st.spinner("Chargement des marchés...", show_time=True):
    market_structure = load_market_structure()

# Render sidebar et récupérer la sélection
selected_stock = render_sidebar(market_structure)

# Afficher un message si aucune action n'est sélectionnée
if not selected_stock:
    st.info("⬅️ Sélectionnez une action dans le menu latéral pour commencer l'analyse")
    st.markdown("""
    ## Bienvenue dans le Dashboard d'Analyse Financière
    
    Ce dashboard vous permet d'analyser en profondeur les actions des marchés mondiaux avec :
    
    - **📊 Analyse Technique** : Graphiques avancés, indicateurs techniques, tendances
    - **📑 Analyse Fondamentale** : États financiers, valorisation, métriques clés
    - **📰 Actualités** : Dernières nouvelles et événements importants
    - **🔎 Screener** : Recherche d'actions selon les critères Buffett-Graham
    - **🧩 Portefeuille** : Gestion et analyse de votre portefeuille
    
    ### Comment commencer ?
    1. Choisissez une méthode de filtrage dans le menu latéral
    2. Sélectionnez une action
    3. Explorez les différents onglets d'analyse
    """)
else:
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
