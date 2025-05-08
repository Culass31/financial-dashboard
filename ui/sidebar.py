"""
Composant sidebar pour la sélection d'actions
"""
import streamlit as st
from core.market_data import flatten_market_structure

def render_sidebar(market_structure):
    """Render la sidebar et retourne l'action sélectionnée"""
    
    with st.sidebar:
        st.title("🔍 Sélection du Marché")
        
        # Initialiser les valeurs par défaut dans session_state
        if 'filter_method' not in st.session_state:
            st.session_state.filter_method = "Géographique"
        
        # Méthode de filtrage
        st.markdown('<p class="sidebar-title">Méthode de filtrage</p>', unsafe_allow_html=True)
        filter_method = st.radio(
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
            regions = list(market_structure['regions'].keys())
            
            # Sélection de la région
            selected_region = st.selectbox(
                "Région", 
                regions, 
                index=regions.index("Europe") if "Europe" in regions else 0,
                key="selected_region"
            )
            
            if selected_region:
                pays_list = list(market_structure['regions'][selected_region].keys())
                
                # Sélection du pays
                selected_pays = st.selectbox(
                    "Pays", 
                    pays_list, 
                    index=pays_list.index("France") if "France" in pays_list else 0,
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
            secteurs = list(market_structure['secteurs'].keys())
            selected_secteur = st.selectbox("Secteur", secteurs)
            
            if selected_secteur:
                industries = list(market_structure['secteurs'][selected_secteur].keys())
                selected_industrie = st.selectbox("Industrie", industries)
                
                # Récupérer les actions filtrées
                flattened_stocks = flatten_market_structure(
                    market_structure, 
                    filter_type='secteur', 
                    level1=selected_secteur, 
                    level2=selected_industrie
                )
        
        # Approche par marché
        elif filter_method == "Place de marché":
            marches = list(market_structure['marches'].keys())
            selected_marche = st.selectbox("Place de marché", marches)
            
            if selected_marche:
                # Récupérer les actions filtrées
                flattened_stocks = flatten_market_structure(
                    market_structure, 
                    filter_type='marche', 
                    level1=selected_marche
                )
        
        # Liste complète
        else:
            st.write("Affichage de toutes les actions disponibles")
            flattened_stocks = market_structure['all_stocks']
        
        # Affichage du nombre d'actions trouvées
        st.write(f"**{len(flattened_stocks)} actions** correspondent à vos critères")
        
        # Sélection de l'action
        if flattened_stocks:
            stock_names = list(flattened_stocks.keys())
            stock_names.sort()
            
            # Recherche avec autocomplétion
            selected_stock_name = st.selectbox(
                "Sélectionnez une action",
                stock_names,
                format_func=lambda x: f"{x} ({flattened_stocks[x]['ticker']})"
            )
            
            if selected_stock_name:
                selected_stock = flattened_stocks[selected_stock_name]
                st.session_state['selected_stock'] = selected_stock
                st.session_state['selected_stock_name'] = selected_stock_name
                st.session_state['ticker'] = selected_stock['ticker']
                
                # Afficher les informations de l'action sélectionnée
                st.markdown('<div class="selected-info">', unsafe_allow_html=True)
                st.markdown(f"**{selected_stock_name}** ({selected_stock['ticker']})")
                st.markdown(f"**Marché:** {selected_stock['marche']}")
                st.markdown(f"**Secteur:** {selected_stock['secteur']}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    return selected_stock
