"""
Composant pour l'onglet gestionnaire de portefeuille
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from services import PortfolioService
from utils.common import (
    create_columns_layout, 
    create_expander, 
    display_success_message,
    display_error_message,
    display_info_message,
    format_number,
    session_state_init
)


# Clé de cache pour le portefeuille
PORTFOLIO_CACHE_KEY = "user_portfolio"


def render_portfolio_tab(market_structure):
    """Render l'onglet gestionnaire de portefeuille"""
    
    # Initialiser le portefeuille dans la session
    session_state_init(PORTFOLIO_CACHE_KEY, pd.DataFrame(
        columns=['Ticker', 'Nom', 'Quantité', 'PRU']
    ))
    
    # Créer les sous-onglets
    portfolio_tabs = st.tabs(["Mon Portefeuille", "Analyse Détaillée", "Optimisation"])
    
    with portfolio_tabs[0]:
        render_portfolio_management(market_structure)
    
    with portfolio_tabs[1]:
        render_portfolio_analysis()
    
    with portfolio_tabs[2]:
        render_portfolio_optimization()


def render_portfolio_management(market_structure):
    """Render la gestion du portefeuille"""
    
    st.markdown("### Mon Portefeuille")
    
    col1, col2 = create_columns_layout([1, 1])
    
    with col1:
        render_add_position_form(market_structure)
    
    with col2:
        render_import_portfolio()
    
    render_current_positions()


def render_add_position_form(market_structure):
    """Render le formulaire d'ajout de position"""
    
    st.markdown("#### Ajouter une action")
    
    # Sélection d'action
    flattened_stocks = market_structure['all_stocks']
    stock_options = list(flattened_stocks.keys())
    
    selected_stock_name = st.selectbox(
        "Sélectionner une action", 
        [""] + stock_options
    )
    
    # Récupérer le ticker
    ticker = ""
    if selected_stock_name and selected_stock_name in flattened_stocks:
        ticker = flattened_stocks[selected_stock_name]['ticker']
    
    # Champs quantité et prix
    col_qty, col_price = create_columns_layout([1, 1])
    
    with col_qty:
        quantity = st.number_input("Quantité", min_value=1, value=1, step=1)
    
    with col_price:
        entry_price = st.number_input("Prix d'achat unitaire", min_value=0.01, value=100.0, step=0.01)
    
    # Bouton d'ajout
    if st.button("Ajouter au portefeuille"):
        if selected_stock_name and ticker:
            add_position_to_portfolio(ticker, selected_stock_name, quantity, entry_price)
        else:
            display_error_message("Veuillez sélectionner une action valide")


def add_position_to_portfolio(ticker, stock_name, quantity, entry_price):
    """Ajoute ou met à jour une position dans le portefeuille"""
    
    portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
    
    # Vérifier si l'action existe déjà
    existing = portfolio_df[portfolio_df['Ticker'] == ticker]
    
    if len(existing) > 0:
        # Mettre à jour la position existante
        idx = existing.index[0]
        old_qty = portfolio_df.loc[idx, 'Quantité']
        old_price = portfolio_df.loc[idx, 'PRU']
        
        # Calculer le nouveau PRU
        new_qty = old_qty + quantity
        new_pru = ((old_qty * old_price) + (quantity * entry_price)) / new_qty
        
        portfolio_df.loc[idx, 'Quantité'] = new_qty
        portfolio_df.loc[idx, 'PRU'] = new_pru
        
        display_success_message(f"Position mise à jour pour {stock_name} ({ticker})")
    else:
        # Ajouter une nouvelle position
        new_row = pd.DataFrame({
            'Ticker': [ticker],
            'Nom': [stock_name],
            'Quantité': [quantity],
            'PRU': [entry_price]
        })
        
        st.session_state[PORTFOLIO_CACHE_KEY] = pd.concat(
            [portfolio_df, new_row], 
            ignore_index=True
        )
        
        display_success_message(f"Action ajoutée au portefeuille: {stock_name} ({ticker})")


def render_import_portfolio():
    """Render l'import de portefeuille CSV"""
    
    st.markdown("#### Importer un portefeuille")
    
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Lire le fichier CSV
            import_df = read_portfolio_csv(uploaded_file)
            
            # Remplacer le portefeuille existant
            st.session_state[PORTFOLIO_CACHE_KEY] = import_df
            
            display_success_message(f"Portefeuille importé avec succès: {len(import_df)} positions")
        
        except Exception as e:
            display_error_message(f"Erreur lors de l'importation: {str(e)}")


def read_portfolio_csv(uploaded_file):
    """Lit un fichier CSV de portefeuille"""
    
    # Déterminer le séparateur
    first_line = uploaded_file.readline().decode('utf-8-sig')
    uploaded_file.seek(0)
    
    sep = ',' if ',' in first_line else ';'
    
    # Lire le fichier
    import_df = pd.read_csv(uploaded_file, sep=sep)
    
    # Mapper les colonnes
    col_mapping = {
        'ticker': 'Ticker', 'symbol': 'Ticker',
        'nom': 'Nom', 'name': 'Nom',
        'quantité': 'Quantité', 'quantite': 'Quantité', 'quantity': 'Quantité',
        'pru': 'PRU', 'prix': 'PRU', 'price': 'PRU'
    }
    
    # Normaliser les noms de colonnes
    import_df.columns = [col.lower() for col in import_df.columns]
    
    # Appliquer le mapping
    for col in import_df.columns:
        if col in col_mapping:
            import_df.rename(columns={col: col_mapping[col]}, inplace=True)
    
    # Vérifier les colonnes requises
    required_cols = ['Ticker', 'Nom', 'Quantité', 'PRU']
    missing_cols = [col for col in required_cols if col not in import_df.columns]
    
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing_cols)}")
    
    # Filtrer et convertir
    import_df = import_df[required_cols].copy()
    import_df['Quantité'] = pd.to_numeric(import_df['Quantité'], errors='coerce')
    import_df['PRU'] = pd.to_numeric(import_df['PRU'], errors='coerce')
    
    return import_df.dropna()


def render_current_positions():
    """Render les positions actuelles du portefeuille"""
    
    st.markdown("### Positions actuelles")
    
    portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
    
    if portfolio_df.empty:
        display_info_message("Votre portefeuille est vide. Ajoutez des actions pour commencer l'analyse.")
        return
    
    # Afficher le tableau éditable
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
    
    # Mettre à jour si modifié
    if not edited_df.equals(portfolio_df):
        st.session_state[PORTFOLIO_CACHE_KEY] = edited_df
    
    # Options supplémentaires
    render_position_actions(portfolio_df)


def render_position_actions(portfolio_df):
    """Render les actions sur les positions"""
    
    col1, col2 = create_columns_layout([2, 1])
    
    with col1:
        st.markdown("#### Actions sur les positions")
        
        # Supprimer une position
        if not portfolio_df.empty:
            positions = [f"{row['Nom']} ({row['Ticker']})" for _, row in portfolio_df.iterrows()]
            selected_position = st.selectbox("Sélectionner une position", [""] + positions)
            
            if selected_position and st.button("Supprimer"):
                ticker = selected_position.split('(')[1].split(')')[0]
                delete_position(ticker)
    
    with col2:
        st.markdown("#### Export")
        
        # Exporter le portefeuille
        if not portfolio_df.empty and st.button("Exporter (CSV)"):
            csv = portfolio_df.to_csv(index=False)
            filename = f"portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="Télécharger",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )


def delete_position(ticker):
    """Supprime une position du portefeuille"""
    
    portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
    st.session_state[PORTFOLIO_CACHE_KEY] = portfolio_df[portfolio_df['Ticker'] != ticker]
    display_success_message(f"Position supprimée: {ticker}")
    st.rerun()


def render_portfolio_analysis():
    """Render l'analyse détaillée du portefeuille"""
    
    st.markdown("### Analyse Détaillée du Portefeuille")
    
    portfolio_df = st.session_state[PORTFOLIO_CACHE_KEY]
    
    if portfolio_df.empty:
        display_info_message("Veuillez d'abord ajouter des actions à votre portefeuille.")
        return
    
    # Bouton d'analyse
    if st.button("Analyser le portefeuille"):
        with st.spinner("Analyse en cours..."):
            results_df, portfolio_stats = analyze_portfolio(portfolio_df)
            
            if not results_df.empty:
                st.session_state['portfolio_analysis'] = {
                    'results': results_df,
                    'stats': portfolio_stats
                }
                display_success_message("Analyse terminée!")
    
    # Afficher les résultats s'ils existent
    if 'portfolio_analysis' in st.session_state:
        render_analysis_results()


def render_analysis_results():
    """Render les résultats d'analyse"""
    
    analysis = st.session_state['portfolio_analysis']
    results_df = analysis['results']
    portfolio_stats = analysis['stats']
    
    # Vue d'ensemble
    st.markdown("#### Vue d'ensemble")
    
    col1, col2, col3 = create_columns_layout([1, 1, 1])
    
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
        if 'Potentiel (%)' in results_df.columns:
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
    render_portfolio_visualizations(results_df)
    
    # Tableau détaillé
    render_positions_table(results_df)
    
    # Export
    render_analysis_export(results_df)


def render_portfolio_visualizations(results_df):
    """Render les visualisations du portefeuille"""
    
    st.markdown("#### Visualisations")
    
    viz_col1, viz_col2 = create_columns_layout([1, 1])
    
    with viz_col1:
        # Allocation par recommandation
        fig1 = create_allocation_chart(results_df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
    
    with viz_col2:
        # Performance vs Potentiel
        fig2 = create_performance_vs_potential_chart(results_df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)


def create_allocation_chart(portfolio_df):
    """Crée un graphique d'allocation par recommandation"""
    
    if portfolio_df.empty or 'Recommandation' not in portfolio_df.columns:
        return None
    
    # Regrouper par recommandation
    grouped = portfolio_df.groupby('Recommandation')['Valeur position'].sum().reset_index()
    
    # Calculer les pourcentages
    total = grouped['Valeur position'].sum()
    grouped['Pourcentage'] = (grouped['Valeur position'] / total * 100).round(1)
    
    # Créer le graphique
    fig = go.Figure(data=[go.Pie(
        labels=grouped['Recommandation'],
        values=grouped['Pourcentage'],
        hole=.4,
        marker=dict(
            colors=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#9E9E9E']
        )
    )])
    
    fig.update_layout(
        title="Allocation par recommandation (%)",
        height=350
    )
    
    return fig


def create_performance_vs_potential_chart(portfolio_df):
    """Crée un graphique performance vs potentiel"""
    
    if portfolio_df.empty or 'Potentiel (%)' not in portfolio_df.columns:
        return None
    
    # Filtrer les données valides
    filtered_df = portfolio_df[pd.notna(portfolio_df['Potentiel (%)'])]
    
    if filtered_df.empty:
        return None
    
    # Créer le graphique
    fig = go.Figure()
    
    # Couleurs par recommandation
    color_map = {
        'ACHETER': '#4CAF50',
        'RENFORCER': '#8BC34A',
        'CONSERVER': '#FFC107',
        'ALLEGER': '#FF9800',
        'VENDRE': '#F44336',
        'DONNÉES INSUFFISANTES': '#9E9E9E'
    }
    
    fig.add_trace(go.Scatter(
        x=filtered_df['Performance (%)'],
        y=filtered_df['Potentiel (%)'],
        mode='markers',
        marker=dict(
            size=filtered_df['Valeur position'] / filtered_df['Valeur position'].max() * 40 + 10,
            color=[color_map.get(rec, '#9E9E9E') for rec in filtered_df['Recommandation']],
            opacity=0.7
        ),
        text=filtered_df['Ticker'],
        hovertemplate="<b>%{text}</b><br>Performance: %{x:.1f}%<br>Potentiel: %{y:.1f}%<extra></extra>"
    ))
    
    # Lignes de référence
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Performance vs Potentiel",
        xaxis_title="Ticker",
        yaxis_title="Allocation (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_optimization_export(optimized_df):
    """Render l'export de l'optimisation"""
    
    if st.button("Exporter l'allocation optimisée (CSV)"):
        csv = optimized_df.to_csv(index=False)
        filename = f"allocation_optimisee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="Télécharger l'allocation",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

def render_positions_table(results_df):
    """Render le tableau des positions analysées"""
    
    st.markdown("#### Analyse détaillée des positions")
    
    # Formater les colonnes numériques
    display_df = results_df.copy()
    
    format_cols = {
        'Prix actuel': '€',
        'Valeur position': '€',
        'Performance (%)': '%',
        'Valeur intrinsèque': '€',
        'VI avec MOS': '€',
        'Potentiel (%)': '%'
    }
    
    for col, suffix in format_cols.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.2f} {suffix}" if pd.notna(x) else "N/A"
            )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_analysis_export(results_df):
    """Render l'option d'export de l'analyse"""
    
    if st.button("Exporter l'analyse (CSV)"):
        csv = results_df.to_csv(index=False)
        filename = f"analyse_portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="Télécharger l'analyse",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )


def render_portfolio_optimization():
    """Render l'optimisation du portefeuille"""
    
    st.markdown("### Optimisation du Portefeuille")
    
    # Vérifier si l'analyse existe
    if 'portfolio_analysis' not in st.session_state:
        display_info_message("Veuillez d'abord analyser votre portefeuille dans l'onglet 'Analyse Détaillée'.")
        return
    
    # Profil de risque
    risk_profile = st.radio(
        "Sélectionnez votre profil de risque",
        ["conservative", "moderate", "aggressive"],
        captions=["Prudent", "Modéré", "Dynamique"],
        horizontal=True
    )
    
    # Bouton d'optimisation
    if st.button("Optimiser l'allocation"):
        with st.spinner("Optimisation en cours..."):
            results_df = st.session_state['portfolio_analysis']['results']
            optimized_df = optimize_portfolio_allocation(results_df, risk_profile)
            
            st.session_state['portfolio_optimization'] = optimized_df
            display_success_message("Optimisation terminée!")
    
    # Afficher les résultats d'optimisation
    if 'portfolio_optimization' in st.session_state:
        render_optimization_results()


def render_optimization_results():
    """Render les résultats d'optimisation"""
    
    optimized_df = st.session_state['portfolio_optimization']
    
    st.markdown("#### Allocation optimisée")
    
    # Formater le tableau
    display_df = optimized_df.copy()
    
    display_df['Poids actuel (%)'] = display_df['Poids actuel (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Poids cible (%)'] = display_df['Poids cible (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Ajustement (%)'] = display_df['Ajustement (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Montant à ajuster'] = display_df['Montant à ajuster'].apply(lambda x: f"{x:,.2f} €")
    
    # Colorier les actions
    def style_action(val):
        if val == "ACHETER":
            return 'background-color: #d4edda; color: #155724'
        elif val == "VENDRE":
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_df = display_df.style.map(style_action, subset=['Action'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Visualisation
    render_allocation_visualization(optimized_df)
    
    # Export
    render_optimization_export(optimized_df)


def render_allocation_visualization(optimized_df):
    """Render la visualisation de l'allocation optimisée"""
    
    st.markdown("#### Visualisation de l'allocation")
    
    fig = go.Figure()
    
    # Convertir les pourcentages en float si nécessaire
    if isinstance(optimized_df['Poids actuel (%)'].iloc[0], str):
        optimized_df['Poids actuel (%)'] = optimized_df['Poids actuel (%)'].apply(
            lambda x: float(x.rstrip('%')) if isinstance(x, str) else x
        )
        optimized_df['Poids cible (%)'] = optimized_df['Poids cible (%)'].apply(
            lambda x: float(x.rstrip('%')) if isinstance(x, str) else x
        )
    
    # Barres pour allocation actuelle
    fig.add_trace(go.Bar(
        x=optimized_df['Ticker'],
        y=optimized_df['Poids actuel (%)'],
        name='Allocation actuelle',
        marker_color='rgb(158,202,225)'
    ))
    
    # Barres pour allocation cible
    fig.add_trace(go.Bar(
        x=optimized_df['Ticker'],
        y=optimized_df['Poids cible (%)'],
        name='Allocation cible',
        marker_color='rgb(94,158,217)'
    ))
    
    fig.update_layout(
        title="Comparaison des allocations",
        xaxis_title="Ticker",
        yaxis_title="Allocation (%)",
        barmode='group',
        height=500,
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Fonctions utilitaires
def analyze_portfolio(portfolio_df):
    """Analyse un portefeuille d'actions"""
    from services import DataService, PortfolioService
    
    data_service = DataService()
    portfolio_service = PortfolioService(data_service)
    
    return portfolio_service.analyze_portfolio(portfolio_df)


def optimize_portfolio_allocation(results_df, risk_profile):
    """Optimise l'allocation du portefeuille"""
    from services import DataService, PortfolioService
    
    data_service = DataService()
    portfolio_service = PortfolioService(data_service)
    
    return portfolio_service.optimize_portfolio_allocation(results_df, risk_profile)