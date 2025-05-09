"""
Composant pour l'onglet screener d'actions Buffett-Graham
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

from services import AnalysisService, DataService
from utils.common import create_columns_layout, create_expander


def render_screener_tab(market_structure):
    """Render l'onglet screener d'actions"""
    
    st.markdown("### 🔎 Screener d'actions Buffett-Graham")
    
    with create_expander("📊 À propos de ce screener", expanded=True):
        st.markdown("""
        Ce screener d'actions combine les principes d'investissement de Warren Buffett et Benjamin Graham pour identifier des entreprises de qualité à des valorisations raisonnables. Il analyse :
        
        - **Qualité financière** : ROE, marges, stabilité des bénéfices
        - **Santé du bilan** : Niveaux d'endettement, liquidité, capacité à générer du cash
        - **Régularité des performances** : Marges brutes stables, croissance constante
        - **Valorisation** : P/E ratio, PEG ratio, price-to-book
        
        Ajustez les critères ci-dessous pour trouver des actions correspondant à votre stratégie d'investissement.
        """)
    
    # Section des filtres de marché
    stocks_dict = render_market_filters(market_structure)
    
    if not stocks_dict:
        st.warning("Aucune action ne correspond aux filtres sélectionnés. Veuillez modifier vos critères.")
        return
    
    st.info(f"{len(stocks_dict)} actions trouvées dans l'univers sélectionné.")
    
    # Section des critères de screening
    screening_params = render_screening_criteria()
    
    # Options d'affichage
    display_options = render_display_options()
    
    # Bouton pour lancer l'analyse
    if st.button("Lancer l'analyse"):
        run_screening_analysis(stocks_dict, screening_params, display_options)


def render_market_filters(market_structure):
    """Render les filtres de marché et retourne le dictionnaire d'actions"""
    
    st.subheader("Filtrage de l'univers d'investissement")
    
    filter_col1, filter_col2 = create_columns_layout([1, 1])
    
    with filter_col1:
        filter_type = st.selectbox(
            "Type de filtre",
            ["Toutes les actions", "Par région", "Par secteur", "Par marché"],
            index=0
        )
    
    level1, level2 = None, None
    
    with filter_col2:
        if filter_type == "Par région":
            regions = list(market_structure['regions'].keys())
            selected_region = st.selectbox("Sélectionner une région", regions)
            level1 = selected_region
            
            if selected_region:
                pays_options = ["Tous les pays"] + list(market_structure['regions'][selected_region].keys())
                selected_pays = st.selectbox("Sélectionner un pays", pays_options)
                if selected_pays != "Tous les pays":
                    level2 = selected_pays
        
        elif filter_type == "Par secteur":
            secteurs = list(market_structure['secteurs'].keys())
            selected_secteur = st.selectbox("Sélectionner un secteur", secteurs)
            level1 = selected_secteur
            
            if selected_secteur:
                industrie_options = ["Toutes les industries"] + list(market_structure['secteurs'][selected_secteur].keys())
                selected_industrie = st.selectbox("Sélectionner une industrie", industrie_options)
                if selected_industrie != "Toutes les industries":
                    level2 = selected_industrie
        
        elif filter_type == "Par marché":
            marches = list(market_structure['marches'].keys())
            selected_marche = st.selectbox("Sélectionner un marché", marches)
            level1 = selected_marche
    
    # Obtenir le dictionnaire d'actions filtré
    filter_type_mapping = {
        "Par région": "region",
        "Par secteur": "secteur",
        "Par marché": "marche",
        "Toutes les actions": None
    }
    
    return flatten_market_structure(
        market_structure,
        filter_type=filter_type_mapping[filter_type],
        level1=level1,
        level2=level2
    )


def render_screening_criteria():
    """Render les critères de screening et retourne les paramètres"""
    
    st.subheader("Critères de screening")
    
    col1, col2, col3 = create_columns_layout([1, 1, 1])
    
    params = {}
    
    with col1:
        st.markdown("**Critères financiers fondamentaux**")
        params['min_revenue_growth'] = st.slider("Croissance min. des revenus (%)", 0, 30, 5)
        params['min_net_profit_margin'] = st.slider("Marge nette min. (%)", 0, 40, 10)
        params['min_roe'] = st.slider("ROE min. (%)", 5, 50, 15)
        params['min_operating_margin'] = st.slider("Marge opérationnelle min. (%)", 0, 40, 15)
        params['min_earnings_stability_years'] = st.slider("Années de bénéfices positifs consécutifs", 0, 10, 5)
    
    with col2:
        st.markdown("**Critères d'endettement et valorisation**")
        params['max_debt_to_equity'] = st.slider("Ratio dette/fonds propres max.", 0.0, 2.0, 0.5, 0.1)
        params['max_debt_ebitda_ratio'] = st.slider("Ratio dette/EBITDA max.", 0.0, 10.0, 5.0, 0.5)
        params['max_pe_ratio'] = st.slider("P/E ratio max.", 5, 50, 25)
        params['max_pe_to_growth'] = st.slider("PEG ratio max.", 0.0, 3.0, 1.0, 0.1)
        params['max_price_to_book'] = st.slider("Ratio cours/valeur comptable max.", 0.0, 5.0, 1.5, 0.1)
    
    with col3:
        st.markdown("**Critères de consistance**")
        params['min_gross_margin_consistency'] = st.slider("Marge brute min. consistante (%)", 20, 60, 40)
        params['min_net_income_growth_consistency'] = st.slider("Croissance min. du résultat net (%)", 0, 20, 8)
        params['consistent_low_debt'] = st.checkbox("Exiger une dette constamment faible", True)
        params['min_operating_cashflow_growth'] = st.slider("Croissance min. du cash-flow op. (%)", 0, 20, 5)
        params['min_dividend_years'] = st.slider("Années consécutives de dividendes", 0, 20, 0)
    
    # Filtres avancés
    with create_expander("🔍 Filtres avancés"):
        adv_col1, adv_col2 = create_columns_layout([1, 1])
        
        with adv_col1:
            params['min_free_cashflow'] = st.number_input("Free Cash Flow minimal", 0, 1000000000, 0, 1000000)
            params['min_current_ratio'] = st.slider("Ratio de liquidité courante min.", 0.5, 5.0, 1.5, 0.1)
            params['max_rd_sales_ratio'] = st.slider("Ratio R&D/Ventes max.", 0.0, 0.20, 0.05, 0.01)
        
        with adv_col2:
            params['max_capex_net_income_ratio'] = st.slider("Ratio CAPEX/Résultat Net max.", 0.0, 2.0, 0.5, 0.1)
            params['max_interest_ebit_ratio'] = st.slider("Ratio Intérêts/EBIT max.", 0.0, 0.5, 0.20, 0.05)
            params['min_interest_coverage_ratio'] = st.slider("Ratio de couverture d'intérêts min.", 1, 20, 5)
    
    return params


def render_display_options():
    """Render les options d'affichage"""
    
    with create_expander("⚙️ Options d'affichage"):
        display_cols = st.multiselect(
            "Colonnes à afficher dans les résultats",
            ['Nom', 'Ticker', 'ISIN', 'Région', 'Pays', 'Score Buffett-Graham', 
             'Croissance CA (%)', 'Marge nette (%)', 'ROE (%)', 'Dette/Fonds propres', 
             'P/E', 'PEG', 'P/B', 'Free Cash Flow', 'Années bénéfices positifs',
             'Marge Brute Consistante', 'Croissance RN Consistante', 'Croissance FCF',
             'Forces', 'Faiblesses'],
            default=['Nom', 'Ticker', 'Région', 'Score Buffett-Graham', 'ROE (%)', 
                    'Dette/Fonds propres', 'P/E', 'PEG', 'Années bénéfices positifs', 'Forces']
        )
        
        sort_by = st.selectbox(
            "Trier les résultats par",
            ['Score Buffett-Graham', 'ROE (%)', 'Croissance CA (%)', 'Marge nette (%)', 
             'Dette/Fonds propres', 'P/E', 'PEG', 'P/B', 'Années bénéfices positifs'],
            index=0
        )
        
        sort_ascending = st.checkbox("Ordre croissant", False)
    
    return {
        'display_cols': display_cols,
        'sort_by': sort_by,
        'sort_ascending': sort_ascending
    }


def run_screening_analysis(stocks_dict, params, display_options):
    """Lance l'analyse de screening avec les paramètres donnés"""
    
    with st.spinner("Analyse en cours..."):
        # Créer une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyser par lots
        stock_list = list(stocks_dict.keys())
        total_stocks = len(stock_list)
        batch_size = max(1, total_stocks // 100)
        
        results = []
        detailed_results = {}
        
        for i in range(0, total_stocks, batch_size):
            # Mise à jour de la progression
            progress = int((i / total_stocks) * 100)
            progress_bar.progress(progress)
            current_stock = stock_list[i] if i < total_stocks else ""
            status_text.text(f"Analyse en cours: {i+1}/{total_stocks} actions - {current_stock}")
            
            # Analyser le lot
            batch_stocks = {k: stocks_dict[k] for k in stock_list[i:min(i+batch_size, total_stocks)]}
            batch_results_df, batch_detailed_results = buffett_graham_stock_screener(batch_stocks, **params)
            
            if not batch_results_df.empty:
                results.append(batch_results_df)
            detailed_results.update(batch_detailed_results)
        
        # Nettoyer la progression
        status_text.empty()
        progress_bar.empty()
        
        # Combiner les résultats
        if results:
            results_df = pd.concat(results, ignore_index=True)
            display_screening_results(results_df, detailed_results, display_options)
        else:
            st.warning("Aucune action ne correspond à vos critères. Essayez d'assouplir certains paramètres.")


def display_screening_results(results_df, detailed_results, display_options):
    """Affiche les résultats du screening"""
    
    # Trier les résultats
    sort_by = display_options['sort_by']
    sort_ascending = display_options['sort_ascending']
    
    if 'Score Buffett-Graham' in results_df.columns and sort_by == 'Score Buffett-Graham':
        results_df['Score Numérique'] = results_df['Score Buffett-Graham'].apply(
            lambda x: float(x.split('/')[0]) if isinstance(x, str) else 0
        )
        results_df = results_df.sort_values('Score Numérique', ascending=sort_ascending)
        results_df = results_df.drop('Score Numérique', axis=1)
    else:
        results_df = results_df.sort_values(sort_by, ascending=sort_ascending)
    
    # Afficher les résultats
    st.subheader(f"Résultats : {len(results_df)} actions correspondent à vos critères")
    st.dataframe(results_df[display_options['display_cols']], height=600)
    
    # Visualisations
    if len(results_df) > 0:
        render_screening_visualizations(results_df)
    
    # Analyse détaillée
    render_detailed_analysis(results_df, detailed_results)
    
    # Options d'export
    render_export_options(results_df, detailed_results)


def render_screening_visualizations(results_df):
    """Render les visualisations des résultats"""
    
    with create_expander("📈 Visualisations", expanded=True):
        viz_col1, viz_col2 = create_columns_layout([1, 1])
        
        with viz_col1:
            # Distribution des scores
            if 'Score Buffett-Graham' in results_df.columns:
                scores = results_df['Score Buffett-Graham'].apply(
                    lambda x: float(x.split('/')[0]) if isinstance(x, str) else 0
                )
                
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=scores,
                    nbinsx=10,
                    name='Distribution des scores',
                    marker_color='blue'
                ))
                
                fig1.update_layout(
                    title='Distribution des scores Buffett-Graham',
                    xaxis_title='Score',
                    yaxis_title="Nombre d'actions",
                    height=400
                )
                
                st.plotly_chart(fig1, use_container_width=True)
        
        with viz_col2:
            # Relation ROE vs P/E
            if 'ROE (%)' in results_df.columns and 'P/E' in results_df.columns:
                # Convertir les colonnes en float
                results_df['ROE_num'] = results_df['ROE (%)'].apply(
                    lambda x: float(x) if isinstance(x, (int, float)) else 0
                )
                results_df['PE_num'] = results_df['P/E'].apply(
                    lambda x: float(x) if isinstance(x, (int, float)) else 0
                )
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=results_df['ROE_num'],
                    y=results_df['PE_num'],
                    mode='markers',
                    text=results_df['Nom'],
                    marker=dict(
                        size=10,
                        color=scores if 'scores' in locals() else results_df.index,
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
                
                fig2.update_layout(
                    title='ROE vs P/E',
                    xaxis_title='ROE (%)',
                    yaxis_title='P/E Ratio',
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)


def render_detailed_analysis(results_df, detailed_results):
    """Render l'analyse détaillée pour chaque action"""
    
    with create_expander("🔍 Analyse détaillée par action"):
        if results_df.empty:
            st.info("Aucun résultat à analyser.")
            return
        
        selected_ticker = st.selectbox(
            "Sélectionner une action pour voir l'analyse détaillée",
            options=results_df['Ticker'].tolist(),
            index=0
        )
        
        if selected_ticker in detailed_results:
            details = detailed_results[selected_ticker]
            
            st.subheader(f"Analyse détaillée pour {selected_ticker}")
            
            # Métriques clés
            metric_col1, metric_col2, metric_col3 = create_columns_layout([1, 1, 1])
            
            with metric_col1:
                st.metric("Score Buffett-Graham", f"{details['score']}/100")
                st.metric("ROE", f"{details['financial_metrics']['roe']:.2f}%")
                st.metric("Marge Nette", f"{details['financial_metrics']['net_profit_margin']:.2f}%")
            
            with metric_col2:
                st.metric("P/E Ratio", f"{details['intrinsic_value_metrics']['pe_ratio']:.2f}")
                st.metric("PEG Ratio", f"{details['intrinsic_value_metrics']['pe_to_growth']:.2f}")
                st.metric("Dette/Fonds Propres", f"{details['financial_metrics']['debt_to_equity']:.2f}")
            
            with metric_col3:
                from ui_components.common import format_number
                st.metric("FCF", format_number(details['financial_metrics']['free_cashflow']))
                st.metric("Marge Opérationnelle", f"{details['financial_metrics']['operating_margin']:.2f}%")
                st.metric("Années Bénéfices +", f"{details['consistency_metrics']['earnings_stability_years']}")
            
            # Forces et faiblesses
            st.subheader("Forces et Faiblesses")
            force_col, faiblesse_col = create_columns_layout([1, 1])
            
            with force_col:
                st.markdown("### Forces")
                st.write(details['strengths'] or "Aucune force identifiée")
            
            with faiblesse_col:
                st.markdown("### Faiblesses")
                st.write(details['weaknesses'] or "Aucune faiblesse identifiée")
            
            # Métriques détaillées
            with create_expander("Toutes les métriques"):
                # Métriques financières
                st.markdown("### Métriques Financières")
                financial_df = pd.DataFrame({
                    'Métrique': list(details['financial_metrics'].keys()),
                    'Valeur': [format_metric_value(v) for v in details['financial_metrics'].values()]
                })
                st.dataframe(financial_df, use_container_width=True)
                
                # Métriques de valorisation
                st.markdown("### Métriques de Valorisation")
                value_df = pd.DataFrame({
                    'Métrique': list(details['intrinsic_value_metrics'].keys()),
                    'Valeur': [format_metric_value(v) for v in details['intrinsic_value_metrics'].values()]
                })
                st.dataframe(value_df, use_container_width=True)
                
                # Métriques de consistance
                st.markdown("### Métriques de Consistance")
                consistency_df = pd.DataFrame({
                    'Métrique': list(details['consistency_metrics'].keys()),
                    'Valeur': list(details['consistency_metrics'].values())
                })
                st.dataframe(consistency_df, use_container_width=True)


def render_export_options(results_df, detailed_results):
    """Render les options d'export des résultats"""
    
    with create_expander("💾 Exporter les résultats"):
        # Export CSV simple
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Télécharger les résultats (CSV)",
            data=csv,
            file_name="buffett_graham_screener_results.csv",
            mime="text/csv"
        )
        
        # Export Excel complet
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            # Onglet résultats
            results_df.to_excel(writer, sheet_name="Résultats", index=False)
            
            # Onglet analyse détaillée
            detailed_df = create_detailed_dataframe(detailed_results)
            detailed_df.to_excel(writer, sheet_name="Analyse Détaillée", index=False)
            
            # Onglet paramètres (nécessiterait de passer les params)
            params_df = pd.DataFrame({
                'Paramètre': ['Information'],
                'Valeur': ['Voir les paramètres utilisés dans l\'interface']
            })
            params_df.to_excel(writer, sheet_name="Paramètres", index=False)
        
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="Télécharger rapport complet (Excel)",
            data=excel_data,
            file_name="buffett_graham_screener_rapport.xlsx",
            mime="application/vnd.ms-excel"
        )


def create_detailed_dataframe(detailed_results):
    """Crée un DataFrame détaillé à partir des résultats d'analyse"""
    
    detailed_data = []
    
    for ticker, details in detailed_results.items():
        row = {
            'Ticker': ticker,
            'Score': details['score']
        }
        
        # Ajouter les métriques financières
        for key, value in details['financial_metrics'].items():
            row[f"Fin_{key}"] = value
        
        # Ajouter les métriques de valorisation
        for key, value in details['intrinsic_value_metrics'].items():
            row[f"Val_{key}"] = value
        
        # Ajouter les métriques de consistance
        for key, value in details['consistency_metrics'].items():
            row[f"Cons_{key}"] = value
        
        detailed_data.append(row)
    
    return pd.DataFrame(detailed_data)


def format_metric_value(value):
    """Formate une valeur métrique pour l'affichage"""
    
    if isinstance(value, float):
        if value > 1e9:
            return f"{value/1e9:.2f}B"
        elif value > 1e6:
            return f"{value/1e6:.2f}M"
        elif value > 1e3:
            return f"{value/1e3:.2f}K"
        else:
            return f"{value:.2f}"
    return str(value)
