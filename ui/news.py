"""
Composant pour l'onglet des actualités récentes
"""
import streamlit as st
import pandas as pd

from services import NewsService


def render_news_tab(selected_stock):
    """Render l'onglet des actualités récentes"""
    
    if selected_stock is None:
        st.info("Veuillez sélectionner une action dans le menu latéral pour voir les actualités.")
        return
    
    ticker = selected_stock['ticker']
    stock_name = st.session_state.get('selected_stock_name', ticker)
    
    # Titre principal
    st.markdown(f"### 📰 Actualités récentes pour {stock_name} ({ticker})")
    
    # Options de filtrage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_category = st.selectbox(
            "Filtrer par catégorie",
            ["Toutes", "Changement Fondamental", "Dividendes", "Rachat d'Actions", 
             "Résultats Financiers", "Actualité Générale"]
        )
    
    with col2:
        filter_impact = st.selectbox(
            "Filtrer par impact",
            ["Tous", "Majeur", "Significatif", "Modéré", "Faible/Neutre"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            ["Date (plus récent)", "Importance", "Impact"]
        )
    
    # Récupérer les actualités
    with st.spinner("Chargement des actualités..."):
        news_df = get_stock_news(ticker)
    
    if news_df.empty:
        st.info(f"Aucune actualité récente trouvée pour {stock_name}.")
        return
    
    # Appliquer les filtres
    filtered_news = filter_news(news_df, filter_category, filter_impact)
    
    # Appliquer le tri
    filtered_news = sort_news(filtered_news, sort_by)
    
    # Statistiques
    render_news_statistics(filtered_news)
    
    # Afficher les actualités
    st.markdown("### 📋 Actualités")
    
    if filtered_news.empty:
        st.warning("Aucune actualité ne correspond aux critères sélectionnés.")
    else:
        render_news_items(filtered_news)
    
    # Exporter les actualités
    if not filtered_news.empty:
        render_export_options(filtered_news)


def filter_news(news_df, category_filter, impact_filter):
    """Applique les filtres aux actualités"""
    
    filtered_df = news_df.copy()
    
    # Filtrer par catégorie
    if category_filter != "Toutes":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    # Filtrer par impact
    if impact_filter != "Tous":
        filtered_df = filtered_df[filtered_df['impact_on_valuation'] == impact_filter]
    
    return filtered_df


def sort_news(news_df, sort_option):
    """Trie les actualités selon l'option sélectionnée"""
    
    if sort_option == "Date (plus récent)":
        return news_df.sort_values('published_at', ascending=False)
    elif sort_option == "Importance":
        return news_df.sort_values('importance_score', ascending=False)
    elif sort_option == "Impact":
        # Définir un ordre pour les impacts
        impact_order = {
            "Majeur": 4,
            "Significatif": 3,
            "Modéré": 2,
            "Faible/Neutre": 1
        }
        news_df['impact_order'] = news_df['impact_on_valuation'].map(impact_order)
        return news_df.sort_values('impact_order', ascending=False)
    
    return news_df


def render_news_statistics(news_df):
    """Affiche les statistiques des actualités"""
    
    st.markdown("### 📊 Statistiques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total actualités", len(news_df))
    
    with col2:
        high_importance = len(news_df[news_df['importance_score'] >= 4])
        st.metric("Haute importance", high_importance)
    
    with col3:
        major_impact = len(news_df[news_df['impact_on_valuation'] == 'Majeur'])
        st.metric("Impact majeur", major_impact)
    
    with col4:
        # Calculer la moyenne d'actualités par jour
        if not news_df.empty:
            days_span = (news_df['published_at'].max() - news_df['published_at'].min()).days + 1
            avg_per_day = len(news_df) / days_span if days_span > 0 else 0
            st.metric("Moy. par jour", f"{avg_per_day:.1f}")


def render_news_items(news_df):
    """Affiche les actualités avec un style visuel distinct selon leur importance"""
    
    for idx, news in news_df.iterrows():
        # Déterminer le style en fonction de l'importance
        importance_styles = {
            5: {"bg_color": "#ffe6e6", "border_color": "#ff0000", "badge_color": "#d32f2f"},
            4: {"bg_color": "#ffe6e6", "border_color": "#ffb3b3", "badge_color": "#f44336"},
            3: {"bg_color": "#fff2e6", "border_color": "#ffcc99", "badge_color": "#ff9800"},
            2: {"bg_color": "#f2f2f2", "border_color": "#e6e6e6", "badge_color": "#757575"},
            1: {"bg_color": "#f2f2f2", "border_color": "#e6e6e6", "badge_color": "#bdbdbd"}
        }
        
        style = importance_styles.get(news['importance_score'], importance_styles[1])
        
        # Créer le HTML pour l'actualité
        news_html = f"""
        <div style="background-color: {style['bg_color']}; padding: 15px; margin-bottom: 15px; border-left: 5px solid {style['border_color']}; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                <h4 style="margin: 0; padding-right: 20px;">{news['title']}</h4>
                <div style="display: flex; gap: 10px;">
                    <span style="background-color: {style['badge_color']}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                        {news['category']}
                    </span>
                    <span style="background-color: #616161; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                        Impact: {news['impact_on_valuation']}
                    </span>
                </div>
            </div>
            <p style="margin: 10px 0; color: #555;">
                <small>Publié le {news['date_formatted']} | Source: {news.get('provider', 'N/A')}</small>
            </p>
            <p style="margin: 10px 0;">{news['description']}</p>
            <div style="margin-top: 10px;">
                <a href="{news['link']}" target="_blank" style="color: #1E88E5; text-decoration: none;">
                    Lire l'article complet →
                </a>
            </div>
        </div>
        """
        
        st.markdown(news_html, unsafe_allow_html=True)
    
    # Pagination si nécessaire
    if len(news_df) > 10:
        if 'news_page' not in st.session_state:
            st.session_state.news_page = 0
        
        total_pages = (len(news_df) - 1) // 10 + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Précédent", disabled=(st.session_state.news_page == 0)):
                st.session_state.news_page -= 1
                st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.news_page + 1} sur {total_pages}")
        
        with col3:
            if st.button("Suivant →", disabled=(st.session_state.news_page >= total_pages - 1)):
                st.session_state.news_page += 1
                st.rerun()


def render_export_options(news_df):
    """Options d'export des actualités"""
    
    st.markdown("### 💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export CSV
        csv = news_df.to_csv(index=False)
        st.download_button(
            label="Télécharger en CSV",
            data=csv,
            file_name=f"actualites_{news_df.iloc[0]['ticker']}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export format résumé
        summary = generate_news_summary(news_df)
        st.download_button(
            label="Télécharger le résumé",
            data=summary,
            file_name=f"resume_actualites_{news_df.iloc[0]['ticker']}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )


def generate_news_summary(news_df):
    """Génère un résumé textuel des actualités"""
    
    summary = []
    summary.append(f"RÉSUMÉ DES ACTUALITÉS - {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
    summary.append("=" * 50)
    summary.append(f"Total actualités: {len(news_df)}")
    summary.append("")
    
    # Actualités par catégorie
    summary.append("PAR CATÉGORIE:")
    category_counts = news_df['category'].value_counts()
    for cat, count in category_counts.items():
        summary.append(f"- {cat}: {count}")
    summary.append("")
    
    # Actualités par impact
    summary.append("PAR IMPACT:")
    impact_counts = news_df['impact_on_valuation'].value_counts()
    for impact, count in impact_counts.items():
        summary.append(f"- {impact}: {count}")
    summary.append("")
    
    # Actualités importantes
    summary.append("ACTUALITÉS IMPORTANTES (score >= 4):")
    important_news = news_df[news_df['importance_score'] >= 4]
    for _, news in important_news.iterrows():
        summary.append(f"\n[{news['date_formatted']}] {news['title']}")
        summary.append(f"Catégorie: {news['category']} | Impact: {news['impact_on_valuation']}")
        summary.append(f"Résumé: {news['description'][:200]}...")
    
    return "\n".join(summary)
