"""
Module de styles CSS personnalisés pour l'application
"""
import streamlit as st

def apply_custom_styles():
    """Applique les styles CSS personnalisés à l'application"""
    st.markdown("""
    <style>
    /* Styles généraux */
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    
    .medium-font {
        font-size: 18px !important;
        font-weight: bold;
    }
    
    .custom-metric {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Couleurs pour les indicateurs */
    .positive {
        color: green;
        font-weight: bold;
    }
    
    .negative {
        color: red;
        font-weight: bold;
    }
    
    .neutral {
        color: gray;
        font-weight: bold;
    }
    
    .warning {
        color: orange;
        font-weight: bold;
    }
    
    /* Sidebar */
    .sidebar-title {
        margin-top: 15px;
        margin-bottom: 5px;
        color: #333;
        font-weight: bold;
    }
    
    .selected-info {
        background-color: #f0f9ff;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    
    /* Containers */
    .block-container {
        padding-top: 0;
    }
    
    /* News items */
    .news-item {
        background-color: #f9f9f9;
        border-left: 3px solid #4e8cff;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    
    .news-date {
        color: #666;
        font-size: 0.8rem;
    }
    
    .news-title {
        font-weight: bold;
        margin: 5px 0;
    }
    
    .news-summary {
        font-size: 0.9rem;
        color: #333;
    }
    
    /* Recommandations */
    .recommendation-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    
    .recommendation-buy {
        background-color: #d4edda;
        color: #155724;
    }
    
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Métriques de qualité */
    .quality-badge {
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        display: inline-block;
    }
    
    .quality-excellent {
        background-color: #4CAF50;
        color: white;
    }
    
    .quality-good {
        background-color: #8BC34A;
        color: white;
    }
    
    .quality-average {
        background-color: #FFC107;
        color: white;
    }
    
    .quality-poor {
        background-color: #F44336;
        color: white;
    }
    
    /* Graphiques */
    .chart-container {
        margin: 10px 0;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
    }
    
    .dataframe th {
        background-color: #f0f2f6;
        font-weight: bold;
    }
    
    /* Alertes personnalisées */
    .custom-alert {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-info {
        background-color: #cfe2ff;
        color: #084298;
        border: 1px solid #b6d4fe;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        color: #664d03;
        border: 1px solid #ffecb5;
    }
    
    .alert-success {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
    }
    
    /* Boutons personnalisés */
    .custom-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .custom-button:hover {
        background-color: #45a049;
    }
    
    /* Cards */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-card-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    
    .metric-card-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .metric-card-change {
        font-size: 14px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f5ff;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff8e6;
        border-left: 4px solid #FF9800;
        padding: 10px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #F44336;
        padding: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Liens */
    a {
        color: #1E88E5;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .big-font {
            font-size: 20px !important;
        }
        
        .medium-font {
            font-size: 16px !important;
        }
        
        .metric-card {
            margin: 5px 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def get_recommendation_style(recommendation):
    """Retourne le style CSS pour une recommandation donnée"""
    styles = {
        "ACHAT FORT": "color: darkgreen; font-weight: bold;",
        "ACHETER": "color: green; font-weight: bold;",
        "RENFORCER": "color: lightgreen; font-weight: bold;",
        "CONSERVER": "color: gold; font-weight: bold;",
        "ALLEGER": "color: orange; font-weight: bold;",
        "VENDRE": "color: red; font-weight: bold;",
        "VENTE FORTE": "color: darkred; font-weight: bold;"
    }
    return styles.get(recommendation, "color: gray; font-weight: bold;")

def format_metric_card(title, value, change=None, color=None):
    """Formate une carte métrique"""
    change_html = ""
    if change is not None:
        change_color = "green" if change > 0 else "red"
        change_arrow = "↑" if change > 0 else "↓"
        change_html = f"""
        <div class="metric-card-change" style="color: {change_color};">
            {change_arrow} {abs(change):.2f}%
        </div>
        """
    
    value_color = color if color else "inherit"
    
    return f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value" style="color: {value_color};">{value}</div>
        {change_html}
    </div>
    """

def create_info_box(content, box_type="info"):
    """Crée une boîte d'information stylisée"""
    box_class = f"{box_type}-box"
    return f"""
    <div class="{box_class}">
        {content}
    </div>
    """

def quality_badge(quality):
    """Crée un badge de qualité coloré"""
    quality_map = {
        'Excellent': 'excellent',
        'Bon': 'good',
        'Moyen': 'average',
        'Faible': 'poor',
        'Non évalué': 'average'
    }
    css_class = quality_map.get(quality, 'average')
    return f'<span class="quality-badge quality-{css_class}">{quality}</span>'
