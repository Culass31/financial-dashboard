"""
Fonctions communes réutilisables dans l'application
"""
import streamlit as st
import math


def format_number(num):
    """
    Formate les grands nombres de manière lisible
    
    :param num: Nombre à formater
    :return: Chaîne formatée
    """
    if num is None:
        return 'N/A'
    if isinstance(num, str):
        return num
    
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"


def format_percentage(value, decimals=2):
    """
    Formate une valeur en pourcentage
    
    :param value: Valeur à formater (0-1 ou 0-100)
    :param decimals: Nombre de décimales
    :return: Chaîne formatée avec %
    """
    if value is None:
        return 'N/A'
    
    # Si la valeur est entre 0 et 1, multiplier par 100
    if abs(value) < 1:
        value = value * 100
    
    return f"{value:.{decimals}f}%"


def format_currency(value, currency="€", decimals=2):
    """
    Formate une valeur monétaire
    
    :param value: Valeur à formater
    :param currency: Symbole de la devise
    :param decimals: Nombre de décimales
    :return: Chaîne formatée avec devise
    """
    if value is None:
        return 'N/A'
    
    return f"{value:,.{decimals}f} {currency}"


def color_value(value, positive_is_good=True):
    """
    Retourne une couleur en fonction d'une valeur
    
    :param value: Valeur à évaluer
    :param positive_is_good: Si True, les valeurs positives sont vertes
    :return: Couleur CSS
    """
    if value is None:
        return 'gray'
    
    if positive_is_good:
        return 'green' if value > 0 else 'red' if value < 0 else 'gray'
    else:
        return 'red' if value > 0 else 'green' if value < 0 else 'gray'


def display_loading_message(message="Chargement en cours...", show_time=False):
    """
    Affiche un message de chargement avec spinner
    
    :param message: Message à afficher
    :param show_time: Afficher le temps écoulé
    :return: Contexte manager du spinner
    """
    return st.spinner(message, show_time=show_time)


def create_tabs_layout(tab_names):
    """
    Crée une disposition avec onglets
    
    :param tab_names: Liste des noms d'onglets
    :return: Liste des conteneurs d'onglets
    """
    return st.tabs(tab_names)


def display_error_message(message):
    """
    Affiche un message d'erreur formaté
    
    :param message: Message d'erreur
    """
    st.error(f"❌ {message}")


def display_success_message(message):
    """
    Affiche un message de succès formaté
    
    :param message: Message de succès
    """
    st.success(f"✅ {message}")


def display_warning_message(message):
    """
    Affiche un message d'avertissement formaté
    
    :param message: Message d'avertissement
    """
    st.warning(f"⚠️ {message}")


def display_info_message(message):
    """
    Affiche un message d'information formaté
    
    :param message: Message d'information
    """
    st.info(f"ℹ️ {message}")


def create_download_button(label, data, filename, mime_type="text/csv"):
    """
    Crée un bouton de téléchargement
    
    :param label: Libellé du bouton
    :param data: Données à télécharger
    :param filename: Nom du fichier
    :param mime_type: Type MIME
    :return: Bouton de téléchargement
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )


def create_columns_layout(weights):
    """
    Crée une disposition en colonnes avec des poids spécifiques
    
    :param weights: Liste des poids relatifs des colonnes
    :return: Liste des colonnes
    """
    return st.columns(weights)


def create_expander(title, expanded=False):
    """
    Crée un conteneur extensible
    
    :param title: Titre de l'expander
    :param expanded: État initial (ouvert ou fermé)
    :return: Contexte manager de l'expander
    """
    return st.expander(title, expanded=expanded)


def validate_data(data, required_fields):
    """
    Valide que les données contiennent les champs requis
    
    :param data: Dictionnaire de données
    :param required_fields: Liste des champs requis
    :return: Tuple (is_valid, missing_fields)
    """
    if not data:
        return False, required_fields
    
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    return len(missing_fields) == 0, missing_fields


def safe_get(data, *keys, default=None):
    """
    Récupère une valeur de manière sécurisée dans une structure imbriquée
    
    :param data: Dictionnaire ou objet
    :param keys: Clés à traverser
    :param default: Valeur par défaut si non trouvé
    :return: Valeur trouvée ou default
    """
    current = data
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                current = getattr(current, key, default)
            
            if current is None:
                return default
        except:
            return default
    
    return current


def truncate_text(text, max_length=100, suffix="..."):
    """
    Tronque un texte à une longueur maximale
    
    :param text: Texte à tronquer
    :param max_length: Longueur maximale
    :param suffix: Suffixe à ajouter si tronqué
    :return: Texte tronqué
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_date(date, format_string="%d/%m/%Y"):
    """
    Formate une date
    
    :param date: Date à formater
    :param format_string: Format de sortie
    :return: Date formatée
    """
    if not date:
        return 'N/A'
    
    try:
        return date.strftime(format_string)
    except:
        return str(date)


def calculate_change_percentage(current, previous):
    """
    Calcule le pourcentage de variation entre deux valeurs
    
    :param current: Valeur actuelle
    :param previous: Valeur précédente
    :return: Pourcentage de variation
    """
    if current is None or previous is None or previous == 0:
        return None
    
    return ((current - previous) / abs(previous)) * 100


def group_data_by_period(data, period='monthly'):
    """
    Regroupe des données par période
    
    :param data: DataFrame avec index temporel
    :param period: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    :return: DataFrame regroupé
    """
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }
    
    freq = period_map.get(period, 'M')
    return data.resample(freq).agg({
        col: 'sum' if col in ['Volume'] else 'last'
        for col in data.columns
    })


def create_metric_card_html(title, value, change=None, prefix="", suffix=""):
    """
    Crée le HTML pour une carte métrique
    
    :param title: Titre de la métrique
    :param value: Valeur principale
    :param change: Variation (optionnel)
    :param prefix: Préfixe (optionnel)
    :param suffix: Suffixe (optionnel)
    :return: HTML string
    """
    change_html = ""
    if change is not None:
        change_color = "green" if change > 0 else "red" if change < 0 else "gray"
        change_arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        change_html = f"""
        <div style='font-size: 0.9rem; color: {change_color};'>
            {change_arrow} {abs(change):.2f}%
        </div>
        """
    
    return f"""
    <div style='background-color: white; border-radius: 10px; padding: 15px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
        <div style='font-size: 0.9rem; color: #666; margin-bottom: 5px;'>{title}</div>
        <div style='font-size: 1.5rem; font-weight: bold;'>{prefix}{value}{suffix}</div>
        {change_html}
    </div>
    """


def display_dataframe_with_formatting(df, format_dict=None):
    """
    Affiche un DataFrame avec formatage personnalisé
    
    :param df: DataFrame à afficher
    :param format_dict: Dictionnaire de format {colonne: format_string}
    """
    if format_dict:
        styled_df = df.style
        for col, fmt in format_dict.items():
            if col in df.columns:
                styled_df = styled_df.format({col: fmt})
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def create_progress_indicator(current, total, label=""):
    """
    Crée un indicateur de progression
    
    :param current: Valeur actuelle
    :param total: Valeur totale
    :param label: Étiquette optionnelle
    :return: Barre de progression
    """
    if total == 0:
        progress = 0
    else:
        progress = min(current / total, 1.0)
    
    if label:
        st.write(f"{label}: {current}/{total}")
    
    return st.progress(progress)


def create_select_box_with_default(label, options, default_value=None, key=None):
    """
    Crée une selectbox avec une valeur par défaut
    
    :param label: Libellé de la selectbox
    :param options: Liste des options
    :param default_value: Valeur par défaut
    :param key: Clé unique pour la selectbox
    :return: Valeur sélectionnée
    """
    if default_value and default_value in options:
        index = options.index(default_value)
    else:
        index = 0
    
    return st.selectbox(label, options, index=index, key=key)


def display_key_value_pairs(data, columns=2):
    """
    Affiche des paires clé-valeur dans une disposition en colonnes
    
    :param data: Dictionnaire de données {clé: valeur}
    :param columns: Nombre de colonnes
    """
    items = list(data.items())
    cols = st.columns(columns)
    
    for i, (key, value) in enumerate(items):
        with cols[i % columns]:
            st.markdown(f"**{key}:** {value}")


def create_confirmation_dialog(message):
    """
    Crée un dialogue de confirmation
    
    :param message: Message de confirmation
    :return: Booléen indiquant si confirmé
    """
    return st.button(message)


def cache_data(ttl=3600):
    """
    Décorateur pour mettre en cache des données
    
    :param ttl: Durée de vie du cache en secondes
    :return: Décorateur
    """
    return st.cache_data(ttl=ttl)


def session_state_init(key, default_value):
    """
    Initialise une valeur dans session_state si elle n'existe pas
    
    :param key: Clé dans session_state
    :param default_value: Valeur par défaut
    """
    if key not in st.session_state:
        st.session_state[key] = default_value


def clear_session_state(*keys):
    """
    Efface des clés spécifiques de session_state
    
    :param keys: Clés à effacer
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
