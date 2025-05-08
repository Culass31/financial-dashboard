import streamlit as st
import plotly.graph_objects as go

class AppMetrics:
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

    def display_price_with_trend(label, current_price, previous_price, format_func=None, background=True, size="medium"):
        """
        Affiche un prix avec son indicateur de tendance (flèche et pourcentage)
        
        :param label: Libellé à afficher (ex: "Prix actuel")
        :param current_price: Prix actuel
        :param previous_price: Prix de référence pour calculer la tendance
        :param format_func: Fonction optionnelle pour formater le prix
        :param background: Ajouter un fond de couleur pour plus de visibilité
        :param size: Taille de l'affichage ("small", "medium", "large")
        """
        if current_price is not None and previous_price is not None:
            # Calculer le pourcentage de variation
            change_pct = ((current_price - previous_price) / previous_price) * 100
            
            # Déterminer la flèche et la couleur en fonction de la tendance
            if change_pct > 0:
                arrow = "↑"
                color = "green"
                bg_color = "rgba(0, 128, 0, 0.1)"
            elif change_pct < 0:
                arrow = "↓"
                color = "red"
                bg_color = "rgba(255, 0, 0, 0.1)"
            else:
                arrow = "→"
                color = "gray"
                bg_color = "rgba(128, 128, 128, 0.1)"
                
            # Définir la taille de la police
            font_sizes = {
                "small": {"label": "0.9rem", "value": "1.1rem", "change": "0.9rem", "previous": "0.8rem"},
                "medium": {"label": "1rem", "value": "1.4rem", "change": "1rem", "previous": "0.9rem"},
                "large": {"label": "1.2rem", "value": "1.8rem", "change": "1.2rem", "previous": "1rem"}
            }
            
            sizes = font_sizes.get(size, font_sizes["medium"])
            
            # Formater la valeur
            formatted_price = format_func(current_price) if format_func else f"{current_price:.2f}"
            formatted_previous = format_func(previous_price) if format_func else f"{previous_price:.2f}"
            
            # Créer le style pour l'affichage
            if background:
                style = f"""
                <div style='padding: 10px; border-radius: 5px; background-color: {bg_color}; margin-bottom: 10px;'>
                    <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                    <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                    <div style='display: flex; align-items: center;'>
                        <div style='font-size: {sizes["change"]}; color: {color};'>
                            {arrow} {abs(change_pct):.2f}%
                        </div>
                        <div style='font-size: {sizes["previous"]}; margin-left: 10px; color: #666;'>
                            Préc: {formatted_previous}
                        </div>
                    </div>
                </div>
                """
            else:
                style = f"""
                <div style='margin-bottom: 10px;'>
                    <div style='font-size: {sizes["label"]}; font-weight: bold;'>{label}</div>
                    <div style='font-size: {sizes["value"]}; font-weight: bold;'>{formatted_price}</div>
                    <div style='display: flex; align-items: center;'>
                        <div style='font-size: {sizes["change"]}; color: {color};'>
                            {arrow} {abs(change_pct):.2f}%
                        </div>
                        <div style='font-size: {sizes["previous"]}; margin-left: 10px; color: #666;'>
                            Préc: {formatted_previous}
                        </div>
                    </div>
                </div>
                """
            
            # Afficher avec le style approprié
            st.markdown(style, unsafe_allow_html=True)
        else:
            st.markdown(f"**{label}:** N/A")


    def display_metric_card(title, value, change=None, suffix="", prefix="", color=None):
        """
        Affiche une carte métrique stylisée
        
        :param title: Titre de la métrique
        :param value: Valeur à afficher
        :param change: Changement par rapport à une valeur précédente
        :param suffix: Suffixe à ajouter (ex: "%", "€")
        :param prefix: Préfixe à ajouter (ex: "$", "€")
        :param color: Couleur de la valeur
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
        
        value_color = color if color else "inherit"
        
        card_html = f"""
        <div style='background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
            <div style='font-size: 0.9rem; color: #666; margin-bottom: 5px;'>{title}</div>
            <div style='font-size: 1.5rem; font-weight: bold; color: {value_color};'>{prefix}{value}{suffix}</div>
            {change_html}
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)


    def display_kpi_metrics(kpis):
        """
        Affiche un ensemble de KPIs dans une disposition en colonnes
        
        :param kpis: Liste de dictionnaires avec les clés: title, value, change, suffix, prefix, color
        """
        cols = st.columns(len(kpis))
        
        for i, kpi in enumerate(kpis):
            with cols[i]:
                display_metric_card(
                    title=kpi.get('title', ''),
                    value=kpi.get('value', 0),
                    change=kpi.get('change'),
                    suffix=kpi.get('suffix', ''),
                    prefix=kpi.get('prefix', ''),
                    color=kpi.get('color')
                )


    def display_comparison_metric(label, actual, target, format_func=None):
        """
        Affiche une métrique comparant une valeur actuelle à une cible
        
        :param label: Libellé de la métrique
        :param actual: Valeur actuelle
        :param target: Valeur cible
        :param format_func: Fonction de formatage
        """
        if actual is not None and target is not None:
            # Calculer l'écart
            difference = actual - target
            difference_pct = (difference / target) * 100 if target != 0 else 0
            
            # Déterminer la couleur
            color = "green" if difference > 0 else "red" if difference < 0 else "gray"
            
            # Formater les valeurs
            formatted_actual = format_func(actual) if format_func else f"{actual:.2f}"
            formatted_target = format_func(target) if format_func else f"{target:.2f}"
            formatted_diff = format_func(abs(difference)) if format_func else f"{abs(difference):.2f}"
            
            # Créer le HTML
            html = f"""
            <div style='background-color: #f8f9fa; border-radius: 5px; padding: 10px; margin: 10px 0;'>
                <div style='font-weight: bold; margin-bottom: 5px;'>{label}</div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <span style='font-size: 1.2rem; font-weight: bold;'>{formatted_actual}</span>
                        <span style='color: #666; font-size: 0.9rem;'> / {formatted_target}</span>
                    </div>
                    <div style='color: {color}; text-align: right;'>
                        <div>{difference_pct:+.1f}%</div>
                        <div style='font-size: 0.8rem;'>{formatted_diff}</div>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(f"**{label}:** N/A")


    def create_progress_bar(label, current, target, color="blue"):
        """
        Crée une barre de progression
        
        :param label: Libellé
        :param current: Valeur actuelle
        :param target: Valeur cible
        :param color: Couleur de la barre
        """
        if current is not None and target is not None and target != 0:
            percentage = (current / target) * 100
            percentage = min(max(percentage, 0), 100)  # Limiter entre 0 et 100
            
            # Déterminer la couleur si non spécifiée
            if color == "auto":
                if percentage >= 100:
                    color = "green"
                elif percentage >= 75:
                    color = "lightgreen"
                elif percentage >= 50:
                    color = "orange"
                else:
                    color = "red"
            
            html = f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='font-weight: bold;'>{label}</span>
                    <span>{percentage:.1f}%</span>
                </div>
                <div style='background-color: #e0e0e0; border-radius: 10px; overflow: hidden;'>
                    <div style='width: {percentage}%; background-color: {color}; height: 20px; border-radius: 10px;'></div>
                </div>
            </div>
            """
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(f"**{label}:** N/A")


    def create_mini_chart(data, chart_type="line", height=100, color="blue"):
        """
        Crée un mini graphique
        
        :param data: Données à afficher
        :param chart_type: Type de graphique ("line", "bar", "area")
        :param height: Hauteur du graphique
        :param color: Couleur du graphique
        """
        fig = go.Figure()
        
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                line=dict(color=color, width=2),
                fill=None
            ))
        elif chart_type == "bar":
            fig.add_trace(go.Bar(
                y=data,
                marker_color=color
            ))
        elif chart_type == "area":
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                line=dict(color=color, width=2),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}"
            ))
        
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


    def create_delta_metric(current, previous, label=""):
        """
        Crée un indicateur de variation
        
        :param current: Valeur actuelle
        :param previous: Valeur précédente
        :param label: Libellé optionnel
        """
        if current is not None and previous is not None and previous != 0:
            delta = current - previous
            delta_pct = (delta / previous) * 100
            
            color = "green" if delta > 0 else "red" if delta < 0 else "gray"
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            
            html = f"""
            <div style='display: inline-block; color: {color}; font-weight: bold;'>
                {arrow} {abs(delta_pct):.2f}%
                {f" ({label})" if label else ""}
            </div>
            """
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("N/A")