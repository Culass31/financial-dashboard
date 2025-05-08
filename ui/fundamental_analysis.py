        
        # Site web
        if general_data.get('Site web'):
            st.markdown(f"**Site web:** [{general_data.get('Site web')}]({general_data.get('Site web')})")
        
        # Description avec expander
        if general_data.get('Description'):
            with st.expander("Description de l'entreprise"):
                st.write(general_data.get('Description'))
        
        # Données de marché
        st.subheader("📈 Données de marché")
        market_data = fundamental_data["Données de marché"]
        
        # Prix actuel avec tendance
        current_price = market_data.get('Prix actuel')
        previous_close = market_data.get('Précédente clôture')
        
        if current_price and previous_close:
            change = ((current_price - previous_close) / previous_close) * 100
            arrow = "↑" if change >= 0 else "↓"
            color = "green" if change >= 0 else "red"
            st.markdown(f"**Prix actuel:** {format_number(current_price)} {arrow} <span style='color:{color};'>({change:.2f}%)</span>", unsafe_allow_html=True)
        
        # Afficher les principales données de marché
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
    
    with col2:
        # Ratios fondamentaux
        st.subheader("📊 Ratios fondamentaux")
        fundamental_metrics = fundamental_data["Données fondamentales"]
        
        ratios_col1, ratios_col2 = st.columns(2)
        
        with ratios_col1:
            st.markdown("**PER:** " + format_number(fundamental_metrics.get('PER', 'N/A')))
            st.markdown("**PEG Ratio:** " + format_number(fundamental_metrics.get('PEG Ratio', 'N/A')))
            st.markdown("**P/B (Cours/Valeur Comptable):** " + format_number(fundamental_metrics.get('P/B Cours/Valeur Comptable', 'N/A')))
            st.markdown("**P/S (Price to Sales):** " + format_number(fundamental_metrics.get('P/S (Price to Sales)', 'N/A')))
            st.markdown("**Marge nette (%):** " + format_number(fundamental_metrics.get('Marge nette (%)', 'N/A')))
        
        with ratios_col2:
            st.markdown("**ROE (%):** " + format_number(fundamental_metrics.get('ROE (%)', 'N/A')))
            st.markdown("**ROA (%):** " + format_number(fundamental_metrics.get('ROA (%)', 'N/A')))
            st.markdown("**Ratio d'endettement:** " + format_number(fundamental_metrics.get('Ratio d\'endettement', 'N/A')))
            st.markdown("**Dividende:** " + format_number(fundamental_metrics.get('Dividende', 'N/A')))
            st.markdown("**Rendement du dividende (%):** " + format_number(fundamental_metrics.get('Rendement du dividende (%)', 'N/A')))
        
        # Données financières
        st.subheader("💰 Données financières")
        financial_data = fundamental_data["Données financières"]
        
        financial_col1, financial_col2 = st.columns(2)
        
        with financial_col1:
            st.markdown("**Chiffre d'affaires:** " + format_number(financial_data.get('Chiffre d\'affaires', 'N/A')))
            st.markdown("**Résultat net:** " + format_number(financial_data.get('Résultat net', 'N/A')))
            st.markdown("**EBITDA:** " + format_number(financial_data.get('EBITDA', 'N/A')))
            st.markdown("**BPA:** " + format_number(fundamental_metrics.get('BPA', 'N/A')))
        
        with financial_col2:
            st.markdown("**Total Actif:** " + format_number(financial_data.get('Total Actif', 'N/A')))
            st.markdown("**Total Dette:** " + format_number(financial_data.get('Total Dette', 'N/A')))
            st.markdown("**Fonds propres:** " + format_number(financial_data.get('Fonds propres', 'N/A')))
            st.markdown("**Free Cash Flow:** " + format_number(financial_data.get('Free Cash Flow', 'N/A')))


def render_financial_statements(income_stmt, balance_sheet, cashflow):
    """Render les états financiers historiques"""
    
    st.subheader("📜 États financiers historiques")
    financial_tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie"])
    
    with financial_tabs[0]:
        if not income_stmt.empty:
            income_display = income_stmt.copy() / 1e6
            st.dataframe(income_display.style.format("{:.2f}"))
            st.caption("Montants en millions")
        else:
            st.info("Données du compte de résultat non disponibles")
    
    with financial_tabs[1]:
        if not balance_sheet.empty:
            balance_display = balance_sheet.copy() / 1e6
            st.dataframe(balance_display.style.format("{:.2f}"))
            st.caption("Montants en millions")
        else:
            st.info("Données du bilan non disponibles")
    
    with financial_tabs[2]:
        if not cashflow.empty:
            cashflow_display = cashflow.copy() / 1e6
            st.dataframe(cashflow_display.style.format("{:.2f}"))
            st.caption("Montants en millions")
        else:
            st.info("Données des flux de trésorerie non disponibles")


def render_valuation_analysis(ticker, fundamental_data, income_stmt, balance_sheet, cashflow):
    """Render l'analyse de valorisation et les recommandations"""
    
    # Créer des onglets pour différentes analyses
    valuation_tabs = st.tabs(["Valorisation", "Méthodes détaillées", "Avantage compétitif", "Dividendes"])
    
    # Onglet 1: Valorisation intrinsèque
    with valuation_tabs[0]:
        render_intrinsic_valuation(ticker, fundamental_data, income_stmt, balance_sheet, cashflow)
    
    # Onglet 2: Méthodes de valorisation détaillées
    with valuation_tabs[1]:
        render_valuation_methods(ticker, fundamental_data, income_stmt, balance_sheet, cashflow)
    
    # Onglet 3: Avantage compétitif
    with valuation_tabs[2]:
        render_competitive_advantage(ticker, fundamental_data, income_stmt, balance_sheet, cashflow)
    
    # Onglet 4: Dividendes
    with valuation_tabs[3]:
        render_dividend_analysis(ticker)


def render_intrinsic_valuation(ticker, fundamental_data, income_stmt, balance_sheet, cashflow):
    """Render l'analyse de valorisation intrinsèque"""
    
    # Layout en deux colonnes
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        st.write("#### Valorisation intrinsèque calculée")
        
        # Calculer la valeur intrinsèque
        intrinsic_value_result = calculate_intrinsic_value(
            ticker, 
            fundamental_data["Données générales"]
        )
    
    if moat_analysis:
        # Affichage de l'avantage compétitif
        moat_existence = moat_analysis.get('existence', 'Indéterminé')
        moat_strength = moat_analysis.get('strength', 0)
        moat_factors = moat_analysis.get('factors', [])
        moat_explanation = moat_analysis.get('explanation', '')
        
        # Couleur selon la force de l'avantage
        moat_color = {
            0: 'gray',
            1: 'orange',
            2: 'blue',
            3: 'green'
        }.get(moat_strength, 'gray')
        
        # Affichage principal
        st.markdown(f"""
        <div style='background-color: {moat_color}; padding: 10px; border-radius: 5px; color: white;'>
            <h4 style='margin: 0;'>🛡️ Avantage compétitif: {moat_existence}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Détails
        moat_col1, moat_col2 = st.columns(2)
        
        with moat_col1:
            if moat_factors:
                st.markdown("##### Facteurs contribuant à l'avantage compétitif:")
                for factor in moat_factors:
                    st.markdown(f"- {factor}")
        
        with moat_col2:
            st.markdown("##### Indicateurs clés")
            
            # ROIC
            from core.analysis import calculate_roic
            roic = calculate_roic(income_stmt, balance_sheet)
            
            if roic is not None:
                roic_color = "green" if roic > 10 else "orange" if roic > 5 else "red"
                st.markdown(f"""
                <div style='background-color: {roic_color}; padding: 5px; border-radius: 5px; color: white; text-align: center;'>
                    <h4 style='margin: 0;'>ROIC: {roic:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyse détaillée
        st.markdown(f"**Analyse:** {moat_explanation}")
        
        # Graphiques complémentaires
        render_competitive_charts(income_stmt, balance_sheet, cashflow)


def render_competitive_charts(income_stmt, balance_sheet, cashflow):
    """Render les graphiques liés à l'avantage compétitif"""
    
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        # Graphique FCF historique
        if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            render_fcf_chart(cashflow)
        
        # Stabilité des marges
        if not income_stmt.empty and len(income_stmt.columns) >= 4:
            render_margin_stability_chart(income_stmt)
    
    with main_col2:
        # Evolution du ROE
        if not income_stmt.empty and not balance_sheet.empty:
            render_roe_evolution_chart(income_stmt, balance_sheet)
        
        # Analyse de l'endettement
        if not balance_sheet.empty:
            render_debt_analysis(income_stmt, balance_sheet)


def render_margin_stability_chart(income_stmt):
    """Render le graphique de stabilité des marges brutes"""
    
    try:
        if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
            margins = []
            years = []
            
            for col in income_stmt.columns[:5]:
                try:
                    gross_profit = income_stmt.loc['Gross Profit', col]
                    total_revenue = income_stmt.loc['Total Revenue', col]
                    if gross_profit and total_revenue and total_revenue != 0:
                        margin = gross_profit / total_revenue * 100
                        margins.append(margin)
                        years.append(str(col.year))
                except:
                    continue
            
            if margins:
                st.markdown("##### Stabilité des marges brutes")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=years, y=margins, name='Marge brute (%)', marker_color='blue'))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse de stabilité
                stability = np.std(margins)
                st.markdown(f"**Stabilité des marges (écart-type):** {stability:.2f}%")
                
                if stability < 3:
                    st.markdown("🟢 **Marges très stables** (indicateur positif d'avantage compétitif)")
                elif stability < 5:
                    st.markdown("🟡 **Marges relativement stables**")
                else:
                    st.markdown("🔴 **Marges volatiles** (possible absence d'avantage compétitif)")
    except Exception as e:
        st.error(f"Erreur lors du calcul de la stabilité des marges: {e}")


def render_roe_evolution_chart(income_stmt, balance_sheet):
    """Render le graphique d'évolution du ROE"""
    
    try:
        if 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance_sheet.index:
            roe_values = []
            roe_years = []
            
            for year in range(min(5, len(income_stmt.columns), len(balance_sheet.columns))):
                try:
                    net_income = income_stmt.loc['Net Income', income_stmt.columns[year]]
                    equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[year]]
                    if net_income and equity and equity != 0:
                        roe = net_income / equity * 100
                        roe_values.append(roe)
                        roe_years.append(str(income_stmt.columns[year].year))
                except:
                    continue
            
            if roe_values:
                st.markdown("##### Evolution du ROE")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=roe_years, 
                    y=roe_values, 
                    name='ROE (%)', 
                    mode='lines+markers', 
                    line=dict(color='green')
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse du ROE moyen
                avg_roe = np.mean(roe_values)
                st.markdown(f"**ROE moyen sur la période:** {avg_roe:.2f}%")
                
                if avg_roe > 15:
                    st.markdown("🟢 **ROE excellent** (entreprise très rentable)")
                elif avg_roe > 10:
                    st.markdown("🟡 **ROE bon** (entreprise rentable)")
                else:
                    st.markdown("🔴 **ROE faible** (rentabilité à améliorer)")
    except Exception as e:
        st.error(f"Erreur lors du calcul de l'évolution du ROE: {e}")


def render_debt_analysis(income_stmt, balance_sheet):
    """Render l'analyse de l'endettement"""
    
    try:
        st.markdown("##### Ratios d'endettement")
        
        debt_metrics = {}
        
        # Ratio dette à long terme / bénéfice net
        if 'Long Term Debt' in balance_sheet.index and 'Net Income' in income_stmt.index:
            long_term_debt = balance_sheet.loc['Long Term Debt', balance_sheet.columns[0]]
            net_income = income_stmt.loc['Net Income', income_stmt.columns[0]]
            if long_term_debt is not None and net_income is not None and net_income != 0:
                debt_to_income = long_term_debt / net_income
                debt_metrics["Dette LT / Bénéfice net"] = f"{debt_to_income:.2f}x"
                
                if debt_to_income < 3:
                    debt_metrics["Évaluation"] = "🟢 Excellent (< 3x selon Buffett)"
                elif debt_to_income < 5:
                    debt_metrics["Évaluation"] = "🟡 Acceptable (< 5x)"
                else:
                    debt_metrics["Évaluation"] = "🔴 Élevé (> 5x)"
        
        # Afficher les métriques
        if debt_metrics:
            df_debt = pd.DataFrame(debt_metrics.items(), columns=["Métrique", "Valeur"])
            st.dataframe(df_debt, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse d'endettement: {e}")


def render_dividend_analysis(ticker):
    """Render l'analyse des dividendes"""
    
    st.subheader("💰 Analyse des Dividendes")
    
    with st.spinner("Analyse de la politique de dividendes en cours..."):
        dividend_analysis = get_dividend_policy_analysis(ticker)
        
        if dividend_analysis.get('error'):
            st.error(f"Erreur lors de l'analyse des dividendes: {dividend_analysis['error']}")
            return
        
        if not dividend_analysis.get('has_dividends', False):
            st.info(f"Cette entreprise ne verse pas de dividendes actuellement.")
            return
        
        # Layout en deux colonnes
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Indicateurs clés")
            
            # Métriques principales
            metrics_data = [
                ("Rendement du dividende", f"{dividend_analysis.get('dividend_yield', 0):.2f}%", ""),
                ("Santé du dividende", "", quality_badge(dividend_analysis.get('dividend_health', 'Non évalué'))),
                ("Croissance du dividende", f"{dividend_analysis.get('dividend_growth', 0):.2f}%/an", quality_badge(dividend_analysis.get('dividend_growth_score', 'Non évalué'))),
                ("Consistance", "", quality_badge(dividend_analysis.get('dividend_consistency', 'Non évalué'))),
                ("Qualité globale", "", quality_badge(dividend_analysis.get('dividend_quality', 'Non évalué'))),
            ]
            
            # Afficher les métriques
            for name, value, badge in metrics_data:
                if value and badge:
                    st.markdown(f"**{name}:** {value} {badge}", unsafe_allow_html=True)
                elif value:
                    st.markdown(f"**{name}:** {value}", unsafe_allow_html=True)
                elif badge:
                    st.markdown(f"**{name}:** {badge}", unsafe_allow_html=True)
            
            # Informations supplémentaires
            st.markdown("#### Détails supplémentaires")
            
            additional_metrics = [
                ("Ratio de distribution", f"{dividend_analysis.get('payout_ratio', 0):.2f}%"),
                ("Durabilité", dividend_analysis.get('sustainability', 'Non évalué')),
                ("Tendance", dividend_analysis.get('dividend_trend')),
                ("Années consécutives d'augmentation", dividend_analysis.get('consecutive_years_increase', 0)),
                ("Aristocrate du dividende", "Oui" if dividend_analysis.get('dividend_aristocrat', False) else "Non"),
                ("Fréquence de versement", f"{dividend_analysis.get('dividend_frequency', 0):.1f} fois/an"),
                ("Rachat d'actions", dividend_analysis.get('buyback_trend', 'Indéterminé'))
            ]
            
            for name, value in additional_metrics:
                st.markdown(f"**{name}:** {value}")
        
        with col2:
            st.markdown("### 📈 Visualisation des dividendes")
            
            # Graphiques des dividendes
            if dividend_analysis.get('dividend_history'):
                render_dividend_charts(dividend_analysis['dividend_history'])
        
        # Analyse et recommandation
        render_dividend_recommendation(dividend_analysis)


def render_dividend_charts(dividend_history):
    """Render les graphiques de dividendes"""
    
    div_history = dividend_history
    dates = list(div_history.keys())
    values = list(div_history.values())
    
    # Convertir en DataFrame
    dividend_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'dividend': values
    })
    
    dividend_df = dividend_df.sort_values('date')
    
    # Graphique 1: Historique des dividendes
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=dividend_df['date'],
        y=dividend_df['dividend'],
        marker_color='#1E88E5',
        name='Dividende'
    ))
    
    fig1.update_layout(
        title="Historique des versements de dividendes",
        xaxis_title="Date",
        yaxis_title="Dividende par action",
        height=300
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Graphique 2: Croissance annuelle
    dividend_df['year'] = dividend_df['date'].dt.year
    annual_div = dividend_df.groupby('year')['dividend'].sum().reset_index()
    
    if len(annual_div) > 1:
        annual_div['growth'] = annual_div['dividend'].pct_change() * 100
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(
            go.Bar(
                x=annual_div['year'],
                y=annual_div['dividend'],
                name="Dividende annuel",
                marker_color='#43A047'
            ),
            secondary_y=False
        )
        
        fig2.add_trace(
            go.Scatter(
                x=annual_div['year'][1:],
                y=annual_div['growth'][1:],
                name="Taux de croissance",
                mode='lines+markers',
                line=dict(color='#E65100', width=3)
            ),
            secondary_y=True
        )
        
        fig2.update_layout(
            title="Croissance annuelle des dividendes",
            height=300
        )
        
        fig2.update_yaxes(title_text="Dividende par action", secondary_y=False)
        fig2.update_yaxes(title_text="Croissance annuelle (%)", secondary_y=True)
        
        st.plotly_chart(fig2, use_container_width=True)


def render_dividend_recommendation(dividend_analysis):
    """Render la recommandation sur les dividendes"""
    
    st.markdown("### 🔍 Analyse et recommandation")
    
    # Points forts et points faibles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Points forts")
        strengths = []
        
        if dividend_analysis.get('dividend_aristocrat', False):
            strengths.append("✅ Aristocrate du dividende (25+ ans d'augmentations)")
        if dividend_analysis.get('dividend_health') in ['Excellent', 'Bon']:
            strengths.append(f"✅ Santé du dividende {dividend_analysis.get('dividend_health').lower()}")
        if dividend_analysis.get('dividend_growth_score') in ['Excellent', 'Bon']:
            strengths.append(f"✅ Croissance du dividende {dividend_analysis.get('dividend_growth_score').lower()}")
        
        if strengths:
            for strength in strengths:
                st.markdown(strength)
        else:
            st.markdown("*Aucun point fort significatif identifié*")
    
    with col2:
        st.markdown("#### Points à surveiller")
        weaknesses = []
        
        if dividend_analysis.get('sustainability') == 'Risqué':
            weaknesses.append("⚠️ Taux de distribution élevé (>80%)")
        if dividend_analysis.get('dividend_consistency') in ['Moyen', 'Faible']:
            weaknesses.append(f"⚠️ Consistance des versements {dividend_analysis.get('dividend_consistency').lower()}")
        if dividend_analysis.get('buyback_trend') == 'Dilution':
            weaknesses.append("⚠️ Dilution des actionnaires")
        
        if weaknesses:
            for weakness in weaknesses:
                st.markdown(weakness)
        else:
            st.markdown("*Aucun point faible significatif identifié*")
    
    # Recommandation finale
    st.markdown("#### Recommandation")
    
    quality = dividend_analysis.get('dividend_quality', 'Non évalué')
    
    if quality == 'Excellent':
        recommendation = "Cet investissement présente une **excellente politique de dividende** qui combine santé financière, croissance et consistance."
    elif quality == 'Bon':
        recommendation = "Cet investissement présente une **bonne politique de dividende** avec une balance appropriée entre distribution et réinvestissement."
    elif quality == 'Moyen':
        recommendation = "Cet investissement présente une politique de dividende **acceptable mais avec des points d'attention**."
    elif quality == 'Faible':
        recommendation = "Cet investissement présente une politique de dividende **présentant des risques**."
    else:
        recommendation = "L'analyse complète de la politique de dividende n'a pas pu être réalisée."
    
    st.info(recommendation)
data, 
            (income_stmt, balance_sheet, cashflow)
        )
        
        if intrinsic_value_result and intrinsic_value_result.get('success', False):
            # Afficher les résultats principaux
            iv_col1, iv_col2, iv_col3 = st.columns(3)
            
            current_price = intrinsic_value_result.get('current_price', 0)
            
            with iv_col1:
                intrinsic_value = intrinsic_value_result.get('intrinsic_value', None)
                if intrinsic_value:
                    value_color = "green" if intrinsic_value > current_price else "red"
                    st.markdown(f"**Valeur intrinsèque:** <span style='color:{value_color};'>{intrinsic_value:.2f} €</span>", unsafe_allow_html=True)
            
            with iv_col2:
                mos_value = intrinsic_value_result.get('intrinsic_value_with_mos', None)
                if mos_value:
                    mos_color = "green" if mos_value > current_price else "orange"
                    st.markdown(f"**Avec marge de sécurité:** <span style='color:{mos_color};'>{mos_value:.2f} €</span>", unsafe_allow_html=True)
            
            with iv_col3:
                deviation = intrinsic_value_result.get('deviation', None)
                if deviation is not None:
                    dev_color = "green" if deviation < 0 else "red"
                    st.markdown(f"**Écart de valorisation:** <span style='color:{dev_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
            
            # Afficher la recommandation
            recommendation = intrinsic_value_result.get('recommendation', '')
            rec_colors = {
                "ACHAT FORT": "darkgreen",
                "ACHETER": "green",
                "RENFORCER": "lightgreen",
                "CONSERVER": "gold",
                "ALLEGER": "orange",
                "VENDRE": "red"
            }
            
            st.markdown(f"**Recommandation:** <span style='color:{rec_colors.get(recommendation, 'gray')};font-weight:bold;'>{recommendation}</span>", unsafe_allow_html=True)
            
            # Graphique comparatif des valorisations
            render_valuation_comparison_chart(intrinsic_value_result, current_price)
            
        else:
            st.error(f"Impossible de calculer la valeur intrinsèque pour {ticker}")
            if intrinsic_value_result:
                st.write(intrinsic_value_result.get('message', ''))
    
    with main_col2:
        st.write("#### Consensus des analystes")
        render_analyst_consensus(ticker, current_price if 'current_price' in locals() else None)


def render_valuation_comparison_chart(intrinsic_value_result, current_price):
    """Render le graphique comparatif des valorisations"""
    
    # Préparer les données
    price_data = [
        {
            'type': 'Valeur intrinsèque',
            'valeur': intrinsic_value_result.get('intrinsic_value', 0),
            'couleur': 'green' if intrinsic_value_result.get('intrinsic_value', 0) > current_price else 'red'
        },
        {
            'type': 'Valeur avec MOS',
            'valeur': intrinsic_value_result.get('intrinsic_value_with_mos', 0),
            'couleur': 'green' if intrinsic_value_result.get('intrinsic_value_with_mos', 0) > current_price else 'orange'
        }
    ]
    
    # Créer le graphique
    fig = go.Figure()
    
    for item in price_data:
        fig.add_trace(go.Bar(
            x=[item['type']],
            y=[item['valeur']],
            name=item['type'],
            marker_color=item['couleur']
        ))
    
    # Ajouter une ligne pour le prix actuel
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=current_price,
        x1=len(price_data)-0.5,
        y1=current_price,
        line=dict(color="blue", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=len(price_data)-0.5,
        y=current_price,
        text=f"Prix actuel: {current_price:.2f} €",
        showarrow=False,
        yshift=10,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    fig.update_layout(
        title='Comparaison des différentes valorisations',
        xaxis_title='Source de valorisation',
        yaxis_title='Valeur (€)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_analyst_consensus(ticker, current_price):
    """Render le consensus des analystes"""
    
    import yfinance as yf
    ticker_obj = yf.Ticker(ticker)
    
    # Objectifs de cours
    if hasattr(ticker_obj, 'analyst_price_targets') and ticker_obj.analyst_price_targets:
        target_data = ticker_obj.analyst_price_targets
        
        pt_col1, pt_col2, pt_col3 = st.columns(3)
        
        with pt_col1:
            mean_target = target_data.get('mean', None)
            if mean_target and current_price:
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
    
    # Recommandations
    if hasattr(ticker_obj, 'recommendations_summary') and not ticker_obj.recommendations_summary.empty:
        render_recommendations_chart(ticker_obj.recommendations_summary)


def render_recommendations_chart(recommendations_summary):
    """Render le graphique des recommandations des analystes"""
    
    latest_rec = recommendations_summary.iloc[0]
    
    # Créer le graphique
    categories = ['Achat fort', 'Achat', 'Conserver', 'Vendre', 'Vente forte']
    values = [latest_rec['strongBuy'], latest_rec['buy'], latest_rec['hold'], 
              latest_rec['sell'], latest_rec['strongSell']]
    colors = ['darkgreen', 'green', 'gold', 'orange', 'red']
    
    fig = go.Figure()
    
    total_recs = sum(values)
    
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
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
    
    fig.update_layout(
        title='Répartition des recommandations',
        barmode='stack',
        height=120,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(title="Nombre d'analystes"),
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_valuation_methods(ticker, fundamental_data, income_stmt, balance_sheet, cashflow):
    """Render les différentes méthodes de valorisation"""
    
    st.write("#### Détail des méthodes de valorisation")
    
    # Calculer la valeur intrinsèque
    valuation_result = calculate_intrinsic_value(
        ticker, 
        fundamental_data, 
        historical_financials=(income_stmt, balance_sheet, cashflow)
    )
    
    if not valuation_result['success']:
        st.error(f"Erreur de valorisation: {valuation_result['message']}")
        return
    
    # Créer des onglets pour chaque méthode
    method_tabs = st.tabs(["DCF", "Graham", "PER", "Valeur patrimoniale"])
    
    # Extraire les valeurs
    current_price = valuation_result['current_price']
    methods = valuation_result['methods']
    
    # Onglet DCF
    with method_tabs[0]:
        render_dcf_method(methods.get('dcf', {}), current_price, cashflow)
    
    # Onglet Graham
    with method_tabs[1]:
        render_graham_method(methods.get('graham', {}), current_price)
    
    # Onglet PER
    with method_tabs[2]:
        render_per_method(methods.get('per_based', {}), current_price, ticker)
    
    # Onglet Valeur patrimoniale
    with method_tabs[3]:
        render_asset_method(methods.get('asset_based', {}), current_price, balance_sheet)


def render_dcf_method(dcf_data, current_price, cashflow):
    """Render la méthode DCF"""
    
    st.write("##### Modèle DCF (Discounted Cash Flow)")
    
    dcf_value = dcf_data.get('value')
    
    if dcf_value:
        dcf_color = "green" if dcf_value > current_price else "red"
        st.markdown(f"**Valeur DCF calculée:** <span style='color:{dcf_color};font-weight:bold;'>{dcf_value:.2f} €</span>", unsafe_allow_html=True)
        
        # Potentiel
        potential = ((dcf_value / current_price) - 1) * 100
        potential_color = "green" if potential > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
        
        # Graphique FCF historique
        render_fcf_chart(cashflow)
    else:
        st.warning("Impossible de calculer la valeur DCF pour cette entreprise.")


def render_fcf_chart(cashflow):
    """Render le graphique des Free Cash Flow historiques"""
    
    if not cashflow.empty and 'Free Cash Flow' in cashflow.index:
        fcf_data = cashflow.loc['Free Cash Flow'].to_dict()
        
        fcf_df = pd.DataFrame({
            'Année': [str(year.year) for year in fcf_data.keys()],
            'FCF (millions)': [float(value) / 1e6 for value in fcf_data.values()]
        })
        
        fcf_df = fcf_df.sort_values('Année')
        
        # Calculer la croissance annuelle moyenne
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
        
        # Créer le graphique
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
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_graham_method(graham_data, current_price):
    """Render la méthode Graham"""
    
    st.write("##### Méthode de Graham (Valeur Intrinsèque)")
    
    graham_value = graham_data.get('value')
    netnet_value = graham_data.get('netnet_value')
    
    if graham_value:
        graham_color = "green" if graham_value > current_price else "red"
        st.markdown(f"**Valeur selon Graham:** <span style='color:{graham_color};font-weight:bold;'>{graham_value:.2f} €</span>", unsafe_allow_html=True)
        
        # Écart avec le prix actuel
        deviation = ((graham_value / current_price) - 1) * 100
        deviation_color = "green" if deviation > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{deviation_color};'>{deviation:.1f}%</span>", unsafe_allow_html=True)
        
        # Valeur Net-Net si disponible
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


def render_per_method(per_data, current_price, ticker):
    """Render la méthode PER"""
    
    st.write("##### Valorisation basée sur le PER (Price Earnings Ratio)")
    
    per_value = per_data.get('value')
    
    if per_value:
        # Afficher différents scénarios de valorisation PER
        st.info("La valorisation PER utilise le PER sectoriel pour estimer la valeur de l'action.")
        
        # Créer un tableau de scénarios
        scenarios = pd.DataFrame({
            'Scénario': ['PER sectoriel', 'PER -20%', 'PER +20%'],
            'Valorisation': [per_value, per_value * 0.8, per_value * 1.2],
            'Potentiel': [
                ((per_value / current_price) - 1) * 100,
                ((per_value * 0.8 / current_price) - 1) * 100,
                ((per_value * 1.2 / current_price) - 1) * 100
            ]
        })
        
        # Formater le tableau
        scenarios['Valorisation'] = scenarios['Valorisation'].apply(lambda x: f"{x:.2f} €")
        scenarios['Potentiel'] = scenarios['Potentiel'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(scenarios)
    else:
        st.warning("Données insuffisantes pour calculer la valorisation basée sur le PER.")


def render_asset_method(asset_data, current_price, balance_sheet):
    """Render la méthode patrimoniale"""
    
    st.write("##### Valorisation basée sur les actifs (Valeur patrimoniale)")
    
    asset_value = asset_data.get('value')
    
    if asset_value:
        asset_color = "green" if asset_value > current_price else "red"
        st.markdown(f"**Valeur patrimoniale par action:** <span style='color:{asset_color};font-weight:bold;'>{asset_value:.2f} €</span>", unsafe_allow_html=True)
        
        # Potentiel
        potential = ((asset_value / current_price) - 1) * 100
        potential_color = "green" if potential > 0 else "red"
        st.markdown(f"**Potentiel:** <span style='color:{potential_color};'>{potential:.1f}%</span>", unsafe_allow_html=True)
        
        with st.expander("À propos de la valeur patrimoniale"):
            st.write("""
            La méthode de valorisation patrimoniale évalue une entreprise sur la base de ses actifs nets tangibles.
            Elle est particulièrement pertinente pour les entreprises des secteurs financiers, immobiliers ou industriels.
            """)
        
        # Afficher les composantes du bilan
        render_balance_sheet_chart(balance_sheet)
    else:
        st.warning("Impossible de calculer la valeur patrimoniale pour cette entreprise.")


def render_balance_sheet_chart(balance_sheet):
    """Render le graphique des composantes du bilan"""
    
    if not balance_sheet.empty:
        st.write("##### Principales composantes du bilan (dernière année)")
        
        try:
            balance_data = {}
            
            key_metrics = {
                'Total Assets': 'Actifs totaux',
                'Total Liabilities Net Minority Interest': 'Passifs totaux',
                'Goodwill And Intangible Assets': 'Goodwill et Intangibles',
                'Stockholders Equity': 'Capitaux propres'
            }
            
            for key, label in key_metrics.items():
                if key in balance_sheet.index:
                    balance_data[label] = balance_sheet.loc[key, balance_sheet.columns[0]] / 1e6
            
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
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de l'affichage du bilan: {str(e)}")


def render_competitive_advantage(ticker, fundamental_data, income_stmt, balance_sheet, cashflow):
    """Render l'analyse de l'avantage compétitif"""
    
    with st.spinner("Analyse de l'avantage compétitif en cours..."):
        moat_analysis = analyze_competitive_advantage(
            income_stmt, 
            balance_sheet, 
            cashflow, 
            fundamental_"""
Composant pour l'onglet d'analyse fondamentale
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.market_data import get_fundamental_data, get_historical_financials
from core.analysis import calculate_intrinsic_value, analyze_competitive_advantage
from core.dividend_analysis import get_dividend_policy_analysis
from ui_components.metrics import display_metric_card, display_kpi_metrics
from ui_components.styles import format_metric_card, quality_badge
from ui_components.common import format_number


def render_fundamental_analysis_tab(selected_stock):
    """Render l'onglet d'analyse fondamentale"""
    
    if selected_stock is None:
        st.info("Veuillez sélectionner une action dans le menu latéral pour voir l'analyse fondamentale.")
        return
    
    ticker = selected_stock['ticker']
    stock_name = st.session_state.get('selected_stock_name', ticker)
    
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>Analyse Fondamentale: {stock_name} ({ticker})</h2>", unsafe_allow_html=True)
    
    # Sous-onglets
    fundamental_tab1, fundamental_tab2 = st.tabs(["📋 Profil & Données Financières", "🔮 Évaluation & Recommandation"])
    
    # Récupération des données
    with st.spinner("Chargement des données fondamentales..."):
        fundamental_data = get_fundamental_data(ticker)
        income_stmt, balance_sheet, cashflow = get_historical_financials(ticker)
    
    # Sous-onglet 1: Profil et Données Financières
    with fundamental_tab1:
        if fundamental_data:
            render_company_profile(fundamental_data)
            render_financial_statements(income_stmt, balance_sheet, cashflow)
        else:
            st.error(f"Impossible de récupérer les données fondamentales pour {ticker}")
    
    # Sous-onglet 2: Évaluation et Recommandation
    with fundamental_tab2:
        render_valuation_analysis(ticker, fundamental_data, income_stmt, balance_sheet, cashflow)


def render_company_profile(fundamental_data):
    """Render le profil de l'entreprise et les données financières"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Informations générales
        st.subheader("📝 Profil de l'entreprise")
        general_data = fundamental_data["Données générales"]
        
        # Affichage des informations de base
        st.markdown(f"**{general_data.get('Nom', 'N/A')}**")
        st.markdown(f"**Secteur:** {general_data.get('Secteur', 'N/A')} | **Industrie:** {general_data.get('Industrie', 'N/A')} | **Pays:** {general_data.get('Pays', 'N/A')}")