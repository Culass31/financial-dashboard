# services/portfolio_service.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional, List

from .data_service import DataService
from .analysis_service import AnalysisService


class PortfolioService:
    """Service responsible for portfolio management operations"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.analysis_service = AnalysisService(data_service)
    
    def analyze_portfolio(self, portfolio_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze a portfolio of stocks"""
        if portfolio_df.empty:
            return pd.DataFrame(), {}
        
        results = []
        total_portfolio_value = 0
        
        for index, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            quantity = row['Quantité']
            entry_price = row['PRU']
            
            # Get fundamental data and history
            fundamental_data = self.data_service.get_fundamental_data(ticker)
            historical_financials = self.data_service.get_historical_financials(ticker)
            stock_history = self.data_service.get_stock_history(ticker)
            
            # Technical analysis
            if not stock_history.empty:
                df_reg, tech_indicators = self.analysis_service.calculate_regression(stock_history)
                tech_recommendation = self._get_technical_recommendation(tech_indicators)
            else:
                tech_indicators = {}
                tech_recommendation = "DONNÉES INSUFFISANTES"
            
            # Fundamental analysis
            intrinsic_value_result = self.analysis_service.calculate_intrinsic_value(
                ticker, fundamental_data, historical_financials
            )
            
            # Current price
            current_price = None
            if fundamental_data and 'Données de marché' in fundamental_data:
                current_price = fundamental_data['Données de marché'].get('Prix actuel')
            
            if not current_price and not stock_history.empty:
                current_price = stock_history['Close'].iloc[-1]
            
            if current_price:
                position_value = quantity * current_price
                total_portfolio_value += position_value
                
                performance = ((current_price / entry_price) - 1) * 100
                
                if intrinsic_value_result['success']:
                    intrinsic_value = intrinsic_value_result['intrinsic_value']
                    intrinsic_value_with_mos = intrinsic_value_result['intrinsic_value_with_mos']
                    fundamental_recommendation = intrinsic_value_result['recommendation']
                else:
                    intrinsic_value = None
                    intrinsic_value_with_mos = None
                    fundamental_recommendation = "DONNÉES INSUFFISANTES"
                
                upside_potential = None
                if intrinsic_value:
                    upside_potential = ((intrinsic_value / current_price) - 1) * 100
                
                # Combine recommendations
                combined_recommendation = self._combine_recommendations(
                    tech_recommendation, fundamental_recommendation
                )
                
                results.append({
                    'Ticker': ticker,
                    'Nom': fundamental_data['Données générales'].get('Nom', ''),
                    'Quantité': quantity,
                    'PRU': entry_price,
                    'Prix actuel': current_price,
                    'Valeur position': position_value,
                    'Performance (%)': performance,
                    'Valeur intrinsèque': intrinsic_value,
                    'VI avec MOS': intrinsic_value_with_mos,
                    'Potentiel (%)': upside_potential,
                    'Rec. Technique': tech_recommendation,
                    'Rec. Fondamentale': fundamental_recommendation,
                    'Recommandation': combined_recommendation
                })
        
        results_df = pd.DataFrame(results)
        
        portfolio_stats = {
            'total_value': total_portfolio_value,
            'performance': None
        }
        
        if not results_df.empty:
            total_cost = sum(results_df['Quantité'] * results_df['PRU'])
            portfolio_stats['performance'] = ((total_portfolio_value / total_cost) - 1) * 100
        
        return results_df, portfolio_stats
    
    def optimize_portfolio_allocation(
        self, 
        portfolio_df: pd.DataFrame, 
        risk_tolerance: str = 'moderate'
    ) -> pd.DataFrame:
        """Optimize portfolio allocation based on recommendations and risk profile"""
        if portfolio_df.empty:
            return pd.DataFrame()
        
        optimized_df = portfolio_df.copy()
        total_value = optimized_df['Valeur position'].sum()
        
        optimized_df['Poids actuel (%)'] = (optimized_df['Valeur position'] / total_value) * 100
        
        # Define target weights based on recommendations
        recommendation_weights = {
            'conservative': {
                'ACHETER': 1.5,
                'RENFORCER': 1.2,
                'CONSERVER': 1.0,
                'ALLEGER': 0.7,
                'VENDRE': 0.3,
                'DONNÉES INSUFFISANTES': 0.8
            },
            'moderate': {
                'ACHETER': 2.0,
                'RENFORCER': 1.5,
                'CONSERVER': 1.0,
                'ALLEGER': 0.5,
                'VENDRE': 0.2,
                'DONNÉES INSUFFISANTES': 0.7
            },
            'aggressive': {
                'ACHETER': 2.5,
                'RENFORCER': 1.8,
                'CONSERVER': 1.0,
                'ALLEGER': 0.3,
                'VENDRE': 0.1,
                'DONNÉES INSUFFISANTES': 0.6
            }
        }
        
        weights = recommendation_weights.get(risk_tolerance, recommendation_weights['moderate'])
        
        # Calculate raw weights
        optimized_df['Poids brut'] = optimized_df['Recommandation'].map(weights)
        
        # Adjust by upside potential
        for idx, row in optimized_df.iterrows():
            if pd.notna(row['Potentiel (%)']) and row['Potentiel (%)'] > 0:
                potential_factor = min(1 + (row['Potentiel (%)'] / 100), 1.5)
                optimized_df.at[idx, 'Poids brut'] *= potential_factor
        
        # Normalize weights
        total_weight = optimized_df['Poids brut'].sum()
        optimized_df['Poids cible (%)'] = (optimized_df['Poids brut'] / total_weight) * 100
        
        # Calculate adjustments
        optimized_df['Ajustement (%)'] = optimized_df['Poids cible (%)'] - optimized_df['Poids actuel (%)']
        optimized_df['Montant à ajuster'] = (optimized_df['Ajustement (%)'] / 100) * total_value
        
        # Determine actions
        optimized_df['Action'] = optimized_df['Ajustement (%)'].apply(
            lambda x: "ACHETER" if x > 1 else "VENDRE" if x < -1 else "MAINTENIR"
        )
        
        # Calculate shares to adjust
        optimized_df['Qté à ajuster'] = (optimized_df['Montant à ajuster'] / optimized_df['Prix actuel']).round().fillna(0).astype(int)
        
        return optimized_df[['Ticker', 'Poids actuel (%)', 'Poids cible (%)', 'Ajustement (%)', 
                          'Montant à ajuster', 'Action', 'Qté à ajuster', 'Recommandation']]
    
    def _get_technical_recommendation(self, indicators: Dict) -> str:
        """Determine technical recommendation based on indicators"""
        if not indicators:
            return "DONNÉES INSUFFISANTES"
        
        correlation = indicators.get('correlation', 0)
        model_growth = indicators.get('model_growth', 0)
        deviation = indicators.get('deviation', 0)
        
        if correlation > 0.7 and model_growth > 5:
            if deviation < -1:
                return "ACHETER"
            elif deviation < 0:
                return "RENFORCER"
            elif deviation > 1.5:
                return "ALLEGER"
            else:
                return "CONSERVER"
        elif correlation < -0.7 and model_growth < -5:
            if deviation < -1.5:
                return "CONSERVER"  # Possible bounce
            else:
                return "VENDRE"
        else:
            if deviation < -1.5:
                return "RENFORCER"
            elif deviation > 1.5:
                return "ALLEGER"
            else:
                return "NEUTRE"
    
    def _combine_recommendations(
        self, 
        tech_recommendation: str, 
        fundamental_recommendation: str
    ) -> str:
        """Combine technical and fundamental recommendations"""
        if tech_recommendation != "DONNÉES INSUFFISANTES" and fundamental_recommendation != "DONNÉES INSUFFISANTES":
            recommendation_points = {
                "ACHAT FORT": 2,
                "ACHETER": 2,
                "RENFORCER": 1,
                "CONSERVER": 0,
                "ALLEGER": -1,
                "VENDRE": -2,
                "NEUTRE": 0,
                "DONNÉES INSUFFISANTES": 0
            }
            
            tech_points = recommendation_points.get(tech_recommendation, 0)
            fundamental_points = recommendation_points.get(fundamental_recommendation, 0)
            
            # Weight: 40% technical, 60% fundamental
            weighted_points = (tech_points * 0.4) + (fundamental_points * 0.6)
            
            if weighted_points >= 1.5:
                return "ACHETER"
            elif weighted_points >= 0.5:
                return "RENFORCER"
            elif weighted_points <= -1.5:
                return "VENDRE"
            elif weighted_points <= -0.5:
                return "ALLEGER"
            else:
                return "CONSERVER"
        elif tech_recommendation != "DONNÉES INSUFFISANTES":
            return tech_recommendation
        elif fundamental_recommendation != "DONNÉES INSUFFISANTES":
            return fundamental_recommendation
        else:
            return "DONNÉES INSUFFISANTES"