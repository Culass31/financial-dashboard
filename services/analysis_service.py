# services/analysis_service.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
import yfinance as yf

from .data_service import DataService


class AnalysisService:
    """Service responsible for all analysis operations"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
    
    def calculate_regression(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Calculate linear regression and technical indicators"""
        if df.empty:
            return df, {}
        
        df_reg = df.copy()
        df_reg['index'] = range(len(df_reg))
        
        X = df_reg['index'].values.reshape(-1, 1)
        y = df_reg['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        df_reg['Regression'] = model.predict(X)
        df_reg['Deviation'] = df_reg['Close'] - df_reg['Regression']
        
        std_dev = df_reg['Deviation'].std()
        
        if std_dev > 0:
            df_reg['DeviationScore'] = df_reg['Deviation'] / std_dev
        else:
            df_reg['DeviationScore'] = 0
        
        # Calculate performance indicators
        current_price = df_reg['Close'].iloc[-1]
        
        days_1y = min(252, len(df_reg) - 1)
        price_1y_ago = df_reg['Close'].iloc[-days_1y-1] if days_1y > 0 else current_price
        prog_1y = ((current_price / price_1y_ago) - 1) * 100 if price_1y_ago > 0 else 0
        
        days_5y = min(1260, len(df_reg) - 1)
        price_5y_ago = df_reg['Close'].iloc[-days_5y-1] if days_5y > 0 else current_price
        prog_5y = ((current_price / price_5y_ago) - 1) * 100 if price_5y_ago > 0 else 0
        
        # Model growth
        if len(df_reg) > 252:
            initial_regression = df_reg['Regression'].iloc[0]
            final_regression = df_reg['Regression'].iloc[-1]
            years = len(df_reg) / 252
            
            if initial_regression > 0:
                model_growth = (((final_regression / initial_regression) ** (1/years)) - 1) * 100
            else:
                model_growth = 0
        else:
            model_growth = 0
        
        correlation = np.corrcoef(df_reg['index'], df_reg['Close'])[0, 1]
        current_deviation = df_reg['DeviationScore'].iloc[-1]
        
        indicators = {
            'model_growth': model_growth,
            'correlation': correlation,
            'deviation': current_deviation,
            'prog_1y': prog_1y,
            'prog_5y': prog_5y
        }
        
        return df_reg, indicators
    
    def calculate_intrinsic_value(
        self, 
        ticker: str, 
        fundamental_data: Optional[Dict] = None, 
        historical_financials: Optional[Tuple] = None, 
        market_structure: Optional[Dict] = None
    ) -> Dict:
        """Calculate intrinsic value using multiple methods"""
        try:
            if fundamental_data is None:
                fundamental_data = self.data_service.get_fundamental_data(ticker)
            
            if historical_financials is None:
                income_stmt, balance_sheet, cashflow = self.data_service.get_historical_financials(ticker)
            else:
                income_stmt, balance_sheet, cashflow = historical_financials
            
            if not fundamental_data or 'Données de marché' not in fundamental_data:
                return {
                    'success': False,
                    'message': f"Données fondamentales insuffisantes pour {ticker}"
                }
            
            current_price = fundamental_data['Données de marché'].get('Prix actuel', None)
            
            if not current_price:
                return {
                    'success': False,
                    'message': f"Prix actuel non disponible pour {ticker}"
                }
            
            sector = fundamental_data['Données générales'].get('Secteur', None)
            
            # Method 1: DCF Model
            dcf_value = self._calculate_dcf_value(
                ticker, income_stmt, cashflow, balance_sheet, 
                fundamental_data, current_price, market_structure
            )
            
            # Method 2: Graham Model
            graham_value = self._calculate_graham_value(fundamental_data, income_stmt)
            
            # Method 3: PER-based valuation
            per_based_value = self._calculate_per_value(ticker, fundamental_data)
            
            # Method 4: Asset-based valuation
            asset_based_value = self._calculate_asset_value(
                balance_sheet, fundamental_data, sector
            )
            
            # Combine methods for final intrinsic value
            values_and_weights = []
            
            # Sector-specific weights
            dcf_weight = 0.5
            per_weight = 0.3
            graham_weight = 0.1
            asset_weight = 0.1
            
            if sector:
                if sector in ['Technology', 'Healthcare', 'Communication Services']:
                    dcf_weight = 0.6
                    per_weight = 0.3
                    graham_weight = 0.1
                    asset_weight = 0.0
                elif sector in ['Financial Services', 'Real Estate', 'Utilities']:
                    dcf_weight = 0.3
                    per_weight = 0.3
                    graham_weight = 0.1
                    asset_weight = 0.3
            
            # Add available values with weights
            if per_based_value is not None:
                values_and_weights.append((per_based_value, per_weight))
            if graham_value is not None:
                values_and_weights.append((graham_value, graham_weight))
            if dcf_value is not None:
                values_and_weights.append((dcf_value, dcf_weight))
            if asset_based_value is not None:
                values_and_weights.append((asset_based_value, asset_weight))
            
            if not values_and_weights:
                return {
                    'success': False,
                    'message': f"Impossible de calculer la valeur intrinsèque pour {ticker}"
                }
            
            # Calculate weighted average
            total_weight = sum(w for _, w in values_and_weights)
            intrinsic_value = sum(v * w for v, w in values_and_weights) / total_weight
            
            # Calculate margin of safety
            sector_mos_map = {
                'Technology': 0.25, 'Healthcare': 0.25, 'Consumer Cyclical': 0.25,
                'Financial Services': 0.20, 'Communication Services': 0.25,
                'Industrials': 0.20, 'Consumer Defensive': 0.15, 'Energy': 0.30,
                'Basic Materials': 0.25, 'Utilities': 0.15, 'Real Estate': 0.20
            }
            
            mos_percentage = sector_mos_map.get(sector, 0.20)
            intrinsic_value_with_mos = intrinsic_value * (1 - mos_percentage)
            
            # Recommendation
            deviation = (current_price / intrinsic_value - 1) * 100
            
            if current_price <= intrinsic_value_with_mos:
                recommendation = "ACHAT FORT" if current_price < intrinsic_value_with_mos * 0.8 else "ACHETER"
            elif current_price <= intrinsic_value:
                recommendation = "RENFORCER"
            elif current_price <= intrinsic_value * 1.1:
                recommendation = "CONSERVER"
            elif current_price <= intrinsic_value * 1.3:
                recommendation = "ALLEGER"
            else:
                recommendation = "VENDRE"
            
            return {
                'success': True,
                'ticker': ticker,
                'current_price': current_price,
                'intrinsic_value': intrinsic_value,
                'intrinsic_value_with_mos': intrinsic_value_with_mos,
                'mos_percentage': mos_percentage * 100,
                'deviation': deviation,
                'recommendation': recommendation,
                'methods': {
                    'per_based': {'value': per_based_value, 'weight': per_weight if per_based_value else 0},
                    'graham': {'value': graham_value, 'weight': graham_weight if graham_value else 0},
                    'dcf': {'value': dcf_value, 'weight': dcf_weight if dcf_value else 0},
                    'asset_based': {'value': asset_based_value, 'weight': asset_weight if asset_based_value else 0}
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Erreur lors du calcul de la valeur intrinsèque pour {ticker}: {str(e)}"
            }
    
    def _calculate_dcf_value(
        self, ticker: str, income_stmt: pd.DataFrame, cashflow: pd.DataFrame,
        balance_sheet: pd.DataFrame, fundamental_data: Dict,
        current_price: float, market_structure: Optional[Dict] = None
    ) -> Optional[float]:
        """Calculate DCF value"""
        try:
            if not income_stmt.empty and len(income_stmt.columns) >= 3 and 'Free Cash Flow' in cashflow.index:
                fcf_values = []
                for col in cashflow.columns[:3]:
                    if 'Free Cash Flow' in cashflow.index:
                        fcf_values.append(cashflow.loc['Free Cash Flow', col])
                
                if len(fcf_values) == 3 and all(v is not None for v in fcf_values):
                    # Calculate growth rate
                    growth_rates = []
                    for i in range(len(fcf_values)-1):
                        if fcf_values[i] > 0 and fcf_values[i+1] > 0:
                            growth_rates.append((fcf_values[i] / fcf_values[i+1]) - 1)
                    
                    avg_growth_rate = np.median(growth_rates) if growth_rates else 0.05
                    avg_growth_rate = max(min(avg_growth_rate, 0.15), 0.02)
                    
                    # Get sector PE
                    sector_pe, sector_name = self._get_sector_pe_ratio(ticker, market_structure)
                    
                    # Calculate WACC
                    wacc = self._calculate_wacc(fundamental_data, income_stmt, balance_sheet)
                    
                    # Project future cash flows
                    fcf_current = fcf_values[0]
                    future_fcf = []
                    
                    for i in range(5):
                        fcf_value = fcf_current * ((1 + avg_growth_rate) ** (i + 1))
                        future_fcf.append(fcf_value / ((1 + wacc) ** (i + 1)))
                    
                    # Terminal value
                    terminal_growth = 0.02
                    terminal_fcf = fcf_current * ((1 + avg_growth_rate) ** 5) * (1 + terminal_growth)
                    terminal_value = terminal_fcf / (wacc - terminal_growth) / ((1 + wacc) ** 5)
                    
                    # Enterprise value
                    enterprise_value = sum(future_fcf) + terminal_value
                    
                    # Equity value
                    total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]] if 'Total Debt' in balance_sheet.index else 0
                    cash = balance_sheet.loc['Cash And Cash Equivalents', balance_sheet.columns[0]] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                    equity_value = enterprise_value + cash - total_debt
                    
                    # Value per share
                    market_cap = fundamental_data['Données de marché'].get('Capitalisation boursière', None)
                    if market_cap and current_price and current_price > 0:
                        shares_outstanding = market_cap / current_price
                        dcf_value = equity_value / shares_outstanding if shares_outstanding else None
                        return dcf_value
            
            return None
        except Exception as e:
            print(f"DCF calculation error: {e}")
            return None
    
    def _calculate_graham_value(self, fundamental_data: Dict, income_stmt: pd.DataFrame) -> Optional[float]:
        """Calculate Graham intrinsic value"""
        try:
            eps = fundamental_data['Données fondamentales'].get('BPA', None)
            
            if eps is not None and eps > 0:
                bond_yield = 0.04  # 4% for France
                growth_estimate = 0.05  # Default 5%
                
                if not income_stmt.empty and len(income_stmt.columns) >= 3 and 'Net Income' in income_stmt.index:
                    net_incomes = []
                    for col in income_stmt.columns[:3]:
                        if 'Net Income' in income_stmt.index:
                            net_incomes.append(income_stmt.loc['Net Income', col])
                    
                    if len(net_incomes) == 3 and all(ni is not None and ni > 0 for ni in net_incomes):
                        cagr = ((net_incomes[0] / net_incomes[2]) ** (1/2)) - 1
                        growth_estimate = min(max(cagr, 0.02), 0.15)
                
                graham_value = eps * (8.5 + 2 * growth_estimate * 100) * 4.4 / (bond_yield * 100)
                return graham_value
            
            return None
        except Exception:
            return None
    
    def _calculate_per_value(self, ticker: str, fundamental_data: Dict) -> Optional[float]:
        """Calculate PER-based valuation"""
        try:
            eps = fundamental_data['Données fondamentales'].get('BPA', None)
            
            if eps is not None and eps > 0:
                sector_per, _ = self._get_sector_pe_ratio(ticker)
                per_based_value = eps * sector_per
                
                per_fwd = fundamental_data['Données fondamentales'].get('PER (Forward)', None)
                if per_fwd is not None and per_fwd > 0:
                    per_fwd_value = eps * per_fwd
                    per_based_value = (per_based_value * 0.4 + per_fwd_value * 0.6)
                
                return per_based_value
            
            return None
        except Exception:
            return None
    
    def _calculate_asset_value(
        self, balance_sheet: pd.DataFrame, fundamental_data: Dict, sector: str
    ) -> Optional[float]:
        """Calculate asset-based valuation"""
        try:
            if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                total_assets = balance_sheet.loc['Total Assets', balance_sheet.columns[0]] if 'Total Assets' in balance_sheet.index else None
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', balance_sheet.columns[0]] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                goodwill = balance_sheet.loc['Goodwill And Intangible Assets', balance_sheet.columns[0]] if 'Goodwill And Intangible Assets' in balance_sheet.index else 0
                
                if total_assets is not None and total_liabilities is not None:
                    tangible_net_assets = total_assets - total_liabilities - goodwill
                    
                    shares_outstanding = fundamental_data['Données de marché'].get('Actions ordinaires', None)
                    
                    if shares_outstanding:
                        asset_value_per_share = tangible_net_assets / shares_outstanding
                        
                        sector_multiplier = {
                            'Financial Services': 1.2,
                            'Real Estate': 1.5,
                            'Utilities': 1.3,
                            'Basic Materials': 1.1,
                            'Energy': 1.1
                        }
                        
                        multiplier = sector_multiplier.get(sector, 1.0)
                        asset_based_value = asset_value_per_share * multiplier
                        return asset_based_value
            
            return None
        except Exception:
            return None
    
    def _get_sector_pe_ratio(self, ticker: str, market_structure: Optional[Dict] = None) -> Tuple[float, str]:
        """Get sector average PE ratio"""
        try:
            company = yf.Ticker(ticker)
            info = company.info
            sector = info.get('sector', None)
            
            default_sector_pes = {
                'Technology': 25.3,
                'Healthcare': 20.1,
                'Consumer Cyclical': 22.5,
                'Consumer Defensive': 18.7,
                'Financial Services': 14.2,
                'Industrials': 17.5,
                'Energy': 12.3,
                'Utilities': 16.8,
                'Real Estate': 19.2,
                'Communication Services': 21.4,
                'Basic Materials': 15.9
            }
            
            default_pe = default_sector_pes.get(sector, 15)
            return (default_pe, sector)
        except Exception:
            return (15, "Unknown")
    
    def _calculate_wacc(
        self, fundamental_data: Dict, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> float:
        """Calculate Weighted Average Cost of Capital"""
        try:
            market_cap = fundamental_data['Données de marché'].get('Capitalisation boursière', 0)
            total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]] if 'Total Debt' in balance_sheet.index else 0
            
            total_capital = market_cap + total_debt
            equity_weight = market_cap / total_capital if total_capital > 0 else 0.8
            debt_weight = total_debt / total_capital if total_capital > 0 else 0.2
            
            # Tax rate
            tax_rate = 0.25
            if 'Income Tax Expense' in income_stmt.index and 'Income Before Tax' in income_stmt.index:
                income_tax = income_stmt.loc['Income Tax Expense', income_stmt.columns[0]]
                income_before_tax = income_stmt.loc['Income Before Tax', income_stmt.columns[0]]
                if income_before_tax and income_before_tax != 0:
                    tax_rate = min(max(abs(income_tax / income_before_tax), 0.15), 0.35)
            
            # Cost of debt
            debt_cost = 0.05
            if 'Interest Expense' in income_stmt.index and total_debt > 0:
                interest_expense = abs(income_stmt.loc['Interest Expense', income_stmt.columns[0]])
                debt_cost = min(interest_expense / total_debt, 0.10)
            
            # Risk-free rate
            risk_free_rate = 0.04
            
            # Market risk premium
            market_risk_premium = 0.05
            
            # Beta
            beta = fundamental_data['Données de marché'].get('Beta', 1.0)
            if not beta or beta < 0.3:
                beta = 1.0
            
            # Cost of equity
            equity_cost = risk_free_rate + beta * market_risk_premium
            
            # WACC
            wacc = (equity_weight * equity_cost) + (debt_weight * debt_cost * (1 - tax_rate))
            wacc = max(min(wacc, 0.15), 0.06)
            
            return wacc
        except Exception:
            return 0.08  # Default WACC
    
    def analyze_competitive_advantage(
        self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
        cashflow: pd.DataFrame, ticker_info: Dict
    ) -> Dict:
        """Analyze competitive advantage (moat)"""
        competitive_analysis = {
            'existence': 'Indéterminé',
            'strength': 0,
            'factors': [],
            'explanation': ''
        }
        
        try:
            # Margin stability
            margin_stability = 0
            if not income_stmt.empty and 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                margins = []
                for col in income_stmt.columns[:min(5, len(income_stmt.columns))]:
                    gross_profit = income_stmt.loc['Gross Profit', col]
                    total_revenue = income_stmt.loc['Total Revenue', col]
                    if gross_profit is not None and total_revenue is not None and total_revenue != 0:
                        margins.append(gross_profit / total_revenue)
                
                if len(margins) >= 3:
                    std_margin = np.std(margins)
                    if std_margin < 0.03:
                        competitive_analysis['factors'].append('Marges très stables')
                        margin_stability = 2
                    elif std_margin < 0.05:
                        competitive_analysis['factors'].append('Marges assez stables')
                        margin_stability = 1
            
            # ROE analysis
            roe_quality = 0
            if not income_stmt.empty and not balance_sheet.empty and 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance_sheet.index:
                roes = []
                for i in range(min(5, len(income_stmt.columns), len(balance_sheet.columns))):
                    net_income = income_stmt.loc['Net Income', income_stmt.columns[i]]
                    equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[i]]
                    if net_income is not None and equity is not None and equity != 0:
                        roes.append(net_income / equity)
                
                if len(roes) >= 3:
                    avg_roe = np.mean(roes) * 100
                    if avg_roe > 15:
                        competitive_analysis['factors'].append(f'ROE élevé ({avg_roe:.1f}%)')
                        roe_quality = 2
                    elif avg_roe > 10:
                        competitive_analysis['factors'].append(f'ROE bon ({avg_roe:.1f}%)')
                        roe_quality = 1
            
            # ROIC analysis
            roic = self.calculate_roic(income_stmt, balance_sheet)
            roic_quality = 0
            if roic is not None:
                if roic > 15:
                    competitive_analysis['factors'].append(f'ROIC excellent ({roic:.1f}%)')
                    roic_quality = 2
                elif roic > 10:
                    competitive_analysis['factors'].append(f'ROIC bon ({roic:.1f}%)')
                    roic_quality = 1
            
            # FCF efficiency
            fcf_efficiency = 0
            if not cashflow.empty and not income_stmt.empty:
                if 'Free Cash Flow' in cashflow.index and 'Total Revenue' in income_stmt.index:
                    fcf = cashflow.loc['Free Cash Flow', cashflow.columns[0]]
                    revenue = income_stmt.loc['Total Revenue', income_stmt.columns[0]]
                    if fcf is not None and revenue is not None and revenue != 0:
                        fcf_to_sales = fcf / revenue
                        if fcf_to_sales > 0.15:
                            competitive_analysis['factors'].append(f'Forte conversion en FCF ({fcf_to_sales:.1%})')
                            fcf_efficiency = 2
                        elif fcf_to_sales > 0.08:
                            competitive_analysis['factors'].append(f'Bonne conversion en FCF ({fcf_to_sales:.1%})')
                            fcf_efficiency = 1
            
            # Calculate total score
            total_score = margin_stability + roe_quality + roic_quality + fcf_efficiency
            
            # Determine competitive advantage
            if total_score >= 5:
                competitive_analysis['existence'] = 'Fort'
                competitive_analysis['strength'] = 3
                competitive_analysis['explanation'] = "L'entreprise présente plusieurs indicateurs d'un avantage compétitif durable (moat)."
            elif total_score >= 3:
                competitive_analysis['existence'] = 'Modéré'
                competitive_analysis['strength'] = 2
                competitive_analysis['explanation'] = "L'entreprise montre certains signes d'un avantage compétitif."
            elif total_score >= 1:
                competitive_analysis['existence'] = 'Faible'
                competitive_analysis['strength'] = 1
                competitive_analysis['explanation'] = "L'entreprise présente peu d'indicateurs d'avantage compétitif durable."
            else:
                competitive_analysis['existence'] = 'Non détecté'
                competitive_analysis['strength'] = 0
                competitive_analysis['explanation'] = "Aucun signe clair d'avantage compétitif n'a été détecté."
        
        except Exception as e:
            competitive_analysis['explanation'] = f"Erreur lors de l'analyse: {str(e)}"
        
        return competitive_analysis
    
    def calculate_roic(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[float]:
        """Calculate Return on Invested Capital"""
        try:
            if not income_stmt.empty and not balance_sheet.empty:
                ebit = income_stmt.loc['EBIT', income_stmt.columns[0]] if 'EBIT' in income_stmt.index else None
                
                income_tax = income_stmt.loc['Income Tax Expense', income_stmt.columns[0]] if 'Income Tax Expense' in income_stmt.index else None
                income_before_tax = income_stmt.loc['Income Before Tax', income_stmt.columns[0]] if 'Income Before Tax' in income_stmt.index else None
                
                tax_rate = 0.25
                if income_tax is not None and income_before_tax is not None and income_before_tax != 0:
                    tax_rate = min(max(abs(income_tax / income_before_tax), 0.15), 0.35)
                
                total_debt = balance_sheet.loc['Total Debt', balance_sheet.columns[0]] if 'Total Debt' in balance_sheet.index else 0
                stockholders_equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]] if 'Stockholders Equity' in balance_sheet.index else None
                
                if ebit is not None and stockholders_equity is not None:
                    nopat = ebit * (1 - tax_rate)
                    invested_capital = total_debt + stockholders_equity
                    
                    if invested_capital > 0:
                        roic = (nopat / invested_capital) * 100
                        return roic
        except Exception:
            pass
        
        return None