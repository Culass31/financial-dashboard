# services/screening_service.py
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import time
from typing import Dict, List, Optional, Tuple

from .data_service import DataService
from .analysis_service import AnalysisService


class ScreeningService:
    """Service responsible for stock screening operations"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.analysis_service = AnalysisService(data_service)
    
    def buffett_graham_stock_screener(
        self,
        stocks_dict: Dict,
        # Financial criteria
        min_revenue_growth: float = 5,
        min_net_profit_margin: float = 10,
        min_roe: float = 15,
        max_debt_to_equity: float = 0.5,
        min_free_cashflow: float = 0,
        min_operating_margin: float = 15,
        min_earnings_stability_years: int = 5,
        # Consistency criteria
        min_gross_margin_consistency: float = 40,
        min_net_income_growth_consistency: float = 8,
        consistent_low_debt: bool = True,
        min_operating_cashflow_growth: float = 5,
        # Valuation criteria  
        max_pe_ratio: float = 25,
        max_pe_to_growth: float = 1.0,
        max_price_to_book: float = 1.5,
        # Operational efficiency criteria
        max_rd_sales_ratio: float = 0.05,
        max_capex_net_income_ratio: float = 0.5,
        max_interest_ebit_ratio: float = 0.20,
        min_interest_coverage_ratio: float = 5,
        # Additional criteria
        max_debt_ebitda_ratio: float = 5,
        min_dividend_years: int = 0,
        min_current_ratio: float = 1.5
    ) -> Tuple[pd.DataFrame, Dict]:
        """Screen stocks using Buffett-Graham criteria"""
        
        eligible_stocks = []
        detailed_analysis = {}
        
        for stock_name, stock_info in stocks_dict.items():
            ticker = stock_info['ticker']
            try:
                # Get stock data
                stock_data = yf.Ticker(ticker, session=self.session)
                info = stock_data.info
                time.sleep(0.5)  # Rate limiting
                
                # Get historical financials
                income_history, balance_history, cashflow_history = self.data_service.get_historical_financials(ticker)
                
                financial_metrics = {}
                consistency_metrics = {}
                intrinsic_value_metrics = {}
                
                # 1. Fundamental financial metrics
                financial_metrics['revenue_growth'] = info.get('earningsGrowth', 0) * 100
                financial_metrics['net_profit_margin'] = info.get('profitMargins', 0) * 100
                financial_metrics['roe'] = info.get('returnOnEquity', 0) * 100
                financial_metrics['operating_margin'] = info.get('operatingMargins', 0) * 100
                
                total_equity = info.get('totalEquity', 1)
                total_debt = info.get('totalDebt', 0)
                financial_metrics['debt_to_equity'] = total_debt / total_equity if total_equity != 0 else float('inf')
                
                financial_metrics['free_cashflow'] = info.get('freeCashflow', 0)
                financial_metrics['current_ratio'] = info.get('currentRatio', 0)
                
                ebit = info.get('ebit', 1)
                interest_expense = info.get('interestExpense', 0)
                financial_metrics['interest_ebit_ratio'] = interest_expense / ebit if ebit != 0 else float('inf')
                financial_metrics['interest_coverage_ratio'] = ebit / interest_expense if interest_expense != 0 else float('inf')
                
                rd_expenses = info.get('researchDevelopment', 0)
                total_revenue = info.get('totalRevenue', 1)
                financial_metrics['rd_sales_ratio'] = rd_expenses / total_revenue if total_revenue != 0 else 0
                
                ebitda = info.get('ebitda', 1)
                financial_metrics['debt_ebitda_ratio'] = total_debt / ebitda if ebitda != 0 else float('inf')
                
                # 2. Valuation metrics
                price = info.get('currentPrice', 0)
                earnings_per_share = info.get('trailingEPS', 0)
                book_value_per_share = info.get('bookValue', 0)
                
                intrinsic_value_metrics['pe_ratio'] = price / earnings_per_share if earnings_per_share != 0 else float('inf')
                intrinsic_value_metrics['price_to_book'] = price / book_value_per_share if book_value_per_share != 0 else float('inf')
                
                growth_5y = info.get('earningsGrowth', 0) * 100
                intrinsic_value_metrics['pe_to_growth'] = intrinsic_value_metrics['pe_ratio'] / growth_5y if growth_5y > 0 else float('inf')
                
                # 3. Consistency analysis
                consistent_gross_margin, gross_margin_reason = self._analyze_gross_margin_consistency(
                    income_history, min_gross_margin_consistency
                )
                consistency_metrics['gross_margin_consistency'] = gross_margin_reason
                
                consistent_net_income_growth, net_income_growth_reason = self._analyze_net_income_growth(
                    income_history, min_net_income_growth_consistency
                )
                consistency_metrics['net_income_growth_consistency'] = net_income_growth_reason
                
                consistent_low_debt_result, debt_consistency_reason = self._analyze_debt_consistency(
                    balance_history, max_debt_to_equity
                )
                consistency_metrics['debt_consistency'] = debt_consistency_reason
                
                consistent_operating_cashflow, operating_cashflow_reason = self._analyze_cashflow_growth(
                    cashflow_history, min_operating_cashflow_growth
                )
                consistency_metrics['operating_cashflow_consistency'] = operating_cashflow_reason
                
                earnings_stability = self._count_consecutive_positive_earnings(income_history)
                consistency_metrics['earnings_stability_years'] = earnings_stability
                
                dividend_years = self._count_dividend_years(cashflow_history)
                consistency_metrics['dividend_years'] = dividend_years
                
                capex_net_income_ratio = self._calculate_capex_to_net_income_ratio(
                    cashflow_history, income_history
                )
                financial_metrics['capex_net_income_ratio'] = capex_net_income_ratio
                
                # 4. Check all criteria
                all_criteria_met = (
                    financial_metrics['revenue_growth'] >= min_revenue_growth and
                    financial_metrics['net_profit_margin'] >= min_net_profit_margin and
                    financial_metrics['roe'] >= min_roe and
                    financial_metrics['debt_to_equity'] <= max_debt_to_equity and
                    financial_metrics['free_cashflow'] >= min_free_cashflow and
                    financial_metrics['operating_margin'] >= min_operating_margin and
                    financial_metrics['current_ratio'] >= min_current_ratio and
                    financial_metrics['rd_sales_ratio'] <= max_rd_sales_ratio and
                    financial_metrics['interest_ebit_ratio'] <= max_interest_ebit_ratio and
                    financial_metrics['interest_coverage_ratio'] >= min_interest_coverage_ratio and
                    financial_metrics['debt_ebitda_ratio'] <= max_debt_ebitda_ratio and
                    consistent_gross_margin and
                    consistent_net_income_growth and
                    (not consistent_low_debt or consistent_low_debt_result) and
                    consistent_operating_cashflow and
                    earnings_stability >= min_earnings_stability_years and
                    (min_dividend_years == 0 or dividend_years >= min_dividend_years) and
                    capex_net_income_ratio <= max_capex_net_income_ratio and
                    intrinsic_value_metrics['pe_ratio'] <= max_pe_ratio and
                    intrinsic_value_metrics['pe_to_growth'] <= max_pe_to_growth and
                    intrinsic_value_metrics['price_to_book'] <= max_price_to_book
                )
                
                if all_criteria_met:
                    # Get ISIN
                    isin = self.data_service.get_isin_for_ticker(ticker)
                    region = stock_info.get('region', 'N/A')
                    country = stock_info.get('pays', 'N/A')
                    
                    # Calculate Buffett-Graham score
                    buffett_graham_score = self._calculate_buffett_graham_score(
                        financial_metrics, 
                        consistency_metrics, 
                        intrinsic_value_metrics
                    )
                    
                    # Identify strengths and weaknesses
                    strengths, weaknesses = self._identify_strengths_weaknesses(
                        financial_metrics, 
                        consistency_metrics, 
                        intrinsic_value_metrics
                    )
                    
                    stock_details = {
                        'Nom': stock_name,
                        'Ticker': ticker,
                        'ISIN': isin,
                        'Région': region,
                        'Pays': country,
                        'Score Buffett-Graham': f"{buffett_graham_score}/100",
                        'Croissance CA (%)': f"{financial_metrics['revenue_growth']:.2f}",
                        'Marge nette (%)': f"{financial_metrics['net_profit_margin']:.2f}",
                        'ROE (%)': f"{financial_metrics['roe']:.2f}",
                        'Dette/Fonds propres': f"{financial_metrics['debt_to_equity']:.2f}",
                        'P/E': f"{intrinsic_value_metrics['pe_ratio']:.2f}",
                        'PEG': f"{intrinsic_value_metrics['pe_to_growth']:.2f}",
                        'P/B': f"{intrinsic_value_metrics['price_to_book']:.2f}",
                        'Free Cash Flow': self._format_number(financial_metrics['free_cashflow']),
                        'Années bénéfices positifs': consistency_metrics['earnings_stability_years'],
                        'Marge Brute Consistante': consistency_metrics['gross_margin_consistency'],
                        'Croissance RN Consistante': consistency_metrics['net_income_growth_consistency'],
                        'Croissance FCF': consistency_metrics['operating_cashflow_consistency'],
                        'Forces': strengths,
                        'Faiblesses': weaknesses
                    }
                    eligible_stocks.append(stock_details)
                    
                    detailed_analysis[ticker] = {
                        'financial_metrics': financial_metrics,
                        'consistency_metrics': consistency_metrics,
                        'intrinsic_value_metrics': intrinsic_value_metrics,
                        'strengths': strengths,
                        'weaknesses': weaknesses,
                        'score': buffett_graham_score
                    }
            
            except Exception as e:
                st.warning(f"Error for {stock_name}: {e}")
        
        return pd.DataFrame(eligible_stocks), detailed_analysis
    
    def _analyze_gross_margin_consistency(
        self, 
        income_history: pd.DataFrame, 
        min_value: float
    ) -> Tuple[bool, str]:
        """Analyze gross margin consistency"""
        if income_history.empty:
            return False, "N/A"
        
        if 'Gross Profit' in income_history.index and 'Total Revenue' in income_history.index:
            gross_margins = ((income_history.loc['Gross Profit'] / income_history.loc['Total Revenue']) * 100).dropna()
            if not gross_margins.empty:
                return self._analyze_consistency(gross_margins, min_value=min_value)
        
        return False, "N/A"
    
    def _analyze_net_income_growth(
        self, 
        income_history: pd.DataFrame, 
        min_growth: float
    ) -> Tuple[bool, str]:
        """Analyze net income growth consistency"""
        if income_history.empty:
            return False, "N/A"
        
        if 'Net Income' in income_history.index:
            net_incomes = income_history.loc['Net Income'].dropna()
            if len(net_incomes) > 1:
                growth_rates = net_incomes.pct_change().dropna() * 100
                return self._analyze_consistency(growth_rates, min_growth=min_growth, min_length=4)
        
        return False, "N/A"
    
    def _analyze_debt_consistency(
        self, 
        balance_history: pd.DataFrame, 
        max_value: float
    ) -> Tuple[bool, str]:
        """Analyze debt consistency"""
        if balance_history.empty:
            return False, "N/A"
        
        if 'Total Debt' in balance_history.index and 'Total Stockholder Equity' in balance_history.index:
            debt_equity_ratios = (balance_history.loc['Total Debt'] / balance_history.loc['Total Stockholder Equity']).dropna()
            if not debt_equity_ratios.empty:
                return self._analyze_consistency(debt_equity_ratios, max_value=max_value)
        
        return False, "N/A"
    
    def _analyze_cashflow_growth(
        self, 
        cashflow_history: pd.DataFrame, 
        min_growth: float
    ) -> Tuple[bool, str]:
        """Analyze operating cash flow growth"""
        if cashflow_history.empty:
            return False, "N/A"
        
        if 'Operating Cash Flow' in cashflow_history.index:
            operating_cashflows = cashflow_history.loc['Operating Cash Flow'].dropna()
            if len(operating_cashflows) > 1:
                growth_rates = operating_cashflows.pct_change().dropna() * 100
                return self._analyze_consistency(growth_rates, min_growth=min_growth)
        
        return False, "N/A"
    
    def _analyze_consistency(
        self, 
        series: pd.Series, 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_growth: Optional[float] = None,
        min_length: int = 5
    ) -> Tuple[bool, str]:
        """Generic consistency analysis"""
        if series.empty or len(series) < min_length:
            return False, f"Données insuffisantes (min. {min_length} ans requis)"
        
        series = series.sort_index()
        
        if min_value is not None:
            all_above_min = all(value >= min_value for value in series)
            avg_value = series.mean()
            if all_above_min:
                return True, f"Consistante et élevée (moy. {avg_value:.2f}%)"
            else:
                pct_above = sum(1 for v in series if v >= min_value) / len(series) * 100
                return False, f"Inconstante ({pct_above:.1f}% au-dessus du min.)"
        
        if max_value is not None:
            all_below_max = all(value <= max_value for value in series)
            avg_value = series.mean()
            if all_below_max:
                return True, f"Constamment faible (moy. {avg_value:.2f})"
            else:
                pct_below = sum(1 for v in series if v <= max_value) / len(series) * 100
                return False, f"Inconstante ({pct_below:.1f}% en-dessous du max.)"
        
        if min_growth is not None:
            avg_growth = series.mean()
            growth_consistency = series.std() / (abs(avg_growth) + 1e-10)
            
            if avg_growth >= min_growth and growth_consistency < 2:
                return True, f"Croissance consistante (moy. {avg_growth:.2f}%)"
            elif avg_growth >= min_growth:
                return False, f"Croissance volatile (moy. {avg_growth:.2f}%)"
            else:
                return False, f"Croissance insuffisante (moy. {avg_growth:.2f}%)"
        
        mean_value = series.mean()
        std_value = series.std()
        cv = std_value / (abs(mean_value) + 1e-10)
        
        if cv < 0.3:
            return True, f"Très consistante (CV={cv:.2f})"
        elif cv < 0.5:
            return True, f"Assez consistante (CV={cv:.2f})"
        else:
            return False, f"Inconstante (CV={cv:.2f})"
    
    def _count_consecutive_positive_earnings(self, income_history: pd.DataFrame) -> int:
        """Count consecutive years of positive earnings"""
        if income_history.empty or 'Net Income' not in income_history.index:
            return 0
        
        net_incomes = income_history.loc['Net Income'].sort_index(ascending=False)
        count = 0
        for value in net_incomes:
            if value > 0:
                count += 1
            else:
                break
        
        return count
    
    def _count_dividend_years(self, cashflow_history: pd.DataFrame) -> int:
        """Count consecutive years of dividend payments"""
        if cashflow_history.empty or 'Dividends Paid' not in cashflow_history.index:
            return 0
        
        dividends = cashflow_history.loc['Dividends Paid'].abs().sort_index(ascending=False)
        count = 0
        for value in dividends:
            if value > 0:
                count += 1
            else:
                break
        
        return count
    
    def _calculate_capex_to_net_income_ratio(
        self, 
        cashflow_history: pd.DataFrame, 
        income_history: pd.DataFrame
    ) -> float:
        """Calculate CAPEX to Net Income ratio"""
        if cashflow_history.empty or income_history.empty:
            return float('inf')
        
        if 'Capital Expenditure' in cashflow_history.index and 'Net Income' in income_history.index:
            common_dates = set(cashflow_history.columns).intersection(set(income_history.columns))
            if common_dates:
                capex = cashflow_history.loc['Capital Expenditure', list(common_dates)].abs()
                net_income = income_history.loc['Net Income', list(common_dates)]
                ratio = (capex / net_income).mean()
                return ratio
        
        return float('inf')
    
    def _calculate_buffett_graham_score(
        self, 
        financial_metrics: Dict, 
        consistency_metrics: Dict, 
        intrinsic_value_metrics: Dict
    ) -> int:
        """Calculate overall Buffett-Graham score"""
        score = 0
        
        # Financial metrics (40 points max)
        score += min(financial_metrics['roe'] / 2, 10)
        score += min(financial_metrics['net_profit_margin'] / 2, 10)
        score += min(15 - financial_metrics['debt_to_equity'] * 5, 10)
        score += min(financial_metrics['operating_margin'] / 3, 10)
        
        # Consistency metrics (40 points max)
        if 'earnings_stability_years' in consistency_metrics:
            score += min(consistency_metrics['earnings_stability_years'] * 2, 10)
        
        if 'gross_margin_consistency' in consistency_metrics and "Consistante" in consistency_metrics['gross_margin_consistency']:
            score += 10
        
        if 'net_income_growth_consistency' in consistency_metrics and "Consistante" in consistency_metrics['net_income_growth_consistency']:
            score += 10
        
        if 'operating_cashflow_consistency' in consistency_metrics and "Consistant" in consistency_metrics['operating_cashflow_consistency']:
            score += 10
        
        # Valuation metrics (20 points max)
        if intrinsic_value_metrics['pe_ratio'] < float('inf'):
            score += min(20 - intrinsic_value_metrics['pe_ratio'] / 2, 7)
        
        if intrinsic_value_metrics['pe_to_growth'] < float('inf'):
            score += min(10 - intrinsic_value_metrics['pe_to_growth'] * 7, 7)
        
        if intrinsic_value_metrics['price_to_book'] < float('inf'):
            score += min(10 - intrinsic_value_metrics['price_to_book'] * 3, 6)
        
        return max(0, min(round(score), 100))
    
    def _identify_strengths_weaknesses(
        self, 
        financial_metrics: Dict, 
        consistency_metrics: Dict, 
        intrinsic_value_metrics: Dict
    ) -> Tuple[str, str]:
        """Identify strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze strengths
        if financial_metrics['roe'] > 20:
            strengths.append("ROE excellent")
        
        if financial_metrics['net_profit_margin'] > 15:
            strengths.append("Marge nette élevée")
        
        if financial_metrics['debt_to_equity'] < 0.3:
            strengths.append("Endettement très faible")
        
        if 'earnings_stability_years' in consistency_metrics and consistency_metrics['earnings_stability_years'] > 8:
            strengths.append("Bénéfices très stables")
        
        if intrinsic_value_metrics['pe_to_growth'] < 0.8:
            strengths.append("PEG ratio attractif")
        
        # Analyze weaknesses
        if financial_metrics['roe'] < 10:
            weaknesses.append("ROE insuffisant")
        
        if financial_metrics['debt_to_equity'] > 1:
            weaknesses.append("Endettement élevé")
        
        if intrinsic_value_metrics['pe_ratio'] > 30:
            weaknesses.append("Valorisation élevée")
        
        if 'earnings_stability_years' in consistency_metrics and consistency_metrics['earnings_stability_years'] < 3:
            weaknesses.append("Bénéfices instables")
        
        return ", ".join(strengths[:3]), ", ".join(weaknesses[:3])
    
    def _format_number(self, num):
        """Format large numbers"""
        if num is None:
            return 'N/A'
        if isinstance(num, str):
            return num