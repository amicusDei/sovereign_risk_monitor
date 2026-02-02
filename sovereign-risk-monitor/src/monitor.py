"""
Sovereign Risk Monitor - Main Interface
Combines data fetching, scoring, and analysis into unified API.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
import logging

from src.data.fetchers import EikonFetcher, create_fetcher
from src.scoring.aggregator import RiskScorer, create_scorer
from src.analysis.crisis_comparison import CrisisAnalyzer, create_analyzer
from src.indicators.early_warning import EarlyWarningSystem, create_early_warning

logger = logging.getLogger(__name__)


class SovereignRiskMonitor:
    """
    Main interface for sovereign risk monitoring.

    Provides:
    - Real-time risk scores and grades for 17 countries
    - Crisis comparison analysis
    - Historical context for current market conditions
    """

    def __init__(self, app_key: Optional[str] = None):
        self.fetcher = create_fetcher(app_key) if app_key else create_fetcher()
        self.scorer = create_scorer()
        self.analyzer = create_analyzer()
        self.early_warning = create_early_warning()

        # Country metadata
        self.countries = self.fetcher.countries

    def get_risk_dashboard(self) -> pd.DataFrame:
        """
        Get current risk dashboard for all countries.

        Returns:
            DataFrame with scores, grades, and stress levels
        """
        # Fetch live data
        snapshot = self.fetcher.get_live_snapshot()

        if snapshot.empty:
            logger.error("No data received from Eikon")
            return pd.DataFrame()

        # Score all countries
        results = []

        for _, row in snapshot.iterrows():
            country = row['country']
            region = self.countries[country].get('region', 'other')

            # Calculate scores
            scored = self.scorer.score_country(
                country=country,
                yield_pct=row['yield_pct'],
                spread_bps=row.get('spread_bps', 0) or 0,
                cds_bps=row.get('cds_spread_bps'),
                region=region
            )

            results.append({
                'country': country,
                'name': row['name'],
                'yield_pct': round(row['yield_pct'], 2),
                'spread_bps': round(row.get('spread_bps', 0) or 0, 0),
                'cds_bps': round(row['cds_spread_bps'], 0) if pd.notna(row.get('cds_spread_bps')) else None,
                'market_score': scored['market_score'],
                'composite_score': scored['composite_score'],
                'grade': scored['grade'],
                'stress': scored['stress_levels']['spread'],
            })

        df = pd.DataFrame(results)

        # Sort by risk score (highest first)
        df = df.sort_values('composite_score', ascending=False, na_position='last')

        return df

    def analyze_country(self, country: str) -> Dict:
        """
        Get detailed risk analysis for a single country.

        Args:
            country: ISO2 country code (e.g., 'IT', 'GR')

        Returns:
            Dict with full analysis including crisis comparisons
        """
        # Fetch current data
        snapshot = self.fetcher.get_live_snapshot(countries=[country])

        if snapshot.empty:
            return {'error': f'No data for {country}'}

        row = snapshot.iloc[0]
        region = self.countries[country].get('region', 'other')

        # Get scored assessment
        scored = self.scorer.score_country(
            country=country,
            yield_pct=row['yield_pct'],
            spread_bps=row.get('spread_bps', 0) or 0,
            cds_bps=row.get('cds_spread_bps'),
            region=region
        )

        # Get crisis analysis
        crisis_report = self.analyzer.generate_risk_report(
            country=country,
            yield_pct=row['yield_pct'],
            spread_bps=row.get('spread_bps', 0) or 0,
            cds_bps=row.get('cds_spread_bps'),
        )

        return {
            'country': country,
            'name': self.countries[country]['name'],
            'timestamp': datetime.now().isoformat(),

            # Current levels
            'yield_10y': round(row['yield_pct'], 3),
            'spread_vs_benchmark_bps': round(row.get('spread_bps', 0) or 0, 1),
            'cds_5y_bps': round(row['cds_spread_bps'], 1) if pd.notna(row.get('cds_spread_bps')) else None,

            # Scores
            'scores': {
                'spread': scored['spread_score'],
                'cds': scored['cds_score'],
                'market': scored['market_score'],
                'composite': scored['composite_score'],
            },

            # Grade
            'grade': scored['grade'],
            'grade_description': scored['grade_description'],

            # Stress assessment
            'stress_regime': crisis_report['stress_regime'],

            # Crisis context
            'crisis_comparisons': crisis_report['crisis_comparisons'],

            # Key insights
            'insights': crisis_report['insights'],
        }

    def compare_to_crisis(
        self,
        crisis: str,
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare current levels to a specific historical crisis.

        Args:
            crisis: Crisis key ('global_financial_crisis', 'eurozone_debt_crisis',
                   'covid_crash', 'uk_gilt_crisis')
            countries: Countries to compare. None = affected countries from that crisis

        Returns:
            DataFrame with current vs peak comparison
        """
        crisis_data = self.analyzer.crisis_periods.get(crisis)
        if not crisis_data:
            return pd.DataFrame({'error': [f'Unknown crisis: {crisis}']})

        if countries is None:
            countries = crisis_data.get('affected_countries', [])

        # Fetch current data
        snapshot = self.fetcher.get_live_snapshot(countries=countries)

        if snapshot.empty:
            return pd.DataFrame()

        results = []
        for _, row in snapshot.iterrows():
            country = row['country']
            peaks = crisis_data.get('peak_levels', {}).get(country, {})

            if not peaks:
                continue

            current_spread = row.get('spread_bps', 0) or 0
            current_cds = row.get('cds_spread_bps')

            result = {
                'country': country,
                'crisis': crisis_data['name'],
                'current_spread': round(current_spread, 0),
                'peak_spread': peaks.get('spread_vs_de'),
                'current_cds': round(current_cds, 0) if pd.notna(current_cds) else None,
                'peak_cds': peaks.get('cds_5y'),
            }

            # Calculate % of peak
            if peaks.get('spread_vs_de'):
                result['spread_pct_of_peak'] = round(
                    100 * current_spread / peaks['spread_vs_de'], 1
                )

            if pd.notna(current_cds) and peaks.get('cds_5y'):
                result['cds_pct_of_peak'] = round(
                    100 * current_cds / peaks['cds_5y'], 1
                )

            results.append(result)

        return pd.DataFrame(results)

    def get_alert_countries(self, threshold: str = 'elevated') -> pd.DataFrame:
        """
        Get countries currently above a stress threshold.

        Args:
            threshold: 'elevated', 'stressed', 'crisis', 'extreme'

        Returns:
            DataFrame with countries above threshold
        """
        dashboard = self.get_risk_dashboard()

        if dashboard.empty:
            return dashboard

        threshold_map = {
            'elevated': ['elevated', 'stressed', 'crisis', 'extreme'],
            'stressed': ['stressed', 'crisis', 'extreme'],
            'crisis': ['crisis', 'extreme'],
            'extreme': ['extreme'],
        }

        levels = threshold_map.get(threshold, ['elevated', 'stressed', 'crisis', 'extreme'])
        alerts = dashboard[dashboard['stress'].isin(levels)]

        return alerts

    def print_dashboard(self):
        """Print formatted risk dashboard to console."""
        df = self.get_risk_dashboard()

        if df.empty:
            print("No data available")
            return

        print("\n" + "=" * 80)
        print("SOVEREIGN RISK MONITOR - Live Dashboard")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Format for display
        display_df = df[['country', 'name', 'yield_pct', 'spread_bps', 'cds_bps',
                         'composite_score', 'grade', 'stress']].copy()

        display_df.columns = ['Code', 'Country', 'Yield%', 'Spread', 'CDS',
                              'Score', 'Grade', 'Stress']

        print(display_df.to_string(index=False))

        # Summary
        print("\n" + "-" * 80)
        alerts = self.get_alert_countries('elevated')
        if not alerts.empty:
            print(f"⚠️  {len(alerts)} countries with elevated stress: {', '.join(alerts['country'].tolist())}")
        else:
            print("✓ All countries within normal stress levels")

        print("=" * 80)

    def get_early_warnings(
        self,
        countries: Optional[List[str]] = None,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get early warning signals based on historical trends.

        Args:
            countries: List of country codes. None = periphery + stressed.
            lookback_days: Historical data to analyze.

        Returns:
            DataFrame with early warning signals per country.
        """
        if countries is None:
            # Focus on historically vulnerable countries
            countries = ['IT', 'ES', 'PT', 'GR', 'IE', 'FR', 'GB']

        results = []

        for country in countries:
            logger.info(f"Analyzing early warning signals for {country}...")

            try:
                # Fetch historical yields
                hist = self.fetcher.get_historical_yields(
                    countries=[country],
                    start_date=(datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                )

                if hist.empty:
                    continue

                # Calculate spread vs benchmark
                de_hist = self.fetcher.get_historical_yields(
                    countries=['DE'],
                    start_date=(datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                )

                if not de_hist.empty:
                    merged = hist.merge(
                        de_hist[['date', 'yield_pct']].rename(columns={'yield_pct': 'de_yield'}),
                        on='date',
                        how='left'
                    )
                    merged['spread_bps'] = (merged['yield_pct'] - merged['de_yield']) * 100
                else:
                    merged = hist.copy()
                    merged['spread_bps'] = merged['yield_pct'] * 100  # Fallback

                # Generate signals
                signals = self.early_warning.generate_signals(merged, country)

                results.append({
                    'country': country,
                    'name': self.countries[country]['name'],
                    'alert_level': signals.get('alert_level', 'unknown'),
                    'trend': signals.get('trend', 'unknown'),
                    'momentum_5d': signals.get('details', {}).get('spread_momentum_5d_bps'),
                    'momentum_20d': signals.get('details', {}).get('spread_momentum_20d_bps'),
                    'vol_zscore': signals.get('details', {}).get('volatility_zscore'),
                    'percentile': signals.get('details', {}).get('spread_percentile_1y'),
                    'signals': '; '.join(signals.get('signals', [])),
                })

            except Exception as e:
                logger.warning(f"Error analyzing {country}: {e}")

        return pd.DataFrame(results)

    def print_early_warnings(self, countries: Optional[List[str]] = None):
        """Print early warning dashboard to console."""
        df = self.get_early_warnings(countries)

        if df.empty:
            print("No early warning data available")
            return

        print("\n" + "=" * 80)
        print("EARLY WARNING SYSTEM - Trend Analysis")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Sort by alert level
        level_order = {'critical': 0, 'alert': 1, 'warning': 2, 'normal': 3, 'unknown': 4}
        df['_sort'] = df['alert_level'].map(level_order)
        df = df.sort_values('_sort').drop(columns=['_sort'])

        # Format display
        display_cols = ['country', 'name', 'alert_level', 'trend', 'momentum_5d', 'momentum_20d', 'percentile']
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        display_df.columns = ['Code', 'Country', 'Alert', 'Trend', '5D Δ', '20D Δ', 'Pctl']

        print(display_df.to_string(index=False))

        # Show detailed signals
        alerts = df[df['alert_level'].isin(['warning', 'alert', 'critical'])]
        if not alerts.empty:
            print("\n" + "-" * 80)
            print("⚠️  ACTIVE SIGNALS:")
            for _, row in alerts.iterrows():
                if row['signals']:
                    print(f"\n  {row['country']} ({row['name']}):")
                    for sig in row['signals'].split('; '):
                        print(f"    • {sig}")

        print("\n" + "=" * 80)


def create_monitor(app_key: Optional[str] = None) -> SovereignRiskMonitor:
    """Factory function to create monitor."""
    return SovereignRiskMonitor(app_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = create_monitor()
    monitor.print_dashboard()

    # Detailed analysis example
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: Italy")
    print("=" * 80)

    analysis = monitor.analyze_country('IT')
    print(f"\nGrade: {analysis['grade']} - {analysis['grade_description']}")
    print(f"Composite Score: {analysis['scores']['composite']}/100")
    print(f"\nStress Regime: {analysis['stress_regime']['overall_regime']}")
    print(f"Historical Analog: {analysis['stress_regime']['historical_analog']}")

    print("\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")

    if analysis['crisis_comparisons']:
        print("\nCrisis Comparisons:")
        for comp in analysis['crisis_comparisons']:
            print(f"  vs {comp['crisis']}:")
            if 'spread_pct_of_peak' in comp:
                print(f"    Spread: {comp['spread_pct_of_peak']}% of peak")
            if 'cds_pct_of_peak' in comp:
                print(f"    CDS: {comp['cds_pct_of_peak']}% of peak")

    # Early Warning System
    print("\n")
    monitor.print_early_warnings(countries=['IT', 'FR', 'ES', 'GR', 'GB'])
