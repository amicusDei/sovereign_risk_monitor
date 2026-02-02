"""
Sovereign Risk Monitor - Crisis Comparison Analysis
Compares current market conditions to historical crisis periods.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CrisisAnalyzer:
    """Analyzes current conditions vs historical crisis periods."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config"

        with open(config_path / "crisis_benchmarks.yaml", 'r') as f:
            self.config = yaml.safe_load(f)

        self.crisis_periods = self.config['crisis_periods']
        self.stress_thresholds = self.config['stress_thresholds']

    def get_crisis_peak(self, crisis_name: str, country: str) -> Optional[Dict]:
        """Get peak stress levels for a country during a specific crisis."""
        if crisis_name not in self.crisis_periods:
            return None

        crisis = self.crisis_periods[crisis_name]
        peaks = crisis.get('peak_levels', {})

        if country in peaks:
            return {
                'crisis': crisis['name'],
                'peak_date': crisis['peak'],
                **peaks[country]
            }
        return None

    def compare_to_crisis(
        self,
        country: str,
        current_yield: float,
        current_spread: float,
        current_cds: Optional[float] = None
    ) -> List[Dict]:
        """
        Compare current levels to all relevant historical crises.

        Returns:
            List of comparisons with % of crisis peak
        """
        comparisons = []

        for crisis_key, crisis_data in self.crisis_periods.items():
            if country not in crisis_data.get('affected_countries', []):
                continue

            peaks = crisis_data.get('peak_levels', {}).get(country, {})
            if not peaks:
                continue

            comparison = {
                'crisis': crisis_data['name'],
                'crisis_period': f"{crisis_data['start']} to {crisis_data['end']}",
                'peak_date': crisis_data['peak'],
            }

            # Yield comparison
            if 'yield_10y' in peaks and peaks['yield_10y']:
                comparison['peak_yield'] = peaks['yield_10y']
                comparison['current_yield'] = current_yield
                comparison['yield_pct_of_peak'] = round(
                    100 * current_yield / peaks['yield_10y'], 1
                )

            # Spread comparison
            if 'spread_vs_de' in peaks and peaks['spread_vs_de']:
                comparison['peak_spread'] = peaks['spread_vs_de']
                comparison['current_spread'] = current_spread
                comparison['spread_pct_of_peak'] = round(
                    100 * current_spread / peaks['spread_vs_de'], 1
                )

            # CDS comparison
            if pd.notna(current_cds) and 'cds_5y' in peaks and peaks['cds_5y']:
                comparison['peak_cds'] = peaks['cds_5y']
                comparison['current_cds'] = current_cds
                comparison['cds_pct_of_peak'] = round(
                    100 * current_cds / peaks['cds_5y'], 1
                )

            comparisons.append(comparison)

        return comparisons

    def classify_stress_regime(
        self,
        spread_bps: float,
        cds_bps: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        Classify current market regime based on stress indicators.

        Returns:
            Dict with regime classification and nearest historical analog
        """
        # Classify each indicator
        spread_level = self._classify_value('yield_spread_bps', spread_bps)
        cds_level = self._classify_value('cds_spread_bps', cds_bps) if pd.notna(cds_bps) else None
        vol_level = self._classify_value('yield_volatility_30d', volatility) if pd.notna(volatility) else None

        # Overall regime (worst of the components)
        levels = ['normal', 'elevated', 'stressed', 'crisis', 'extreme']
        components = [spread_level]
        if cds_level is not None:
            components.append(cds_level)
        if vol_level is not None:
            components.append(vol_level)

        max_level = max(components, key=lambda x: levels.index(x) if x in levels else -1)

        # Find nearest historical analog
        analog = self._find_historical_analog(spread_bps, cds_bps)

        return {
            'spread_regime': spread_level,
            'cds_regime': cds_level,
            'volatility_regime': vol_level,
            'overall_regime': max_level,
            'historical_analog': analog,
        }

    def _classify_value(self, indicator: str, value: float) -> str:
        """Classify a value into its stress category."""
        if value is None:
            return None

        thresholds = self.stress_thresholds.get(indicator, {})

        for level, bounds in thresholds.items():
            lower, upper = bounds
            if upper is None:
                if value >= lower:
                    return level
            elif lower <= value < upper:
                return level

        return 'normal'

    def _find_historical_analog(
        self,
        spread_bps: float,
        cds_bps: Optional[float]
    ) -> Optional[str]:
        """Find the closest historical crisis analog to current conditions."""
        if spread_bps < 100:
            return "Normal conditions - no crisis analog"

        if spread_bps < 200:
            return "Early COVID-19 (Feb 2020) or post-ECB intervention (2013-2015)"

        if spread_bps < 400:
            if pd.notna(cds_bps) and cds_bps > 200:
                return "Pre-Draghi Eurozone crisis (early 2012)"
            return "COVID peak (March 2020) or late Eurozone crisis (H2 2012)"

        if spread_bps < 600:
            return "Eurozone crisis peak (Italy/Spain July 2012)"

        return "Severe stress - comparable to peripheral crisis peaks"

    def generate_risk_report(
        self,
        country: str,
        yield_pct: float,
        spread_bps: float,
        cds_bps: Optional[float] = None,
        volatility: Optional[float] = None,
        debt_to_gdp: Optional[float] = None
    ) -> Dict:
        """
        Generate comprehensive risk report for a country.

        Returns:
            Dict with all analysis components
        """
        # Crisis comparisons
        crisis_comps = self.compare_to_crisis(
            country=country,
            current_yield=yield_pct,
            current_spread=spread_bps,
            current_cds=cds_bps
        )

        # Stress regime
        regime = self.classify_stress_regime(
            spread_bps=spread_bps,
            cds_bps=cds_bps,
            volatility=volatility
        )

        # Key insights
        insights = self._generate_insights(
            country=country,
            spread_bps=spread_bps,
            cds_bps=cds_bps,
            crisis_comps=crisis_comps,
            regime=regime,
            debt_to_gdp=debt_to_gdp
        )

        return {
            'country': country,
            'timestamp': datetime.now().isoformat(),
            'current_levels': {
                'yield_10y': yield_pct,
                'spread_bps': spread_bps,
                'cds_5y': cds_bps,
                'volatility_30d': volatility,
                'debt_to_gdp': debt_to_gdp,
            },
            'stress_regime': regime,
            'crisis_comparisons': crisis_comps,
            'insights': insights,
        }

    def _generate_insights(
        self,
        country: str,
        spread_bps: float,
        cds_bps: Optional[float],
        crisis_comps: List[Dict],
        regime: Dict,
        debt_to_gdp: Optional[float]
    ) -> List[str]:
        """Generate human-readable insights."""
        insights = []

        # Overall regime
        regime_text = regime['overall_regime']
        if regime_text == 'normal':
            insights.append(f"{country} markets show normal stress levels")
        elif regime_text == 'elevated':
            insights.append(f"{country} shows elevated but manageable stress")
        elif regime_text == 'stressed':
            insights.append(f"{country} is under significant market stress")
        elif regime_text in ['crisis', 'extreme']:
            insights.append(f"WARNING: {country} shows crisis-level stress indicators")

        # Crisis comparison insights
        for comp in crisis_comps:
            if 'spread_pct_of_peak' in comp:
                pct = comp['spread_pct_of_peak']
                crisis = comp['crisis']
                if pct >= 80:
                    insights.append(f"Spreads near {crisis} peak levels ({pct}%)")
                elif pct >= 50:
                    insights.append(f"Spreads at {pct}% of {crisis} peak")
                elif pct < 30:
                    insights.append(f"Spreads well below {crisis} levels ({pct}%)")

        # Debt sustainability
        if pd.notna(debt_to_gdp):
            if debt_to_gdp > 120:
                insights.append(f"High debt burden ({debt_to_gdp}% of GDP) limits fiscal space")
            elif debt_to_gdp > 90:
                insights.append(f"Elevated debt ({debt_to_gdp}% of GDP) - monitor trajectory")

        # Historical analog
        if regime['historical_analog']:
            insights.append(f"Historical analog: {regime['historical_analog']}")

        return insights


def create_analyzer() -> CrisisAnalyzer:
    """Factory function."""
    return CrisisAnalyzer()
