"""
Sovereign Risk Monitor - Score Normalization
Transforms raw indicators into comparable 0-100 scores using historical percentiles.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """Normalizes raw indicators to 0-100 scale using historical context."""

    def __init__(self):
        # Historical reference distributions (percentiles from 2000-2024 data)
        # Format: [p5, p25, p50, p75, p95]
        self.historical_distributions = {
            'yield_spread_bps': {
                'eurozone': [5, 25, 60, 120, 300],      # vs Bund
                'other': [10, 40, 80, 150, 350],        # vs Treasury
            },
            'cds_spread_bps': {
                'safe_haven': [5, 15, 25, 40, 80],      # DE, CH, US
                'core': [10, 30, 50, 90, 200],          # FR, NL, AT, BE
                'periphery': [30, 80, 150, 300, 600],   # IT, ES, PT, GR, IE
            },
            'yield_volatility_30d': {
                'all': [15, 30, 50, 80, 150],
            },
            'debt_to_gdp': {
                'all': [30, 50, 70, 100, 140],
            },
        }

        # Crisis peak values for extreme stress reference
        self.crisis_peaks = {
            'GR': {'yield_spread_bps': 3350, 'cds_spread_bps': 25000},
            'IT': {'yield_spread_bps': 550, 'cds_spread_bps': 590},
            'ES': {'yield_spread_bps': 590, 'cds_spread_bps': 640},
            'PT': {'yield_spread_bps': 1450, 'cds_spread_bps': 1520},
            'IE': {'yield_spread_bps': 1250, 'cds_spread_bps': 1200},
            'GB': {'yield_spread_bps': 250, 'cds_spread_bps': 165},
        }

    def percentile_score(
        self,
        value: float,
        distribution: List[float],
        higher_is_riskier: bool = True
    ) -> float:
        """
        Convert value to 0-100 score based on historical distribution.

        Args:
            value: Raw indicator value
            distribution: [p5, p25, p50, p75, p95] reference percentiles
            higher_is_riskier: If True, higher values = higher risk score

        Returns:
            Score from 0 (lowest risk) to 100 (highest risk)
        """
        if pd.isna(value):
            return np.nan

        p5, p25, p50, p75, p95 = distribution

        # Handle negative values (e.g., negative spreads = lower risk)
        if value < 0:
            score = 0.0
        elif value <= p5:
            score = 5 * (value / p5) if p5 > 0 else 0
        elif value <= p25:
            score = 5 + 20 * (value - p5) / (p25 - p5)
        elif value <= p50:
            score = 25 + 25 * (value - p25) / (p50 - p25)
        elif value <= p75:
            score = 50 + 25 * (value - p50) / (p75 - p50)
        elif value <= p95:
            score = 75 + 20 * (value - p75) / (p95 - p75)
        else:
            # Beyond p95: scale up to 100, cap at 100
            excess = (value - p95) / (p95 - p75) if (p95 - p75) > 0 else 0
            score = min(95 + 5 * excess, 100)

        if not higher_is_riskier:
            score = 100 - score

        return round(score, 1)

    def normalize_spread(
        self,
        spread_bps: float,
        country: str,
        region: str = 'eurozone'
    ) -> float:
        """Normalize yield spread to 0-100 score."""
        dist_key = 'eurozone' if region == 'eurozone' else 'other'
        distribution = self.historical_distributions['yield_spread_bps'][dist_key]
        return self.percentile_score(spread_bps, distribution, higher_is_riskier=True)

    def normalize_cds(
        self,
        cds_bps: float,
        country: str
    ) -> float:
        """Normalize CDS spread to 0-100 score."""
        # Classify country
        if country in ['DE', 'CH', 'US']:
            tier = 'safe_haven'
        elif country in ['FR', 'NL', 'AT', 'BE', 'SE', 'CA', 'AU', 'JP']:
            tier = 'core'
        else:
            tier = 'periphery'

        distribution = self.historical_distributions['cds_spread_bps'][tier]
        return self.percentile_score(cds_bps, distribution, higher_is_riskier=True)

    def normalize_volatility(self, vol: float) -> float:
        """Normalize yield volatility to 0-100 score."""
        distribution = self.historical_distributions['yield_volatility_30d']['all']
        return self.percentile_score(vol, distribution, higher_is_riskier=True)

    def normalize_debt_gdp(self, debt_gdp: float) -> float:
        """Normalize debt/GDP to 0-100 score."""
        distribution = self.historical_distributions['debt_to_gdp']['all']
        return self.percentile_score(debt_gdp, distribution, higher_is_riskier=True)

    def crisis_comparison(
        self,
        country: str,
        current_spread: float,
        current_cds: Optional[float] = None
    ) -> Dict:
        """
        Compare current levels to historical crisis peaks.

        Returns:
            Dict with crisis comparison metrics
        """
        result = {
            'country': country,
            'current_spread_bps': current_spread,
            'current_cds_bps': current_cds,
        }

        if country in self.crisis_peaks:
            peaks = self.crisis_peaks[country]

            # Spread as % of crisis peak
            if 'yield_spread_bps' in peaks and peaks['yield_spread_bps'] > 0:
                result['spread_vs_crisis_peak_pct'] = round(
                    100 * current_spread / peaks['yield_spread_bps'], 1
                )

            # CDS as % of crisis peak
            if pd.notna(current_cds) and 'cds_spread_bps' in peaks and peaks['cds_spread_bps'] > 0:
                result['cds_vs_crisis_peak_pct'] = round(
                    100 * current_cds / peaks['cds_spread_bps'], 1
                )

        return result


def create_normalizer() -> ScoreNormalizer:
    """Factory function."""
    return ScoreNormalizer()
