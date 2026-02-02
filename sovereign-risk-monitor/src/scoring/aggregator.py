"""
Sovereign Risk Monitor - Score Aggregation & Grading
Combines normalized indicators into composite score and assigns letter grade.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

from src.scoring.normalizer import ScoreNormalizer

logger = logging.getLogger(__name__)


class RiskScorer:
    """Aggregates indicator scores and assigns risk grades."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config"

        with open(config_path / "weights.yaml", 'r') as f:
            self.weights_config = yaml.safe_load(f)

        with open(config_path / "crisis_benchmarks.yaml", 'r') as f:
            self.crisis_config = yaml.safe_load(f)

        self.normalizer = ScoreNormalizer()
        self.grading_scale = self.crisis_config['grading']['scale']
        self.grade_descriptions = self.crisis_config['grading']['descriptions']

    def score_to_grade(self, score: float) -> str:
        """Convert numeric score (0-100) to letter grade."""
        if pd.isna(score):
            return 'NR'  # Not Rated

        for grade, (lower, upper) in self.grading_scale.items():
            if lower <= score < upper:
                return grade

        return 'C' if score >= 95 else 'AAA'

    def calculate_market_score(
        self,
        spread_bps: float,
        cds_bps: Optional[float],
        volatility: Optional[float],
        country: str,
        region: str
    ) -> Dict:
        """
        Calculate market-based risk score.

        Returns:
            Dict with component scores and weighted total
        """
        weights = self.weights_config['market_indicators']

        # Normalize components
        spread_score = self.normalizer.normalize_spread(spread_bps, country, region)
        cds_score = self.normalizer.normalize_cds(cds_bps, country) if pd.notna(cds_bps) else None
        vol_score = self.normalizer.normalize_volatility(volatility) if pd.notna(volatility) else None

        components = {
            'spread_score': spread_score,
            'cds_score': cds_score,
            'volatility_score': vol_score,
        }

        # Weighted average (adjust weights if components missing)
        total_weight = 0
        weighted_sum = 0

        if spread_score is not None:
            w = weights['yield_spread']['weight']
            weighted_sum += spread_score * w
            total_weight += w

        if cds_score is not None:
            w = weights['cds_spread']['weight']
            weighted_sum += cds_score * w
            total_weight += w

        if vol_score is not None:
            w = weights['yield_volatility']['weight']
            weighted_sum += vol_score * w
            total_weight += w

        market_score = weighted_sum / total_weight if total_weight > 0 else None

        return {
            **components,
            'market_score': round(market_score, 1) if market_score else None
        }

    def calculate_composite_score(
        self,
        market_score: Optional[float],
        fiscal_score: Optional[float] = None,
        external_score: Optional[float] = None,
        governance_score: Optional[float] = None
    ) -> float:
        """
        Calculate composite risk score from category scores.

        Returns:
            Composite score 0-100
        """
        cat_weights = self.weights_config['category_weights']

        scores = {
            'market': (market_score, cat_weights['market']),
            'fiscal': (fiscal_score, cat_weights['fiscal']),
            'external': (external_score, cat_weights['external']),
            'governance': (governance_score, cat_weights['governance']),
        }

        total_weight = 0
        weighted_sum = 0

        for name, (score, weight) in scores.items():
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return None

        return round(weighted_sum / total_weight, 1)

    def get_stress_level(self, indicator: str, value: float) -> str:
        """Classify value into stress level category."""
        if pd.isna(value):
            return 'unknown'

        # Negative spreads = safe (yield below benchmark)
        if value < 0:
            return 'normal'

        thresholds = self.crisis_config['stress_thresholds'].get(indicator, {})

        for level, (lower, upper) in thresholds.items():
            if upper is None:
                if value >= lower:
                    return level
            elif lower <= value < upper:
                return level

        return 'normal'

    def score_country(
        self,
        country: str,
        yield_pct: float,
        spread_bps: float,
        cds_bps: Optional[float] = None,
        volatility: Optional[float] = None,
        debt_to_gdp: Optional[float] = None,
        region: str = 'eurozone'
    ) -> Dict:
        """
        Generate complete risk assessment for a country.

        Returns:
            Dict with scores, grade, and context
        """
        # Market score
        market = self.calculate_market_score(
            spread_bps=spread_bps,
            cds_bps=cds_bps,
            volatility=volatility,
            country=country,
            region=region
        )

        # Fiscal score (if available)
        fiscal_score = None
        if pd.notna(debt_to_gdp):
            fiscal_score = self.normalizer.normalize_debt_gdp(debt_to_gdp)

        # Composite score
        composite = self.calculate_composite_score(
            market_score=market['market_score'],
            fiscal_score=fiscal_score
        )

        # Grade
        grade = self.score_to_grade(composite)

        # Stress levels
        stress_levels = {
            'spread': self.get_stress_level('yield_spread_bps', spread_bps),
        }
        if pd.notna(cds_bps):
            stress_levels['cds'] = self.get_stress_level('cds_spread_bps', cds_bps)

        # Crisis comparison
        crisis_comp = self.normalizer.crisis_comparison(country, spread_bps, cds_bps)

        return {
            'country': country,
            'timestamp': datetime.now().isoformat(),

            # Raw values
            'yield_pct': yield_pct,
            'spread_bps': spread_bps,
            'cds_bps': cds_bps,

            # Scores
            'spread_score': market['spread_score'],
            'cds_score': market['cds_score'],
            'market_score': market['market_score'],
            'fiscal_score': fiscal_score,
            'composite_score': composite,

            # Grade
            'grade': grade,
            'grade_description': self.grade_descriptions.get(grade, ''),

            # Context
            'stress_levels': stress_levels,
            'crisis_comparison': crisis_comp,
        }

    def score_all_countries(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all countries in a snapshot DataFrame.

        Args:
            snapshot_df: DataFrame with columns: country, yield_pct, spread_bps, cds_spread_bps

        Returns:
            DataFrame with scores and grades for each country
        """
        results = []

        for _, row in snapshot_df.iterrows():
            country = row['country']

            # Determine region
            region = 'eurozone'  # Default, should come from config

            result = self.score_country(
                country=country,
                yield_pct=row.get('yield_pct'),
                spread_bps=row.get('spread_bps', 0),
                cds_bps=row.get('cds_spread_bps'),
                region=region
            )

            results.append({
                'country': country,
                'yield_pct': result['yield_pct'],
                'spread_bps': result['spread_bps'],
                'cds_bps': result['cds_bps'],
                'market_score': result['market_score'],
                'composite_score': result['composite_score'],
                'grade': result['grade'],
                'spread_stress': result['stress_levels']['spread'],
            })

        return pd.DataFrame(results)


def create_scorer() -> RiskScorer:
    """Factory function."""
    return RiskScorer()
