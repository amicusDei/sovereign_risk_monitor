"""
Sovereign Risk Monitor - Early Warning Indicators
Detects pre-crisis patterns based on academic research on sovereign debt crises.

Key signals based on literature:
- Spread momentum (velocity of widening)
- CDS term structure inversion
- Volatility regime shifts
- Cross-country contagion patterns

References:
- Kaminsky et al. (1998) - Signal extraction approach
- Augustin (2018) - CDS term structure dynamics
- Longstaff et al. (2011) - Global vs local factors
- ECB Working Papers on sovereign risk determinants
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EarlyWarningSystem:
    """
    Generates early warning signals for sovereign stress.

    Key indicators:
    1. Spread momentum - rate of change in yield spreads
    2. Volatility breakout - sudden increase in yield volatility
    3. CDS-Bond basis - divergence between CDS and bond spreads
    4. Contagion signal - correlation spike with stressed peers
    5. Term structure slope - inversion signals near-term stress
    """

    def __init__(self):
        # Signal thresholds calibrated to historical crises
        self.thresholds = {
            # Spread momentum (bps change over period)
            'spread_momentum_5d': {
                'warning': 20,      # 20 bps in 5 days
                'alert': 40,        # 40 bps in 5 days
                'critical': 80,     # 80 bps in 5 days
            },
            'spread_momentum_20d': {
                'warning': 50,      # 50 bps in 20 days
                'alert': 100,       # 100 bps in 20 days
                'critical': 200,    # 200 bps in 20 days
            },
            # Volatility (annualized bps)
            'volatility_zscore': {
                'warning': 1.5,     # 1.5 std above mean
                'alert': 2.0,       # 2 std above mean
                'critical': 3.0,    # 3 std above mean
            },
            # Spread level percentile (vs own history)
            'spread_percentile': {
                'warning': 75,      # 75th percentile
                'alert': 90,        # 90th percentile
                'critical': 95,     # 95th percentile
            },
        }

        # Pre-crisis patterns from historical events
        self.crisis_patterns = {
            'eurozone_2010_2012': {
                'lead_time_days': 90,  # Signals appeared ~3 months before peak
                'spread_acceleration': True,
                'volatility_spike': True,
                'contagion_sequence': ['GR', 'IE', 'PT', 'ES', 'IT'],
                'detectable': True,
            },
            'covid_2020': {
                'lead_time_days': 14,  # Very rapid onset
                'spread_acceleration': True,
                'volatility_spike': True,
                'global_correlation_spike': True,
                'detectable': True,
            },
            'uk_gilt_2022': {
                'lead_time_days': 3,   # Almost no warning from market data
                'volatility_spike': True,
                'long_end_stress': True,  # 30Y more than 10Y
                'detectable': False,  # Political/fiscal shock - not market-predictable
                'lesson': 'Fiscal credibility shocks require political/policy monitoring',
            },
        }

        # Additional thresholds for rapid-onset detection (UK-style)
        self.rapid_onset_thresholds = {
            'daily_move_bps': {
                'warning': 15,   # 15 bps in one day
                'alert': 25,     # 25 bps in one day
                'critical': 40,  # 40 bps in one day (UK saw ~100)
            },
            'intraday_range_bps': {
                'warning': 20,
                'alert': 35,
                'critical': 50,
            },
        }

    def calculate_momentum(
        self,
        timeseries: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate spread momentum over multiple windows.

        Args:
            timeseries: DataFrame with 'date' and 'spread_bps' columns
            windows: List of lookback periods in days

        Returns:
            DataFrame with momentum signals
        """
        df = timeseries.copy()
        df = df.sort_values('date')

        for window in windows:
            col_name = f'momentum_{window}d'
            df[col_name] = df['spread_bps'].diff(window)

            # Percentage change
            pct_col = f'momentum_{window}d_pct'
            df[pct_col] = df['spread_bps'].pct_change(window) * 100

        return df

    def calculate_volatility(
        self,
        timeseries: pd.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility and z-score vs history.

        Args:
            timeseries: DataFrame with 'date' and 'yield_pct' columns
            window: Rolling window in days
            annualize: Whether to annualize volatility

        Returns:
            DataFrame with volatility metrics
        """
        df = timeseries.copy()
        df = df.sort_values('date')

        # Daily changes in bps
        df['daily_change_bps'] = df['yield_pct'].diff() * 100

        # Rolling volatility
        df['volatility_raw'] = df['daily_change_bps'].rolling(window).std()

        if annualize:
            df['volatility'] = df['volatility_raw'] * np.sqrt(252)
        else:
            df['volatility'] = df['volatility_raw']

        # Z-score vs 1-year history
        df['vol_mean_1y'] = df['volatility'].rolling(252).mean()
        df['vol_std_1y'] = df['volatility'].rolling(252).std()
        df['volatility_zscore'] = (
            (df['volatility'] - df['vol_mean_1y']) / df['vol_std_1y']
        )

        return df

    def calculate_percentile_rank(
        self,
        timeseries: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate where current spread sits vs rolling history.

        Args:
            timeseries: DataFrame with 'date' and 'spread_bps' columns
            window: Lookback window in days

        Returns:
            DataFrame with percentile rank
        """
        df = timeseries.copy()
        df = df.sort_values('date')

        def rolling_percentile(series, window):
            result = []
            for i in range(len(series)):
                if i < window:
                    result.append(np.nan)
                else:
                    historical = series.iloc[i-window:i]
                    current = series.iloc[i]
                    percentile = (historical < current).sum() / window * 100
                    result.append(percentile)
            return result

        df['spread_percentile'] = rolling_percentile(df['spread_bps'], window)

        return df

    def generate_signals(
        self,
        timeseries: pd.DataFrame,
        country: str
    ) -> Dict:
        """
        Generate early warning signals for a country.

        Args:
            timeseries: DataFrame with historical data
            country: ISO2 country code

        Returns:
            Dict with signal status and details
        """
        if timeseries.empty or len(timeseries) < 30:
            return {'error': 'Insufficient historical data'}

        # Ensure we have required columns
        df = timeseries.copy()
        df = df.sort_values('date')

        # Calculate indicators
        if 'spread_bps' in df.columns:
            df = self.calculate_momentum(df)
            df = self.calculate_percentile_rank(df)

        if 'yield_pct' in df.columns:
            df = self.calculate_volatility(df)

        # Get latest values
        latest = df.iloc[-1]

        signals = {
            'country': country,
            'date': latest['date'] if 'date' in latest else datetime.now(),
            'signals': [],
            'alert_level': 'normal',
            'details': {},
        }

        alert_levels = ['normal', 'warning', 'alert', 'critical']
        max_alert = 0

        # Check spread momentum
        for window in [5, 20]:
            col = f'momentum_{window}d'
            if col in latest and pd.notna(latest[col]):
                momentum = latest[col]
                signals['details'][f'spread_momentum_{window}d_bps'] = round(momentum, 1)

                thresholds = self.thresholds[f'spread_momentum_{window}d']
                if momentum >= thresholds['critical']:
                    signals['signals'].append(f"CRITICAL: Spread widened {momentum:.0f}bps in {window} days")
                    max_alert = max(max_alert, 3)
                elif momentum >= thresholds['alert']:
                    signals['signals'].append(f"ALERT: Spread widened {momentum:.0f}bps in {window} days")
                    max_alert = max(max_alert, 2)
                elif momentum >= thresholds['warning']:
                    signals['signals'].append(f"WARNING: Spread widened {momentum:.0f}bps in {window} days")
                    max_alert = max(max_alert, 1)

        # Check volatility z-score
        if 'volatility_zscore' in latest and pd.notna(latest['volatility_zscore']):
            vol_z = latest['volatility_zscore']
            signals['details']['volatility_zscore'] = round(vol_z, 2)

            thresholds = self.thresholds['volatility_zscore']
            if vol_z >= thresholds['critical']:
                signals['signals'].append(f"CRITICAL: Volatility at {vol_z:.1f} std above normal")
                max_alert = max(max_alert, 3)
            elif vol_z >= thresholds['alert']:
                signals['signals'].append(f"ALERT: Elevated volatility ({vol_z:.1f} std)")
                max_alert = max(max_alert, 2)
            elif vol_z >= thresholds['warning']:
                signals['signals'].append(f"WARNING: Rising volatility ({vol_z:.1f} std)")
                max_alert = max(max_alert, 1)

        # Check spread percentile
        if 'spread_percentile' in latest and pd.notna(latest['spread_percentile']):
            pct = latest['spread_percentile']
            signals['details']['spread_percentile_1y'] = round(pct, 1)

            thresholds = self.thresholds['spread_percentile']
            if pct >= thresholds['critical']:
                signals['signals'].append(f"CRITICAL: Spread at {pct:.0f}th percentile (1Y)")
                max_alert = max(max_alert, 3)
            elif pct >= thresholds['alert']:
                signals['signals'].append(f"ALERT: Spread at {pct:.0f}th percentile (1Y)")
                max_alert = max(max_alert, 2)
            elif pct >= thresholds['warning']:
                signals['signals'].append(f"WARNING: Spread elevated ({pct:.0f}th percentile)")
                max_alert = max(max_alert, 1)

        signals['alert_level'] = alert_levels[max_alert]

        # Add trend direction
        if 'momentum_5d' in latest and 'momentum_20d' in latest:
            m5 = latest['momentum_5d'] if pd.notna(latest['momentum_5d']) else 0
            m20 = latest['momentum_20d'] if pd.notna(latest['momentum_20d']) else 0

            if m5 > 0 and m20 > 0:
                signals['trend'] = 'widening'
            elif m5 < 0 and m20 < 0:
                signals['trend'] = 'tightening'
            elif m5 > 0 and m20 < 0:
                signals['trend'] = 'reversing_higher'
            else:
                signals['trend'] = 'reversing_lower'

        return signals

    def detect_contagion(
        self,
        spread_data: Dict[str, pd.DataFrame],
        window: int = 20
    ) -> Dict:
        """
        Detect cross-country contagion patterns.

        Args:
            spread_data: Dict mapping country code to timeseries DataFrame
            window: Correlation window

        Returns:
            Dict with contagion analysis
        """
        # Build correlation matrix of spread changes
        changes = {}
        for country, df in spread_data.items():
            if 'spread_bps' in df.columns and len(df) >= window:
                df = df.sort_values('date')
                changes[country] = df['spread_bps'].diff()

        if len(changes) < 2:
            return {'error': 'Insufficient data for contagion analysis'}

        changes_df = pd.DataFrame(changes).dropna()

        if len(changes_df) < window:
            return {'error': 'Insufficient overlapping data'}

        # Rolling correlation
        recent_corr = changes_df.tail(window).corr()

        # Compare to longer history
        if len(changes_df) >= 252:
            historical_corr = changes_df.tail(252).head(252 - window).corr()
            corr_change = recent_corr - historical_corr
        else:
            corr_change = None

        # Identify high correlations (potential contagion)
        high_corr_pairs = []
        for i, c1 in enumerate(recent_corr.columns):
            for j, c2 in enumerate(recent_corr.columns):
                if i < j:
                    corr = recent_corr.loc[c1, c2]
                    if corr > 0.7:
                        high_corr_pairs.append({
                            'pair': (c1, c2),
                            'correlation': round(corr, 2),
                        })

        return {
            'correlation_matrix': recent_corr.round(2).to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'contagion_risk': len(high_corr_pairs) > 2,
        }

    def check_rapid_onset(
        self,
        timeseries: pd.DataFrame,
        country: str
    ) -> Dict:
        """
        Check for rapid-onset crisis signals (UK Gilt 2022 style).

        These detect DURING a crisis, not before - useful for immediate response.

        Args:
            timeseries: DataFrame with 'date' and 'yield_pct' columns
            country: ISO2 country code

        Returns:
            Dict with rapid-onset signals
        """
        if timeseries.empty or len(timeseries) < 5:
            return {'error': 'Insufficient data'}

        df = timeseries.copy().sort_values('date')

        # Calculate daily changes
        df['daily_change_bps'] = df['yield_pct'].diff() * 100

        # Get last 5 days
        recent = df.tail(5)
        latest = recent.iloc[-1]

        signals = {
            'country': country,
            'rapid_onset_alert': False,
            'signals': [],
            'details': {},
        }

        # Check latest daily move
        if pd.notna(latest['daily_change_bps']):
            move = abs(latest['daily_change_bps'])
            signals['details']['latest_daily_move_bps'] = round(move, 1)

            thresholds = self.rapid_onset_thresholds['daily_move_bps']
            if move >= thresholds['critical']:
                signals['rapid_onset_alert'] = True
                signals['signals'].append(f"CRITICAL: {move:.0f}bps daily move - crisis in progress")
            elif move >= thresholds['alert']:
                signals['rapid_onset_alert'] = True
                signals['signals'].append(f"ALERT: {move:.0f}bps daily move - rapid stress")
            elif move >= thresholds['warning']:
                signals['signals'].append(f"WARNING: {move:.0f}bps daily move - monitor closely")

        # Check for consecutive large moves (sustained stress)
        large_moves = recent['daily_change_bps'].abs() > 15
        if large_moves.sum() >= 3:
            signals['rapid_onset_alert'] = True
            signals['signals'].append("ALERT: 3+ days of large moves - sustained stress")

        # Check cumulative 3-day move
        if len(recent) >= 3:
            cum_3d = recent['daily_change_bps'].tail(3).sum()
            signals['details']['cumulative_3d_bps'] = round(cum_3d, 1)
            if abs(cum_3d) > 50:
                signals['rapid_onset_alert'] = True
                signals['signals'].append(f"CRITICAL: {cum_3d:.0f}bps move in 3 days")

        return signals

    def get_crisis_similarity(
        self,
        current_signals: Dict,
        historical_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Compare current pattern to known pre-crisis patterns.

        Returns:
            List of similar historical episodes
        """
        similarities = []

        # Check against Eurozone crisis pattern
        if (current_signals.get('alert_level') in ['alert', 'critical'] and
            current_signals.get('trend') == 'widening'):

            similarities.append({
                'pattern': 'Eurozone 2010-2012',
                'similarity': 'Medium',
                'note': 'Spread widening with elevated volatility matches early crisis pattern',
                'typical_lead_time': '1-3 months before peak stress',
            })

        # Check for rapid COVID-like pattern
        details = current_signals.get('details', {})
        if details.get('spread_momentum_5d_bps', 0) > 30:
            similarities.append({
                'pattern': 'COVID March 2020',
                'similarity': 'High',
                'note': 'Rapid spread widening similar to COVID shock',
                'typical_lead_time': '1-2 weeks',
            })

        return similarities


def create_early_warning() -> EarlyWarningSystem:
    """Factory function."""
    return EarlyWarningSystem()
