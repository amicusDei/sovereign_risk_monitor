"""
Sovereign Risk Monitor - Data Fetchers
Eikon API interface for sovereign bond yields and CDS spreads.
"""

import os
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

try:
    import eikon as ek
    EIKON_AVAILABLE = True
except ImportError:
    EIKON_AVAILABLE = False

logger = logging.getLogger(__name__)


class EikonFetcher:
    """Fetches sovereign risk data via LSEG Eikon API."""

    def __init__(self, app_key: Optional[str] = None, config_path: Optional[Path] = None):
        """
        Initialize the Eikon data fetcher.

        Args:
            app_key: Eikon API key. If None, reads from EIKON_API_KEY env variable.
            config_path: Path to countries.yaml config. If None, uses default.
        """
        if not EIKON_AVAILABLE:
            raise ImportError("eikon package not installed. Run: pip install eikon")

        # Get API key from parameter or environment
        self.app_key = app_key or os.getenv("EIKON_API_KEY")
        if not self.app_key:
            raise ValueError(
                "Eikon API key required. Either pass app_key parameter or "
                "set EIKON_API_KEY environment variable."
            )

        ek.set_app_key(self.app_key)

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "countries.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.countries = self.config['countries']
        self._build_ric_mappings()

    def _build_ric_mappings(self):
        """Create bidirectional mappings between RICs and country codes."""
        self.yield_rics = {}
        self.cds_rics = {}
        self.ric_to_country = {}

        for code, info in self.countries.items():
            if info.get('yield_ric'):
                self.yield_rics[code] = info['yield_ric']
                self.ric_to_country[info['yield_ric']] = code
            if info.get('cds_ric'):
                self.cds_rics[code] = info['cds_ric']
                self.ric_to_country[info['cds_ric']] = code

    def get_live_yields(self, countries: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch current 10Y government bond yields.

        Args:
            countries: List of ISO country codes. None = all countries.

        Returns:
            DataFrame with columns: country, name, yield_pct, ric, timestamp
        """
        if countries is None:
            rics = list(self.yield_rics.values())
        else:
            rics = [self.yield_rics[c] for c in countries if c in self.yield_rics]

        if not rics:
            return pd.DataFrame()

        logger.info(f"Fetching yields for {len(rics)} countries...")
        df, err = ek.get_data(rics, ['SEC_YLD_1'])

        if err:
            logger.warning(f"Eikon errors: {err}")

        df = df.rename(columns={'Instrument': 'ric', 'SEC_YLD_1': 'yield_pct'})
        df['country'] = df['ric'].map(self.ric_to_country)
        df['name'] = df['country'].map(lambda c: self.countries[c]['name'])
        df['timestamp'] = datetime.now()

        return df[['country', 'name', 'yield_pct', 'ric', 'timestamp']]

    def get_live_cds(self, countries: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch current 5Y CDS spreads.

        Args:
            countries: List of ISO country codes. None = all with CDS data.

        Returns:
            DataFrame with columns: country, name, cds_spread_bps, ric, timestamp
        """
        if countries is None:
            rics = list(self.cds_rics.values())
        else:
            rics = [self.cds_rics[c] for c in countries if c in self.cds_rics]

        if not rics:
            return pd.DataFrame()

        logger.info(f"Fetching CDS for {len(rics)} countries...")
        df, err = ek.get_data(rics, ['CF_LAST'])

        if err:
            logger.warning(f"Eikon errors: {err}")

        df = df.rename(columns={'Instrument': 'ric', 'CF_LAST': 'cds_spread_bps'})
        df['country'] = df['ric'].map(self.ric_to_country)
        df['name'] = df['country'].map(lambda c: self.countries[c]['name'])
        df['timestamp'] = datetime.now()

        return df[['country', 'name', 'cds_spread_bps', 'ric', 'timestamp']]

    def get_live_snapshot(self, countries: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch combined snapshot of yields and CDS with calculated spreads.

        Args:
            countries: List of ISO country codes. None = all countries.

        Returns:
            DataFrame with yield, CDS, and spread data per country.
        """
        yields_df = self.get_live_yields(countries)
        cds_df = self.get_live_cds(countries)

        if yields_df.empty:
            return cds_df
        if cds_df.empty:
            return yields_df

        merged = yields_df.merge(
            cds_df[['country', 'cds_spread_bps']],
            on='country',
            how='left'
        )

        return self._calculate_spreads(merged)

    def _calculate_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate yield spreads vs regional benchmark (DE for EUR, US for others)."""
        df = df.copy()

        de_yield = df.loc[df['country'] == 'DE', 'yield_pct'].values
        us_yield = df.loc[df['country'] == 'US', 'yield_pct'].values

        de_benchmark = de_yield[0] if len(de_yield) > 0 else None
        us_benchmark = us_yield[0] if len(us_yield) > 0 else None

        def calc_spread(row):
            region = self.countries[row['country']].get('region', '')
            if region == 'eurozone':
                return (row['yield_pct'] - de_benchmark) * 100 if de_benchmark else None
            else:
                return (row['yield_pct'] - us_benchmark) * 100 if us_benchmark else None

        df['spread_bps'] = df.apply(calc_spread, axis=1)
        return df

    def get_historical_yields(
        self,
        countries: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical yield time series.

        Args:
            countries: List of ISO country codes. None = all countries.
            start_date: Start date (YYYY-MM-DD). Default: 1 year ago.
            end_date: End date (YYYY-MM-DD). Default: today.

        Returns:
            DataFrame with columns: date, yield_pct, country, name
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if countries is None:
            country_list = list(self.yield_rics.keys())
        else:
            country_list = [c for c in countries if c in self.yield_rics]

        all_data = []

        for country in country_list:
            ric = self.yield_rics[country]
            logger.info(f"Fetching historical data for {country}...")

            try:
                df = ek.get_timeseries(
                    ric,
                    fields=['CLOSE'],
                    start_date=start_date,
                    end_date=end_date
                )

                if df is not None and not df.empty:
                    df = df.reset_index()
                    df.columns = ['date', 'yield_pct']
                    df['country'] = country
                    df['name'] = self.countries[country]['name']
                    all_data.append(df)

            except Exception as e:
                logger.warning(f"Error fetching {country}: {e}")

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values(['date', 'country'])

    def get_available_countries(self) -> Dict[str, Dict]:
        """Return dictionary of all configured countries with data availability."""
        result = {}
        for code, info in self.countries.items():
            result[code] = {
                'name': info['name'],
                'region': info.get('region'),
                'has_yield': info.get('yield_ric') is not None,
                'has_cds': info.get('cds_ric') is not None,
                'currency': info.get('currency')
            }
        return result


def create_fetcher(app_key: Optional[str] = None) -> EikonFetcher:
    """
    Factory function to create EikonFetcher.

    Args:
        app_key: Eikon API key. If None, reads from EIKON_API_KEY env variable.
    """
    return EikonFetcher(app_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = create_fetcher()

    print("\n=== Available Countries ===")
    for code, info in fetcher.get_available_countries().items():
        cds_status = "CDS" if info['has_cds'] else "   "
        print(f"{code}: {info['name']:20} {cds_status}")

    print("\n=== Live Snapshot ===")
    snapshot = fetcher.get_live_snapshot()
    print(snapshot.to_string(index=False))
