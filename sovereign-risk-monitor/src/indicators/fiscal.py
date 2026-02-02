"""
Sovereign Risk Monitor - Fiscal & Macro Indicators
Data from IMF WEO, World Bank WDI, and Worldwide Governance Indicators.
"""

import pandas as pd
import logging
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# ISO3 codes for our monitored countries
COUNTRY_ISO3 = {
    'DE': 'DEU', 'FR': 'FRA', 'IT': 'ITA', 'ES': 'ESP', 'NL': 'NLD',
    'BE': 'BEL', 'AT': 'AUT', 'PT': 'PRT', 'GR': 'GRC', 'IE': 'IRL',
    'GB': 'GBR', 'CH': 'CHE', 'SE': 'SWE', 'US': 'USA', 'CA': 'CAN',
    'JP': 'JPN', 'AU': 'AUS'
}

# World Bank indicator codes
WB_INDICATORS = {
    # Fiscal
    'debt_to_gdp': 'GC.DOD.TOTL.GD.ZS',           # Central govt debt (% GDP)
    'budget_balance': 'GC.BAL.CASH.GD.ZS',         # Cash surplus/deficit (% GDP)

    # External
    'external_debt': 'DT.DOD.DECT.GD.ZS',          # External debt stocks (% GNI)
    'current_account': 'BN.CAB.XOKA.GD.ZS',        # Current account (% GDP)
    'reserves_months': 'FI.RES.TOTL.MO',           # Reserves (months of imports)
    'reserves_to_debt': 'FI.RES.TOTL.DT.ZS',       # Reserves to external debt (%)

    # Governance (WGI)
    'political_stability': 'PV.EST',               # Political stability estimate
    'govt_effectiveness': 'GE.EST',                # Government effectiveness
    'regulatory_quality': 'RQ.EST',                # Regulatory quality
    'rule_of_law': 'RL.EST',                       # Rule of law
    'corruption_control': 'CC.EST',                # Control of corruption
}

# IMF WEO indicator codes
IMF_INDICATORS = {
    'debt_to_gdp': 'GGXWDG_NGDP',                  # General govt gross debt (% GDP)
    'budget_deficit': 'GGXCNL_NGDP',              # Net lending/borrowing (% GDP)
    'primary_balance': 'GGXONLB_NGDP',            # Primary balance (% GDP)
}


class FiscalDataFetcher:
    """Fetches fiscal and macro indicators from IMF and World Bank APIs."""

    def __init__(self):
        self._check_dependencies()
        self.iso3_map = COUNTRY_ISO3
        self.iso3_reverse = {v: k for k, v in COUNTRY_ISO3.items()}

    def _check_dependencies(self):
        """Check if required packages are installed."""
        self.wbgapi_available = False
        self.weo_available = False

        try:
            import wbgapi
            self.wbgapi_available = True
        except ImportError:
            logger.warning("wbgapi not installed. Run: pip install wbgapi")

        try:
            import weo
            self.weo_available = True
        except ImportError:
            logger.warning("weo not installed. Run: pip install weo")

    def get_world_bank_data(
        self,
        indicators: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        start_year: int = 2015,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch World Bank indicators.

        Args:
            indicators: List of indicator keys from WB_INDICATORS. None = all.
            countries: List of ISO2 country codes. None = all monitored.
            start_year: Start year for data.
            end_year: End year. None = latest available.

        Returns:
            DataFrame with columns: country, year, indicator, value
        """
        if not self.wbgapi_available:
            raise ImportError("wbgapi required. Run: pip install wbgapi")

        import wbgapi as wb

        if end_year is None:
            end_year = datetime.now().year

        if countries is None:
            iso3_codes = list(self.iso3_map.values())
        else:
            iso3_codes = [self.iso3_map[c] for c in countries if c in self.iso3_map]

        if indicators is None:
            indicator_codes = WB_INDICATORS
        else:
            indicator_codes = {k: WB_INDICATORS[k] for k in indicators if k in WB_INDICATORS}

        all_data = []

        for name, code in indicator_codes.items():
            logger.info(f"Fetching {name} from World Bank...")
            try:
                df = wb.data.DataFrame(
                    code,
                    economy=iso3_codes,
                    time=range(start_year, end_year + 1),
                    labels=True
                )

                if df is not None and not df.empty:
                    df = df.reset_index()
                    df = df.melt(
                        id_vars=['economy'],
                        var_name='year',
                        value_name='value'
                    )
                    df['indicator'] = name
                    df['country'] = df['economy'].map(self.iso3_reverse)
                    # Extract year, handle NaN values
                    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
                    df = df.dropna(subset=['year', 'value'])
                    df['year'] = df['year'].astype(int)
                    all_data.append(df[['country', 'year', 'indicator', 'value']])

            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def get_imf_weo_data(
        self,
        indicators: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        year: int = 2024,
        release: str = 'Oct'
    ) -> pd.DataFrame:
        """
        Fetch IMF World Economic Outlook data.

        Args:
            indicators: List of indicator keys. None = all fiscal indicators.
            countries: List of ISO2 country codes. None = all monitored.
            year: WEO release year.
            release: 'Apr' or 'Oct'.

        Returns:
            DataFrame with columns: country, year, indicator, value
        """
        if not self.weo_available:
            raise ImportError("weo required. Run: pip install weo")

        import weo

        if countries is None:
            iso3_codes = list(self.iso3_map.values())
        else:
            iso3_codes = [self.iso3_map[c] for c in countries if c in self.iso3_map]

        if indicators is None:
            indicator_codes = IMF_INDICATORS
        else:
            indicator_codes = {k: IMF_INDICATORS[k] for k in indicators if k in IMF_INDICATORS}

        # Download WEO data
        logger.info(f"Fetching IMF WEO {release} {year}...")
        try:
            filename = f"weo_{year}_{release}.csv"
            weo.download(year=year, release=release, filename=filename)
            w = weo.WEO(filename)
        except Exception as e:
            logger.error(f"Error downloading WEO: {e}")
            return pd.DataFrame()

        all_data = []

        for name, code in indicator_codes.items():
            try:
                df = w.getc(code)
                if df is not None:
                    df = df.reset_index()
                    df = df.melt(id_vars=['ISO'], var_name='year', value_name='value')
                    df = df[df['ISO'].isin(iso3_codes)]
                    df['indicator'] = name
                    df['country'] = df['ISO'].map(self.iso3_reverse)
                    df['year'] = df['year'].astype(int)
                    all_data.append(df[['country', 'year', 'indicator', 'value']])

            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def get_governance_indicators(
        self,
        countries: Optional[List[str]] = None,
        start_year: int = 2015
    ) -> pd.DataFrame:
        """
        Fetch World Governance Indicators (WGI).

        Args:
            countries: List of ISO2 country codes. None = all monitored.
            start_year: Start year.

        Returns:
            DataFrame with governance scores (-2.5 to +2.5 scale).
        """
        if not self.wbgapi_available:
            raise ImportError("wbgapi required. Run: pip install wbgapi")

        import wbgapi as wb

        # WGI indicators (source 3 = WGI database)
        wgi_codes = {
            'political_stability': 'PV.EST',
            'govt_effectiveness': 'GE.EST',
            'regulatory_quality': 'RQ.EST',
            'rule_of_law': 'RL.EST',
            'corruption_control': 'CC.EST',
        }

        if countries is None:
            iso3_codes = list(self.iso3_map.values())
        else:
            iso3_codes = [self.iso3_map[c] for c in countries if c in self.iso3_map]

        end_year = datetime.now().year
        all_data = []

        for name, code in wgi_codes.items():
            logger.info(f"Fetching {name} from World Bank WGI...")
            try:
                # WGI is in source 3
                df = wb.data.DataFrame(
                    code,
                    economy=iso3_codes,
                    time=range(start_year, end_year + 1),
                    db=3  # WGI database
                )

                if df is not None and not df.empty:
                    df = df.reset_index()
                    df = df.melt(
                        id_vars=['economy'],
                        var_name='year',
                        value_name='value'
                    )
                    df['indicator'] = name
                    df['country'] = df['economy'].map(self.iso3_reverse)
                    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
                    df = df.dropna(subset=['year', 'value'])
                    df['year'] = df['year'].astype(int)
                    all_data.append(df[['country', 'year', 'indicator', 'value']])

            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def get_external_vulnerability(
        self,
        countries: Optional[List[str]] = None,
        start_year: int = 2015
    ) -> pd.DataFrame:
        """
        Fetch external vulnerability indicators.

        Args:
            countries: List of ISO2 country codes. None = all monitored.
            start_year: Start year.

        Returns:
            DataFrame with external balance and reserve indicators.
        """
        external_indicators = [
            'external_debt',
            'current_account',
            'reserves_months',
            'reserves_to_debt'
        ]

        return self.get_world_bank_data(
            indicators=external_indicators,
            countries=countries,
            start_year=start_year
        )

    def get_fiscal_snapshot(
        self,
        countries: Optional[List[str]] = None,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get latest fiscal snapshot combining IMF and World Bank data.

        Args:
            countries: List of ISO2 country codes. None = all monitored.
            year: Specific year. None = latest available.

        Returns:
            Wide-format DataFrame with one row per country.
        """
        if year is None:
            year = datetime.now().year - 1  # Use last complete year

        all_data = []

        # IMF WEO fiscal data
        if self.weo_available:
            try:
                imf_data = self.get_imf_weo_data(countries=countries)
                imf_latest = imf_data[imf_data['year'] == year]
                all_data.append(imf_latest)
            except Exception as e:
                logger.warning(f"IMF data unavailable: {e}")

        # World Bank data
        if self.wbgapi_available:
            try:
                wb_data = self.get_world_bank_data(
                    countries=countries,
                    start_year=year - 2,
                    end_year=year
                )
                # Get latest available for each indicator
                wb_latest = wb_data.sort_values('year', ascending=False)
                wb_latest = wb_latest.groupby(['country', 'indicator']).first().reset_index()
                all_data.append(wb_latest)
            except Exception as e:
                logger.warning(f"World Bank data unavailable: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Pivot to wide format
        snapshot = combined.pivot_table(
            index='country',
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()

        return snapshot


def create_fiscal_fetcher() -> FiscalDataFetcher:
    """Factory function to create FiscalDataFetcher."""
    return FiscalDataFetcher()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = create_fiscal_fetcher()

    print("\n=== Fiscal Data Fetcher ===")
    print(f"World Bank API: {'Available' if fetcher.wbgapi_available else 'Not installed'}")
    print(f"IMF WEO API: {'Available' if fetcher.weo_available else 'Not installed'}")

    if fetcher.wbgapi_available:
        print("\n=== Governance Indicators (Sample) ===")
        gov = fetcher.get_governance_indicators(countries=['DE', 'IT', 'GR'])
        if not gov.empty:
            # Get latest year per country/indicator
            latest = gov.sort_values('year', ascending=False).groupby(['country', 'indicator']).first().reset_index()
            print(latest.pivot(index='country', columns='indicator', values='value').round(2))
