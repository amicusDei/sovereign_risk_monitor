"""
Sovereign Risk Monitor - Quick Start Example

This script demonstrates the basic functionality of the monitor.
Make sure you have set EIKON_API_KEY in your environment or .env file.

Usage:
    python examples/quickstart.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    pass  # python-dotenv not installed, use system env vars

from src.monitor import create_monitor


def main():
    # Check for API key
    if not os.getenv('EIKON_API_KEY'):
        print("ERROR: EIKON_API_KEY environment variable not set.")
        print("Set it in your environment or create a .env file.")
        sys.exit(1)

    print("Initializing Sovereign Risk Monitor...")
    monitor = create_monitor()

    # 1. Print the live risk dashboard
    print("\n" + "=" * 60)
    print("1. LIVE RISK DASHBOARD")
    print("=" * 60)
    monitor.print_dashboard()

    # 2. Detailed analysis for Italy
    print("\n" + "=" * 60)
    print("2. DETAILED ANALYSIS: ITALY")
    print("=" * 60)

    italy = monitor.analyze_country('IT')

    print(f"\nCurrent Levels:")
    print(f"  10Y Yield:     {italy['yield_10y']:.2f}%")
    print(f"  Spread vs DE:  {italy['spread_vs_benchmark_bps']:.0f} bps")
    print(f"  5Y CDS:        {italy['cds_5y_bps']} bps")

    print(f"\nRisk Assessment:")
    print(f"  Grade:         {italy['grade']} - {italy['grade_description']}")
    print(f"  Score:         {italy['scores']['composite']:.0f}/100")

    print(f"\nStress Regime:   {italy['stress_regime']['overall_regime']}")
    print(f"Historical Analog: {italy['stress_regime']['historical_analog']}")

    print("\nKey Insights:")
    for insight in italy['insights']:
        print(f"  • {insight}")

    # 3. Compare current levels to Eurozone debt crisis
    print("\n" + "=" * 60)
    print("3. COMPARISON TO EUROZONE DEBT CRISIS (2010-2012)")
    print("=" * 60)

    crisis_comp = monitor.compare_to_crisis('eurozone_debt_crisis')
    if not crisis_comp.empty:
        print(f"\n{'Country':<10} {'Spread Now':>12} {'Peak':>10} {'% of Peak':>12}")
        print("-" * 50)
        for _, row in crisis_comp.iterrows():
            pct = row.get('spread_pct_of_peak', 'N/A')
            pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else pct
            print(f"{row['country']:<10} {row['current_spread']:>12.0f} {row['peak_spread']:>10.0f} {pct_str:>12}")

    # 4. Early warning signals
    print("\n" + "=" * 60)
    print("4. EARLY WARNING SIGNALS")
    print("=" * 60)
    monitor.print_early_warnings(countries=['IT', 'ES', 'GR', 'FR', 'GB'])

    # 5. Get programmatic access to data
    print("\n" + "=" * 60)
    print("5. PROGRAMMATIC ACCESS")
    print("=" * 60)

    dashboard = monitor.get_risk_dashboard()
    print("\nHighest risk countries:")
    print(dashboard.head(3)[['country', 'name', 'composite_score', 'grade']].to_string(index=False))

    alerts = monitor.get_alert_countries(threshold='elevated')
    if not alerts.empty:
        print(f"\nCountries with elevated stress: {', '.join(alerts['country'].tolist())}")
    else:
        print("\nNo countries with elevated stress currently.")


if __name__ == "__main__":
    main()
