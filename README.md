# Sovereign Risk Monitor

Real-time sovereign bond risk scoring and early warning system for 17 developed market countries.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

A Python tool that monitors sovereign bond markets and provides:
- **Risk Scores** (0-100) and letter grades (AAA to C) for each country
- **Crisis Comparison** against historical peaks (GFC 2008, Eurozone 2010-12, COVID 2020)
- **Early Warning Signals** based on spread momentum, volatility, and percentile rankings
- **Fiscal & Governance Data** from IMF WEO and World Bank APIs

## Sample Output

```
================================================================================
SOVEREIGN RISK MONITOR - Live Dashboard
Timestamp: 2025-01-17 14:30:00
================================================================================
Code    Country              Yield%  Spread   CDS   Score  Grade  Stress
GR      Greece                 3.42    115    78      45    BBB   elevated
IT      Italy                  3.58     92    85      42    BBB   elevated
ES      Spain                  3.12     46    38      28     A    normal
PT      Portugal               2.98     32    42      25     A    normal
FR      France                 2.89     23    25      18    AA    normal
...
DE      Germany                2.66      0     -       5   AAA    normal
================================================================================
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sovereign-risk-monitor.git
cd sovereign-risk-monitor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your Eikon API key
```

## Quick Start

```python
from src.monitor import create_monitor

# Initialize (reads EIKON_API_KEY from environment)
monitor = create_monitor()

# Print live risk dashboard
monitor.print_dashboard()

# Detailed analysis for a single country
analysis = monitor.analyze_country('IT')
print(f"Italy Grade: {analysis['grade']}")
print(f"Risk Score: {analysis['scores']['composite']}/100")

# Early warning signals
monitor.print_early_warnings(countries=['IT', 'ES', 'GR', 'FR'])
```

## Architecture

```
sovereign-risk-monitor/
├── config/
│   ├── countries.yaml      # Country RICs and metadata
│   ├── weights.yaml        # Scoring methodology
│   └── crisis_benchmarks.yaml  # Historical crisis data
├── src/
│   ├── data/
│   │   └── fetchers.py     # Eikon API interface
│   ├── indicators/
│   │   ├── fiscal.py       # IMF/World Bank data
│   │   └── early_warning.py # Trend signals
│   ├── scoring/
│   │   ├── normalizer.py   # Percentile scoring
│   │   └── aggregator.py   # Risk grades
│   ├── analysis/
│   │   └── crisis_comparison.py
│   └── monitor.py          # Main interface
└── examples/
    └── quickstart.py
```

## Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| LSEG Eikon | Bond yields, CDS spreads | Real-time |
| IMF WEO | Debt/GDP, fiscal balance | Semi-annual |
| World Bank WDI | External debt, reserves | Annual |
| World Bank WGI | Governance scores | Annual |

## Methodology

### Risk Scoring

Scores are percentile-based (0-100), calibrated against historical distributions:

| Component | Weight | Source |
|-----------|--------|--------|
| CDS Spread | 40% | Market |
| Yield Spread vs Benchmark | 30% | Market |
| Volatility | 30% | Calculated |

### Letter Grades

| Grade | Score Range | Description |
|-------|-------------|-------------|
| AAA | 0-10 | Minimal risk |
| AA | 10-20 | Very low risk |
| A | 20-35 | Low risk |
| BBB | 35-50 | Moderate risk |
| BB | 50-65 | Elevated risk |
| B | 65-80 | High risk |
| CCC | 80-90 | Very high risk |
| CC | 90-95 | Near distress |
| C | 95-100 | Distress |

### Early Warning System

Signals are triggered by:
- **Momentum**: 5-day and 20-day spread changes
- **Volatility Z-score**: Current vs 1-year average
- **Percentile Rank**: Current level vs 1-year distribution

## Countries Monitored

**Eurozone**: Germany, France, Italy, Spain, Netherlands, Belgium, Austria, Portugal, Greece, Ireland

**Non-Euro Europe**: UK, Switzerland, Sweden

**Americas**: USA, Canada

**Asia-Pacific**: Japan, Australia

## Requirements

- Python 3.10+
- LSEG Eikon terminal or API access
- See `requirements.txt` for packages

## License

MIT License - see [LICENSE](LICENSE)

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Past performance and historical crisis patterns do not predict future results.
