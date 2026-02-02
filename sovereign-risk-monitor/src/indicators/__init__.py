"""Indicator modules for fiscal data and early warning signals."""

from .fiscal import FiscalDataFetcher, create_fiscal_fetcher
from .early_warning import EarlyWarningSystem

__all__ = ["FiscalDataFetcher", "create_fiscal_fetcher", "EarlyWarningSystem"]
