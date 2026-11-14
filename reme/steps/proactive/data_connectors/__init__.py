"""Collect external observations into workspace resource packages."""

from .base_connector import BaseConnectorStep
from .screen_capture import ScreenCaptureStep
from .event import ResourceEventStep
from .finance_news import FinanceNewsStep
from .stock_quotes import StockQuotesStep
from .btc_quotes import BtcQuotesStep
from .work_observation import WorkObservationInitStep, WorkObservationStep

__all__ = [
    "BaseConnectorStep",
    "ScreenCaptureStep",
    "ResourceEventStep",
    "FinanceNewsStep",
    "StockQuotesStep",
    "BtcQuotesStep",
    "WorkObservationInitStep",
    "WorkObservationStep",
]
