"""Descarga, normalización, validación y persistencia local."""

from quinielator.data.downloader import DataDownloader
from quinielator.data.repository import DataRepository
from quinielator.data.sources import (
    FifaRankingsSource,
    WorldCup2026ResultsSource,
    WorldCupResultsSource,
)
from quinielator.data.validator import DataValidator, ValidationReport

__all__ = [
    "DataDownloader",
    "DataRepository",
    "DataValidator",
    "FifaRankingsSource",
    "ValidationReport",
    "WorldCup2026ResultsSource",
    "WorldCupResultsSource",
]
