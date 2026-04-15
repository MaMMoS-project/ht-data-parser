from __future__ import annotations

from .basescan import BaseScan
from ..measurements.mokemeas import MokeMeas


class MokeScan(BaseScan):
    """Full HT scan of MOKE measurements. Loads all magnetization .txt files from a folder."""

    MEASUREMENT_CLASS = MokeMeas
    FILE_EXTENSION = "_magnetization.txt"
