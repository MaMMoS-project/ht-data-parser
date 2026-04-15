from __future__ import annotations

from .basescan import BaseScan
from ..measurements.edxmeas import EdxMeas


class EdxScan(BaseScan):
    """Full HT scan of EDX measurements. Loads all .spx files from a folder."""

    MEASUREMENT_CLASS = EdxMeas
    FILE_EXTENSION = ".spx"
