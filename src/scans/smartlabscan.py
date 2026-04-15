from __future__ import annotations

from .basescan import BaseScan
from ..measurements.smartlabmeas import SmartlabMeas


class SmartlabScan(BaseScan):
    """Full HT scan of Rigaku SmartLab XRD measurements. Loads all .ras files from a folder."""

    MEASUREMENT_CLASS = SmartlabMeas
    FILE_EXTENSION = ".ras"
