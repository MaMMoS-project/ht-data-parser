from __future__ import annotations

from .basescan import BaseScan
from ..measurements.profilmeas import ProfilMeas


class ProfilScan(BaseScan):
    """Full HT scan of profilometry measurements. Loads all .asc2d files from a folder."""

    MEASUREMENT_CLASS = ProfilMeas
    FILE_EXTENSION = ".asc2d"
