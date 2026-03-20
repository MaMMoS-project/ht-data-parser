from pathlib import Path
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.profilmeas import ProfilMeas


class ProfilScan(HTMeasurement):
    """
    Class to handle a full scan of profilometry measurements.

    Loads all .asc2d profilometry files from a folder and converts them
    into ProfilMeas objects.
    """

    MEASUREMENT_CLASS = ProfilMeas
    FILE_EXTENSION = ".asc2d"
