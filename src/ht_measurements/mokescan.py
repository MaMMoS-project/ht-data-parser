from pathlib import Path
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.mokemeas import MOKEMeas


class MOKEScan(HTMeasurement):
    """
    Class to handle a full scan of MOKE measurements.
    Loads all MOKEMeas from a folder and provides utilities
    for heatmaps and extracting quantities.
    """

    MEASUREMENT_CLASS = MOKEMeas
    FILE_EXTENSION = "_magnetization.txt"
