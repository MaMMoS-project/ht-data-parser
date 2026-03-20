from pathlib import Path
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.edxmeas import EDXMeas


class EDXScan(HTMeasurement):
    """
    Class to handle a full scan of EDX measurements.
    Loads all EDXMeas files from a folder.
    """

    MEASUREMENT_CLASS = EDXMeas
    FILE_EXTENSION = ".spx"
