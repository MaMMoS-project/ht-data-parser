from pathlib import Path
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.smartlabmeas import SMARTLABMeas


class SMARTLABScan(HTMeasurement):
    """
    Class to handle a full scan of SMARTLAB XRD measurements.
    Loads all SMARTLABMeas files from a folder.
    """

    MEASUREMENT_CLASS = SMARTLABMeas
    FILE_EXTENSION = ".ras"
