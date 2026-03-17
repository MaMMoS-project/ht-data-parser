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

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self._load_measurements()

    def _load_measurements(self):
        files = sorted(self.folder_path.glob("*.spx"))

        for f in tqdm.tqdm(files):
            meas = EDXMeas(f)
            if (
                np.abs(meas.x_position.value) > 40
                or np.abs(meas.y_position.value) > 40
                or (np.abs(meas.x_position.value) + np.abs(meas.y_position.value)) > 60
            ):
                continue

            self.measurements.append(meas)
