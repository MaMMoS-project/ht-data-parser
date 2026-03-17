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

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self._load_measurements()

    def _load_measurements(self):
        """
        Load all MOKEMeas files in the folder.
        """
        files = sorted(self.folder_path.glob("*_magnetization.txt"))

        for f in tqdm.tqdm(files):

            meas = MOKEMeas(f)
            if (
                np.abs(meas.x_position.value) > 40
                or np.abs(meas.y_position.value) > 40
                or (np.abs(meas.x_position.value) + np.abs(meas.y_position.value)) > 60
            ):
                continue

            self.measurements.append(meas)
