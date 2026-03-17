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

    def __init__(self, folder_path, **kwargs):
        super().__init__()
        self.folder_path = Path(folder_path)
        self._load_measurements()

    def _load_measurements(self):
        """
        Load all profilometry measurements (.asc2d files) from the folder.
        """

        files = sorted(self.folder_path.glob("*.asc2d"))

        for f in tqdm.tqdm(files):
            meas = ProfilMeas(f)

            if (
                np.abs(meas.x_position.value) > 40
                or np.abs(meas.y_position.value) > 40
                or (np.abs(meas.x_position.value) + np.abs(meas.y_position.value)) > 60
            ):
                continue

            self.measurements.append(meas)
