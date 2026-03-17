from pathlib import Path
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.xrdmeas import SMARTLABMeas


class SMARTLABScan(HTMeasurement):
    """
    Class to handle a full scan of SMARTLAB XRD measurements.
    Loads all SMARTLABMeas files from a folder.
    """

    MEASUREMENT_CLASS = SMARTLABMeas

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self._load_measurements()

    def _load_measurements(self):
        files = sorted(self.folder_path.glob("*.ras"))

        for f in tqdm.tqdm(files):
            try:
                meas = SMARTLABMeas(f)
            except Exception as e:
                print(f"Skipping {f.name}: {e}")
                continue

            # Skip measurements without positions
            if meas.x_position is None or meas.y_position is None:
                continue

            if (
                np.abs(meas.x_position.value) > 40
                or np.abs(meas.y_position.value) > 40
                or (np.abs(meas.x_position.value) + np.abs(meas.y_position.value)) > 60
            ):
                continue

            self.measurements.append(meas)
