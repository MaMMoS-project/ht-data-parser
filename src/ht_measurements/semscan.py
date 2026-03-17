from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import tqdm

from .htmeasurement import HTMeasurement
from ..base_measurements.semmeas import SEMMeas


class SEMScan(HTMeasurement):
    MEASUREMENT_CLASS = SEMMeas

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.scan_metadata = {}

        self._load_measurements()
        self._read_profile_metadata()
        self._propagate_metadata()

    def _load_measurements(self):
        image_files = sorted(self.folder_path.glob("Sample*.png"))

        for f in tqdm.tqdm(image_files):
            meas = SEMMeas(f)
            if (
                np.abs(meas.x_position.value) > 40
                or np.abs(meas.y_position.value) > 40
                or (np.abs(meas.x_position.value) + np.abs(meas.y_position.value)) > 60
            ):
                continue

            self.measurements.append(meas)

    def _read_profile_metadata(self):
        profile_path = self.folder_path / "Profile.rtj2"

        if not profile_path.exists():
            print("Profile.rtj2 not found")
            self.scan_metadata = {}
            return

        tree = ET.parse(profile_path)
        root = tree.getroot()

        metadata = {}

        def convert_value(text):
            """Try converting text to int/float."""
            text = text.strip()

            if "," in text:
                parts = text.split(",")
                try:
                    return tuple(float(p) for p in parts)
                except ValueError:
                    return text

            try:
                if "." in text:
                    return float(text)
                return int(text)
            except ValueError:
                return text

        for elem in root.iter():
            if elem.text is None:
                continue

            value = elem.text.strip()
            if value == "":
                continue

            if elem.tag == "LineCounter" or elem.tag == "Data":
                continue

            key = elem.tag
            metadata[key] = convert_value(value)

        self.scan_metadata = metadata

    def _propagate_metadata(self):
        for meas in self.measurements:
            if not hasattr(meas, "metadata") or meas.metadata is None:
                meas.metadata = {}

            meas.metadata.update(self.scan_metadata)
