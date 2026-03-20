from pathlib import Path
import xml.etree.ElementTree as ET
import tqdm
from .htmeasurement import HTMeasurement
from ..base_measurements.semmeas import SEMMeas


class SEMScan(HTMeasurement):
    MEASUREMENT_CLASS = SEMMeas
    FILE_EXTENSION = "Sample*.png"

    def __init__(self, folder_path):
        self.scan_metadata = {}
        super().__init__(folder_path)

    def _post_load(self):
        self._read_profile_metadata()
        self._propagate_metadata()

    def _read_profile_metadata(self):
        """Parse scan-level metadata from the Profile.rtj2 XML file."""
        profile_path = self.folder_path / "Profile.rtj2"
        if not profile_path.exists():
            print("Profile.rtj2 not found")
            return

        root = ET.parse(profile_path).getroot()
        metadata = {}

        for elem in root.iter():
            if elem.text is None:
                continue
            value = elem.text.strip()
            if not value or elem.tag in ("LineCounter", "Data"):
                continue
            metadata[elem.tag] = _convert_value(value)

        self.scan_metadata = metadata

    def _propagate_metadata(self):
        for meas in self.measurements:
            if meas.metadata is None:
                meas.metadata = {}
            meas.metadata.update(self.scan_metadata)


def _convert_value(text):
    """Try to parse a string as a numeric scalar or tuple, fall back to string."""
    text = text.strip()

    if "," in text:
        try:
            return tuple(float(p) for p in text.split(","))
        except ValueError:
            return text

    try:
        return int(text) if "." not in text else float(text)
    except ValueError:
        return text
