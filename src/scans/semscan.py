from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from .basescan import BaseScan
from ..measurements.semmeas import SemMeas


class SemScan(BaseScan):
    """
    Full HT scan of SEM images.

    Expects a Profile.rtj2 XML file at the folder root for scan-level metadata,
    which is propagated to every individual SemMeas after loading.
    """

    MEASUREMENT_CLASS = SemMeas
    FILE_EXTENSION = "Sample*.png"

    def __init__(self, folder_path: Path) -> None:
        self.scan_metadata: dict = {}
        super().__init__(folder_path)

    # ------------------------------------------------------------------
    # Post-load hook to retrieve metadata from .rtj2 file
    def _post_load(self) -> None:
        self._read_profile_metadata()
        self._propagate_scan_metadata()

    # ------------------------------------------------------------------
    # Metadata retrieving
    def _read_profile_metadata(self) -> None:
        """Parse scan-level metadata from the Profile.rtj2 XML file."""
        profile_path = self.folder_path / "Profile.rtj2"
        if not profile_path.exists():
            print("Profile.rtj2 not found")
            return

        root = ET.parse(profile_path).getroot()
        metadata: dict = {}

        for elem in root.iter():
            if elem.text is None:
                continue
            text_value = elem.text.strip()
            if not text_value or elem.tag in ("LineCounter", "Data"):
                continue
            metadata[elem.tag] = _parse_xml_value(text_value)

        self.scan_metadata = metadata

    def _propagate_scan_metadata(self) -> None:
        """Copy scan-level metadata into every individual measurement."""
        for meas in self.measurements:
            if meas.metadata is None:
                meas.metadata = {}
            meas.metadata.update(self.scan_metadata)

    # ------------------------------------------------------------------
    # Analysis section
    def plot_image_at(
        self, x_mm: float, y_mm: float, tolerance_mm: float = 0.1
    ) -> None:
        """Display the SEM image at wafer position (x_mm, y_mm)."""
        return self.get_measurement_at(x_mm, y_mm, tolerance_mm).plot()


def _parse_xml_value(text: str) -> tuple | int | float | str:
    """Try to parse an XML text node as a numeric scalar or tuple, fall back to string."""
    text = text.strip()

    if "," in text:
        try:
            return tuple(float(part) for part in text.split(","))
        except ValueError:
            return text

    try:
        return int(text) if "." not in text else float(text)
    except ValueError:
        return text
