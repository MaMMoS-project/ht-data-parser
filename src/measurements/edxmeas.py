from __future__ import annotations

from typing import Optional
from .basemeas import BaseMeas

import re
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go
import xml.etree.ElementTree as et


class EdxMeas(BaseMeas):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "edx.single_measurement"

    index_x: Optional[int] = None
    index_y: Optional[int] = None

    # Known units for metadata fields
    UNIT_MAP = {
        "RealTime": mu.microsecond,
        "LifeTime": mu.microsecond,
        "PrimaryEnergy": mu.keV,
        "AccelerationVoltage": mu.kV,
        "BeamCurrent": mu.nA,
        "WorkingDistance": mu.mm,
        "ElevationAngle": mu.degree,
        "DetectorTemperature": mu.deg_C,
        "CalibAbs": mu.keV,
        "CalibLin": mu.keV,
    }
    RESULTS_UNIT_MAP = {}

    def __init__(self, path: pl.Path, **kwargs) -> None:
        super().__init__(path)
        self.step_x: int = kwargs.pop("step_x", 5)
        self.step_y: int = kwargs.pop("step_y", 5)
        self.start_x: int = kwargs.pop("start_x", -40)
        self.start_y: int = kwargs.pop("start_y", -40)
        self._read()

    # ------------------------------------------------------------------
    # Get wafer position from filename
    def _set_positions_from_path(self, file_path: pl.Path) -> None:
        match = re.search(r".*\((\d+),(\d+)\).*", str(file_path))
        if not match:
            return
        x_idx = int(match.group(1))
        y_idx = int(match.group(2))
        self.index_x, self.index_y = x_idx, y_idx
        self.x_position = -((x_idx - 1) * self.step_x + self.start_x) * mu.mm
        self.y_position = ((y_idx - 1) * self.step_y + self.start_y) * mu.mm

    # ------------------------------------------------------------------
    # Reading .spx datafiles from BRUKER
    def _visit_items(self, item: et.Element, edx_dict: Optional[dict] = None) -> dict:
        """Recursively walk the .spx XML tree into a flat dict."""
        if edx_dict is None:
            edx_dict = {}

        parse_ignore = {
            "",
            "DetLayers",
            "ShiftData",
            "PPRTData",
            "ResponseFunction",
            "WindowLayers",
        }

        if item.tag == "ClassInstance" and item.attrib.get("Type") == "TRTPSEElement":
            parent_name = item.attrib["Type"] + " " + item.attrib["Name"]
        elif item.tag in ("Result", "ExtResults"):
            for child in item.iter():
                if child.tag == "Atom":
                    parent_name = item.tag + " " + child.text
        elif item.tag == "ClassInstance":
            parent_name = item.attrib["Type"]
        else:
            parent_name = item.tag

        edx_dict[parent_name] = {}
        for child in item:
            if child.tag in parse_ignore:
                continue
            if not child.findall("./"):
                edx_dict[parent_name][child.tag] = child.text
            else:
                edx_dict = self._visit_items(child, edx_dict)

        return edx_dict

    def _extract_metadata(self, edx_dict: dict) -> dict:
        HEADER_KEYS = [
            "TRTSpectrumHardwareHeader",
            "TRTDetectorHeader",
            "TRTESMAHeader",
            "TRTSpectrumHeader",
            "TRTKnownHeader",
        ]

        metadata: dict = {
            "DetectorName": "Bruker AXS XFlash Detector 4010",
            "SEMModel": "Zeiss Ultra plus",
        }

        for header_key in HEADER_KEYS:
            header = edx_dict.get(header_key, {})
            if not isinstance(header, dict):
                continue
            for key, value in header.items():
                if value is None or value == "":
                    continue
                metadata[key] = self._apply_units(key, value, self.UNIT_MAP)

        return metadata

    def _extract_spectrum(self, edx_dict: dict) -> dict:
        spec = edx_dict.get("TRTSpectrum")
        header = edx_dict.get("TRTSpectrumHeader")

        counts = np.array([int(x) for x in spec["Channels"].split(",")])
        calib_abs = float(header["CalibAbs"])
        calib_lin = float(header["CalibLin"])
        energy = calib_abs + np.arange(len(counts)) * calib_lin

        return {
            "Energy": me.Entity("EdxEnergy", energy * mu.keV),
            "Counts": me.Entity("EdxCounts", counts * mu.dimensionless_unscaled),
        }

    def _atomic_number_to_symbol(
        self, edx_dict: dict, atom_number: int
    ) -> Optional[str]:
        """Look up element symbol from atomic number via TRTPSEElement entries."""
        for key, value in edx_dict.items():
            if key.startswith("TRTPSEElement") and isinstance(value, dict) and value:
                if int(value["Element"]) == atom_number:
                    return key.replace("TRTPSEElement ", "")
        return None

    def _extract_results(self, edx_dict: dict) -> dict:
        def safe_float(val: object) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return np.nan

        results: dict = {}

        for key, value in edx_dict.items():
            if not key.startswith("Result "):
                continue

            element = self._atomic_number_to_symbol(edx_dict, int(value["Atom"]))
            if element is None:
                continue

            element_entities: dict = {}
            atom_percent = safe_float(value.get("AtomPercent", np.nan))
            mass_percent = safe_float(value.get("MassPercent", np.nan))

            # Converting fractions to percents
            if not np.isnan(atom_percent):
                element_entities["AtomPercent"] = me.Entity(
                    "LocalAtomPercent", atom_percent * 100 * mu.percent
                )
            if not np.isnan(mass_percent):
                element_entities["MassPercent"] = me.Entity(
                    "LocalMassPercent", mass_percent * 100 * mu.percent
                )

            if element_entities:
                results[element] = element_entities

        return results

    def _read(self) -> None:
        root = et.parse(self.path).getroot()[1]
        edx_dict = self._visit_items(root)

        self._set_positions_from_path(self.path)
        self.metadata = self._extract_metadata(edx_dict)
        self.data = self._extract_spectrum(edx_dict)
        self.results = self._extract_results(edx_dict)

    # ------------------------------------------------------------------
    # Plotting data
    def _add_traces(self, fig: go.Figure) -> None:
        fig.add_trace(
            go.Scatter(
                x=self.data["Energy"].value,
                y=self.data["Counts"].value,
                name="edx_spectrum",
                mode="lines",
            )
        )

    def _xaxis_title(self) -> str:
        return f"Energy ({self.data['Energy'].unit})"

    def _yaxis_title(self) -> str:
        return "Counts"
