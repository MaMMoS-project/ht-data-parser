from typing import Optional
from .basemeasurement import BaseMeasurement

import h5py
import re
import pathlib as pl
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as et


class EDXMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "edx.single_measurement"

    index_x: Optional[int] = None
    index_y: Optional[int] = None

    # List of some known units associated with metadata, can be extended
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

    def __init__(self, path: pl.Path, **kwargs):
        # Initialize from the abstract BaseMeasurement class
        super().__init__(path)

        # Default scan parameters
        self.step_x = kwargs.pop("step_x", 5)
        self.step_y = kwargs.pop("step_y", 5)
        self.start_x = kwargs.pop("start_x", -40)
        self.start_y = kwargs.pop("start_y", -40)

        self._read()

    def _set_positions_from_path(self, filepath):
        filepath = str(filepath)
        pattern = r".*\((\d+),(\d+)\).*"
        match = re.search(pattern, filepath)
        x_idx = int(match.group(1))
        y_idx = int(match.group(2))
        x_pos, y_pos = (
            -((x_idx - 1) * self.step_x + self.start_x),
            (y_idx - 1) * self.step_y + self.start_y,
        )
        self.index_x, self.index_y = x_idx, y_idx
        self.x_position, self.y_position = x_pos * mu.mm, y_pos * mu.mm

    def _visit_items(self, item, edx_dict=None):
        if edx_dict is None:
            edx_dict = {}

        parse_ignore = [
            "",
            "DetLayers",
            "ShiftData",
            "PPRTData",
            "ResponseFunction",
            "WindowLayers",
        ]

        # Extract the name of the parent element
        if item.tag == "ClassInstance" and item.attrib["Type"] == "TRTPSEElement":
            parent_name = item.attrib["Type"] + " " + item.attrib["Name"]
        elif item.tag == "Result" or item.tag == "ExtResults":
            for child in item.iter():
                if child.tag == "Atom":
                    parent_name = item.tag + " " + child.text
        elif item.tag == "ClassInstance":
            parent_name = item.attrib["Type"]
        else:
            parent_name = item.tag

        # Builds a nested dictionary with all the edx metadata
        edx_dict.update({parent_name: {}})
        for child in item:
            if child.tag in parse_ignore:
                continue
            elif not child.findall("./"):
                edx_dict[parent_name][child.tag] = child.text
            else:
                # Recursively visit the child elements
                edx_dict = self._visit_items(child, edx_dict)

        return edx_dict

    def _extract_metadata(self, edx_dict):
        metadata = {}

        HEADER_KEYS = [
            "TRTSpectrumHardwareHeader",
            "TRTDetectorHeader",
            "TRTESMAHeader",
            "TRTSpectrumHeader",
            "TRTKnownHeader",
        ]

        # Adding detector name and SEM model manually
        metadata["DetectorName"] = "Bruker AXS XFlash Detector 4010"
        metadata["SEMModel"] = "Zeiss Ultra plus"

        for header_key in HEADER_KEYS:
            header = edx_dict.get(header_key, {})
            if not isinstance(header, dict):
                continue

            for key, value in header.items():
                if value is None or value == "":
                    continue

                # Apply unit to value if unit in UNIT_MAP
                value = self._apply_units(key, value, self.UNIT_MAP)

                metadata[key] = value

        return metadata

    def _extract_spectrum(self, edx_dict):
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

    def _atomic_number_to_symbol(self, edx_dict, atom_number):
        atomic_symbol = None

        # Looks at the TRTPSEElement name to find the atomic symbol from the atom number
        for key, value in edx_dict.items():
            if key.startswith("TRTPSEElement") and isinstance(value, dict) and value:
                if int(value["Element"]) == atom_number:
                    atomic_symbol = key.replace("TRTPSEElement ", "")
                    break

        return atomic_symbol

    def _extract_results(self, edx_dict):
        def safe_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return np.nan

        results = {}

        for key, value in edx_dict.items():
            if key.startswith("Result "):

                atom_number = int(value["Atom"])
                element = self._atomic_number_to_symbol(edx_dict, atom_number)
                if element is None:
                    continue

                element_entities = {}

                atom_percent = safe_float(value.get("AtomPercent", np.nan))
                mass_percent = safe_float(value.get("MassPercent", np.nan))

                # Here we need to convert fractions to percent (*100)
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

    def _read(self):
        # Reading EDX data in .spx files
        root = et.parse(self.path).getroot()[1]
        edx_dict = self._visit_items(root)

        # Extracting x and y wafer positions
        self._set_positions_from_path(self.path)

        # Extracting EDX data from dictionary
        self.metadata = self._extract_metadata(edx_dict)
        self.data = self._extract_spectrum(edx_dict)
        self.results = self._extract_results(edx_dict)

    def _add_traces(self, fig):
        fig.add_trace(
            go.Scatter(
                x=self.data["Energy"].value,
                y=self.data["Counts"].value,
                name="edx_spectrum",
                mode="lines",
            )
        )

    def _xaxis_title(self):
        return f"Energy ({self.data['Energy'].unit})"

    def _yaxis_title(self):
        return "Counts"
