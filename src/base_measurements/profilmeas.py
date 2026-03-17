from typing import Optional
from .basemeasurement import BaseMeasurement

import pathlib as pl
import re
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go


class ProfilMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "profil.single_measurement"

    index_x: Optional[int] = None
    index_y: Optional[int] = None

    mu.inch = 25.4 * mu.mm  # Defining inch since it is not a standard unit
    UNIT_MAP = {
        "Micrometer": mu.um,
        "Milligram": mu.mg,
        "Second": mu.second,
        "Degree": mu.degree,
        "Inch": mu.inch,
    }

    def __init__(self, path: pl.Path, **kwargs):
        super().__init__(path)

        self._read()

    def _set_positions_from_target(self, target_name):
        pattern = r"\((\d+),(\d+)\)"
        match = re.search(pattern, target_name)

        if not match:
            return

        # Header tuple has the format (y,x)
        x_idx = int(match.group(2))
        y_idx = int(match.group(1))
        x_pos = (x_idx - 10) * 5
        y_pos = (10 - y_idx) * 5

        self.index_x = x_idx
        self.index_y = y_idx
        self.x_position = x_pos * mu.mm
        self.y_position = y_pos * mu.mm

    def _parse_metadata_value(self, value):
        parts = value.split()

        if len(parts) == 2:
            try:
                val = float(parts[0])
                unit = parts[1]

                if unit in self.UNIT_MAP:
                    return val * self.UNIT_MAP[unit]
            except ValueError:
                pass

        try:
            return float(value)
        except ValueError:
            return value

    def _read(self):
        metadata = {}
        y_values = []
        z_values = []

        data_section = False

        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Start of data table
                if line.startswith("y("):
                    data_section = True
                    continue

                if data_section:
                    y, z = line.split(",")
                    y_values.append(float(y))
                    z_values.append(float(z))

                else:
                    parts = line.split(",", 3)

                    if len(parts) == 4:
                        key = parts[0]
                        value = parts[3]
                        metadata[key] = self._parse_metadata_value(value)

        # Convert to numpy arrays
        y_values = np.array(y_values)
        z_values = np.array(z_values)

        # Converting ScanSpeed to a quantity in metadata (written as string by default)
        if "ScanSpeed" in metadata:
            old_scan_speed = metadata["ScanSpeed"]
            metadata["ScanSpeed"] = (
                float(old_scan_speed.split(" ")[0]) * mu.um / mu.second
            )
        # Save metadata
        self.metadata = metadata

        # Save data
        self.data = {
            "distance": me.Entity("ProfilDistance", y_values * mu.um),
            "profile": me.Entity("ProfilTotalProfile", z_values * mu.um),
        }

        # Fix for DaHU compatibility
        del self.results

        # Extract wafer position
        if "TargetName" in metadata:
            self._set_positions_from_target(metadata["TargetName"])

    def _add_traces(self, fig):
        fig.add_trace(
            go.Scatter(
                x=self.data["distance"].value,
                y=self.data["profile"].value,
                mode="lines",
                name="profile",
            )
        )

    def _xaxis_title(self):
        return f"Distance ({self.data['distance'].unit})"

    def _yaxis_title(self):
        return f"Height ({self.data['profile'].unit})"
