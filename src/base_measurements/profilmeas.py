from typing import Optional
from .basemeasurement import BaseMeasurement

import pathlib as pl
import re
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go

# inch is not a standard mammos_units unit
mu.inch = 25.4 * mu.mm


class ProfilMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "profil.single_measurement"

    index_x: Optional[int] = None
    index_y: Optional[int] = None

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

        # TargetName tuple is (row, col) to (y_idx, x_idx)
        x_idx = int(match.group(2))
        y_idx = int(match.group(1))

        self.index_x = x_idx
        self.index_y = y_idx
        self.x_position = (x_idx - 10) * 5 * mu.mm
        self.y_position = (10 - y_idx) * 5 * mu.mm

    def _parse_metadata_value(self, value):
        # Try "float unit" format (e.g. "10.0 Micrometer")
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

                if line.startswith("y("):
                    data_section = True
                    continue

                if data_section:
                    y, z = line.split(",")
                    y_values.append(float(y))
                    z_values.append(float(z))
                else:
                    # Metadata lines: key,_,_,value
                    parts = line.split(",", 3)
                    if len(parts) == 4:
                        metadata[parts[0]] = self._parse_metadata_value(parts[3])

        # ScanSpeed is written as a plain string ("10.0 um/s"), needs manual conversion
        if "ScanSpeed" in metadata:
            metadata["ScanSpeed"] = (
                float(str(metadata["ScanSpeed"]).split()[0]) * mu.um / mu.second
            )

        self.metadata = metadata
        self.data = {
            "distance": me.Entity("ProfilDistance", np.array(y_values) * mu.um),
            "profile": me.Entity("ProfilTotalProfile", np.array(z_values) * mu.um),
        }

        # results is not used for profilometry
        del self.results

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
