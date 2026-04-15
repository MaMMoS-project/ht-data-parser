from __future__ import annotations

from typing import Optional
from .basemeas import BaseMeas

import pathlib as pl
import re
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go

# Inch is not a standard mammos_units unit, defining it from mm
mu.inch = 25.4 * mu.mm


class ProfilMeas(BaseMeas):
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
    RESULTS_UNIT_MAP: dict = {
        "measured_thickness": (mu.nm, "LocalThickness"),
    }

    def __init__(self, path: pl.Path, **kwargs) -> None:
        super().__init__(path)
        self._read()

    # ------------------------------------------------------------------
    # Position
    def _set_positions_from_target(self, target_name: str) -> None:
        """Extract wafer (x, y) indices from TargetName string '(row,col)'."""
        match = re.search(r"\((\d+),(\d+)\)", target_name)
        if not match:
            return

        # TargetName format is (row, col) → (y_idx, x_idx)
        x_idx = int(match.group(2))
        y_idx = int(match.group(1))

        self.index_x = x_idx
        self.index_y = y_idx
        self.x_position = (x_idx - 10) * 5 * mu.mm
        self.y_position = (10 - y_idx) * 5 * mu.mm

    # ------------------------------------------------------------------
    # Reading profilometry data
    def _parse_metadata_value(self, raw_value: str) -> mu.Quantity | float | str:
        """Try to parse 'float unit' format (e.g. '10.0 Micrometer'), then plain float."""
        parts = raw_value.split()
        if len(parts) == 2:
            try:
                numeric = float(parts[0])
                unit_str = parts[1]
                if unit_str in self.UNIT_MAP:
                    return numeric * self.UNIT_MAP[unit_str]
            except ValueError:
                pass

        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    def _read(self) -> None:
        metadata: dict = {}
        lateral_positions: list[float] = []
        height_values: list[float] = []
        in_data_section = False

        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("y("):
                    in_data_section = True
                    continue

                if in_data_section:
                    lateral, height = line.split(",")
                    lateral_positions.append(float(lateral))
                    height_values.append(float(height))
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
            "distance": me.Entity(
                "ProfilDistance", np.array(lateral_positions) * mu.um
            ),
            "profile": me.Entity("ProfilTotalProfile", np.array(height_values) * mu.um),
        }

        # Results is not used for profilometry
        del self.results

        if "TargetName" in metadata:
            self._set_positions_from_target(metadata["TargetName"])

    # ------------------------------------------------------------------
    # Plotting
    def _add_traces(self, fig: go.Figure) -> None:
        fig.add_trace(
            go.Scatter(
                x=self.data["distance"].value,
                y=self.data["profile"].value,
                mode="lines",
                name="profile",
            )
        )

    def _xaxis_title(self) -> str:
        return f"Distance ({self.data['distance'].unit})"

    def _yaxis_title(self) -> str:
        return f"Height ({self.data['profile'].unit})"
