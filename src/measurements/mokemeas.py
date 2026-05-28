from __future__ import annotations

from typing import Optional
from .basemeas import BaseMeas

import re
import pathlib as pl
import numpy as np
import mammos_units as mu
import plotly.graph_objects as go


class MokeMeas(BaseMeas):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "moke.single_measurement"

    point_id: Optional[str] = None

    UNIT_MAP = {
        "Pulse_voltage": mu.V,
        "Coil_to_Sample_gap": mu.um,
        "Scanning_dimension_y": mu.mm,
        "Scanning_dimension_x": mu.mm,
    }
    RESULTS_UNIT_MAP: dict = {
        # LocalCoercivity incompatible with T, has to be fixed, set to None for now
        # MokeKerrSignal unit is V, ontology set to None for now
        # LocalReflectivity unit is V, ontology set to None for now
        "coercivity_m0.mean": (mu.T, None),
        "max_kerr_signal": (mu.V, None),
        "reflectivity": (mu.V, None),
    }

    def __init__(self, magnetization_file: pl.Path) -> None:
        super().__init__(magnetization_file)
        self.point_id = self.path.stem.split("_")[0]
        self._read()

    # ------------------------------------------------------------------
    # Reading datafile
    def _read(self) -> None:
        self.metadata = self._parse_info()
        self._parse_position_from_filename()
        self.data = self._parse_data_files_dahu()

    def _parse_info(self) -> dict:
        info_file = self.path.parent / "info.txt"
        metadata: dict = {}

        with open(info_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                if value.lower() == "true":
                    metadata[key] = True
                    continue
                if value.lower() == "false":
                    metadata[key] = False
                    continue

                # Strip unit annotation from key label (e.g. "Pulse_voltage (V)")
                clean_key = re.sub(r"\(.*?\)", "", key).strip()
                metadata[clean_key] = self._apply_units(clean_key, value, self.UNIT_MAP)

        return metadata

    def _parse_position_from_filename(self) -> None:
        match = re.search(r"x(-?\d+\.?\d*)_y(-?\d+\.?\d*)", self.path.name)
        if not match:
            raise ValueError(f"Could not extract position from '{self.path.name}'")
        self.x_position = float(match.group(1)) * mu.mm
        self.y_position = float(match.group(2)) * mu.mm

    def _parse_data_files_dahu(self) -> dict:
        file_type_map = {
            "magnetization": "magnetization",
            "pulse": "pulse",
            "reflectivity": "sum",
        }

        raw_data: dict = {}
        for signal_key, filename_suffix in file_type_map.items():
            file_path = next(
                self.path.parent.glob(f"{self.point_id}_x*_y*_{filename_suffix}.txt")
            )
            raw_data[signal_key] = np.loadtxt(file_path, skiprows=1).T
            if np.ndim(raw_data[signal_key]) < 2:
                raw_data[signal_key] = raw_data[signal_key].reshape(1, -1)

            raw_data[signal_key] = raw_data[signal_key] * mu.V

        n_shots = raw_data["magnetization"].shape[0]
        n_points = raw_data["magnetization"].shape[1]
        data: dict = {}

        for i in range(n_shots):
            shot_key = f"shot_{i + 1}"
            data[shot_key] = {
                f"magnetization_{i + 1}": raw_data["magnetization"][i],
                f"pulse_{i + 1}": raw_data["pulse"][i],
                f"reflectivity_{i + 1}": raw_data["reflectivity"][i],
                f"integrated_pulse_{i + 1}": self.moke_integrate_pulse_array(
                    raw_data["pulse"][i]
                ),
            }

        # Mean signals across all shots
        data["shot_mean"] = {
            "magnetization_mean": raw_data["magnetization"].mean(axis=0),
            "pulse_mean": raw_data["pulse"].mean(axis=0),
            "reflectivity_mean": raw_data["reflectivity"].mean(axis=0),
            "integrated_pulse_mean": self.moke_integrate_pulse_array(
                raw_data["pulse"].mean(axis=0)
            ),
        }

        time_step = 0.05  # µs per sample
        data["time"] = np.arange(n_points) * time_step * mu.us

        return data

    @staticmethod
    def moke_integrate_pulse_array(pulse_array: mu.Quantity) -> mu.Quantity:
        values = pulse_array.value.copy()
        unit = pulse_array.unit

        # Zero out known spurious voltage levels before integrating
        values[(np.abs(values) >= 0.0016666) & (np.abs(values) <= 0.0016668)] = 0
        values[(np.abs(values) >= 0.0045) & (np.abs(values) <= 0.0055)] = 0

        values[values == 0] = np.nan
        isnan = np.isnan(values)

        integrated = np.cumsum(np.where(isnan, 0, values))
        integrated[isnan] = np.nan

        return integrated * unit

    # ------------------------------------------------------------------
    # Plotting raw MOKE data
    def _add_traces(self, fig: go.Figure) -> None:
        magnetization_offset = self.data["shot_mean"]["magnetization_mean"][0].value

        fig.add_trace(
            go.Scatter(
                x=self.data["time"].value,
                y=self.data["shot_mean"]["magnetization_mean"].value
                - magnetization_offset,
                name="moke_magnetization",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data["time"].value,
                y=self.data["shot_mean"]["pulse_mean"].value,
                name="moke_pulse",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data["time"].value,
                y=self.data["shot_mean"]["reflectivity_mean"].value / 10,
                name="moke_reflectivity",
                mode="lines",
            )
        )

    def _xaxis_title(self) -> str:
        return f"Time ({self.data['time'].unit})"

    def _yaxis_title(self) -> str:
        return "Signal (V)"
