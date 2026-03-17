from .basemeasurement import BaseMeasurement

import h5py
import re
import pathlib as pl
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go
import numpy as np


class MOKEMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "moke.single_measurement"

    point_id: str = None

    UNIT_MAP = {
        "Pulse_voltage": mu.V,
        "Coil_to_Sample_gap": mu.um,
        "Scanning_dimension_y": mu.mm,
        "Scanning_dimension_x": mu.mm,
    }

    def __init__(self, magnetization_file: pl.Path):
        # Initialize from the abstract BaseMeasurement class
        super().__init__(magnetization_file)

        # Extract point id from filename
        self.point_id = self.path.stem.split("_")[0]

        self._read()

    def _read(self):
        self.metadata = self._parse_info()
        self._parse_position_from_filename()
        # self.data = self._parse_data_files()
        self.data = self._parse_data_files_dahu()

    def _parse_info(self):
        info_file = self.path.parent / "info.txt"
        metadata = {}

        with open(info_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line:
                    continue
                if line.startswith("#"):
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert booleans
                    if value.lower() == "true":
                        metadata[key] = True
                        continue
                    elif value.lower() == "false":
                        metadata[key] = False
                        continue

                    # Using clean key to save label with removed units in label (only for float values)
                    clean_key = re.sub(r"\(.*?\)", "", key).strip()
                    value = self._apply_units(clean_key, value, self.UNIT_MAP)

                    metadata[clean_key] = value

        return metadata

    def _parse_position_from_filename(self):
        """
        Extract x and y from one of the point files.
        """

        pattern = r"x(-?\d+\.?\d*)_y(-?\d+\.?\d*)"
        match = re.search(pattern, self.path.name)

        if not match:
            raise ValueError(f"Could not extract position from {self.path.name.name}")

        x_val = float(match.group(1))
        y_val = float(match.group(2))

        self.x_position = x_val * mu.mm
        self.y_position = y_val * mu.mm

    def _parse_data_files(self):
        """
        Load magnetization, pulse, and sum files
        """

        file_types = ["Magnetization", "Pulse", "Sum"]
        data = {}

        for ftype in file_types:
            # Read the datafile and transpose the array to have columns for each acquisition
            file_path = next(
                self.path.parent.glob(f"{self.point_id}_x*_y*_{ftype.lower()}.txt")
            )
            array = np.loadtxt(file_path, skiprows=1).T

            data[ftype] = array * mu.V

        # Adding time to match moke measurement
        time_step = 0.05  # in microseconds (or 50ns)
        n_points = data["Magnetization"].shape[1]
        data["Time"] = np.arange(n_points) * time_step * mu.us

        return data

    def _parse_data_files_dahu(self):
        # For DaHU compatible data
        file_types = {
            "magnetization": "magnetization",
            "pulse": "pulse",
            "reflectivity": "sum",
        }

        raw_data = {}
        for key, fname in file_types.items():
            file_path = next(
                self.path.parent.glob(f"{self.point_id}_x*_y*_{fname}.txt")
            )

            array = np.loadtxt(file_path, skiprows=1).T
            raw_data[key] = array * mu.V

        n_shots = raw_data["magnetization"].shape[0]
        n_points = raw_data["magnetization"].shape[1]

        data = {}
        for i in range(n_shots):
            shot_key = f"shot_{i+1}"
            data[shot_key] = {}

            data[shot_key][f"magnetization_{i+1}"] = raw_data["magnetization"][i]
            data[shot_key][f"pulse_{i+1}"] = raw_data["pulse"][i]
            data[shot_key][f"reflectivity_{i+1}"] = raw_data["reflectivity"][i]
            data[shot_key][f"integrated_pulse_{i+1}"] = self.moke_integrate_pulse_array(
                raw_data["pulse"][i]
            )

        # Mean signals
        data["shot_mean"] = {}
        data["shot_mean"]["magnetization_mean"] = raw_data["magnetization"].mean(axis=0)
        data["shot_mean"]["pulse_mean"] = raw_data["pulse"].mean(axis=0)
        data["shot_mean"]["reflectivity_mean"] = raw_data["reflectivity"].mean(axis=0)
        data["shot_mean"]["integrated_pulse_mean"] = self.moke_integrate_pulse_array(
            raw_data["pulse"].mean(axis=0)
        )

        # Time axis
        time_step = 0.05  # µs
        data["time"] = np.arange(n_points) * time_step * mu.us

        return data

    @staticmethod
    def moke_integrate_pulse_array(pulse_array):
        values = pulse_array.value.copy()
        unit = pulse_array.unit

        values[(np.abs(values) >= 0.0016666) & (np.abs(values) <= 0.0016668)] = 0
        values[(np.abs(values) >= 0.0045) & (np.abs(values) <= 0.0055)] = 0

        values[values == 0] = np.nan
        isnan = np.isnan(values)

        field_array = np.cumsum(np.where(isnan, 0, values))
        field_array[isnan] = np.nan

        return field_array * unit

    def _add_traces(self, fig):
        # Magnetization
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

        # Pulse
        fig.add_trace(
            go.Scatter(
                x=self.data["time"].value,
                y=self.data["shot_mean"]["pulse_mean"].value,
                name="moke_pulse",
                mode="lines",
            )
        )

        # Reflectivity
        fig.add_trace(
            go.Scatter(
                x=self.data["time"].value,
                y=self.data["shot_mean"]["reflectivity_mean"].value / 10,
                name="moke_reflectivity",
                mode="lines",
            )
        )

    def _xaxis_title(self):
        return f"Time ({self.data['time'].unit})"

    def _yaxis_title(self):
        return "Signal (V)"
