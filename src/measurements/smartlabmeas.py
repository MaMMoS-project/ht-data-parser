from __future__ import annotations

from .xrdmeas import XrdMeas

import re
import fabio
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go


class SmartlabMeas(XrdMeas):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "smartlab.single_measurement"

    UNIT_MAP = {
        # X-ray source
        "MEAS_COND_XG_VOLTAGE": mu.kV,
        "MEAS_COND_XG_CURRENT": mu.mA,
        # Scan parameters
        "MEAS_SCAN_START": mu.deg,
        "MEAS_SCAN_STOP": mu.deg,
        "MEAS_SCAN_STEP": mu.deg,
        "MEAS_SCAN_RESOLUTION_X": mu.deg,
        # Wavelengths
        "HW_XG_WAVE_LENGTH_ALPHA1": mu.angstrom,
        "HW_XG_WAVE_LENGTH_ALPHA2": mu.angstrom,
        "HW_XG_WAVE_LENGTH_BETA": mu.angstrom,
    }
    RESULTS_UNIT_MAP = {}

    def __init__(self, ras_file: pl.Path) -> None:
        super().__init__(ras_file)
        self._read()

    # ------------------------------------------------------------------
    # Positions
    def _get_position(self, axes: dict) -> None:
        self.x_position = None
        self.y_position = None

        for name, axis in axes.items():
            position = axis.get("position")

            if name in ("x", "sample_x"):
                self.x_position = position

            elif name in ("y", "sample_y"):
                self.y_position = position

    def _structure_metadata(self) -> None:
        # The .ras header is a flat key-value dump; restructure into logical subgroups
        raw = self.metadata
        structured = {}

        # File info
        structured["file"] = {
            k.replace("FILE_", "").lower(): v
            for k, v in raw.items()
            if k.startswith("FILE_")
        }

        # X-ray source
        structured["xray_source"] = {
            "target": raw.get("HW_XG_TARGET_NAME"),
            "voltage": raw.get("MEAS_COND_XG_VOLTAGE"),
            "current": raw.get("MEAS_COND_XG_CURRENT"),
            "wave_type": raw.get("MEAS_COND_XG_WAVE_TYPE"),
            "wavelength": {
                "alpha1": raw.get("HW_XG_WAVE_LENGTH_ALPHA1"),
                "alpha2": raw.get("HW_XG_WAVE_LENGTH_ALPHA2"),
                "beta": raw.get("HW_XG_WAVE_LENGTH_BETA"),
            },
        }

        # Scan parameters
        structured["scan"] = {
            "axis": raw.get("MEAS_SCAN_AXIS_X"),
            "start": raw.get("MEAS_SCAN_START"),
            "stop": raw.get("MEAS_SCAN_STOP"),
            "step": raw.get("MEAS_SCAN_STEP"),
            "unit_x": raw.get("MEAS_SCAN_UNIT_X"),
            "unit_y": raw.get("MEAS_SCAN_UNIT_Y"),
            "mode": raw.get("MEAS_SCAN_MODE"),
            "speed": raw.get("MEAS_SCAN_SPEED"),
            "speed_unit": raw.get("MEAS_SCAN_SPEED_UNIT"),
            "resolution": raw.get("MEAS_SCAN_RESOLUTION_X"),
        }

        # Axes reconstruction
        axes = {}
        for key in raw:
            if key.startswith("MEAS_COND_AXIS_NAME-"):
                idx = int(key.split("-")[-1])

                name = raw.get(f"MEAS_COND_AXIS_NAME-{idx}")
                if not name:
                    continue

                unit = raw.get(f"MEAS_COND_AXIS_UNIT-{idx}", "")

                position = self._convert_axis_value(
                    raw.get(f"MEAS_COND_AXIS_POSITION-{idx}"),
                    unit,
                )

                offset = self._convert_axis_value(
                    raw.get(f"MEAS_COND_AXIS_OFFSET-{idx}"),
                    unit,
                )

                axes[name.lower()] = {
                    "index": idx,
                    "internal_name": raw.get(f"MEAS_COND_AXIS_NAME_INTERNAL-{idx}"),
                    "position": position,
                    "offset": offset,
                    "unit": unit,
                }

        structured["axes"] = axes

        # Counters
        counters = {}
        for key in raw:
            if key.startswith("HW_COUNTER_NAME-"):
                idx = int(key.split("-")[-1])
                counters[idx] = {
                    "id": raw.get(f"HW_COUNTER_ID-{idx}"),
                    "name": raw.get(f"HW_COUNTER_NAME-{idx}"),
                }

        structured["counters"] = counters

        # Display settings
        structured["display"] = {
            k.replace("DISP_", "").lower(): v
            for k, v in raw.items()
            if k.startswith("DISP_")
        }

        self._get_position(axes)

        self.metadata = structured

    def _convert_axis_value(self, value, unit: str):
        # Convert axis position/offset to mu.Quantity if possible.
        if value in (None, "", "-", "None", "No_unit"):
            return None

        if isinstance(value, mu.Quantity):
            return value

        if unit:
            try:
                numeric = float(value)
                return numeric * getattr(mu, unit)
            except (ValueError, AttributeError):
                pass

        # Case when the value is written as "2.0mm"
        match = re.match(r"([-+]?\d*\.?\d+)\s*([a-zA-Z]+)", str(value))
        if match:
            num, parsed_unit = match.groups()
            try:
                return float(num) * getattr(mu, parsed_unit)
            except AttributeError:
                pass

        return value

    def _try_load_2d_image(self) -> None:
        folder = self.path.parent
        stem = self.path.stem

        img_files = sorted(folder.glob(f"{stem}_*.img"))

        if not img_files:
            return

        img_path = img_files[0]

        try:
            img = fabio.open(str(img_path))
            image_array = np.array(img.data).astype(np.uint32)

            self.data["2Dimage"] = me.Entity(
                "Xrd2dImage",
                image_array * mu.dimensionless_unscaled,
            )

            self.metadata.setdefault("detector", {})
            self.metadata["detector"]["image_file"] = img_path.name

        except Exception:
            pass

    def _read(self) -> None:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        header_start = lines.index("*RAS_HEADER_START\n")
        header_end = lines.index("*RAS_HEADER_END\n")
        data_start = lines.index("*RAS_INT_START\n")

        header_lines = lines[header_start + 1 : header_end]
        data_lines = lines[data_start + 1 :]

        # Parse Header
        for line in header_lines:
            line = line.strip()
            if not line.startswith("*"):
                continue

            try:
                key, value = line[1:].split(" ", 1)
                value = value.strip().strip('"')

                value = self._apply_units(key, value, self.UNIT_MAP)

                self.metadata[key] = value

            except ValueError:
                continue

        self._structure_metadata()

        # Parse Data
        two_theta = []
        intensity = []

        for line in data_lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            two_theta.append(float(parts[0]))
            intensity.append(float(parts[1]))

        two_theta = np.array(two_theta)
        intensity = np.array(intensity)

        # Store with units
        # self.data["two_theta"] = me.Entity("XrdTwoThetaAngles", two_theta * mu.deg)
        # ValueError: Given unit: deg incompatible with ontology. Allowed units for entity XrdTwoThetaAngles are: [Unit(dimensionless)].
        self.data["Two_theta"] = two_theta * mu.deg
        self.data["Counts"] = me.Entity(
            "XrdCounts", intensity * mu.dimensionless_unscaled
        )

        self._try_load_2d_image()

    def _add_traces(self, fig: go.Figure) -> None:
        fig.add_trace(
            go.Scatter(
                x=self.data["Two_theta"].value,
                y=self.data["Counts"].value,
                name="xrd_pattern",
                mode="lines",
            )
        )

    def _xaxis_title(self) -> str:
        return f"2θ ({self.data['Two_theta'].unit})"

    def _yaxis_title(self) -> str:
        return "Counts"

    def plot_2d_image(self, log_scale: bool = True, remove_zeros: bool = True) -> None:
        """
        Display the 2D XRD detector image.

        Parameters
        ----------
        log_scale : bool
            If True, display log10 intensity scale.
        """
        PLOT_STYLE_2D = {
            "height": 600,
            "width": 1150,
            "tick_font_size": 20,
            "title_font_size": 20,
            "colorbar_title_font_size": 20,
            "colorbar_tick_font_size": 20,
        }
        entity = self.data.get("2Dimage")

        if entity is None:
            print("No 2D XRD image found.")
            return

        image = entity.value.astype(float)

        if remove_zeros:
            image = image[
                :, 612:
            ]  # Strip the inactive left-side columns of the detector

        if log_scale:
            image = np.log10(image + 1)

        fig = go.Figure(
            data=go.Heatmap(
                z=image,
                colorscale="Viridis",
                colorbar=dict(
                    title="log10(Cts)" if log_scale else "Counts",
                    tickfont=dict(size=PLOT_STYLE_2D["colorbar_tick_font_size"]),
                ),
            )
        )

        fig.update_layout(
            title="XRD 2D Detector Image",
            xaxis_title="Detector X (pixels)",
            yaxis_title="Detector Y (pixels)",
            yaxis_autorange="reversed",
            height=PLOT_STYLE_2D["height"],
            width=PLOT_STYLE_2D["width"],
        )

        fig.update_layout(
            xaxis=dict(
                tickfont=dict(size=PLOT_STYLE_2D["tick_font_size"]),
                title_font=dict(size=PLOT_STYLE_2D["title_font_size"]),
            ),
            yaxis=dict(
                tickfont=dict(size=PLOT_STYLE_2D["tick_font_size"]),
                title_font=dict(size=PLOT_STYLE_2D["title_font_size"]),
            ),
        )

        fig.show()
