from .basemeasurement import BaseMeasurement

import h5py
import re
import fabio
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go


class SMARTLABMeas(BaseMeasurement):
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

    def __init__(self, ras_file: pl.Path):
        super().__init__(ras_file)
        self._read()

    def _get_position(self, axes):
        # Extract stage positions (x/y) from axes
        self.x_position = None
        self.y_position = None

        for name, axis in axes.items():
            position = axis.get("position")

            if name in ("x", "sample_x"):
                self.x_position = position

            elif name in ("y", "sample_y"):
                self.y_position = position

    def _structure_metadata(self):
        # Restructure the fucking mess from .ras metadata file from SMARTLAB
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

        # Setting x and y positions
        self._get_position(axes)

        # Overwriting metadata with structured metadata
        self.metadata = structured

    def _convert_axis_value(self, value, unit):
        # Convert axis position/offset to mu.Quantity if possible.
        if value in (None, "", "-", "None", "No_unit"):
            return None

        # If already a quantity we return it directly
        if isinstance(value, mu.Quantity):
            return value

        # Case when explicit unit is provided
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

        # Otherwise return raw string
        return value

    def _try_load_2d_image(self):
        folder = self.path.parent
        stem = self.path.stem

        # Look for files like: Areamap_011006_XXXX.img
        img_files = sorted(folder.glob(f"{stem}_*.img"))

        if not img_files:
            return  # Nothing to load

        # If multiple images exist, take the first one
        img_path = img_files[0]

        try:
            img = fabio.open(str(img_path))
            image_array = np.array(img.data).astype(np.uint32)

            self.data["2Dimage"] = me.Entity(
                "Xrd2dImage",
                image_array * mu.dimensionless_unscaled,
            )

            # Optional: store filename in metadata
            self.metadata.setdefault("detector", {})
            self.metadata["detector"]["image_file"] = img_path.name

        except Exception:
            # Fail silently (optional logging if you want)
            pass

    def _read(self):
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

        # Sorting the metadata dict
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

        # Try to load 2D image, optional
        self._try_load_2d_image()

    def _parse_lst(self):
        lst_file = self.path.with_suffix(".lst")

        if not lst_file.exists():
            return

        results = {
            "r_coefficients": {},
            "phases": {},
        }

        current_phase = None
        with open(lst_file) as f:
            for line in f:
                line = line.strip()

                # Global R coefficients
                if "Rp=" in line:
                    tokens = re.findall(r"([A-Za-z\-]+)=([\d\.]+)", line)
                    for key, val in tokens:
                        results["r_coefficients"][key] = float(val)

                # Phase fractions
                match = re.match(r"Q([A-Za-z0-9_]+)=([\d\.E+-]+)\+\-([\d\.E+-]+)", line)
                if match:
                    phase, val, err = match.groups()

                    results["phases"][phase] = {
                        "phase_fraction": float(val),
                        "phase_fraction_err": float(err),
                    }
                    continue

                # Phase parameters block
                match = re.search(r"Local parameters.*phase\s+([A-Za-z0-9_]+)", line)

                if match:
                    current_phase = match.group(1)
                    results["phases"].setdefault(current_phase, {})
                    results["phases"][current_phase].setdefault("lattice", {})
                    continue

                if current_phase is None:
                    continue

                # Spacegroup
                match = re.search(r"SpacegroupNo=(\d+)", line)
                if match:
                    results["phases"][current_phase]["spacegroup_number"] = int(
                        match.group(1)
                    )

                # Density
                match = re.search(r"XrayDensity=([\d\.Ee+-]+)", line)
                if match:
                    results["phases"][current_phase]["density"] = float(match.group(1))

                # Lattice parameters
                if any(p in line for p in ("A=", "B=", "C=")):
                    lattice = results["phases"][current_phase].setdefault("lattice", {})

                    for token in line.split():
                        if not token.startswith(("A=", "B=", "C=")):
                            continue

                        axis, val = token.split("=")
                        key = axis.lower()

                        if val == "UNDEF":
                            lattice[key] = np.nan * mu.nm
                            continue

                        if "+-" in val:
                            value, err = val.split("+-")
                            lattice[key] = float(value) * mu.nm
                            lattice[f"{key}_err"] = float(err) * mu.nm

                        else:
                            lattice[key] = float(val) * mu.nm

        self.results = results

    def _parse_par(self):
        def _get_wavelength(line):
            wavelength = None

            anode_wavelengths = {
                "cu": 0.15406,
                "co": 0.178897,
                "mo": 0.07093,
                "fe": 0.19360,
                "cr": 0.22897,
            }  # in nm

            match = re.search(r"SYNCHROTRON=([\d\.Ee+-]+)", line)
            if match:
                wavelength = float(match.group(1))

            match = re.search(r"LAMBDA=([^\s]+)", line)
            if match and wavelength is None:
                val = match.group(1).lower()

                try:
                    wavelength = float(val)
                except ValueError:
                    wavelength = anode_wavelengths.get(val)

            return wavelength

        par_file = self.path.with_suffix(".par")
        if not par_file.exists():
            return

        texture = {}
        with open(par_file) as f:
            for line in f:

                if "LAMBDA=" in line or "SYNCHROTRON=" in line:
                    wavelength = _get_wavelength(line)

                if "TEXTUR=" not in line:
                    continue

                try:
                    parts = line.split()

                    d_spacing = 1 / float(parts[2])
                    hkl = tuple(map(int, parts[-3:]))

                    tex_val = float(re.search(r"TEXTUR=([\d\.]+)", line).group(1))
                    phase = re.search(r"PHASE=([^\s]+)", line).group(1)

                    two_theta = np.degrees(2 * np.arcsin(wavelength / (2 * d_spacing)))

                    # create phase dict
                    texture.setdefault(phase, {})

                    hkl_key = f"{hkl[0]}_{hkl[1]}_{hkl[2]}"

                    texture[phase][hkl_key] = {
                        "d_spacing": d_spacing * mu.nm,
                        "two_theta": two_theta * mu.deg,
                        "texture": tex_val * mu.dimensionless_unscaled,
                    }

                except Exception:
                    print("error parsing line:", line)
                    continue

        self.results["texture"] = texture

    def _parse_dia(self):
        dia_file = self.path.with_suffix(".dia")

        if not dia_file.exists():
            return

        two_theta = []
        measured = []
        calculated = []
        background = []

        phase_names = []
        phase_patterns = {}

        with open(dia_file) as f:
            header = f.readline()

            phase_matches = re.findall(r"STRUC\[\d+\]=([^\s]+)", header)
            phase_names = phase_matches

            for p in phase_names:
                phase_patterns[p] = []

            for line in f:
                parts = line.split()

                if len(parts) < 4:
                    continue
                try:
                    two_theta.append(float(parts[0]))
                    measured.append(float(parts[1]))
                    calculated.append(float(parts[2]))
                    background.append(float(parts[3]))

                    # Single phase extraction
                    for i, phase in enumerate(phase_names):
                        idx = 4 + i

                        if idx < len(parts):
                            phase_patterns[phase].append(float(parts[idx]))
                        else:
                            phase_patterns[phase].append(0.0)

                except ValueError:
                    continue

        two_theta = np.array(two_theta)
        results = {
            "Angle": np.array(two_theta),
            "Total Counts": np.array(measured),
            "Calculated": np.array(calculated),
            "Background": np.array(background),
        }

        for phase in phase_names:
            results[phase] = np.array(phase_patterns[phase])

        self.results["fits"] = results

    def add_results(self):
        self._parse_lst()
        self._parse_par()
        self._parse_dia()
        pass

    def _add_traces(self, fig):
        fig.add_trace(
            go.Scatter(
                x=self.data["Two_theta"].value,
                y=self.data["Counts"].value,
                name="xrd_pattern",
                mode="lines",
            )
        )

    def _xaxis_title(self):
        return f"2θ ({self.data['Two_theta'].unit})"

    def _yaxis_title(self):
        return "Counts"

    def plot_2d_image(self, log_scale: bool = True, remove_zeros: bool = True):
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
            image = image[:, 612:]

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


class ESRFMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "esrf.single_measurement"

    UNIT_MAP = {}

    scan_group: str = None

    def __init__(self, hdf5_file: pl.Path, scan_group: str):
        super().__init__(hdf5_file)
        self.scan_group = scan_group

        # self._read()
