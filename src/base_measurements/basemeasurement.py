from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import h5py
import csv
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go


@dataclass(init=False)
class BaseMeasurement(ABC):
    path: pl.Path
    FORMAT_VERSION = "1.0.0"
    SCHEMA = None

    x_position: float = None
    y_position: float = None

    metadata: Dict = field(default_factory=dict)
    data: Dict = field(default_factory=dict)
    results: Dict = field(default_factory=dict)

    @property
    def position(self):
        return (self.x_position, self.y_position)

    def __init__(self, path: pl.Path):
        self.path = pl.Path(path)

        self.metadata = {}
        self.data = {}
        self.results = {}

        self.x_position = None
        self.y_position = None

    @abstractmethod
    def _read(self):
        pass

    def to_hdf5(self, hdf5_path, group_name=None, mode="w"):
        """
        Write the measurement class to an HDF5 file.

        Parameters
        ----------
        hdf5_path : Path or str
            Path to the HDF5 file.
        group_name : str, optional
            Name of the group in the HDF5 file. If not provided, uses the point ID if available,
            otherwise uses the stem of the path.
        mode : str, optional
            Mode to open the HDF5 file. Defaults to "w".

        Notes
        -----
        If the group already exists in the HDF5 file, it is deleted before writing.
        """
        if isinstance(hdf5_path, str):
            hdf5_path = pl.Path(hdf5_path)

        if group_name is None:
            if hasattr(self, "x_position") and hasattr(self, "y_position"):
                if self.x_position is not None and self.y_position is not None:
                    group_name = (
                        f"({self.x_position.value:.1f},{self.y_position.value:.1f})"
                    )

            else:
                group_name = "group_default"

        with h5py.File(hdf5_path, mode) as h5:
            if group_name in h5:
                del h5[group_name]

            grp = h5.create_group(group_name)

            self._write_root_attributes(grp)

            self._write_positions(grp)
            self._write_metadata(grp)
            self._write_measurement(grp)

            # Fix for DaHU compatibility
            if hasattr(self, "results"):
                self._write_results(grp)

    # Internal functions
    def _write_dict(self, h5parent, dictionary):
        for key, value in dictionary.items():
            # Convert int keys (axes indices) to string
            key = str(key)

            if isinstance(value, dict):
                subgrp = h5parent.create_group(key)
                self._write_dict(subgrp, value)

            else:
                self._write_value(h5parent, key, value)

    def _apply_units(self, key, value, unit_map):
        if value is None:
            return value

        if key in unit_map:
            try:
                return float(value) * unit_map[key]
            except Exception:
                return value

        return value

    def _write_value(self, h5parent, key, value):
        if value is None:
            return

        # Only compress large detector images
        compress = False
        if isinstance(value, me.Entity):
            if str(getattr(value, "ontology", None)).split(".")[-1] in (
                "Xrd2dImage",
                "SEMImage",
            ):
                compress = True
                data_to_store = np.array(value.value, dtype=np.uint32)
            else:
                data_to_store = value.value
        elif isinstance(value, mu.Quantity):
            data_to_store = value.value
        else:
            data_to_store = value

        # Skip if still None after extracting value
        if data_to_store is None:
            return

        # Decide how to write
        if isinstance(data_to_store, np.ndarray) and compress:
            dset = h5parent.create_dataset(
                key, data=data_to_store, compression="gzip", compression_opts=1
            )
        else:
            try:
                dset = h5parent.create_dataset(key, data=data_to_store)
            except TypeError:
                # fallback: convert to string if data type is unsupported
                dset = h5parent.create_dataset(key, data=str(data_to_store))

        # Add attributes if present
        if isinstance(value, me.Entity) and hasattr(value, "unit"):
            dset.attrs["unit"] = str(value.unit)
        if (
            isinstance(value, me.Entity)
            and hasattr(value, "ontology")
            and value.ontology
        ):
            dset.attrs["ontology"] = str(value.ontology)
        if isinstance(value, mu.Quantity):
            dset.attrs["unit"] = str(value.unit)

    def _write_root_attributes(self, grp):
        grp.attrs["source_file"] = str(self.path)
        grp.attrs["type"] = self.__class__.__name__
        grp.attrs["format_version"] = self.FORMAT_VERSION
        grp.attrs["schema"] = self.SCHEMA

        # For backwards compatibility with DaHU
        grp.attrs["ignored"] = False

    def _write_positions(self, grp):
        # Global position identifiers
        if hasattr(self, "x_position") and self.x_position is not None:
            self._write_value(grp, "x_position", self.x_position)

        if hasattr(self, "y_position") and self.y_position is not None:
            self._write_value(grp, "y_position", self.y_position)

        # Measurement-specific identifiers
        if hasattr(self, "point_id") and self.point_id is not None:
            grp.attrs["point_id"] = self.point_id
        if hasattr(self, "index_x") and self.index_x is not None:
            grp.attrs["index_x"] = self.index_x
        if hasattr(self, "index_y") and self.index_y is not None:
            grp.attrs["index_y"] = self.index_y

    def _write_metadata(self, grp):
        meta_grp = grp.create_group("instrument")
        meta_grp.attrs["NX_class"] = "HTinstrument"
        # For backwards compatibility with DaHU
        meta_grp.create_dataset("x_pos", data=self.x_position.value)
        meta_grp.create_dataset("y_pos", data=self.y_position.value)

        self._write_dict(meta_grp, self.metadata)

    def _write_measurement(self, grp):
        meas_grp = grp.create_group("measurement")
        meas_grp.attrs["NX_class"] = "HTdata"

        for key, value in getattr(self, "data", {}).items():
            if isinstance(value, dict):
                subgrp = meas_grp.create_group(key)
                self._write_dict(subgrp, value)
            else:
                self._write_value(meas_grp, key, value)

    def _write_results(self, grp):
        results_grp = grp.create_group("results")
        results_grp.attrs["NX_class"] = "HTresults"

        self._write_dict(results_grp, self.results)

    @classmethod
    def from_hdf5(cls, hdf5_path, group_name, load_2d_images=True):
        """
        Reads an HDF5 file into a BaseMeasurement object.

        Parameters
        ----------
        hdf5_path : str or pathlib.Path
            Path to the HDF5 file.
        group_name : str
            Name of the HDF5 group to read from.

        Returns
        -------
        obj : BaseMeasurement
            A BaseMeasurement object with data from the HDF5 file.
        """

        def _read_group(h5group):
            result = {}
            for key, item in h5group.items():
                if isinstance(item, h5py.Group):
                    result[key] = _read_group(item)
                else:
                    result[key] = _read_value(item)

            return result

        if isinstance(hdf5_path, str):
            hdf5_path = pl.Path(hdf5_path)

        with h5py.File(hdf5_path, "r") as h5:
            grp = h5[group_name]

            if grp.attrs.get("schema") != cls.SCHEMA:
                raise ValueError("Incompatible schema")

            obj = cls.__new__(cls)

            def _read_value(h5item):
                data = h5item[()]
                unit = h5item.attrs.get("unit")
                ontology = h5item.attrs.get("ontology")

                # Looking for special patterns in unit (e.g. "percent", "m/s")
                if unit:
                    if unit == "%":
                        unit = "percent"
                    if "/" in unit:
                        unit = unit.replace(" ", "").split("/")
                        quantity = data * getattr(mu, unit[0]) / getattr(mu, unit[1])
                    else:
                        quantity = data * getattr(mu, unit)
                else:
                    quantity = data

                if ontology:
                    return me.Entity(ontology.split(".")[-1], quantity)

                return quantity

            obj.path = pl.Path(grp.attrs.get("source_file", "from_hdf5"))
            obj.x_position = None
            obj.y_position = None

            if "x_position" in grp:
                obj.x_position = _read_value(grp["x_position"])
            if "y_position" in grp:
                obj.y_position = _read_value(grp["y_position"])

            # Metadata from instrument group
            obj.metadata = {}
            if "instrument" in grp:
                obj.metadata = _read_group(grp["instrument"])
            else:
                obj.metadata = {}

            # Measurement
            obj.data = {}
            if "measurement" in grp:
                for key in grp["measurement"]:
                    # Skipping large detector image if requested
                    if not load_2d_images and key == "2Dimage":
                        continue

                    if isinstance(grp["measurement"][key], h5py.Group):
                        obj.data[key] = _read_group(grp["measurement"][key])
                    else:
                        obj.data[key] = _read_value(grp["measurement"][key])

            # Results
            obj.results = {}
            if "results" in grp:
                obj.results = _read_group(grp["results"])
            else:
                obj.results = {}

            # MOKE: point_id
            if hasattr(obj, "point_id"):
                obj.point_id = grp.attrs.get("point_id", None)

            # EDX: index_x and index_y
            if hasattr(obj, "index_x"):
                obj.index_x = grp.attrs.get("index_x", None)
            if hasattr(obj, "index_y"):
                obj.index_y = grp.attrs.get("index_y", None)

            return obj

    def list_quantities(self):
        """
        Recursively list all plottable quantities in the measurement.

        Returns
        -------
        list[str]
            List of dot-separated paths to quantities.
        """
        paths = []

        def explore(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    explore(v, new_prefix)

            # Entity or Quantity
            elif hasattr(obj, "value"):
                paths.append(prefix)

            # Numpy array or scalar
            elif isinstance(obj, (int, float, np.ndarray)):
                paths.append(prefix)

            # Object with attributes
            elif hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    if k.startswith("_"):
                        continue
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    explore(v, new_prefix)

        explore(self)

        return sorted(set(paths))

    def plot(self):
        """
        Plot a measurement.

        This function will create a figure with a single trace based on
        the measurement data. The title of the figure is the name of
        the measurement file. The x-axis and y-axis labels are
        generated based on the measurement data.

        The plot style is defined in PLOT_STYLE with the following
        parameters:
        - height: 800
        - width: 1150
        - tick_font_size: 20
        - title_font_size: 20

        Returns:
        fig: go.Figure()
            The figure object containing the plot.
        """
        # Default plot style
        PLOT_STYLE = {
            "height": 800,
            "width": 1150,
            "tick_font_size": 20,
            "title_font_size": 20,
        }

        fig = go.Figure()

        # Instrument will be defined by child class
        self._add_traces(fig)

        # Base layout
        fig.update_layout(
            title=self.path.stem,
            xaxis_title=self._xaxis_title(),
            yaxis_title=self._yaxis_title(),
            height=PLOT_STYLE["height"],
            width=PLOT_STYLE["width"],
        )

        fig.update_layout(
            xaxis=dict(
                tickfont=dict(size=PLOT_STYLE["tick_font_size"]),
                title_font=dict(size=PLOT_STYLE["title_font_size"]),
            ),
            yaxis=dict(
                tickfont=dict(size=PLOT_STYLE["tick_font_size"]),
                title_font=dict(size=PLOT_STYLE["title_font_size"]),
            ),
        )

        return fig

    def _add_traces(self, fig):
        raise NotImplementedError

    def _xaxis_title(self):
        return ""

    def _yaxis_title(self):
        return ""
