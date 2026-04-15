from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import h5py
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go

# Allows conversion between T and A/m in mammos_unit
mu.set_enabled_equivalencies(mu.magnetic_flux_field())


@dataclass(init=False)
class BaseMeas(ABC):
    FORMAT_VERSION = "1.0.0"
    SCHEMA: Optional[str] = None
    RESULTS_UNIT_MAP: Dict[str, tuple[Any, Optional[str]]] = field(default_factory=dict)

    path: pl.Path
    x_position: Optional[mu.Quantity] = None
    y_position: Optional[mu.Quantity] = None

    metadata: Dict = field(default_factory=dict)
    data: Dict = field(default_factory=dict)
    results: Dict = field(default_factory=dict)

    @property
    def position(self) -> tuple[Optional[mu.Quantity], Optional[mu.Quantity]]:
        return (self.x_position, self.y_position)

    def __init__(self, path: pl.Path) -> None:
        self.path = pl.Path(path)
        self.metadata = {}
        self.data = {}
        self.results = {}
        self.x_position = None
        self.y_position = None

    @abstractmethod
    def _read(self) -> None:
        pass

    # ------------------------------------------------------------------
    # HDF5 export from python class
    def to_hdf5(
        self,
        hdf5_path: pl.Path | str,
        group_name: Optional[str] = None,
        mode: str = "w",
    ) -> None:
        """
        Write the measurement to an HDF5 file.

        Parameters
        ----------
        hdf5_path : Path or str
        group_name : str, optional
            Defaults to '(x,y)' from position if available, else 'group_default'.
        mode : str
            h5py file open mode. Defaults to 'w'.

        Notes
        -----
        If the group already exists it is deleted before writing.
        """
        hdf5_path = pl.Path(hdf5_path)

        if group_name is None:
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

    @classmethod
    def from_hdf5(
        cls,
        hdf5_path: pl.Path | str,
        group_name: str,
        load_2d_images: bool = True,
    ) -> BaseMeas:
        """
        Read a measurement from an HDF5 group.

        Parameters
        ----------
        hdf5_path : Path or str
        group_name : str
        load_2d_images : bool
            If False, skip loading '2Dimage' datasets to save memory.
        """

        def _read_group(h5group: h5py.Group) -> dict:
            result = {}
            for key, item in h5group.items():
                result[key] = (
                    _read_group(item)
                    if isinstance(item, h5py.Group)
                    else _read_value(item)
                )
            return result

        def _read_value(h5item: h5py.Dataset) -> Any:
            raw = h5item[()]
            unit = h5item.attrs.get("unit")
            ontology = h5item.attrs.get("ontology")

            if unit:
                if unit == "%":
                    unit = "percent"
                if "/" in unit:
                    parts = unit.replace(" ", "").split("/")
                    quantity = raw * getattr(mu, parts[0]) / getattr(mu, parts[1])
                else:
                    quantity = raw * getattr(mu, unit)
            else:
                quantity = raw

            if ontology:
                return me.Entity(ontology.split(".")[-1], quantity)
            return quantity

        hdf5_path = pl.Path(hdf5_path)

        with h5py.File(hdf5_path, "a") as h5:
            grp = h5[group_name]

            if grp.attrs.get("schema") != cls.SCHEMA:
                raise ValueError("Incompatible schema")

            obj = cls.__new__(cls)

            obj.path = pl.Path(grp.attrs.get("source_file", "from_hdf5"))
            obj.x_position = None
            obj.y_position = None

            if "x_position" in grp:
                obj.x_position = _read_value(grp["x_position"])
            if "y_position" in grp:
                obj.y_position = _read_value(grp["y_position"])

            obj.metadata = _read_group(grp["instrument"]) if "instrument" in grp else {}

            obj.data = {}
            if "measurement" in grp:
                for key in grp["measurement"]:
                    if not load_2d_images and key == "2Dimage":
                        continue
                    item = grp["measurement"][key]
                    obj.data[key] = (
                        _read_group(item)
                        if isinstance(item, h5py.Group)
                        else _read_value(item)
                    )

            obj.results = _read_group(grp["results"]) if "results" in grp else {}

            # Convert plain results values to Quantity/Entity if not yet done
            if not grp.attrs.get("results_enriched", False) and obj.results:
                obj._enrich_results(h5, group_name)

            # Measurement-specific identifier attributes
            if hasattr(obj, "point_id"):
                obj.point_id = grp.attrs.get("point_id", None)
            if hasattr(obj, "index_x"):
                obj.index_x = grp.attrs.get("index_x", None)
            if hasattr(obj, "index_y"):
                obj.index_y = grp.attrs.get("index_y", None)

            return obj

    def _enrich_results(self, h5_file: h5py.File, group_name: str) -> None:
        """
        Convert plain float results to mu.Quantity / me.Entity using RESULTS_UNIT_MAP,
        then write units/ontology attributes back to the HDF5 dataset and set
        the 'results_enriched' flag so this runs only once.
        """
        if not self.RESULTS_UNIT_MAP or not self.results:
            return

        changed = False
        for dot_path, (unit, ontology) in self.RESULTS_UNIT_MAP.items():
            value = _get_nested(self.results, dot_path)

            # Skip if already converted or missing
            if value is None or isinstance(value, (mu.Quantity, me.Entity)):
                continue

            try:
                raw_value = float(value)
            except (TypeError, ValueError):
                continue

            new_value = (
                me.Entity(ontology, raw_value * unit) if ontology else raw_value * unit
            )
            _set_nested(self.results, dot_path, new_value)
            changed = True

        if not changed:
            return

        # Write attributes back to HDF5 and set the flag
        grp = h5_file.get(group_name)
        if grp is None:
            return

        for dot_path, (unit, ontology) in self.RESULTS_UNIT_MAP.items():
            # dot path maps directly to HDF5 path under results/
            h5_dataset_path = "results/" + dot_path.replace(".", "/")
            if h5_dataset_path not in grp:
                continue
            dset = grp[h5_dataset_path]
            dset.attrs["unit"] = str(unit)
            if ontology:
                dset.attrs["ontology"] = ontology

        grp.attrs["results_enriched"] = True

    # ------------------------------------------------------------------
    # Analysis
    def list_quantities(self) -> list[str]:
        """Recursively list all plottable quantities (dot-separated paths)."""
        paths: list[str] = []

        def explore(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    explore(v, f"{prefix}.{k}" if prefix else k)
            elif hasattr(obj, "value"):
                paths.append(prefix)
            elif isinstance(obj, (int, float, np.ndarray)):
                paths.append(prefix)
            elif hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    if k.startswith("_"):
                        continue
                    explore(v, f"{prefix}.{k}" if prefix else k)

        explore(self)
        return sorted(set(paths))

    def list_entities(self) -> list[str]:
        """Recursively list all quantities backed by a mammos_entity Entity (dot-separated paths)."""
        paths: list[str] = []

        def explore(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, me.Entity):
                paths.append(prefix)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    explore(v, f"{prefix}.{k}" if prefix else k)
            elif hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    if k.startswith("_"):
                        continue
                    explore(v, f"{prefix}.{k}" if prefix else k)

        explore(self)
        return sorted(set(paths))

    # ------------------------------------------------------------------
    # Plotting
    def plot(self) -> go.Figure:
        """
        Plot the measurement.

        Returns
        -------
        fig : go.Figure
        """
        PLOT_STYLE = {
            "height": 800,
            "width": 1150,
            "tick_font_size": 20,
            "title_font_size": 20,
        }

        fig = go.Figure()
        self._add_traces(fig)
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

    def _add_traces(self, fig: go.Figure) -> None:
        raise NotImplementedError

    def _xaxis_title(self) -> str:
        return ""

    def _yaxis_title(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Internal HDF5 write helpers
    def _write_root_attributes(self, grp: h5py.Group) -> None:
        grp.attrs["source_file"] = str(self.path)
        grp.attrs["type"] = self.__class__.__name__
        grp.attrs["format_version"] = self.FORMAT_VERSION
        grp.attrs["schema"] = self.SCHEMA
        grp.attrs["ignored"] = False  # DaHU compatibility

    def _write_positions(self, grp: h5py.Group) -> None:
        if self.x_position is not None:
            self._write_value(grp, "x_position", self.x_position)
        if self.y_position is not None:
            self._write_value(grp, "y_position", self.y_position)

        if hasattr(self, "point_id") and self.point_id is not None:
            grp.attrs["point_id"] = self.point_id
        if hasattr(self, "index_x") and self.index_x is not None:
            grp.attrs["index_x"] = self.index_x
        if hasattr(self, "index_y") and self.index_y is not None:
            grp.attrs["index_y"] = self.index_y

    def _write_metadata(self, grp: h5py.Group) -> None:
        meta_grp = grp.create_group("instrument")
        meta_grp.attrs["NX_class"] = "HTinstrument"
        meta_grp.create_dataset(
            "x_pos", data=self.x_position.value
        )  # DaHU compatibility
        meta_grp.create_dataset("y_pos", data=self.y_position.value)
        self._write_dict(meta_grp, self.metadata)

    def _write_measurement(self, grp: h5py.Group) -> None:
        meas_grp = grp.create_group("measurement")
        meas_grp.attrs["NX_class"] = "HTdata"
        for key, value in getattr(self, "data", {}).items():
            if isinstance(value, dict):
                self._write_dict(meas_grp.create_group(key), value)
            else:
                self._write_value(meas_grp, key, value)

    def _write_results(self, grp: h5py.Group) -> None:
        results_grp = grp.create_group("results")
        results_grp.attrs["NX_class"] = "HTresults"
        self._write_dict(results_grp, self.results)

    def _write_dict(self, h5parent: h5py.Group, dictionary: dict) -> None:
        for key, value in dictionary.items():
            key = str(key)  # int keys (axis indices) must be strings in HDF5
            if isinstance(value, dict):
                self._write_dict(h5parent.create_group(key), value)
            else:
                self._write_value(h5parent, key, value)

    def _write_value(self, h5parent: h5py.Group, key: str, value: Any) -> None:
        if value is None:
            return

        compress = False
        if isinstance(value, me.Entity):
            ontology_name = str(getattr(value, "ontology", None)).split(".")[-1]
            if ontology_name in ("Xrd2dImage", "SEMImage"):
                compress = True
                data_to_store = np.array(value.value, dtype=np.uint32)
            else:
                data_to_store = value.value
        elif isinstance(value, mu.Quantity):
            data_to_store = value.value
        else:
            data_to_store = value

        if data_to_store is None:
            return

        if isinstance(data_to_store, np.ndarray) and compress:
            dset = h5parent.create_dataset(
                key, data=data_to_store, compression="gzip", compression_opts=1
            )
        else:
            try:
                dset = h5parent.create_dataset(key, data=data_to_store)
            except TypeError:
                dset = h5parent.create_dataset(key, data=str(data_to_store))

        if isinstance(value, me.Entity):
            if hasattr(value, "unit"):
                dset.attrs["unit"] = str(value.unit)
            if hasattr(value, "ontology") and value.ontology:
                dset.attrs["ontology"] = str(value.ontology)
        elif isinstance(value, mu.Quantity):
            dset.attrs["unit"] = str(value.unit)

    def _apply_units(self, key: str, value: Any, unit_map: dict) -> Any:
        if value is None or key not in unit_map:
            return value
        try:
            return float(value) * unit_map[key]
        except Exception:
            return value


def _get_nested(d: dict, dot_path: str) -> Any:
    obj: Any = d
    for part in dot_path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
        if obj is None:
            return None
    return obj


def _set_nested(d: dict, dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    obj: Any = d
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value
