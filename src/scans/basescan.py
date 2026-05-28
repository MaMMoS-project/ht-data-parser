from __future__ import annotations

from typing import Optional, Any
from abc import ABC
from pathlib import Path

import h5py
import tqdm
import numpy as np
import mammos_entity as me
import plotly.graph_objects as go

from ..plotting.plotmixin import PlotMixin


class BaseScan(PlotMixin, ABC):
    MEASUREMENT_CLASS = None
    FILE_EXTENSION: Optional[str] = None

    def __init__(self, folder_path: Path | str) -> None:
        self.measurements: list = []
        self.folder_path = Path(folder_path)
        self._load_measurements()
        self._post_load()

    def _post_load(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Loading all scans (excluding some edge positions of the wafer)
    def _is_within_bounds(
        self,
        meas: object,
        max_single: float = 40.0,
        max_sum: float = 60.0,
    ) -> bool:
        if meas.x_position is None or meas.y_position is None:
            return False
        x = abs(meas.x_position.value)
        y = abs(meas.y_position.value)
        return  x <= max_single and y <= max_single and (x + y) < max_sum

    def _load_measurements(self) -> None:
        """Default loader: glob FILE_EXTENSION, instantiate MEASUREMENT_CLASS, filter by position."""
        if self.FILE_EXTENSION is None or self.MEASUREMENT_CLASS is None:
            raise NotImplementedError("Define FILE_EXTENSION and MEASUREMENT_CLASS.")

        for file_path in tqdm.tqdm(
            sorted(self.folder_path.glob(f"*{self.FILE_EXTENSION}"))
        ):
            try:
                meas = self.MEASUREMENT_CLASS(file_path)
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")
                continue
            if self._is_within_bounds(meas):
                self.measurements.append(meas)

    # ------------------------------------------------------------------
    # Analysis section
    def get_measurement_at(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float = 0.1,
    ) -> object:
        """
        Return the measurement at wafer position (x_mm, y_mm).

        Raises ValueError if no measurement is found within tolerance_mm.
        """
        for meas in self.measurements:
            if (
                meas.x_position is not None
                and meas.y_position is not None
                and abs(meas.x_position.value - x_mm) <= tolerance_mm
                and abs(meas.y_position.value - y_mm) <= tolerance_mm
            ):
                return meas
        raise ValueError(
            f"No measurement found at ({x_mm}, {y_mm}) within tolerance={tolerance_mm} mm."
        )

    def add_results(self, show_progress: bool = True) -> None:
        iterator = tqdm.tqdm(self.measurements) if show_progress else self.measurements
        for meas in iterator:
            if hasattr(meas, "add_results"):
                meas.add_results()

    def get_quantity(self, dot_path: str, values_only=False) -> list:
        """
        Extract a quantity from all measurements by dot-separated attribute path.

        Example: get_quantity("results.phases.Fe.phase_fraction")
        """
        values = []
        for meas in self.measurements:
            obj = meas
            for attr in dot_path.split("."):
                obj = (
                    obj.get(attr, np.nan)
                    if isinstance(obj, dict)
                    else getattr(obj, attr, np.nan)
                )
            if values_only and hasattr(obj, "value"):
                values.append(obj.value)
            else:
                values.append(obj)
        return values

    def list_quantities(self) -> list[str]:
        quantities: set[str] = set()
        for meas in self.measurements:
            quantities.update(meas.list_quantities())
        return sorted(quantities)

    def list_entities(self) -> list[str]:
        entities: set[str] = set()
        for meas in self.measurements:
            entities.update(meas.list_entities())
        return sorted(entities)

    def list_scalar_quantities(self) -> list[str]:
        scalar_quantities = []
        for quantity_path in self.list_quantities():
            values = self.get_quantity(quantity_path)
            if all(np.ndim(v.value if hasattr(v, "value") else v) == 0 for v in values):
                scalar_quantities.append(quantity_path)
        return sorted(scalar_quantities)

    # ------------------------------------------------------------------
    # HDF5 export from Scan class, calling export method from Meas classes
    def to_hdf5(
        self,
        hdf5_path: Path | str,
        scan_group_name: str = None,
        mode: str = "a",
        overwrite: bool = False,
    ) -> None:
        hdf5_path = Path(hdf5_path)
        with h5py.File(hdf5_path, mode) as h5:
            if scan_group_name is None:
                scan_group_name = (
                    type(self).__name__.split("Scan")[0].upper()
                    + "_"
                    + self.folder_path.stem
                )

            if scan_group_name in h5:
                if overwrite:
                    del h5[scan_group_name]
                    print(f"Scan group '{scan_group_name}' overwritten.")
                else:
                    print(
                        f"Scan group '{scan_group_name}' already exists. "
                        "Existing measurements will be skipped. "
                        "Set overwrite=True to replace."
                    )

            scan_grp = h5.require_group(scan_group_name)
            scan_grp.attrs["type"] = type(self).__name__
            scan_grp.attrs["HT_type"] = type(self).__name__.split("Scan")[0].lower()

            # For backwards compatibility with DaHU
            if scan_grp.attrs.get("HT_type") == "smartlab":
                scan_grp.attrs["HT_type"] = "xrd"
                scan_grp.attrs["instrument"] = "Rigaku Smartlab"

            for meas in tqdm.tqdm(self.measurements):
                if meas.x_position is not None and meas.y_position is not None:
                    meas_group_name = (
                        f"({meas.x_position.value:.1f},{meas.y_position.value:.1f})"
                    )
                else:
                    meas_group_name = getattr(meas, "path", Path("measurement")).stem

                if meas_group_name in scan_grp:
                    if overwrite:
                        del scan_grp[meas_group_name]
                    else:
                        continue

                meas_grp = scan_grp.create_group(meas_group_name)
                meas._write_root_attributes(meas_grp)
                meas._write_positions(meas_grp)
                meas._write_metadata(meas_grp)
                meas._write_measurement(meas_grp)
                if getattr(meas, "results", None):
                    meas._write_results(meas_grp)

    @classmethod
    def from_hdf5(
        cls,
        hdf5_path: Path | str,
        scan_group: Optional[str] = None,
        load_2d_images: bool = True,
    ) -> BaseScan:
        hdf5_path = Path(hdf5_path)

        with h5py.File(hdf5_path, "a") as h5file:
            if scan_group is None:
                for grp_name, grp in h5file.items():
                    if grp.attrs.get("type") == cls.__name__:
                        scan_group = grp_name
                        break
                if scan_group is None:
                    raise ValueError(f"No scan of type {cls.__name__} found.")

            scan_grp = h5file[scan_group]
            scan_obj = cls.__new__(cls)
            scan_obj.measurements = []
            scan_obj.folder_path = scan_group

            meas_type = getattr(cls, "MEASUREMENT_CLASS", None)
            if meas_type is None:
                raise ValueError("Scan class must define MEASUREMENT_CLASS.")

            for meas_name, meas_grp in tqdm.tqdm(scan_grp.items()):
                if meas_grp.attrs.get("type") != meas_type.__name__:
                    continue  # skip alignment_scans and other non-measurement groups

                meas = meas_type.from_hdf5(
                    hdf5_path,
                    f"{scan_group}/{meas_name}",
                    load_2d_images=load_2d_images,
                )
                scan_obj.measurements.append(meas)

        return scan_obj
