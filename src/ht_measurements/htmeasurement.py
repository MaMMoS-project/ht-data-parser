from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import plotly.graph_objects as go
import numpy as np
import h5py
from pathlib import Path
import tqdm
import time


class HTMeasurement(ABC):
    def __init__(self):
        self.measurements = []

    @abstractmethod
    def _load_measurements(self):
        pass

    def add_results(self, show_progress=True):
        iterator = self.measurements
        if show_progress:
            iterator = tqdm.tqdm(self.measurements)

        for meas in iterator:
            if hasattr(meas, "add_results"):
                meas.add_results()

    def get_quantity(self, path):
        values = []
        for meas in getattr(self, "measurements", [self]):
            obj = meas
            for attr in path.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(attr, np.nan)
                else:
                    obj = getattr(obj, attr, np.nan)
            if hasattr(obj, "value"):
                obj = obj.value
            values.append(obj)
        return values

    def list_quantities(self):
        quantities = set()
        for m in self.measurements:
            quantities.update(m.list_quantities())
        return sorted(quantities)

    def list_scalar_quantities(self):
        scalar_quantities = []
        for q in self.list_quantities():
            values = self.get_quantity(q)
            is_scalar = True
            for v in values:
                if hasattr(v, "value"):
                    v = v.value
                if np.ndim(v) > 0:
                    is_scalar = False
                    break
            if is_scalar:
                scalar_quantities.append(q)
        return sorted(scalar_quantities)

    def to_hdf5(self, hdf5_path, mode="a", overwrite=False):
        hdf5_path = Path(hdf5_path)
        with h5py.File(hdf5_path, mode) as h5:
            scan_group_name = getattr(self, "folder_path", "Scan")
            if isinstance(scan_group_name, Path):
                scan_group_name = (
                    type(self).__name__.split("Scan")[0].upper()
                    + "_"
                    + scan_group_name.stem
                )

            if scan_group_name in h5:
                if overwrite:
                    del h5[scan_group_name]
                    scan_grp = h5.create_group(scan_group_name)
                    scan_grp.attrs["type"] = type(self).__name__

                    scan_grp.attrs["HT_type"] = (
                        type(self).__name__.split("Scan")[0].lower()
                    )
                    # For backwards compatibility with DaHU (smartlab HT_type set to xrd)
                    if scan_grp.attrs.get("HT_type") == "smartlab":
                        scan_grp.attrs["instrument"] = "Rigaku Smartlab"
                        scan_grp.attrs["HT_type"] = "xrd"

                    print(f"Scan group '{scan_group_name}' overwritten.")
                else:
                    scan_grp = h5[scan_group_name]
                    print(
                        f"Scan group '{scan_group_name}' already exists. "
                        "Existing measurements will be skipped. "
                        "Set overwrite=True to replace."
                    )
            else:
                scan_grp = h5.create_group(scan_group_name)
                scan_grp.attrs["type"] = type(self).__name__
                scan_grp.attrs["HT_type"] = type(self).__name__.split("Scan")[0].lower()
                # For backwards compatibility with DaHU (smartlab HT_type set to xrd)
                if scan_grp.attrs.get("HT_type") == "smartlab":
                    scan_grp.attrs["HT_type"] = "xrd"
                    scan_grp.attrs["instrument"] = "Rigaku Smartlab"

            for meas in self.measurements:
                if (
                    hasattr(meas, "x_position")
                    and hasattr(meas, "y_position")
                    and meas.x_position is not None
                    and meas.y_position is not None
                ):
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
                if hasattr(meas, "results"):
                    meas._write_results(meas_grp)

    @classmethod
    def from_hdf5(cls, hdf5_path, scan_group=None):
        hdf5_path = Path(hdf5_path)

        with h5py.File(hdf5_path, "r") as h5file:

            # Determine scan group
            if scan_group is None:
                scan_group = None
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

            # Iterate over measurements
            for meas_name, meas_grp in tqdm.tqdm(scan_grp.items()):
                meas_type = getattr(cls, "MEASUREMENT_CLASS", None)
                if meas_type is None:
                    raise ValueError("Scan class must define MEASUREMENT_CLASS.")

                if meas_grp.attrs.get("type") != meas_type.__name__:
                    raise ValueError(
                        f"Unexpected measurement type '{meas_grp.attrs.get('type')}' "
                        f"in group {meas_name}"
                    )

                meas = meas_type.from_hdf5(hdf5_path, f"{scan_group}/{meas_name}")
                scan_obj.measurements.append(meas)

        return scan_obj

    def heatmap(self, quantity, xlim=[-42.5, 42.5], ylim=[-42.5, 42.5]):
        values = self.get_quantity(quantity)
        values = [v.value if hasattr(v, "value") else v for v in values]

        for v in values:
            if np.ndim(v) > 0:
                raise ValueError(
                    f"{quantity} is not scalar per measurement (got shape {np.shape(v)}). "
                    "Heatmap requires scalar quantities."
                )
        unit = values[0].unit if hasattr(values[0], "unit") else None

        xs = [m.x_position.value for m in self.measurements]
        ys = [m.y_position.value for m in self.measurements]
        x_unique = sorted(set(xs))
        y_unique = sorted(set(ys))
        x_map = {x: i for i, x in enumerate(x_unique)}
        y_map = {y: i for i, y in enumerate(y_unique)}
        z = np.full((len(y_unique), len(x_unique)), np.nan)
        for x, y, v in zip(xs, ys, values):
            z[y_map[y], x_map[x]] = v

        fig = go.Figure(
            data=go.Heatmap(
                x=x_unique,
                y=y_unique,
                z=z,
                colorscale="Plasma",
                colorbar=dict(
                    title=unit, x=1.05, y=0.5, len=1.0, tickfont=dict(size=20)
                ),
            )
        )
        fig.update_layout(
            title=f"Heatmap of {quantity}",
            xaxis=dict(
                range=xlim,
                title="X position",
                tickfont=dict(size=20),
                title_font=dict(size=20),
            ),
            yaxis=dict(
                range=ylim,
                scaleratio=1,
                title="Y position",
                tickfont=dict(size=20),
                title_font=dict(size=20),
            ),
            width=600,
            height=600,
        )
        fig.show()
