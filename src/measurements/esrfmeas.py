from __future__ import annotations

from typing import Any, Optional
from .xrdmeas import XrdMeas

import h5py
import pathlib as pl
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go


class EsrfMeas(XrdMeas):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "esrf.single_measurement"
    UNIT_MAP = {}
    RESULTS_UNIT_MAP = {}

    UNIT_ALIASES: dict[str, Optional[str]] = {
        "A^-1": None,  # Handled at q-read time, not as metadata unit
        "keV": "keV",
        "eV": "eV",
        "mm": "mm",
        "um": "um",
        "deg": "deg",
        "mrad": "mrad",
        "s": "s",
        "ms": "ms",
        "A": "angstrom",
        "nm": "nm",
    }

    def __init__(
        self,
        raw_h5_path: pl.Path,
        processed_h5_path: pl.Path,
        group_name: str,
    ) -> None:
        super().__init__(raw_h5_path)
        self._raw_h5_path = pl.Path(raw_h5_path)
        self._processed_h5_path = pl.Path(processed_h5_path)
        self._group_name = group_name
        self._read()

    # ------------------------------------------------------------------
    # Reading raw hdf5 datafile from ESRF
    def _read(self) -> None:
        with h5py.File(self._raw_h5_path, "r") as raw_h5:
            raw_grp = raw_h5[self._group_name]
            self._read_position(raw_grp)
            self._read_raw_metadata(raw_grp)
            self._read_raw_measurement(raw_grp)

        with h5py.File(self._processed_h5_path, "r") as proc_h5:
            if self._group_name not in proc_h5:
                print(
                    f"Warning: group '{self._group_name}' not in processed H5, skipping spectrum."
                )
                return
            self._read_processed_data(proc_h5[self._group_name])

    def _read_position(self, raw_grp: h5py.Group) -> None:
        inst = raw_grp.get("instrument")
        if inst is None:
            raise ValueError(
                f"No 'instrument' group in raw H5 entry '{self._group_name}'"
            )

        x_pos = float(np.round(inst["positioners/xsamp"][()].flat[0], 2))
        y_pos = float(np.round(inst["positioners/ysamp"][()].flat[0], 2))

        # Guard against negative zero
        self.x_position = (0.0 if x_pos == 0.0 else x_pos) * mu.mm
        self.y_position = (0.0 if y_pos == 0.0 else y_pos) * mu.mm

    def _read_raw_metadata(self, raw_grp: h5py.Group) -> None:
        inst = raw_grp.get("instrument")
        if inst is None:
            return

        if "energy/data" in inst:
            self.metadata["energy"] = self._read_h5_value(inst["energy/data"])

        if "positioners" in inst:
            self.metadata["positioners"] = {
                key: self._read_h5_value(inst[f"positioners/{key}"])
                for key in inst["positioners"]
            }

    def _read_raw_measurement(self, raw_grp: h5py.Group) -> None:
        meas_grp = raw_grp.get("measurement")
        inst_grp = raw_grp.get("instrument")
        if meas_grp is None:
            return

        for subname in list(meas_grp):
            if subname == "CdTe":
                self._read_cdte_image(meas_grp, subname)
                continue

            if inst_grp and subname in inst_grp:
                continue

            if subname.startswith("CdTe_") and "integrate" not in subname:
                if "CdTe_rois" not in self.data:
                    self.data["CdTe_rois"] = {}
                try:
                    item = meas_grp[subname]
                    if isinstance(item, h5py.Dataset):
                        self.data["CdTe_rois"][subname] = self._read_h5_value(item)
                    elif isinstance(item, h5py.Group):
                        self.data["CdTe_rois"][subname] = {
                            k: self._read_h5_value(item[k]) for k in item
                        }
                except Exception as e:
                    print(f"Warning: could not read '{subname}': {e}")

            elif "falconx" in subname:
                if "falconx" not in self.data:
                    self.data["falconx"] = {}
                try:
                    item = meas_grp[subname]
                    if isinstance(item, h5py.Dataset):
                        self.data["falconx"][subname] = self._read_h5_value(item)
                    elif isinstance(item, h5py.Group):
                        self.data["falconx"][subname] = {
                            k: self._read_h5_value(item[k]) for k in item
                        }
                except Exception as e:
                    print(f"Warning: could not read '{subname}': {e}")

    def _read_cdte_image(self, meas_grp: h5py.Group, subname: str) -> None:
        try:
            item = meas_grp[subname]
            if isinstance(item, h5py.Dataset):
                image = np.squeeze(item[()]).astype(np.uint32)
                self.data["2Dimage"] = me.Entity(
                    "Xrd2dImage", image * mu.dimensionless_unscaled
                )
        except Exception as e:
            print(f"Warning: could not read CdTe: {e}")

    def _read_processed_data(self, proc_grp: h5py.Group) -> None:
        integrate_grp = proc_grp.get("CdTe_integrate")
        if integrate_grp is None:
            print(f"Warning: no CdTe_integrate in processed group '{self._group_name}'")
            return

        integrated = integrate_grp.get("integrated")
        if integrated is None:
            return

        if "q" in integrated:
            q_data = np.squeeze(integrated["q"][()])
            if integrated["q"].attrs.get("units") == "A^-1":
                q_data = q_data * 10  # Convert Å⁻¹ → nm⁻¹
            self.data["q"] = q_data

        if "intensity" in integrated:
            self.data["intensity"] = np.squeeze(integrated["intensity"][()])

        if "q" in self.data and "energy" in self.metadata:
            self.data["Two_theta"] = (
                self._q_to_tth(self.data["q"], self.metadata["energy"]) * mu.deg
            )

        if "intensity" in self.data and "2Dimage" in self.data:
            total_counts = float(np.sum(self.data["2Dimage"].value))
            self.data["Counts"] = me.Entity(
                "XrdCounts",
                self.data["intensity"] * total_counts * mu.dimensionless_unscaled,
            )

    @staticmethod
    def _read_h5_value(dataset: h5py.Dataset) -> Any:
        """Read a scalar HDF5 dataset, wrapping it in a mu.Quantity if a units attribute is present."""
        value = float(dataset[()].flat[0])
        unit_str = dataset.attrs.get("units") or dataset.attrs.get("unit")

        if not unit_str:
            return value

        normalised = EsrfMeas.UNIT_ALIASES.get(unit_str.strip(), unit_str.strip())
        if normalised is None:
            return value

        try:
            return value * getattr(mu, normalised)
        except AttributeError:
            print(f"Warning: unknown unit '{unit_str}', storing as plain float.")
            return value

    @staticmethod
    def _q_to_tth(q_nm_inv: np.ndarray, energy_keV: float) -> np.ndarray:
        """Convert q (nm⁻¹) to 2θ (degrees)."""
        wavelength_nm = 1.23984193 / energy_keV
        sin_theta = np.clip(q_nm_inv * wavelength_nm / (4 * np.pi), -1.0, 1.0)
        return np.degrees(2 * np.arcsin(sin_theta))

    # ------------------------------------------------------------------
    # Calling add_results() method from individual Meas classes
    def add_results(self, results_path: pl.Path) -> None:
        """
        Parameters
        ----------
        results_path : Path
            Folder containing the .lst, .par and .dia result files.
        """
        self._results_path = pl.Path(results_path)
        super().add_results()

    @property
    def _results_base_path(self) -> pl.Path:
        if not hasattr(self, "_results_path"):
            raise ValueError(
                "Call add_results(results_path=...) to provide result folder path."
            )
        return self._results_path / f"{self._results_path.stem}_{self._group_name}"

    # ------------------------------------------------------------------
    # Writing to hdf5 from class
    def _write_root_attributes(self, grp: h5py.Group) -> None:
        super()._write_root_attributes(grp)
        grp.attrs["index"] = self._group_name

    def _write_metadata(self, grp: h5py.Group) -> None:
        with h5py.File(self._raw_h5_path, "r") as raw_h5:
            raw_inst = raw_h5[self._group_name].get("instrument")
            if raw_inst is not None:
                raw_h5.copy(raw_inst, grp, name="instrument", expand_soft=True)

        inst_grp = grp.require_group("instrument")
        inst_grp.attrs["NX_class"] = "HTinstrument"
        for key, val in (
            ("x_pos", self.x_position.value),
            ("y_pos", self.y_position.value),
        ):
            if key not in inst_grp:
                inst_grp.create_dataset(key, data=val)

    def _write_measurement(self, grp: h5py.Group) -> None:
        meas_grp = grp.require_group("measurement")
        meas_grp.attrs["NX_class"] = "HTdata"

        if "2Dimage" in self.data:
            meas_grp.create_dataset(
                "2Dimage",
                data=self.data["2Dimage"].value,
                compression="gzip",
                compression_opts=4,
            )

        if "falconx" in self.data:
            fx_grp = meas_grp.create_group("falconx")
            for k, v in self.data["falconx"].items():
                try:
                    if isinstance(v, dict):
                        sub = fx_grp.create_group(k)
                        for sk, sv in v.items():
                            sub.create_dataset(sk, data=sv)
                    else:
                        fx_grp.create_dataset(k, data=v)
                except Exception as e:
                    print(f"Warning: could not write falconx '{k}': {e}")

        if "CdTe_rois" in self.data:
            for key, item in self.data["CdTe_rois"].items():
                try:
                    if isinstance(item, dict):
                        roi_grp = meas_grp.create_group(key)
                        for k, v in item.items():
                            roi_grp.create_dataset(k, data=v)
                    else:
                        meas_grp.create_dataset(
                            key, data=item.value if hasattr(item, "value") else item
                        )
                except Exception as e:
                    print(f"Warning: could not write '{key}': {e}")

        if "q" in self.data:
            integrated = meas_grp.create_group("integrated")
            q_ds = integrated.create_dataset("q", data=self.data["q"])
            q_ds.attrs["units"] = "nm^-1"

            if "intensity" in self.data:
                integrated.create_dataset("intensity", data=self.data["intensity"])
            if "Two_theta" in self.data:
                tth_ds = integrated.create_dataset(
                    "tth", data=self.data["Two_theta"].value
                )
                tth_ds.attrs["units"] = "deg"
            if "Counts" in self.data:
                integrated.create_dataset("counts", data=self.data["Counts"].value)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _add_traces(self, fig: go.Figure) -> None:
        if "Two_theta" not in self.data or "Counts" not in self.data:
            return
        fig.add_trace(
            go.Scatter(
                x=self.data["Two_theta"].value,
                y=self.data["Counts"].value,
                name="esrf_xrd",
                mode="lines",
            )
        )

    def _xaxis_title(self) -> str:
        return (
            f"2θ ({self.data['Two_theta'].unit})"
            if "Two_theta" in self.data
            else "2θ (deg)"
        )

    def _yaxis_title(self) -> str:
        return "Counts"
