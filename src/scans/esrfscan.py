from __future__ import annotations

from typing import Optional
from pathlib import Path

import h5py
import tqdm

from .basescan import BaseScan
from ..measurements.esrfmeas import EsrfMeas

_ESRF_WRITER_VERSION = "2.0.0"


class EsrfScan(BaseScan):
    """
    Full ESRF BM02 HT-XRD scan.

    Expected folder layout:
        folder_path/
            RAW_DATA/<name>/<name>_<index>/<name>.h5
            PROCESSED_DATA/<name>/<name>_<index>/<name>.h5
    """

    MEASUREMENT_CLASS = EsrfMeas

    def __init__(self, folder_path: Path | str, index: str = "0001") -> None:
        self.index = index
        self._raw_h5_path: Optional[Path] = None
        self._processed_h5_path: Optional[Path] = None
        self._alignment_group_names: list[tuple[str, str]] = []
        super().__init__(folder_path)

    # ------------------------------------------------------------------
    # Loading hdf5 datafile
    def _find_h5_files(self) -> None:
        tail_path = (
            Path(self.folder_path.name) / f"{self.folder_path.name}_{self.index}"
        )
        processed_dir = self.folder_path / "PROCESSED_DATA" / tail_path
        raw_dir = self.folder_path / "RAW_DATA" / tail_path

        proc_files = sorted(processed_dir.glob("*.h5"))
        if not proc_files:
            raise FileNotFoundError(f"No *.h5 found in {processed_dir}")

        proc_h5 = proc_files[0]
        raw_h5 = raw_dir / proc_h5.name
        if not raw_h5.exists():
            raise FileNotFoundError(f"Expected RAW_DATA file not found: {raw_h5}")

        self._processed_h5_path = proc_h5
        self._raw_h5_path = raw_h5

    def _load_measurements(self) -> None:
        self._find_h5_files()

        with h5py.File(self._raw_h5_path, "r") as raw_h5:
            entries = list(raw_h5.items())

        for group_name, _ in tqdm.tqdm(entries, desc="Loading ESRF scan"):
            with h5py.File(self._raw_h5_path, "r") as raw_h5:
                grp = raw_h5[group_name]
                is_align, align_type = self._is_alignment(grp)
                is_meas = self._is_measurement(grp)

            if is_align:
                self._alignment_group_names.append((group_name, align_type))
                continue

            if not is_meas:
                print(f"Could not identify group '{group_name}'. Skipping.")
                continue

            try:
                meas = EsrfMeas(self._raw_h5_path, self._processed_h5_path, group_name)
            except Exception as e:
                print(f"Skipping '{group_name}': {e}")
                continue

            if self._is_within_bounds(meas):
                self.measurements.append(meas)

    @staticmethod
    def _is_alignment(grp: h5py.Group) -> tuple[bool, str]:
        if "title" not in grp:
            return False, ""
        title = str(grp["title"][()])
        for keyword in ("kth", "tsz"):
            if keyword in title:
                return True, keyword
        return False, ""

    @staticmethod
    def _is_measurement(grp: h5py.Group) -> bool:
        meas = grp.get("measurement")
        return meas is not None and "CdTe" in meas

    # ------------------------------------------------------------------
    # Calling add_results() method from individual Meas classes
    def add_results(self, results_path: Path | str, show_progress: bool = True) -> None:
        """
        Parameters
        ----------
        results_path : Path
            Folder containing the .lst, .par and .dia result files.
        show_progress : bool
        """
        iterator = tqdm.tqdm(self.measurements) if show_progress else self.measurements
        for meas in iterator:
            meas.add_results(results_path)

    # ------------------------------------------------------------------
    # Writing data to .hdf5 file from class
    def to_hdf5(
        self,
        hdf5_path: Path | str,
        scan_group_name: Optional[str] = None,
        mode: str = "a",
        overwrite: bool = False,
    ) -> None:
        """Write the full ESRF scan to an HDF5 file."""
        hdf5_path = Path(hdf5_path)
        if scan_group_name is None:
            scan_group_name = (
                type(self).__name__.split("Scan")[0].upper()
                + "_"
                + self.folder_path.stem
            )

        with (
            h5py.File(hdf5_path, mode) as out_h5,
            h5py.File(self._raw_h5_path, "r") as raw_h5,
        ):
            if scan_group_name in out_h5:
                if overwrite:
                    del out_h5[scan_group_name]
                else:
                    print(
                        f"Scan group '{scan_group_name}' already exists. Use overwrite=True to replace."
                    )
                    return

            esrf_grp = out_h5.create_group(scan_group_name)
            esrf_grp.attrs["type"] = type(self).__name__
            esrf_grp.attrs["HT_type"] = "xrd"
            esrf_grp.attrs["instrument"] = "bm02 esrf"
            esrf_grp.attrs["esrf_writer"] = _ESRF_WRITER_VERSION

            self._write_alignment_scans(esrf_grp, raw_h5)
            self._write_measurement_groups(esrf_grp)

    def _write_alignment_scans(self, esrf_grp: h5py.Group, raw_h5: h5py.File) -> None:
        align_root = esrf_grp.create_group("alignment_scans")
        counters: dict[str, int] = {}

        for group_name, align_type in self._alignment_group_names:
            n = counters.get(align_type, 1)
            counters[align_type] = n + 1
            target = align_root.create_group(f"{align_type}_alignment_{n}")
            target.attrs["index"] = group_name
            target.attrs["ignored"] = False

            for subname, subitem in raw_h5[group_name].items():
                if subname == "measurement":
                    continue
                raw_h5.copy(subitem, target, name=subname, expand_soft=True)

    def _write_measurement_groups(self, esrf_grp: h5py.Group) -> None:
        for meas in tqdm.tqdm(self.measurements, desc="Writing measurements"):
            x = meas.x_position.value
            y = meas.y_position.value
            pos_grp = esrf_grp.create_group(f"({x},{y})")

            meas._write_root_attributes(pos_grp)
            meas._write_positions(pos_grp)
            meas._write_metadata(pos_grp)
            meas._write_measurement(pos_grp)
            if meas.results:
                meas._write_results(pos_grp)
