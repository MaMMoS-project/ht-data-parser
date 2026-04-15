from __future__ import annotations

from typing import Optional, Any
from pathlib import Path

import mammos_entity as me
import mammos_units as mu
import numpy as np
import h5py
import tqdm
import yaml

from .scans.edxscan import EdxScan
from .scans.esrfscan import EsrfScan
from .scans.mokescan import MokeScan
from .scans.profilscan import ProfilScan
from .scans.semscan import SemScan
from .scans.smartlabscan import SmartlabScan


# Maps 'type' HDF5 attribute to scan class
_SCAN_REGISTRY: dict[str, type] = {
    "EdxScan": EdxScan,
    "EsrfScan": EsrfScan,
    "MokeScan": MokeScan,
    "ProfilScan": ProfilScan,
    "SemScan": SemScan,
    "SmartlabScan": SmartlabScan,
}

# NOMAD definitions block shared by all YAML files
_NOMAD_DEFINITIONS = {
    "name": "SmFeV compositionnally graded hard magnetic film fabricated and characterised at Institut Neel, CNRS",
    "sections": {
        "Sample": {
            "base_sections": ["nomad.datamodel.data.EntryData"],
            "quantities": {
                "name": {"type": "str", "default": "NEEL-Sample"},
                "description": {"type": "str"},
                "lab_id": {"type": "str"},
                "short_name": {"type": "str"},
                "owner": {"type": "str"},
                "xpos": {
                    "type": "np.float64",
                    "unit": "mm",
                    "description": "x-position of the sample on the wafer",
                },
                "ypos": {
                    "type": "np.float64",
                    "unit": "mm",
                    "description": "y-position of the sample on the wafer",
                },
                "CoercivityHcExternal": {
                    "type": "np.float64",
                    "unit": "A/m",
                    "description": "Coercivity from MOKE measurements",
                },
                "annealing_temperature": {
                    "type": "np.float64",
                    "unit": "K",
                    "description": "Annealing temperature",
                },
                "annealing_time": {
                    "type": "np.float64",
                    "unit": "s",
                    "description": "Annealing duration",
                },
            },
            "sub_sections": {
                "elemental_composition": {
                    "repeats": True,
                    "section": {
                        "base_sections": [
                            "nomad.datamodel.metainfo.basesections.ElementalComposition"
                        ],
                        "more": {"label_quantity": "element, atomic_fraction"},
                        "quantities": {
                            "element": {"type": "str"},
                            "atomic_fraction": {"type": "np.float64"},
                        },
                    },
                },
                "phase_fractions": {
                    "repeats": True,
                    "section": {
                        "base_sections": ["nomad.datamodel.data.ArchiveSection"],
                        "more": {"label_quantity": "phase"},
                        "quantities": {
                            "phase": {"type": "str"},
                            "phase_fraction": {"type": "np.float64"},
                        },
                    },
                },
                "lattice_parameters": {
                    "repeats": True,
                    "section": {
                        "base_sections": ["nomad.datamodel.data.ArchiveSection"],
                        "more": {"label_quantity": "lattice"},
                        "quantities": {
                            "lattice": {"type": "str"},
                            "lattice_parameter": {"type": "np.float64", "unit": "nm"},
                        },
                    },
                },
            },
        }
    },
}


class Film:
    """
    Master container for all HT measurements on a single film / wafer.

    Reads an HDF5 file produced by any combination of scan classes and
    reconstructs the appropriate scan objects from the 'type' attribute
    stored in each top-level group.

    Parameters
    ----------
    hdf5_path : Path or str
        Path to the HDF5 file.
    load_2d_images : bool
        Passed through to each scan's from_hdf5 call. Set to False to
        skip loading large detector images.
    """

    def __init__(
        self,
        hdf5_path: Path | str,
        load_2d_images: bool = True,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.sample: Optional[dict] = self._read_sample_metadata()
        self.scans: dict[str, Any] = {}
        self._load(load_2d_images)

    # ------------------------------------------------------------------
    # Loading full hdf5 file
    def _load(self, load_2d_images: bool) -> None:
        with h5py.File(self.hdf5_path, "r") as h5:
            for group_name, grp in tqdm.tqdm(h5.items(), desc="Loading Film"):
                type_name = grp.attrs.get("type")
                if type_name is None:
                    print(f"Skipping '{group_name}': no 'type' attribute.")
                    continue

                if type_name == "SampleMetadata":
                    continue

                scan_cls = _SCAN_REGISTRY.get(type_name)
                if scan_cls is None:
                    print(f"Skipping '{group_name}': unknown scan type '{type_name}'.")
                    continue

                try:
                    scan = scan_cls.from_hdf5(
                        self.hdf5_path,
                        scan_group=group_name,
                        load_2d_images=load_2d_images,
                    )
                    self.scans[group_name] = scan
                except Exception as e:
                    print(f"Could not load '{group_name}': {e}")

    # ------------------------------------------------------------------
    # Sample metadata (full-wafer processing parameters)
    def write_sample_metadata(
        self,
        metadata: dict,
        group_name: str = "sample",
        overwrite: bool = False,
    ) -> None:
        """
        Write a flat or nested metadata dictionary to the HDF5 file as a
        top-level group alongside the scan groups.

        Handles str, int, float, bool, mu.Quantity and me.Entity values.

        Parameters
        ----------
        metadata : dict
            Processing parameters, e.g. deposition time, temperature.
        group_name : str
            Name of the HDF5 group. Defaults to 'sample'.
        overwrite : bool
            If True, delete and replace an existing group of the same name.
        """
        with h5py.File(self.hdf5_path, "a") as h5:
            if group_name in h5:
                if overwrite:
                    del h5[group_name]
                else:
                    print(
                        f"Group '{group_name}' already exists. "
                        "Use overwrite=True to replace."
                    )
                    return

            grp = h5.create_group(group_name)
            grp.attrs["type"] = "SampleMetadata"
            self._write_metadata_dict(h5, grp, metadata)
            self.sample = metadata

        print(f"Sample metadata written to '{group_name}' in {self.hdf5_path.name}")

    def _read_sample_metadata(self, group_name: str = "sample") -> Optional[dict]:
        with h5py.File(self.hdf5_path, "r") as h5:
            if group_name not in h5:
                return None
            grp = h5[group_name]
            if grp.attrs.get("type") != "SampleMetadata":
                return None
            return self._read_metadata_dict(grp)

    def _write_metadata_dict(
        self,
        h5file: h5py.File,
        h5parent: h5py.Group,
        dictionary: dict,
    ) -> None:
        for key, value in dictionary.items():
            key = str(key)

            if isinstance(value, dict):
                self._write_metadata_dict(h5file, h5parent.create_group(key), value)

            elif isinstance(value, bool):
                # bool must come before int — bool is a subclass of int
                dset = h5parent.create_dataset(key, data=int(value))
                dset.attrs["python_type"] = "bool"

            elif isinstance(value, (int, float, str, np.ndarray)):
                try:
                    h5parent.create_dataset(key, data=value)
                except TypeError:
                    h5parent.create_dataset(key, data=str(value))

            elif isinstance(value, mu.Quantity):
                dset = h5parent.create_dataset(key, data=value.value)
                dset.attrs["unit"] = str(value.unit)

            elif isinstance(value, me.Entity):
                dset = h5parent.create_dataset(key, data=value.value)
                if hasattr(value, "unit") and value.unit:
                    dset.attrs["unit"] = str(value.unit)
                if hasattr(value, "ontology") and value.ontology:
                    dset.attrs["ontology"] = str(value.ontology)

            else:
                dset = h5parent.create_dataset(key, data=str(value))
                dset.attrs["python_type"] = type(value).__name__

    def _read_metadata_dict(self, h5group: h5py.Group) -> dict:
        result: dict = {}
        for key, item in h5group.items():
            if isinstance(item, h5py.Group):
                result[key] = self._read_metadata_dict(item)
                continue

            raw = item[()]
            unit = item.attrs.get("unit")
            ontology = item.attrs.get("ontology")
            py_type = item.attrs.get("python_type")

            if ontology:
                quantity = raw * getattr(mu, unit) if unit else raw
                result[key] = me.Entity(ontology.split(".")[-1], quantity)
            elif unit:
                result[key] = raw * getattr(mu, unit)
            elif py_type == "bool":
                result[key] = bool(raw)
            else:
                result[key] = raw

        return result

    # ------------------------------------------------------------------
    # Accessors (used in the code by myself)
    def get_scans_by_type(self, scan_cls: type) -> list:
        """Return all loaded scans that are instances of scan_cls."""
        return [s for s in self.scans.values() if isinstance(s, scan_cls)]

    @property
    def edx_scans(self) -> list[EdxScan]:
        return self.get_scans_by_type(EdxScan)

    @property
    def esrf_scans(self) -> list[EsrfScan]:
        return self.get_scans_by_type(EsrfScan)

    @property
    def moke_scans(self) -> list[MokeScan]:
        return self.get_scans_by_type(MokeScan)

    @property
    def profil_scans(self) -> list[ProfilScan]:
        return self.get_scans_by_type(ProfilScan)

    @property
    def sem_scans(self) -> list[SemScan]:
        return self.get_scans_by_type(SemScan)

    @property
    def smartlab_scans(self) -> list[SmartlabScan]:
        return self.get_scans_by_type(SmartlabScan)

    # ------------------------------------------------------------------
    # Extract compo, xrd results, coercivity and thickness to build .yaml files
    def collect_properties(
        self,
        tolerance_mm: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Collect all physical properties for every wafer position.

        For each position the following are extracted where available:
        - x_mm, y_mm
        - elemental composition  (from EdxScan or EsrfScan results)
        - all phase fractions     (from SmartlabScan or EsrfScan .lst results)
        - all lattice parameters  (from SmartlabScan or EsrfScan .lst results)
        - coercivity              (from MokeScan results)
        - thickness               (from ProfilScan results)

        Returns
        -------
        list[dict]
            One dict per position. Missing values are None.
        """
        positions = self._collect_positions()
        rows = []

        for x_mm, y_mm in tqdm.tqdm(positions, desc="Collecting properties"):
            row: dict[str, Any] = {"x_mm": x_mm, "y_mm": y_mm}
            row.update(self._extract_composition(x_mm, y_mm, tolerance_mm))
            row.update(self._extract_xrd_results(x_mm, y_mm, tolerance_mm))
            row.update(self._extract_coercivity(x_mm, y_mm, tolerance_mm))
            row.update(self._extract_thickness(x_mm, y_mm, tolerance_mm))
            rows.append(row)

        return rows

    def _collect_positions(self) -> list[tuple[float, float]]:
        positions: set[tuple[float, float]] = set()
        for scan in self.scans.values():
            for meas in scan.measurements:
                if meas.x_position is not None and meas.y_position is not None:
                    positions.add(
                        (
                            round(float(meas.x_position.value), 2),
                            round(float(meas.y_position.value), 2),
                        )
                    )
        return sorted(positions)

    def _extract_composition(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float,
    ) -> dict[str, Any]:
        """
        Elemental atomic fractions from EdxScan or EsrfScan.
        Keys: 'composition_<Element>' (e.g. 'composition_Fe': 0.72)
        """
        for scan in self.edx_scans + self.esrf_scans:
            try:
                meas = scan.get_measurement_at(x_mm, y_mm, tolerance_mm)
            except ValueError:
                continue

            if not meas.results:
                continue

            result: dict[str, Any] = {}
            for element, element_data in meas.results.items():
                if not isinstance(element_data, dict):
                    continue
                atom_pct = element_data.get("AtomPercent")
                if atom_pct is None:
                    continue
                value = (
                    float(atom_pct.value)
                    if hasattr(atom_pct, "value")
                    else float(atom_pct)
                )
                result[f"composition_{element}"] = round(value / 100.0, 6)

            if result:
                return result

        return {}

    def _extract_xrd_results(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float,
    ) -> dict[str, Any]:
        """
        All phase fractions and lattice parameters from SmartlabScan or EsrfScan.

        Keys:
            'phase_fraction_<PhaseName>'
            'lattice_<PhaseName>_<param>'      (e.g. 'lattice_NdFe14B_a')
        """
        for scan in self.smartlab_scans + self.esrf_scans:
            try:
                meas = scan.get_measurement_at(x_mm, y_mm, tolerance_mm)
            except ValueError:
                continue

            phases = meas.results.get("phases", {})
            if not phases:
                continue

            result: dict[str, Any] = {}
            for phase_name, phase_data in phases.items():
                if not isinstance(phase_data, dict):
                    continue

                frac = phase_data.get("phase_fraction")
                if frac is not None:
                    result[f"phase_fraction_{phase_name}"] = (
                        float(frac.value) if hasattr(frac, "value") else float(frac)
                    )

                for param, val in phase_data.get("lattice", {}).items():
                    # Skipping error values and param where phase content was not quantify (Ta phases)
                    if val is None or frac is None:
                        continue
                    if "err" in param:
                        continue

                    result[f"lattice_{phase_name}_{param}"] = (
                        float(val.value) if hasattr(val, "value") else float(val)
                    )

            if result:
                return result

        return {}

    def _extract_coercivity(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float,
    ) -> dict[str, Any]:
        """
        Coercivity from MokeScan results.
        Key: 'coercivity_Am'
        """
        for scan in self.moke_scans:
            try:
                meas = scan.get_measurement_at(x_mm, y_mm, tolerance_mm)
            except ValueError:
                continue

            coercivity = meas.results.get("coercivity_m0", {}).get("mean")
            if coercivity is not None:
                value = (
                    float(coercivity.value)
                    if hasattr(coercivity, "value")
                    else float(coercivity)
                )
                return {"coercivity_Am": value * 795775}  # T to A/m

        return {"coercivity_Am": None}

    def _extract_thickness(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float,
    ) -> dict[str, Any]:
        """
        Film thickness from ProfilScan results.
        Key: 'thickness_um'
        """
        for scan in self.profil_scans:
            try:
                meas = scan.get_measurement_at(x_mm, y_mm, tolerance_mm)
            except ValueError:
                continue

            thickness = meas.results.get("thickness")
            if thickness is not None:
                value = (
                    float(thickness.value)
                    if hasattr(thickness, "value")
                    else float(thickness)
                )
                return {"thickness_um": value}

        return {"thickness_um": None}

    def _extract_annealing_from_sample(self) -> dict[str, Optional[float]]:
        """
        Pull annealing temperature (°C) and time (s) from self.sample if present.
        Returns plain floats stripped of units for YAML serialisation.
        """
        result: dict[str, Optional[float]] = {
            "temperature": None,
            "time": None,
        }

        if not self.sample:
            return result

        annealing = self.sample.get("annealing", {})
        if not isinstance(annealing, dict):
            return result

        temp = annealing.get("temperature")
        if temp is not None:
            result["temperature"] = (
                float(temp.value) if hasattr(temp, "value") else float(temp)
            )

        time = annealing.get("time")
        if time is not None:
            result["time"] = (
                float(time.value) if hasattr(time, "value") else float(time)
            )

        return result

    # ------------------------------------------------------------------
    # NOMAD .yaml export
    def to_nomad_yaml(
        self,
        output_folder: Path | str,
        description: str = "",
        lab_id: str = "",
        owner: str = "",
        short_name_prefix: str = "NEEL-Sample",
        tolerance_mm: float = 0.5,
    ) -> None:
        """
        Write one NOMAD-compatible YAML file per wafer position.

        Pulls composition, XRD phase results, coercivity and thickness from
        collect_properties(). Annealing temperature and time are pulled from
        self.sample metadata if present.

        Parameters
        ----------
        output_folder : Path or str
            Folder where .yaml files will be written (created if absent).
        description : str
            Free-text description written into every YAML file.
        lab_id : str
            Institution identifier (e.g. 'CNRS Institut Neel').
        owner : str
            Comma-separated list of owners.
        short_name_prefix : str
            Prefix for the per-position sample name.
        tolerance_mm : float
            Radius for co-location matching across scan types.
        """

        def sorting_prop(props: dict, prefix: str):
            PREFIX_MAP = {
                "composition": {"label": "element", "quantity": "atomic_fraction"},
                "phase_fraction": {"label": "phase", "quantity": "phase_fraction"},
                "lattice": {"label": "lattice", "quantity": "lattice_parameter"},
            }
            if prefix not in PREFIX_MAP.keys():
                return {}

            sorted_prop = sorted(
                [
                    {
                        PREFIX_MAP[prefix]["label"]: key.replace(f"{prefix}_", ""),
                        PREFIX_MAP[prefix]["quantity"]: value,
                    }
                    for key, value in props.items()
                    if key.startswith(f"{prefix}_")
                ],
                key=lambda d: d[PREFIX_MAP[prefix]["label"]],
            )

            return sorted_prop

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        annealing_info = self._extract_annealing_from_sample()
        all_properties = self.collect_properties(tolerance_mm=tolerance_mm)

        for props in tqdm.tqdm(all_properties, desc="Writing NOMAD YAML"):
            x_mm = props["x_mm"]
            y_mm = props["y_mm"]
            sample_name = f"{short_name_prefix}_{x_mm}_{y_mm}"

            composition = sorting_prop(props, "composition")
            phase_fractions = sorting_prop(props, "phase_fraction")
            lattice_parameters = sorting_prop(props, "lattice")

            yaml_data = _build_nomad_yaml(
                sample_name=sample_name,
                description=description,
                lab_id=lab_id,
                owner=owner,
                x_mm=x_mm,
                y_mm=y_mm,
                composition=composition,
                coercivity_Am=props.get("coercivity_Am"),
                phase_fractions=phase_fractions,
                lattice_parameters=lattice_parameters,
                annealing_temperature=annealing_info.get("temperature"),
                annealing_time=annealing_info.get("time"),
            )

            out_path = output_folder / f"{sample_name}.archive.yaml"
            with open(out_path, "w") as f:
                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

        print(f"Wrote {len(all_properties)} YAML files to {output_folder}")


# ------------------------------------------------------------------
# .yaml structure builder
def _build_nomad_yaml(
    sample_name: str,
    description: str,
    lab_id: str,
    owner: str,
    x_mm: float,
    y_mm: float,
    composition: list[dict[str, Any]],
    coercivity_Am: Optional[float],
    phase_fractions: Optional[list[dict[str, Any]]],
    lattice_parameters: Optional[list[dict[str, Any]]],
    annealing_temperature: Optional[float],
    annealing_time: Optional[float],
) -> dict:
    """Build the full NOMAD YAML dict for one sample position."""
    data: dict[str, Any] = {
        "m_def": "Sample",
        "description": description,
        "lab_id": lab_id,
        "short_name": sample_name,
        "sample_name": sample_name,
        "owner": owner,
        "xpos": float(x_mm),
        "ypos": float(y_mm),
    }

    if coercivity_Am is not None:
        data["CoercivityHcExternal"] = float(coercivity_Am)

    if annealing_temperature is not None:
        data["annealing_temperature"] = float(annealing_temperature)

    if annealing_time is not None:
        data["annealing_time"] = float(annealing_time)

    if composition:
        data["elemental_composition"] = composition

    if phase_fractions:
        data["phase_fractions"] = phase_fractions

    if lattice_parameters:
        data["lattice_parameters"] = lattice_parameters

    return {
        "definitions": _NOMAD_DEFINITIONS,
        "data": data,
    }
