from __future__ import annotations

from typing import Optional
from .basemeas import BaseMeas

import re
import pathlib as pl
import numpy as np
import mammos_units as mu
import mammos_entity as me


class XrdMeas(BaseMeas):
    """
    Abstract base for XRD single measurements.
    Provides shared add_results() parsing for .lst, .par and .dia files
    produced by FullProf/MAUD, used for both SmartLab and ESRF data.

    Subclasses may override _results_base_path to redirect where result
    files are located (e.g. ESRF keeps them in a separate refinement folder).
    """

    @property
    def _results_base_path(self) -> pl.Path:
        """Path stem used to find .lst/.par/.dia files. Defaults to self.path."""
        return self.path

    # ------------------------------------------------------------------
    # Parsing XRD results output from Profex software (lst, par and dia files)
    def add_results(self) -> None:
        """Parse all available result files (.lst, .par, .dia) into self.results."""
        self._parse_lst()
        self._parse_par()
        self._parse_dia()

    def _parse_lst(self) -> None:
        lst_file = self._results_base_path.with_suffix(".lst")
        if not lst_file.exists():
            print(lst_file)
            return

        results: dict = {"r_coefficients": {}, "phases": {}}
        current_phase: Optional[str] = None

        with open(lst_file) as f:
            for line in f:
                line = line.strip()

                if "Rp=" in line:
                    for key, val in re.findall(r"([A-Za-z\-]+)=([\d\.]+)", line):
                        results["r_coefficients"][key] = float(val)

                # Phase fractions check for with or without uncertainty cases
                match_with_err = re.match(
                    r"Q([A-Za-z0-9_]+)=([\d\.E+-]+)\+\-([\d\.E+-]+)", line
                )
                match_no_err = re.match(r"Q([A-Za-z0-9_]+)=([\d\.E+-]+)$", line)

                if match_with_err:
                    phase, val, err = match_with_err.groups()
                    results["phases"][phase] = {
                        "phase_fraction": float(val),
                        "phase_fraction_err": float(err),
                    }
                    continue

                if match_no_err:
                    phase, val = match_no_err.groups()
                    results["phases"][phase] = {
                        "phase_fraction": float(val),
                        "phase_fraction_err": None,
                    }
                    continue

                match = re.search(r"Local parameters.*phase\s+([A-Za-z0-9_]+)", line)
                if match:
                    current_phase = match.group(1)
                    results["phases"].setdefault(current_phase, {}).setdefault(
                        "lattice", {}
                    )
                    continue

                if current_phase is None:
                    continue

                match = re.search(r"SpacegroupNo=(\d+)", line)
                if match:
                    results["phases"][current_phase]["spacegroup_number"] = int(
                        match.group(1)
                    )

                match = re.search(r"XrayDensity=([\d\.Ee+-]+)", line)
                if match:
                    results["phases"][current_phase]["density"] = float(match.group(1))

                # Lattice parameters (A, B and C)
                if any(p in line for p in ("A=", "B=", "C=")):
                    lattice = results["phases"][current_phase].setdefault("lattice", {})
                    for token in line.split():
                        if not token.startswith(("A=", "B=", "C=")):
                            continue
                        axis, val = token.split("=")
                        key = axis.lower()
                        if val == "UNDEF":
                            lattice[key] = np.nan * mu.nm
                        elif "+-" in val:
                            value, err = val.split("+-")
                            lattice[key] = me.Entity(
                                f"LocalLatticeConstant{axis}", float(value) * mu.nm
                            )
                            lattice[f"{key}_err"] = float(err) * mu.nm
                        else:
                            lattice[key] = me.Entity(
                                f"LocalLatticeConstant{axis}", float(val) * mu.nm
                            )

        self.results = results

    def _parse_par(self) -> None:
        par_file = self._results_base_path.with_suffix(".par")
        if not par_file.exists():
            return

        anode_wavelengths = {
            "cu": 0.15406,
            "co": 0.178897,
            "mo": 0.07093,
            "fe": 0.19360,
            "cr": 0.22897,
        }  # in nm

        def _get_wavelength(line: str) -> Optional[float]:
            match = re.search(r"SYNCHROTRON=([\d\.Ee+-]+)", line)
            if match:
                return float(match.group(1))
            match = re.search(r"LAMBDA=([^\s]+)", line)
            if match:
                val = match.group(1).lower()
                try:
                    return float(val)
                except ValueError:
                    return anode_wavelengths.get(val)
            return None

        texture: dict = {}
        wavelength: Optional[float] = None

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

                    texture.setdefault(phase, {})
                    texture[phase][f"{hkl[0]}_{hkl[1]}_{hkl[2]}"] = {
                        "d_spacing": d_spacing * mu.nm,
                        "two_theta": two_theta * mu.deg,
                        "texture": tex_val * mu.dimensionless_unscaled,
                    }
                except Exception:
                    print("error parsing line:", line)

        self.results["texture"] = texture

    def _parse_dia(self) -> None:
        dia_file = self._results_base_path.with_suffix(".dia")
        if not dia_file.exists():
            return

        two_theta: list[float] = []
        measured: list[float] = []
        calculated: list[float] = []
        background: list[float] = []
        phase_patterns: dict[str, list[float]] = {}

        with open(dia_file) as f:
            header = f.readline()
            phase_names = re.findall(r"STRUC\[\d+\]=([^\s]+)", header)
            for phase in phase_names:
                phase_patterns[phase] = []

            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    two_theta.append(float(parts[0]))
                    measured.append(float(parts[1]))
                    calculated.append(float(parts[2]))
                    background.append(float(parts[3]))
                    for i, phase in enumerate(phase_names):
                        idx = 4 + i
                        phase_patterns[phase].append(
                            float(parts[idx]) if idx < len(parts) else 0.0
                        )
                except ValueError:
                    continue

        results: dict = {
            "Angle": np.array(two_theta),
            "Total Counts": np.array(measured),
            "Calculated": np.array(calculated),
            "Background": np.array(background),
        }
        for phase in phase_names:
            results[phase] = np.array(phase_patterns[phase])

        self.results["fits"] = results
