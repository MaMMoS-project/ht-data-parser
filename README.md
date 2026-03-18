# HT Parsers – High Throughput Experiment Data Parser

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

**HT Parsers** is a Python package developed at **Institut Néel** to parse and structure data from **high-throughput experiments**.

The package supports multiple characterization techniques and stores data using the **MaMMoS ontology**, enabling consistent, machine-readable datasets that can be exported to **HDF5 (NeXus-inspired format)**.

Supported techniques include:

* **EDX** – elemental composition
* **MOKE** – magnetic measurements
* **XRD** – structural characterization
* **Profilometry (DEKTAK)** – film thickness
* **SEM** – microstructure imaging

Ontology used in this project:
https://github.com/MaMMoS-project/MagneticMaterialsOntology

---

# Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/MaMMoS-project/ht-data-parser.git
cd ht-data-parser
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are managed through **`pyproject.toml`**, please note that you need to use **Python 3.11** or higher.

---

# Basic Usage

A Jupyter Notebook **`DataParser.ipynb`** is providing a full detail on how to use the ht-data-parser, a basic usage has also been written down below:

Each measurement is represented by a **`Meas` class** containing:

* `metadata` – instrument metadata
* `data` – raw measurement data
* `results` – processed quantities

Example with **EDX**:

```python
import pathlib
from src.base_measurements.edxmeas import EDXMeas

path = pathlib.Path("Spectrum_(9,9).spx")

edx = EDXMeas(path)
fig = edx.plot()
fig.show()
```

Quantities are stored as **ontology-aware entities**:

```python
energy = edx.data["Energy"]

energy.value
energy.unit
energy.ontology
```

---

# High Throughput Scans

For wafer-scale experiments (~250 positions), the package provides **Scan classes** that parse entire folders of measurements.

Example:

```python
from src.ht_measurements.edxscan import EDXScan

scan = EDXScan("EDX_folder")
scan.heatmap("results.Nd.AtomPercent")
```

Supported scan classes:

* `EDXScan`
* `MOKEScan`
* `SMARTLABScan`
* `ProfilScan`
* `SEMScan`

---

# Data Export

Measurements and scans can be exported to **HDF5**:

```python
scan.to_hdf5("dataset.hdf5")
```

The resulting structure follows conventions inspired by the **NeXus scientific data format**:

https://www.nexusformat.org/
