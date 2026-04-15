from __future__ import annotations

from typing import Optional
from .basemeas import BaseMeas

import pathlib as pl
import re
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go
from PIL import Image


class SemMeas(BaseMeas):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "sem.single_measurement"
    RESULTS_UNIT_MAP = {}

    index_x: Optional[int] = None
    index_y: Optional[int] = None

    def __init__(self, path: pl.Path, **kwargs) -> None:
        super().__init__(path)
        self.step_x: int = kwargs.pop("step_x", 5)
        self.step_y: int = kwargs.pop("step_y", 5)
        self.start_x: int = kwargs.pop("start_x", -40)
        self.start_y: int = kwargs.pop("start_y", -40)
        self._read()

    # ------------------------------------------------------------------
    # Position
    def _set_positions_from_path(self, file_path: pl.Path) -> None:
        match = re.search(r".*\((\d+),(\d+)\).*", str(file_path))
        if not match:
            return

        x_idx = int(match.group(1))
        y_idx = int(match.group(2))

        self.index_x = x_idx
        self.index_y = y_idx
        self.x_position = -((x_idx - 1) * self.step_x + self.start_x) * mu.mm
        self.y_position = ((y_idx - 1) * self.step_y + self.start_y) * mu.mm

    # ------------------------------------------------------------------
    # Reading
    def _read(self) -> None:
        self._set_positions_from_path(self.path)

        img_array = np.array(Image.open(self.path))

        # SEM images are saved as 3-channel grayscale — keep only one channel
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_array = img_array[..., 0]

        # TODO: switch to me.Entity("SEMImage", ...) once ontology is available
        self.data = {"Image": img_array}
        self.metadata = {}
        self.results = {}

    # ------------------------------------------------------------------
    # Plotting
    def _add_traces(self, fig: go.Figure) -> None:
        fig.add_trace(
            go.Heatmap(
                z=self.data["Image"].astype(float),
                colorscale="Gray",
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")

    def _xaxis_title(self) -> str:
        return "X pixels"

    def _yaxis_title(self) -> str:
        return "Y pixels"
