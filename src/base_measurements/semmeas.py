from typing import Optional
from .basemeasurement import BaseMeasurement

import pathlib as pl
import re
import numpy as np
import mammos_entity as me
import mammos_units as mu
import plotly.graph_objects as go
from PIL import Image


class SEMMeas(BaseMeasurement):
    FORMAT_VERSION = "1.0.0"
    SCHEMA = "sem.single_measurement"

    index_x: Optional[int] = None
    index_y: Optional[int] = None

    def __init__(self, path: pl.Path, **kwargs):
        super().__init__(path)
        self.step_x = kwargs.pop("step_x", 5)
        self.step_y = kwargs.pop("step_y", 5)
        self.start_x = kwargs.pop("start_x", -40)
        self.start_y = kwargs.pop("start_y", -40)
        self._read()

    def _set_positions_from_path(self, filepath):
        pattern = r".*\((\d+),(\d+)\).*"
        match = re.search(pattern, str(filepath))
        if not match:
            return

        x_idx = int(match.group(1))
        y_idx = int(match.group(2))

        self.index_x = x_idx
        self.index_y = y_idx
        self.x_position = -((x_idx - 1) * self.step_x + self.start_x) * mu.mm
        self.y_position = ((y_idx - 1) * self.step_y + self.start_y) * mu.mm

    def _read(self):
        self._set_positions_from_path(self.path)

        img = Image.open(self.path)
        img_array = np.array(img)

        # SEM images are saved as 3-channel grayscale — keep only one channel
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_array = img_array[..., 0]

        # TODO: switch to me.Entity("SEMImage", ...) once ontology is available
        self.data = {"Image": img_array}
        self.metadata = {}
        self.results = {}

    def _add_traces(self, fig):
        fig.add_trace(
            go.Heatmap(
                z=self.data["Image"].astype(float),
                colorscale="Gray",
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")

    def _xaxis_title(self):
        return "X pixels"

    def _yaxis_title(self):
        return "Y pixels"
