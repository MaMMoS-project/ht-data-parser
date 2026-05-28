from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class PlotMixin:
    """
    Mixin providing shared plotting methods for BaseScan and Film.

    Classes that inherit this mixin must implement get_quantity() so that
    heatmap() can call it. For plot_1d() and plot_ternary(), data is passed
    directly as lists or arrays, so any caller can use them.
    """

    @staticmethod
    def _colorbar_layout(
        z_min: float,
        z_max: float,
        precision: int = 1,
        title: str = "",
        prefix: str = "",
    ) -> dict[str, Any]:
        z_mid = (z_min + z_max) / 2
        return dict(
            title=dict(
                text=prefix + title + "<br>&nbsp;<br>",
                font=dict(size=24),
            ),
            tickmode="array",
            tickvals=[
                z_min,
                (z_min + z_mid) / 2,
                z_mid,
                (z_max + z_mid) / 2,
                z_max,
            ],
            ticktext=[
                f"{z_min:.{precision}f}",
                f"{(z_min + z_mid) / 2:.{precision}f}",
                f"{z_mid:.{precision}f}",
                f"{(z_max + z_mid) / 2:.{precision}f}",
                f"{z_max:.{precision}f}",
            ],
            tickfont=dict(size=24),
            ticklen=8,
            thickness=25,
        )

    @staticmethod
    def _drop_nan_rows(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return copies of all arrays with rows removed where any array has NaN."""
        arrays = [np.asarray(a, dtype=float) for a in arrays]
        valid = np.ones(len(arrays[0]), dtype=bool)
        for a in arrays:
            valid &= ~np.isnan(a)
        return tuple(a[valid] for a in arrays)

    def heatmap(
        self,
        quantity: str,
        xlim: list[float] = [-42.5, 42.5],
        ylim: list[float] = [-42.5, 42.5],
        precision: int = 1,
        width: int = 650,
        height: int = 600,
        prefix: str = "",
        show: bool = True,
    ) -> go.Figure:
        """
        Plot a heatmap of any scalar quantity across the wafer.

        Parameters
        ----------
        quantity : str
            Dot-path to the scalar quantity (e.g. 'results.Fe.AtomPercent').
        prefix : str
            Optional prefix prepended to the colorbar unit label.
        show : bool
            If True, call fig.show() before returning.
        """
        quantities = self.get_quantity(quantity)
        values = [q.value if hasattr(q, "value") else q for q in quantities]

        for v in values:
            if np.ndim(v) > 0:
                raise ValueError(
                    f"'{quantity}' is not scalar per measurement (shape {np.shape(v)}). "
                    "Heatmap requires scalar quantities."
                )

        unit = str(quantities[0].unit) if hasattr(quantities[0], "unit") else ""
        measurements = getattr(self, "measurements", None) or [
            m for s in self.scans.values() for m in s.measurements
        ]

        xs = [m.x_position.value for m in measurements]
        ys = [m.y_position.value for m in measurements]
        x_unique = sorted(set(xs))
        y_unique = sorted(set(ys))
        z = np.full((len(y_unique), len(x_unique)), np.nan)
        x_map = {x: i for i, x in enumerate(x_unique)}
        y_map = {y: i for i, y in enumerate(y_unique)}

        for x, y, v in zip(xs, ys, values):
            z[y_map[y], x_map[x]] = v

        fig = go.Figure(
            data=go.Heatmap(
                x=x_unique,
                y=y_unique,
                z=z,
                colorscale="Plasma",
                colorbar=self._colorbar_layout(
                    z_min=np.nanmin(z),
                    z_max=np.nanmax(z),
                    precision=precision,
                    title=unit,
                    prefix=prefix,
                ),
            )
        )
        fig.update_layout(
            title=f"Heatmap of {quantity}",
            xaxis=dict(
                range=xlim,
                title="X (mm)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
                tickmode="array",
                tickvals=[-40, -20, 0, 20, 40],
            ),
            yaxis=dict(
                range=ylim,
                scaleanchor="x",
                scaleratio=1,
                title="Y (mm)",
                tickfont=dict(size=24),
                title_font=dict(size=24),
                tickmode="array",
                tickvals=[-40, -20, 0, 20, 40],
            ),
            width=width,
            height=height,
        )

        if show:
            fig.show()
        return fig

    def plot_1d(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        color: Optional[list | np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        color_label: str = "color",
        range_color: Optional[tuple[float, float]] = None,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        colorscale: str = "plasma",
        marker_size: int = 8,
        width: int = 950,
        height: int = 650,
        title: str = "",
        precision: int = 2,
        show: bool = True,
    ) -> go.Figure:
        """
        Scatter / line plot of two quantities with optional color encoding.

        Parameters
        ----------
        x, y : list or array
            Data for the two axes.
        color : list or array, optional
            Values used for color encoding. Pass None for a plain scatter.
        x_label, y_label, color_label : str
            Axis and colorbar labels.
        range_color : (min, max), optional
            Colorscale range. Computed from data if None.

        Example (Film)
        -------
        props = film.collect_properties()
        film.plot_1d(
            x=[p["x_mm"] for p in props],
            y=[p["coercivity_Am"] for p in props],
            color=[p["composition_Fe"] for p in props],
            x_label="X (mm)", y_label="Hc (A/m)", color_label="Fe (at. fr.)",
        )
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if color is not None:
            color = np.asarray(color, dtype=float)
            x, y, color = self._drop_nan_rows(x, y, color)
        else:
            x, y = self._drop_nan_rows(x, y)

        df = pd.DataFrame({x_label: x, y_label: y})

        if color is not None:
            df[color_label] = color
            if range_color is None:
                range_color = (float(np.nanmin(color)), float(np.nanmax(color)))

        order = np.argsort(x)
        df = df.iloc[order]

        fig = px.scatter(
            df,
            x=x_label,
            y=y_label,
            color=color_label if color is not None else None,
            range_color=range_color,
            color_continuous_scale=colorscale if color is not None else None,
            width=width,
            height=height,
            title=title or f"{y_label} vs {x_label}",
        )
        fig.update_traces(marker={"size": marker_size})
        fig.update_layout(
            xaxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
            yaxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
        )
        if x_range:
            fig.update_xaxes(range=list(x_range))
        if y_range:
            fig.update_yaxes(range=list(y_range))
        if color is not None and range_color is not None:
            fig.update_coloraxes(
                colorbar=self._colorbar_layout(
                    z_min=range_color[0],
                    z_max=range_color[1],
                    precision=precision,
                    title=color_label,
                )
            )

        if show:
            fig.show()
        return fig

    def plot_ternary(
        self,
        a: list | np.ndarray,
        b: list | np.ndarray,
        c: list | np.ndarray,
        color: Optional[list | np.ndarray] = None,
        a_label: str = "A",
        b_label: str = "B",
        c_label: str = "C",
        color_label: str = "color",
        range_color: Optional[tuple[float, float]] = None,
        colorscale: str = "plasma",
        marker_size: int = 10,
        width: int = 1000,
        height: int = 800,
        title: str = "",
        precision: int = 2,
        show: bool = True,
    ) -> go.Figure:
        """
        Ternary scatter plot with optional color encoding.

        Parameters
        ----------
        a, b, c : list or array
            Corner values — must sum to a constant (e.g. 100 or 1).
        color : list or array, optional
            Color encoding (e.g. coercivity).

        Example (Film)
        -------
        props = film.collect_properties()
        film.plot_ternary(
            a=[p.get("composition_Nd", 0) for p in props],
            b=[p.get("composition_Ce", 0) for p in props],
            c=[p.get("composition_Fe", 0) for p in props],
            color=[p["coercivity_Am"] for p in props],
            a_label="Nd (at. fr.)", b_label="Ce (at. fr.)", c_label="Fe (at. fr.)",
            color_label="Hc (A/m)",
        )
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)

        if color is not None:
            color = np.asarray(color, dtype=float)
            a, b, c, color = self._drop_nan_rows(a, b, c, color)
        else:
            a, b, c = self._drop_nan_rows(a, b, c)

        df = pd.DataFrame({a_label: a, b_label: b, c_label: c})

        if color is not None:
            df[color_label] = color
            if range_color is None:
                range_color = (float(np.nanmin(color)), float(np.nanmax(color)))

        fig = px.scatter_ternary(
            df,
            a=a_label,
            b=b_label,
            c=c_label,
            color=color_label if color is not None else None,
            range_color=range_color,
            color_continuous_scale=colorscale if color is not None else None,
            width=width,
            height=height,
            title=title,
        )
        fig.update_traces(marker={"size": marker_size})
        fig.update_layout(
            ternary_aaxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
            ternary_baxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
            ternary_caxis=dict(tickfont=dict(size=24), title_font=dict(size=24)),
        )
        if color is not None and range_color is not None:
            fig.update_coloraxes(
                colorbar=self._colorbar_layout(
                    z_min=range_color[0],
                    z_max=range_color[1],
                    precision=precision,
                    title=color_label,
                )
            )

        if show:
            fig.show()
        return fig
