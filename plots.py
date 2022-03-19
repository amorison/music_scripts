"""Common plot objects."""
from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
from pymusic.io import MusicDump
from pymusic.plotting import Plot, BoundsFromMinMax

if typing.TYPE_CHECKING:
    from typing import Optional
    from derived_fields import FieldGetter


@dataclass(frozen=True)
class SphericalPlot(Plot):
    dump: MusicDump
    get_data: FieldGetter
    cmap: Optional[str] = None
    color_bounds = BoundsFromMinMax()
    with_colorbar: bool = True

    def draw_on(self, ax) -> None:
        rad = self.dump.grid.r_grid.face_points()
        theta = self.dump.grid.theta_grid.face_points()
        data = self.get_data(self.dump.big_array()).array()
        vmin, vmax = self.color_bounds(data)
        r_mesh, t_mesh = np.meshgrid(rad, theta, indexing="ij")
        x_mesh = r_mesh * np.cos(t_mesh)
        y_mesh = r_mesh * np.sin(t_mesh)
        surf = ax.pcolormesh(
            x_mesh, y_mesh, data, cmap=self.cmap,
            vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if self.with_colorbar:
            ax.figure.colorbar(surf, ax=ax)
