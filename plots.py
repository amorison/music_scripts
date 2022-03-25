"""Common plot objects."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np
from pymusic.plotting import Plot, BoundsFromMinMax

if typing.TYPE_CHECKING:
    from typing import Optional, Sequence
    from array_on_grid import DumpArrayOnGrid, ArrayOnGrid
    from derived_fields import FieldGetter, TimeAveragedProfGetter


@dataclass(frozen=True)
class SphericalPlot(Plot):
    dump_arr: DumpArrayOnGrid
    get_data: FieldGetter
    with_vel_arrows: bool = False
    vel_arrows_stride: int = 16
    cmap: Optional[str] = None
    color_bounds = BoundsFromMinMax()
    with_colorbar: bool = True

    def draw_on(self, ax) -> None:
        grid = self.dump_arr.dump.grid
        rad = grid.r_grid.face_points()
        theta = grid.theta_grid.face_points()
        data = self.get_data(self.dump_arr).array()
        vmin, vmax = self.color_bounds(data)
        r_mesh, t_mesh = np.meshgrid(rad, theta, indexing="ij")
        x_mesh = r_mesh * np.cos(t_mesh)
        y_mesh = r_mesh * np.sin(t_mesh)
        surf = ax.pcolormesh(
            x_mesh, y_mesh, data, cmap=self.cmap,
            vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if self.with_vel_arrows:
            rad_c = grid.r_grid.cell_centers()
            theta_c = grid.theta_grid.cell_centers()
            vel_r = FieldGetter("vel_1")(self.dump_arr).array()
            vel_t = FieldGetter("vel_2")(self.dump_arr).array()
            radm, thetam = np.meshgrid(rad_c, theta_c, indexing='ij')
            vel_x = vel_r * np.cos(thetam) - vel_t * np.sin(thetam)
            vel_y = vel_t * np.cos(thetam) + vel_r * np.sin(thetam)
            xc_mesh = radm * np.cos(thetam)
            yc_mesh = radm * np.sin(thetam)
            sset = slice(None, None, self.vel_arrows_stride)
            ax.quiver(xc_mesh[sset, sset], yc_mesh[sset, sset],
                      vel_x[sset, sset], vel_y[sset, sset])
        if self.with_colorbar:
            ax.figure.colorbar(surf, ax=ax)


@dataclass(frozen=True)
class ProfPlot(Plot):
    music_data: ArrayOnGrid
    get_data: TimeAveragedProfGetter
    markers: Sequence[float] = field(default_factory=list)
    length_scale: Optional[float] = None
    log_scale: bool = False

    def draw_on(self, ax) -> None:
        radius = self.music_data.grid.r_grid.cell_centers()
        markers = np.array(self.markers)
        if self.length_scale is not None:
            radius = radius / self.length_scale
            markers /= self.length_scale
        profile = self.get_data(self.music_data).array()
        if self.log_scale:
            ax.semilogy(radius, profile)
        for marker in markers:
            ax.axvline(marker, linewidth=1, linestyle=":", color="k")
