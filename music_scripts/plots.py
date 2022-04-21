"""Common plot objects."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymusic.plotting import Plot, BoundsFromMinMax

from .derived_fields import FieldGetter
if typing.TYPE_CHECKING:
    from typing import Optional, Sequence, Union, Iterable
    from matplotlib.axes import Axes
    from matplotlib.scale import ScaleBase
    from .array_on_grid import DumpArrayOnGrid, ArrayOnGrid, SimArrayOnGrid
    from .derived_fields import TimeAveragedProfGetter, TimeSeriesGetter


@dataclass(frozen=True)
class RawSphericalScalarPlot(Plot):
    r_coord: np.ndarray
    t_coord: np.ndarray
    data: np.ndarray
    cmap: Optional[str] = None
    color_bounds = BoundsFromMinMax()
    with_colorbar: bool = True

    def draw_on(self, ax: Axes) -> None:
        vmin, vmax = self.color_bounds(self.data)

        # project from (r,t) to (x,z)
        r_mesh, t_mesh = np.meshgrid(self.r_coord, self.t_coord, indexing="ij")
        x_mesh = r_mesh * np.sin(t_mesh)
        z_mesh = r_mesh * np.cos(t_mesh)

        surf = ax.pcolormesh(
            x_mesh, z_mesh, self.data, cmap=self.cmap,
            vmin=vmin, vmax=vmax, shading="flat", rasterized=True)

        ax.set_aspect("equal")
        ax.set_axis_off()
        if self.with_colorbar:
            cax = make_axes_locatable(ax).append_axes("right", size="3%",
                                                      pad=0.15)
            ax.figure.colorbar(surf, cax=cax)


@dataclass(frozen=True)
class SphericalScalarPlot(Plot):
    dump_arr: DumpArrayOnGrid
    get_data: FieldGetter
    cmap: Optional[str] = None
    color_bounds = BoundsFromMinMax()
    with_colorbar: bool = True

    def draw_on(self, ax: Axes) -> None:
        grid = self.dump_arr.grid
        RawSphericalScalarPlot(
            r_coord=grid.r_grid.face_points(),
            t_coord=grid.theta_grid.face_points(),
            data=self.get_data(self.dump_arr).array(),
            cmap=self.cmap,
            color_bounds=self.color_bounds,
            with_colorbar=self.with_colorbar,
        ).draw_on(ax)


@dataclass(frozen=True)
class SphericalVectorPlot(Plot):
    dump_arr: DumpArrayOnGrid
    get_rvec: FieldGetter
    get_tvec: FieldGetter
    arrow_stride: int = 16

    def draw_on(self, ax: Axes) -> None:
        grid = self.dump_arr.grid
        rad_c = grid.r_grid.cell_centers()
        theta_c = grid.theta_grid.cell_centers()
        vel_r = self.get_rvec(self.dump_arr).array()
        vel_t = self.get_tvec(self.dump_arr).array()
        radm, thetam = np.meshgrid(rad_c, theta_c, indexing='ij')
        vel_x = vel_r * np.sin(thetam) + vel_t * np.cos(thetam)
        vel_z = vel_r * np.cos(thetam) - vel_t * np.sin(thetam)
        xc_mesh = radm * np.sin(thetam)
        zc_mesh = radm * np.cos(thetam)
        sset = slice(None, None, self.vel_arrows_stride)
        ax.quiver(xc_mesh[sset, sset], zc_mesh[sset, sset],
                  vel_x[sset, sset], vel_z[sset, sset])
        ax.set_aspect("equal")
        ax.set_axis_off()


@dataclass(frozen=True)
class ProfPlot(Plot):
    music_data: ArrayOnGrid
    get_data: TimeAveragedProfGetter
    markers: Sequence[float] = field(default_factory=list)
    length_scale: Optional[float] = None

    def draw_on(self, ax: Axes) -> None:
        radius = self.music_data.grid.r_grid.cell_centers()
        markers = np.array(self.markers)
        if self.length_scale is not None:
            radius = radius / self.length_scale
            markers /= self.length_scale
        profile = self.get_data(self.music_data).array()
        ax.plot(radius, profile)
        for marker in markers:
            ax.axvline(marker, linewidth=1, linestyle=":", color="k")


@dataclass(frozen=True)
class TseriesPlot(Plot):
    music_data: SimArrayOnGrid
    get_data: TimeSeriesGetter

    def draw_on(self, ax: Axes) -> None:
        arr = self.get_data(self.music_data)
        time = np.array(arr.labels_along_axis("time"))
        tseries = arr.array()
        ax.plot(time, tseries, label=self.get_data.var_name)


@dataclass(frozen=True)
class WithScales(Plot):
    plot: Plot
    xscale: Union[str, ScaleBase] = "linear"
    yscale: Union[str, ScaleBase] = "linear"

    def draw_on(self, ax: Axes) -> None:
        self.plot.draw_on(ax)
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)


@dataclass(frozen=True)
class SameAxesPlot(Plot):
    plots: Iterable[Plot]
    legend: bool = True

    def draw_on(self, ax) -> None:
        for plot in self.plots:
            plot.draw_on(ax)
        if self.legend:
            ax.legend()
