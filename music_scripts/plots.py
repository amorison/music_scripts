"""Common plot objects."""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymusic.plotting import Plot

from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from typing import Iterable, Optional, Sequence, Tuple, Union

    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.scale import ScaleBase

    from .musicdata import BaseMusicData, Snap


_LABELS = MappingProxyType(
    {
        "vel_1": "v_r",
        "vel_2": r"v_\theta",
        "vel_3": r"v_\phi",
        "density": r"\rho",
        "e_spec_int": "e_i",
        "scalar_1": "He",
        "temp": "T",
        "press": "P",
        "entropy": "S",
        "ekin": "E_k",
        "adiab_grad": r"\nabla_{\mathrm{ad}}",
    }
)


def _labelizer(var: str, perturbation: bool = False) -> str:
    symbol = _LABELS.get(var, var)
    return rf"$\delta {symbol}/{symbol}$" if perturbation else f"${symbol}$"


@dataclass(frozen=True)
class RawSphericalScalarPlot(Plot):
    r_coord: np.ndarray
    t_coord: np.ndarray
    data: np.ndarray
    data_label: Optional[str] = None
    r_norm: Optional[float] = None
    cmap: Optional[str] = None
    with_colorbar: bool = True
    norm: Optional[Normalize] = None
    costh: bool = False
    rbounds: Tuple[Optional[float], Optional[float]] = (None, None)
    vbounds: Tuple[Optional[float], Optional[float]] = (None, None)

    def draw_on(self, ax: Axes) -> None:
        # project from (r,t) to (x,z)
        r_coord = self.r_coord
        if self.r_norm is not None:
            r_coord /= self.r_norm
        if self.costh:
            x_mesh = np.cos(self.t_coord)
            z_mesh = r_coord
        else:
            r_mesh, t_mesh = np.meshgrid(r_coord, self.t_coord, indexing="ij")
            x_mesh = r_mesh * np.sin(t_mesh)
            z_mesh = r_mesh * np.cos(t_mesh)

        surf = ax.pcolormesh(
            x_mesh,
            z_mesh,
            self.data,
            cmap=self.cmap,
            norm=self.norm,
            vmin=self.vbounds[0],
            vmax=self.vbounds[1],
            shading="flat",
            rasterized=True,
        )

        if self.costh:
            ax.set_ylim(*self.rbounds)
            ax.set_xlabel(r"$\cos\theta$")
            ax.set_ylabel(r"$r/R_{\mathrm{star}}$" if self.r_norm else "$r$")
        else:
            ax.set_aspect("equal")
            ax.set_axis_off()
        if self.with_colorbar:
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
            ax.figure.colorbar(surf, cax=cax, label=self.data_label)


@dataclass(frozen=True)
class RawCartesianScalarPlot(Plot):
    x_coord: np.ndarray
    y_coord: np.ndarray
    data: np.ndarray
    cmap: Optional[str] = None
    with_colorbar: bool = True
    norm: Optional[Normalize] = None
    vbounds: Tuple[Optional[float], Optional[float]] = (None, None)

    def draw_on(self, ax: Axes) -> None:
        surf = ax.pcolormesh(
            self.x_coord,
            self.y_coord,
            self.data,
            cmap=self.cmap,
            norm=self.norm,
            vmin=self.vbounds[0],
            vmax=self.vbounds[1],
            shading="flat",
            rasterized=True,
        )

        ax.set_aspect("equal")
        ax.set_axis_off()
        if self.with_colorbar:
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
            ax.figure.colorbar(surf, cax=cax)


@dataclass(frozen=True)
class ScalarPlot(Plot):
    snap: Snap
    var: str
    cmap: Optional[str] = None
    with_colorbar: bool = True
    norm: Optional[Normalize] = None
    costh: bool = False
    rbounds: Tuple[Optional[float], Optional[float]] = (None, None)
    vbounds: Tuple[Optional[float], Optional[float]] = (None, None)
    normalize_r: Optional[float] = None
    perturbation: bool = False

    def draw_on(self, ax: Axes) -> None:
        field = self.snap.field[self.var].array()
        if self.perturbation:
            prof = self.snap.rprof[self.var].array()
            field = (field - prof[:, np.newaxis]) / field
        grids = self.snap.grid.grids
        plot: Plot
        if not self.snap.cartesian:
            plot = RawSphericalScalarPlot(
                r_coord=grids[0].face_points(),
                t_coord=grids[1].face_points(),
                data=field,
                data_label=_labelizer(self.var, self.perturbation),
                r_norm=self.normalize_r,
                cmap=self.cmap,
                with_colorbar=self.with_colorbar,
                norm=self.norm,
                costh=self.costh,
                rbounds=self.rbounds,
                vbounds=self.vbounds,
            )
        else:
            plot = RawCartesianScalarPlot(
                x_coord=grids[1].face_points(),
                y_coord=grids[0].face_points(),
                data=field,
                cmap=self.cmap,
                with_colorbar=self.with_colorbar,
                norm=self.norm,
                vbounds=self.vbounds,
            )
        plot.draw_on(ax)


@dataclass(frozen=True)
class SphericalVectorPlot(Plot):
    snap: Snap
    vec_r: str
    vec_t: str
    arrow_stride: int = 16

    def draw_on(self, ax: Axes) -> None:
        grids = self.snap.grid.grids
        rad_c = grids[0].cell_centers()
        theta_c = grids[1].cell_centers()
        vel_r = self.snap.field[self.vec_r].array()
        vel_t = self.snap.field[self.vec_t].array()
        radm, thetam = np.meshgrid(rad_c, theta_c, indexing="ij")
        vel_x = vel_r * np.sin(thetam) + vel_t * np.cos(thetam)
        vel_z = vel_r * np.cos(thetam) - vel_t * np.sin(thetam)
        xc_mesh = radm * np.sin(thetam)
        zc_mesh = radm * np.cos(thetam)
        sset = slice(None, None, self.arrow_stride)
        ax.quiver(
            xc_mesh[sset, sset],
            zc_mesh[sset, sset],
            vel_x[sset, sset],
            vel_z[sset, sset],
        )
        ax.set_aspect("equal")
        ax.set_axis_off()


@dataclass(frozen=True)
class ProfPlot(Plot):
    music_data: BaseMusicData
    var: str
    markers: Sequence[float] = field(default_factory=list)
    length_scale: Optional[float] = None

    def draw_on(self, ax: Axes) -> None:
        radius = self.music_data.grid.grids[0].cell_centers()
        markers = np.array(self.markers)
        if self.length_scale is not None:
            radius = radius / self.length_scale
            markers /= self.length_scale
        if isinstance(self.music_data, MusicData):
            profile = self.music_data.rprof_avg[self.var].array()
        else:
            profile = self.music_data.rprof[self.var].array()
        ax.plot(radius, profile)
        for marker in markers:
            ax.axvline(marker, linewidth=1, linestyle=":", color="k")
        ax.set_xlabel("radius" if self.length_scale is None else "normalized radius")
        ax.set_ylabel(_labelizer(self.var))


@dataclass(frozen=True)
class RiverPlot(Plot):
    music_data: MusicData
    var: str

    def draw_on(self, ax: Axes) -> None:
        profile = self.music_data.rprof[self.var]
        time = profile.labels_along_axis("time")
        x_1 = profile.labels_along_axis("x1")
        surf = ax.pcolormesh(
            time, x_1, profile.array().T, shading="nearest", rasterized=True
        )
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
        ax.figure.colorbar(surf, cax=cax)


@dataclass(frozen=True)
class TseriesPlot(Plot):
    music_data: MusicData
    var: str

    def draw_on(self, ax: Axes) -> None:
        arr = self.music_data.tseries[self.var]
        time = np.array(arr.labels_along_axis("time"))
        tseries = arr.array()
        ax.plot(time, tseries, label=self.var)
        ax.set_xlabel("time")
        ax.set_ylabel(_labelizer(self.var))


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

    def draw_on(self, ax: Axes) -> None:
        for plot in self.plots:
            plot.draw_on(ax)
        if self.legend:
            ax.legend()
