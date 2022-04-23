from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses
import typing

from pymusic.plotting import SinglePlotFigure, Plot
import h5py
import numpy as np

from .plots import RawSphericalScalarPlot, SameAxesPlot

if typing.TYPE_CHECKING:
    from typing import Union, Optional, Generator, List, Type
    from os import PathLike
    from loam.manager import ConfigurationManager
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class Contour:
    name: str
    radius: np.ndarray
    theta: np.ndarray

    @staticmethod
    def const_rad(
        radius: float, theta_min: float, theta_max: float, label: str,
        npoints: int = 100,
    ) -> Contour:
        return Contour(
            name=label,
            radius=np.full(npoints, radius),
            theta=np.linspace(theta_min, theta_max, npoints)
        )


@dataclass(frozen=True)
class ContourPlot(Plot):
    contour: Contour

    def draw_on(self, ax: Axes) -> None:
        ax.plot(self.contour.theta, self.contour.radius,
                label=self.contour.name)


@dataclass(frozen=True)
class ContourSphericalPlot(Plot):
    contour: Contour
    color: str = "red"

    def draw_on(self, ax: Axes) -> None:
        rad = self.contour.radius
        theta = self.contour.theta
        x_pos = rad * np.sin(theta)
        z_pos = rad * np.cos(theta)
        ax.plot(x_pos, z_pos, color=self.color, lw=1)


@dataclass(frozen=True)
class Field:
    name: str
    values: np.ndarray
    radius: np.ndarray
    theta: np.ndarray

    @staticmethod
    def _walls_from_centers(centers: np.ndarray) -> np.ndarray:
        # this assumes grid with constant dx and centers midway between walls
        walls = np.zeros(centers.size + 1)
        half_dx = (centers[1] - centers[0]) / 2
        walls[1:] = centers + half_dx
        walls[0] = centers[0] - half_dx
        return walls

    def r_walls(self) -> np.ndarray:
        return self._walls_from_centers(self.radius)

    def t_walls(self) -> np.ndarray:
        return self._walls_from_centers(self.theta)


@dataclass(frozen=True)
class FortPpCheckpoint:
    master_h5: Union[str, PathLike]
    idump: int
    _chkgroup: Optional[h5py.Group] = dataclasses.field(
        default=None, init=False, repr=False, compare=False)

    @contextmanager
    def _chk(self) -> Generator[h5py.Group, None, None]:
        """Reentrant context manager to access the checkpoint data."""
        if self._chkgroup is not None:
            yield self._chkgroup
            return
        try:
            with h5py.File(self.master_h5) as h5f:
                object.__setattr__(
                    self, "_chkgroup", h5f["checkpoints"][f"{self.idump:05d}"])
                yield self._chkgroup
        finally:
            object.__setattr__(
                self, "_chkgroup", None)

    def pp_grid(self, direction: str) -> np.ndarray:
        with self._chk() as chk:
            return chk["pp_parameters"]["eval_grid"][direction][()].squeeze()

    def contour_field(self, name: str) -> Contour:
        with self._chk() as chk:
            contour = Contour(
                name=name,
                radius=chk["Contour_field"][name][()].squeeze(),
                theta=self.pp_grid("theta")
            )
        return contour

    def field(self, name: str) -> Field:
        with self._chk() as chk:
            field = Field(
                name=name,
                values=chk["Field"][name][()].squeeze().T,
                radius=self.pp_grid("rad"),
                theta=self.pp_grid("theta")
            )
        return field


def field_cmd(conf: ConfigurationManager) -> None:
    checkpoint = FortPpCheckpoint(
        master_h5=conf.fort_pp.postfile, idump=conf.fort_pp.idump)
    field = checkpoint.field(conf.field_pp.plot)
    plots: List[Plot] = [
        RawSphericalScalarPlot(
            r_coord=field.r_walls(),
            t_coord=field.t_walls(),
            data=field.values
        ),
    ]
    if conf.plotting.rmarks:
        tmin, tmax = checkpoint.pp_grid("theta")[[0, -1]]
        plots.extend(
            ContourSphericalPlot(Contour.const_rad(rad, tmin, tmax, ""))
            for rad in conf.plotting.rmarks)

    SinglePlotFigure(
        plot=SameAxesPlot(
            plots=plots,
            legend=False,
        )
    ).save_to(f"field_{field.name}.pdf")


def contour_cmd(conf: ConfigurationManager) -> None:
    checkpoint = FortPpCheckpoint(
        master_h5=conf.fort_pp.postfile, idump=conf.fort_pp.idump)
    varstr = "_".join(conf.contour_pp.plot)
    over_str = ""
    plots: List[Plot] = []
    cont_plot: Type[Union[ContourPlot, ContourSphericalPlot]] = ContourPlot
    legend = True
    if conf.contour_pp.over:
        field = checkpoint.field(conf.contour_pp.over)
        plots.append(
            RawSphericalScalarPlot(
                r_coord=field.r_walls(),
                t_coord=field.t_walls(),
                data=field.values
            )
        )
        cont_plot = ContourSphericalPlot
        legend = False
        over_str = f"__over_{conf.contour_pp.over}"
    plots.extend(
        cont_plot(checkpoint.contour_field(var))
        for var in conf.contour_pp.plot)
    if conf.plotting.rmarks:
        tmin, tmax = checkpoint.pp_grid("theta")[[0, -1]]
        plots.extend(
            cont_plot(Contour.const_rad(rad, tmin, tmax, ""))
            for rad in conf.plotting.rmarks)

    SinglePlotFigure(
        plot=SameAxesPlot(
            plots=plots,
            legend=legend,
        ),
    ).save_to(f"contour_{varstr}{over_str}.pdf")
