from __future__ import annotations
from dataclasses import dataclass
import typing

from pymusic.plotting import SinglePlotFigure, Plot
import h5py
import numpy as np

from .plots import RawSphericalScalarPlot, SameAxesPlot

if typing.TYPE_CHECKING:
    from typing import Union
    from os import PathLike
    from loam.manager import ConfigurationManager


@dataclass(frozen=True)
class Contour:
    name: str
    values: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True)
class ContourPlot(Plot):
    contour: Contour

    def draw_on(self, ax) -> None:
        ax.plot(self.contour.theta, self.contour.values,
                label=self.contour.name)


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

    def contour_field(self, name: str) -> Contour:
        with h5py.File(self.master_h5) as h5f:
            chkp = h5f["checkpoints"][f"{self.idump:05d}"]
            contour = Contour(
                name=name,
                values=chkp["Contour_field"][name][()].squeeze(),
                theta=chkp["pp_parameters"]["eval_grid"]["theta"][()].squeeze()
            )
        return contour

    def field(self, name: str) -> Field:
        with h5py.File(self.master_h5) as h5f:
            chkp = h5f["checkpoints"][f"{self.idump:05d}"]
            field = Field(
                name=name,
                values=chkp["Field"][name][()].squeeze().T,
                radius=chkp["pp_parameters"]["eval_grid"]["rad"][()].squeeze(),
                theta=chkp["pp_parameters"]["eval_grid"]["theta"][()].squeeze()
            )
        return field


def field_cmd(conf: ConfigurationManager) -> None:
    checkpoint = FortPpCheckpoint(
        master_h5=conf.fort_pp.postfile, idump=conf.fort_pp.idump)
    field = checkpoint.field(conf.field_pp.plot)
    fig = SinglePlotFigure(
        plot=RawSphericalScalarPlot(
            r_coord=field.r_walls(),
            t_coord=field.t_walls(),
            data=field.values
        ),
    )
    fig.save_to(f"field_{field.name}.pdf")


def contour_cmd(conf: ConfigurationManager) -> None:
    checkpoint = FortPpCheckpoint(
        master_h5=conf.fort_pp.postfile, idump=conf.fort_pp.idump)
    varstr = "_".join(conf.contour_pp.plot)
    SinglePlotFigure(
        plot=SameAxesPlot(
            plots=(ContourPlot(checkpoint.contour_field(var))
                   for var in conf.contour_pp.plot),
        ),
    ).save_to(f"contour_{varstr}.pdf")
