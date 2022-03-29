from __future__ import annotations

import typing
from dataclasses import dataclass

import h5py
import numpy as np
from pymusic.plotting import Plot, SinglePlotFigure

if typing.TYPE_CHECKING:
    from typing import Iterable
    from loam.manager import ConfigurationManager


PENDEPTH_VARS = (
    "pen_depth_conv",
    "pen_depth_conv_max",
    "pen_depth_ke",
    "pen_depth_ke_max",
    "pen_depth_vr_r0neg",
    "pen_depth_vr_r0pos",
)


@dataclass(frozen=True)
class Contour:
    name: str
    values: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True)
class SameAxesPlot(Plot):
    plots: Iterable[Plot]
    legend: bool = True

    def draw_on(self, ax) -> None:
        for plot in self.plots:
            plot.draw_on(ax)
        if self.legend:
            ax.legend()


@dataclass(frozen=True)
class ContourPlot(Plot):
    contour: Contour

    def draw_on(self, ax) -> None:
        ax.plot(self.contour.theta, self.contour.values,
                label=self.contour.name)


def get_contour(h5file: str, idump: int, name: str) -> Contour:
    with h5py.File(h5file) as h5f:
        chkp = h5f["checkpoints"][f"{idump:05d}"]
        data = Contour(
            name=name,
            values=chkp["Contour_field"][name][()].squeeze(),
            theta=chkp["pp_parameters"]["eval_grid"]["theta"][()].squeeze()
        )
    return data


def cmd(conf: ConfigurationManager) -> None:
    idump = 7874
    h5file = "post_es.h5"
    fig = SinglePlotFigure(
        plot=SameAxesPlot(
            plots=(ContourPlot(get_contour(h5file, idump, var))
                   for var in PENDEPTH_VARS),
            legend=True,
        ),
    )
    fig.save_to("pendepth.pdf")
