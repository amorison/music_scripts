from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import h5py
import numpy as np
from pymusic.plotting import Plot, SinglePlotFigure

from .fort_pp import FortPpCheckpoint

if typing.TYPE_CHECKING:
    from typing import Iterable
    from loam.manager import ConfigurationManager
    from .fort_pp import Contour


PENDEPTH_VARS = (
    "pen_depth_conv",
    "pen_depth_ke",
    "r_schwarz_max",
)


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


@dataclass(frozen=True)
class SchwarzSeries:
    values: np.ndarray
    time: np.ndarray

    def append(self, other: SchwarzSeries) -> SchwarzSeries:
        return SchwarzSeries(
            values=np.append(self.values, other.values),
            time=np.append(self.time, other.time),
        )


@dataclass(frozen=True)
class SchwarzSeriesPlot(Plot):
    series: SchwarzSeries

    def draw_on(self, ax) -> None:
        ax.plot(self.series.time, self.series.values)
        ax.set_xlabel("time")
        ax.set_ylabel("Schwarzschild radius")


def schwarz_series_in_file(h5file: Path) -> SchwarzSeries:
    with h5py.File(h5file) as h5f:
        checks = h5f["checkpoints"]
        time = np.zeros(len(checks))
        values = np.zeros(len(checks))
        for i, check in enumerate(checks.values()):
            time[i] = check["parameters"]["time"][()].item()
            values[i] = check["pp_parameters"]["ave_r_schwarz_max"][()].item()
    return SchwarzSeries(values, time)


def schwarz_series_from_set(h5files: Iterable[Path]) -> SchwarzSeries:
    return reduce(
        SchwarzSeries.append,
        map(schwarz_series_in_file, h5files)
    )


def cmd(conf: ConfigurationManager) -> None:
    folder = Path()

    checkpoint = FortPpCheckpoint(
        master_h5=folder / Path("post_es.h5"),
        idump=7800,
    )

    fig = SinglePlotFigure(
        plot=SameAxesPlot(
            plots=(ContourPlot(checkpoint.contour_field(var))
                   for var in PENDEPTH_VARS),
            legend=True,
        ),
    )
    fig.save_to("pendepth.pdf")

    all_h5s = sorted(folder.glob("post_transient*.h5"))
    all_h5s.extend(sorted(folder.glob("post_es*.h5")))
    fig = SinglePlotFigure(
        plot=SchwarzSeriesPlot(schwarz_series_from_set(all_h5s)),
    )
    fig.save_to("series_r_schwarz.pdf")
