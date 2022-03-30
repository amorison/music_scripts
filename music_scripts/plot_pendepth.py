from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pymusic.plotting import Plot, SinglePlotFigure

if typing.TYPE_CHECKING:
    from typing import Iterable
    from loam.manager import ConfigurationManager


PENDEPTH_VARS = (
    "pen_depth_conv",
    "pen_depth_ke",
    "r_schwarz_max",
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


@dataclass(frozen=True)
class SchwarzSeries:
    values: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class SchwarzSeriesPlot(Plot):
    series: SchwarzSeries

    def draw_on(self, ax) -> None:
        ax.plot(self.series.time, self.series.values)
        ax.set_xlabel("time")
        ax.set_ylabel("Schwarzschild radius")


def get_contour(h5file: str, idump: int, name: str) -> Contour:
    with h5py.File(h5file) as h5f:
        chkp = h5f["checkpoints"][f"{idump:05d}"]
        data = Contour(
            name=name,
            values=chkp["Contour_field"][name][()].squeeze(),
            theta=chkp["pp_parameters"]["eval_grid"]["theta"][()].squeeze()
        )
    return data


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
    files_iter = iter(h5files)
    series = schwarz_series_in_file(next(files_iter))
    for file in files_iter:
        new_series = schwarz_series_in_file(file)
        series = SchwarzSeries(
            values=np.append(series.values, new_series.values),
            time=np.append(series.time, new_series.time),
        )
    return series


def cmd(conf: ConfigurationManager) -> None:
    folder = Path()
    idump = 7800
    h5file = folder / Path("post_es.h5")

    fig = SinglePlotFigure(
        plot=SameAxesPlot(
            plots=(ContourPlot(get_contour(h5file, idump, var))
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
