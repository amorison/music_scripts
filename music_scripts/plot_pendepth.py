from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import h5py
import numpy as np
from pymusic.plotting import Plot, SinglePlotFigure

if typing.TYPE_CHECKING:
    from typing import Iterable
    from matplotlib.axes import Axes
    from .config import Config


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

    def draw_on(self, ax: Axes) -> None:
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


def cmd(conf: Config) -> None:
    folder = Path()

    all_h5s = sorted(folder.glob("post_transient*.h5"))
    all_h5s.extend(sorted(folder.glob("post_es*.h5")))
    fig = SinglePlotFigure(
        plot=SchwarzSeriesPlot(schwarz_series_from_set(all_h5s)),
    )
    fig.save_to("series_r_schwarz.pdf")
