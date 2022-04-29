from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pymusic.plotting import Plot, MatrixOfPlotsFigure

from .plots import SameAxesPlot

if typing.TYPE_CHECKING:
    from typing import List
    from matplotlib.axes import Axes
    from .config import Config


@dataclass
class TimeSeries:
    name: str
    values: np.ndarray
    time: np.ndarray


@dataclass
class SeriesPlot(Plot):
    tseries: TimeSeries

    def draw_on(self, ax: Axes) -> None:
        ax.plot(self.tseries.time, self.tseries.values,
                label=self.tseries.name)


@dataclass
class SeriesHist(Plot):
    tseries: TimeSeries

    def draw_on(self, ax: Axes) -> None:
        ax.hist(self.tseries.values, label=self.tseries.name)
        ax.set_xlabel(self.tseries.name)
        ax.set_ylabel("ndumps")


@dataclass
class LMax:
    main_h5: Path
    criteria: str

    def series(self) -> TimeSeries:
        with h5py.File(self.main_h5) as h5f:
            chkpts = h5f["checkpoints"]
            n_points = len(chkpts)
            time = np.zeros(n_points)
            values = np.zeros(n_points)
            for i, chk in enumerate(chkpts.values()):
                r_schwarz = chk["pp_parameters"]["r_schwarz_preset"][()].item()
                time[i] = chk["parameters"]["time"][()].item()
                values[i] = chk["Contour_field"][
                    f"pen_depth_{self.criteria}"][()].max() - r_schwarz
            return TimeSeries(
                name=f"lmax_{self.criteria}",
                values=values,
                time=time,
            )


def cmd(conf: Config) -> None:
    """Implementation of the lmax command."""
    post_h5 = conf.fort_pp.postfile
    all_plots: List[Plot] = []
    series_plots: List[Plot] = []
    for criteria in ("conv", "ke"):
        lmax = LMax(
            main_h5=post_h5,
            criteria=criteria,
        ).series()
        all_plots.append(SeriesHist(tseries=lmax))
        series_plots.append(SeriesPlot(tseries=lmax))
    all_plots.append(SameAxesPlot(plots=series_plots))
    MatrixOfPlotsFigure(
        plots=all_plots, nrows=1, ncols=len(all_plots),
    ).save_to("lmax_hist.pdf")
