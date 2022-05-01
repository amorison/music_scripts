from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pymusic.plotting import Plot, MatrixOfPlotsFigure

from .fort_pp import FortPpCheckpoint
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
        ax.set_xlabel("time")
        ax.set_ylabel(self.tseries.name)


@dataclass
class SeriesHist(Plot):
    tseries: TimeSeries
    group_by_unique: bool = False

    def draw_on(self, ax: Axes) -> None:
        if self.group_by_unique:
            ax.bar(*np.unique(self.tseries.values, return_counts=True),
                   label=self.tseries.name)
        else:
            ax.hist(self.tseries.values, label=self.tseries.name)
        ax.set_xlabel(self.tseries.name)
        ax.set_ylabel("ndumps")


@dataclass
class LMax:
    main_h5: Path
    criteria: str
    normalize_dr: bool = False

    def series(self) -> TimeSeries:
        with h5py.File(self.main_h5) as h5f:
            chkpts = h5f["checkpoints"]
            n_points = len(chkpts)
            time = np.zeros(n_points)
            values = np.zeros(n_points)
            for i, idump in enumerate(map(int, chkpts.keys())):
                chk = FortPpCheckpoint(master_h5=h5f, idump=idump)
                r_schwarz = chk.pp_param("r_schwarz_preset").item()
                time[i] = chk.param("time").item()
                values[i] = chk.contour_field(
                    f"pen_depth_{self.criteria}").radius.max() - r_schwarz
                if self.normalize_dr:
                    dr = np.diff(chk.pp_grid("rad")).mean()
                    values[i] = np.rint(values[i] / dr)
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
            normalize_dr=conf.lmax.normdr,
        ).series()
        all_plots.append(SeriesHist(
            tseries=lmax,
            group_by_unique=conf.lmax.normdr
        ))
        series_plots.append(SeriesPlot(tseries=lmax))
    all_plots.append(SameAxesPlot(plots=series_plots))
    MatrixOfPlotsFigure(
        plots=all_plots, nrows=1, ncols=len(all_plots),
    ).save_to("lmax_hist.pdf")
