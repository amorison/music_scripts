#!/usr/bin/env python3
"""RMS velocity"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymusic.plotting import SinglePlotFigure

from .musicdata import MusicData
from .plots import ProfPlot, TseriesPlot, WithScales


def tau_conv(mdat: MusicData) -> float:
    """Convective time scale."""
    grid = mdat.grid
    d_rad = grid.r_grid.cell_widths()
    core_mask = grid.r_grid.cell_centers() < mdat.prof1d.params["rcore"]
    return (
        mdat.rprof["vrms"].collapse(
            lambda vrms: np.sum(d_rad[core_mask] / vrms[core_mask]), axis="x1")
        ).array().mean()


def plot_prof(mdat: MusicData, var: str) -> None:
    """Plot radial profile of density."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)

    fig = SinglePlotFigure(
        plot=WithScales(
            plot=ProfPlot(
                music_data=mdat,
                var=var,
                markers=[mdat.prof1d.params["rcore"]],
                length_scale=mdat.prof1d.params["rad_surf"],
            ),
            yscale="log",
        ),
    )
    fig.save_to(figdir / f'{var}_prof.pdf')


def plot_dprof(mdat: MusicData, var: str) -> None:
    """Plot radial gradient profile of var."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    rad = mdat.sim.grid.r_grid.cell_centers()
    var_prof = mdat.rprof_avg[var].array()

    grad = (var_prof[1:] - var_prof[:-1]) / (rad[1:] - rad[:-1])
    rad_grad = (rad[1:] + rad[:-1]) / 2

    plt.plot(rad_grad, grad)

    plt.xlabel('radius')
    plt.ylabel(var)
    plt.savefig(figdir / f'{var}_grad_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_tseries(simog: MusicData, var: str) -> None:
    """Plot time series."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)

    fig = SinglePlotFigure(
        plot=TseriesPlot(
            music_data=simog,
            var=var,
        ),
    )
    fig.save_to(figdir / f"tseries_{var}.pdf")


if __name__ == "__main__":
    simfold = Path("transient")
    compute_tconv = False

    simog = MusicData(Path("params.nml"))

    plot_prof(simog, "vel_2")

    plot_tseries(simog, "v2")
    plot_tseries(simog, "vr2")
    plot_tseries(simog, "vt2")
    plot_tseries(simog, "vel_2")

    if compute_tconv:
        tconv = tau_conv(simog)
        print(f'Conv time {simfold.name}: {tconv:.2e}')
