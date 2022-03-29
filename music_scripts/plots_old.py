#!/usr/bin/env python3
"""RMS velocity"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymusic.io.music import MusicSim, PeriodicArrayBC
from pymusic.io.music_new_format import MusicDumpInfo
from pymusic.plotting import SinglePlotFigure

from .array_on_grid import DumpArrayOnGrid, SimArrayOnGrid
from .derived_fields import (
    FieldGetter, ProfGetter, TimeAveragedProfGetter, TimeSeriesGetter
)
from .plots import SphericalPlot, ProfPlot, TseriesPlot
from .prof1d import Prof1d


def tau_conv(simog, rcore: float):
    """Convective time scale."""
    grid = simog.grid
    d_rad = grid.r_grid.cell_widths()
    core_mask = grid.r_grid.cell_centers() < rcore
    return (
        ProfGetter("vrms")(simog).collapse(
            lambda vrms: np.sum(d_rad[core_mask] / vrms[core_mask]), axis="x1")
        ).array().mean()


def plot_prof(simog: SimArrayOnGrid, var: str, profs1d: Prof1d) -> None:
    """Plot radial profile of density."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)

    fig = SinglePlotFigure(
        plot=ProfPlot(
            music_data=simog,
            get_data=TimeAveragedProfGetter(var),
            markers=[profs1d.params["rcore"]],
            length_scale=profs1d.params["rad_surf"],
            log_scale=True,
        ),
    )
    fig.save_to(figdir / f'{var}_prof.pdf')


def plot_dprof(simog, var):
    """Plot radial gradient profile of var."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    rad = simog.sim.grid.r_grid.cell_centers()
    var_prof = TimeAveragedProfGetter(var)(simog).array()

    grad = (var_prof[1:] - var_prof[:-1]) / (rad[1:] - rad[:-1])
    rad_grad = (rad[1:] + rad[:-1]) / 2

    plt.plot(rad_grad, grad)

    plt.xlabel('radius')
    plt.ylabel(var)
    plt.savefig(figdir / f'{var}_grad_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_tseries(simog, var):
    """Plot time series."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)

    fig = SinglePlotFigure(
        plot=TseriesPlot(
            music_data=simog,
            get_data=TimeSeriesGetter(var),
            log_scale=False,
        ),
    )
    fig.save_to(figdir / f"tseries_{var}.pdf")


def plot_var(simog: SimArrayOnGrid, var, vel_arrows=False):
    """Field plots in a given folder."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    for i, dump in enumerate(simog.sim.dumps):
        fig = SinglePlotFigure(
            plot=SphericalPlot(
                dump_arr=DumpArrayOnGrid(dump),
                get_data=FieldGetter(var),
                with_vel_arrows=vel_arrows,
            ),
        )
        fig.save_to(figdir / f"{var}_{i:05d}.png")


if __name__ == "__main__":
    simfold = Path("transient")
    compute_tconv = False

    sim = MusicSim.from_dump_file_names(
        sorted(simfold.glob('*.music')),
        MusicDumpInfo(num_space_dims=2, num_velocities=2, num_scalars=1),
        [PeriodicArrayBC(), PeriodicArrayBC()])
    simog = SimArrayOnGrid(sim)

    profs1d = Prof1d(simfold / "..")

    plot_var(simog, 'e_int', vel_arrows=True)
    plot_var(simog, "vel_2")

    plot_prof(simog, "vel_2", profs1d)

    plot_tseries(simog, "v2")
    plot_tseries(simog, "vr2")
    plot_tseries(simog, "vt2")
    plot_tseries(simog, "vel_2")

    if compute_tconv:
        tconv = tau_conv(simog, profs1d.params["rcore"])
        print(f'Conv time {simfold.name}: {tconv:.2e}')
