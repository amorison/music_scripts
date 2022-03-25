#!/usr/bin/env python3
"""RMS velocity"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymusic.io.music import MusicSim, PeriodicArrayBC
from pymusic.io.music_new_format import MusicDumpInfo
from pymusic.big_array.dtyped_func import FixedDtypedFunc
from pymusic.plotting import SinglePlotFigure

from array_on_grid import DumpArrayOnGrid, SimArrayOnGrid
from derived_fields import FieldGetter, ProfGetter, TimeAveragedProfGetter
from plots import SphericalPlot
from prof1d import Prof1d


def _music_sim(folder) -> SimArrayOnGrid:
    sim = MusicSim.from_dump_file_names(
        sorted(folder.glob('*.music')),
        MusicDumpInfo(num_space_dims=2, num_velocities=2, num_scalars=1),
        [PeriodicArrayBC(), PeriodicArrayBC()])
    return SimArrayOnGrid(sim)


def tau_conv(simog, rcore: float):
    """Convective time scale."""
    grid = simog.grid
    d_rad = grid.r_grid.cell_widths()
    core_mask = grid.r_grid.cell_centers() < rcore
    return (
        ProfGetter("vrms")(simog).collapse(
            FixedDtypedFunc(
                lambda vrms: np.sum(d_rad[core_mask] / vrms[core_mask]),
                np.float64), axis="x1")
        ).array().mean()


def tseries(simog, var):
    """Time series of a var from a given folder."""
    grid = simog.sim.grid
    d_rad = grid.r_grid.cell_widths()
    sim_data = simog.data
    rad = grid.r_grid.cell_centers()
    time = np.array(sim_data.labels_along_axis("time"))
    var_series = (
        ProfGetter(var)(simog)
        .collapse(FixedDtypedFunc(lambda w: np.average(w, weights=d_rad * rad**2), np.float64), axis="x1")
    ).array()
    return time, var_series


def prof(simog, var):
    """Time averaged profile."""
    var_prof = TimeAveragedProfGetter(var)(simog).array()
    return simog.sim.grid.r_grid.cell_centers(), var_prof


def plot_prof(simog, var):
    """Plot radial profile of density."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)

    rad, var_prof = prof(simog, var)
    plt.semilogy(rad, var_prof)
    plt.xlabel('radius')
    plt.ylabel(var)
    plt.legend()
    plt.savefig(figdir / f'{var}_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_dprof(simog, var):
    """Plot radial gradient profile of var."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    rad, var_prof = prof(simog, var)
    grad = (var_prof[1:] - var_prof[:-1]) / (rad[1:] - rad[:-1])
    rad_grad = (rad[1:] + rad[:-1]) / 2
    plt.plot(rad_grad, grad)
    plt.legend()

    plt.xlabel('radius')
    plt.ylabel(var)
    plt.savefig(figdir / f'{var}_grad_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_tseries(simog, var):
    """Plot time series."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)
    time, data = tseries(simog, var)
    plt.plot(time, data)
    plt.legend()
    plt.savefig(figdir / f"tseries_{var}.pdf", bbox_inches='tight')
    plt.close()


def plot_var(simog: SimArrayOnGrid, var, vel_arrows=False):
    """Field plots in a given folder."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    for i, dump in enumerate(simog.sim.dumps):
        fig = SinglePlotFigure(
            plot=SphericalPlot(
                dump=DumpArrayOnGrid(dump),
                get_data=FieldGetter(var),
                with_vel_arrows=vel_arrows,
            ),
        )
        fig.save_to(figdir / f"{var}_{i:05d}.png")


if __name__ == "__main__":
    simfold = Path("transient")
    simog = _music_sim(simfold)
    compute_tconv = False

    plot_var(simog, 'e_int', vel_arrows=True)
    plot_var(simog, "vel_2")

    plot_prof(simog, "vel_2")

    plot_tseries(simog, "v2")
    plot_tseries(simog, "vr2")
    plot_tseries(simog, "vt2")
    plot_tseries(simog, "vel_2")

    if compute_tconv:
        profs1d = Prof1d(simfold / "..")
        tconv = tau_conv(simog, profs1d.params["rcore"])
        print(f'Conv time {simfold.name}: {tconv:.2e}')
