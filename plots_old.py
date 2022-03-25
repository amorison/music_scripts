#!/usr/bin/env python3
"""RMS velocity"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymusic.io.music import MusicSim, PeriodicArrayBC
from pymusic.io.music_new_format import MusicDumpInfo
from pymusic.big_array import BigArray
from pymusic.big_array.dtyped_func import FixedDtypedFunc
from pymusic.math.spherical_quadrature import SphericalMidpointQuad1D
from pymusic.plotting import SinglePlotFigure

from array_on_grid import DumpArrayOnGrid
from derived_fields import FieldGetter, ProfGetter, TimeAveragedProfGetter
from plots import SphericalPlot
from prof1d import Prof1d


def _music_sim(folder) -> MusicSim:
    return MusicSim.from_dump_file_names(
        sorted(folder.glob('*.music')),
        MusicDumpInfo(num_space_dims=2, num_velocities=2, num_scalars=1),
        [PeriodicArrayBC(), PeriodicArrayBC()])


# vrms(r, time) = sqrt(mean_theta(v2))
# also plot radial profile of time average of vrms(r, time)
def tau_conv(folder):
    """Convective time scale."""
    sim = _music_sim(folder)
    sim_data = sim.big_array()
    grid = sim.grid
    d_rad = grid.r_grid.cell_widths()
    params = Prof1d(folder / "..").params
    core_mask = grid.r_grid.cell_centers() < params['rcore']
    return (
        ProfGetter('vel_square')(sim_data)
        .collapse(FixedDtypedFunc(lambda vrms2: np.sum(d_rad[core_mask] / np.sqrt(vrms2[core_mask])),
                                  np.float64), axis="x1")
        ).array().mean()


def get_var(var: str, sim_data: BigArray) -> BigArray:
    """Get a direct output or derived variable."""
    return FieldGetter(var)(sim_data)


def tseries(folder, var):
    """Time series of a var from a given folder."""
    sim = _music_sim(folder)

    grid = sim.grid
    d_rad = grid.r_grid.cell_widths()
    sim_data = sim.big_array()
    full_var = get_var(var, sim_data)
    sph_quad = SphericalMidpointQuad1D(grid.theta_grid)

    rad = grid.r_grid.cell_centers()
    time = np.array(sim_data.labels_along_axis("time"))
    var_series = (
        full_var.collapse(FixedDtypedFunc(sph_quad.average, np.float64),
                          axis="x2")
        .collapse(FixedDtypedFunc(lambda w: np.average(w, weights=d_rad * rad**2), np.float64), axis="x1")
    ).array()
    return time, var_series


def prof(var, folder):
    """Time averaged profile."""
    sim = _music_sim(folder)
    sim_data = sim.big_array()
    var_prof = TimeAveragedProfGetter(var)(sim_data).array()
    return sim.grid.r_grid.cell_centers(), var_prof


def plot_prof(var, *folders):
    """Plot radial profile of density."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        rad, var_prof = prof(var, folder)
        plt.semilogy(rad, var_prof, label=folder.name)
    plt.xlabel('radius')
    plt.ylabel(var)
    plt.legend()
    plt.savefig(figdir / f'{var}_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_dprof(var, *folders):
    """Plot radial gradient profile of var."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        rad, var_prof = prof(var, folder)
        grad = (var_prof[1:] - var_prof[:-1]) / (rad[1:] - rad[:-1])
        rad_grad = (rad[1:] + rad[:-1]) / 2
        plt.plot(rad_grad, grad, label=folder.name)
        plt.legend()
    plt.xlabel('radius')
    plt.ylabel(var)
    plt.savefig(figdir / f'{var}_grad_prof.pdf', bbox_inches='tight')
    plt.close()


def plot_tseries(folder, var):
    """Plot time series."""
    figdir = Path('figures')
    figdir.mkdir(parents=True, exist_ok=True)
    time, data = tseries(folder, var)
    plt.plot(time, data, label=folder.name)
    plt.legend()
    plt.savefig(figdir / f"tseries_{var}.pdf", bbox_inches='tight')
    plt.close()


def plot_var(folder, var, vel_arrows=False):
    """Field plots in a given folder."""
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)
    sim = _music_sim(folder)

    for i, dump in enumerate(sim.dumps):
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
    plot_var(simfold, 'e_int', vel_arrows=True)
    #plot_var(Path(), 'rho', vel_arrows=False)
    plot_tseries(simfold, "v2")
    plot_tseries(simfold, "vr2")
    plot_tseries(simfold, "vt2")
    plot_tseries(simfold, "vel_2")
    plot_var(simfold, "vel_2")
    plot_prof("vel_2", simfold)
    #plot_prof('rho', *sim_folders)
    #plot_prof('scalar_1', *sim_folders)
    #plot_prof('ekin', *sim_folders)
    #plot_prof('v2', *sim_folders)
    #plot_prof('vr2', Path('transient'))
    #plot_var(Path('no_es'), 'vr2')
    #plot_var(Path('no_es'), 'v2')
    #plot_var(Path('transient'), 'vr2_ov2')
    #plot_var(Path('no_es'), 'vt2_ov2')
    #print(*(f'Conv time {fold.name}: {tau_conv(fold):.2e}'
    #        for fold in sim_folders),
    #      sep='\n')
