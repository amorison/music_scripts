from __future__ import annotations

import typing

from pymusic.io import MusicSim, MusicDumpInfo, PeriodicArrayBC

from .array_on_grid import SimArrayOnGrid
from .plots_old import plot_var

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    sim = MusicSim.from_dump_file_names(
        sorted(conf.core.path.glob('*.music')),
        MusicDumpInfo(num_space_dims=2, num_velocities=2, num_scalars=1),
        [PeriodicArrayBC(), PeriodicArrayBC()])
    simog = SimArrayOnGrid(sim)
    plot_var(simog, conf.field.plot, conf.field.velarrow)
