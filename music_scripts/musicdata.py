from __future__ import annotations

from functools import cached_property
import typing

import f90nml
from pymusic.io import MusicSim, MusicDumpInfo, PeriodicArrayBC

from .array_on_grid import SimArrayOnGrid

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Mapping, Any


class MusicData:
    """Data accessor of a MUSIC run."""

    def __init__(self, parfile: Path):
        self.parfile = parfile.resolve()

    @property
    def path(self) -> Path:
        return self.parfile.parent

    @cached_property
    def params(self) -> Mapping[str, Any]:
        """Run parameters from Fortran namelist."""
        return f90nml.read(self.parfile)

    @cached_property
    def sim_arr_on_grid(self) -> SimArrayOnGrid:
        outfiles_ptn = self.params["io"]["dataoutput"] + "*.music"
        sim = MusicSim.from_dump_file_names(
            file_names=sorted(self.path.glob(outfiles_ptn)),
            dump_info=MusicDumpInfo(
                num_space_dims=2,
                num_velocities=2,
                num_scalars=self.params["scalars"]["nscalars"]
            ),
            recenter_bc_list=[PeriodicArrayBC(), PeriodicArrayBC()],
        )
        return SimArrayOnGrid(sim)
