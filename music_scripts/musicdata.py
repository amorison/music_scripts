from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import typing

import f90nml
from pymusic.io import (
    MusicSim, MusicDumpInfo, PeriodicArrayBC, MusicDump,
    MusicNewFormatDumpFile, KnownMusicVariables
)

from .array_on_grid import SimArrayOnGrid, DumpArrayOnGrid
from . import eos

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Mapping, Any, Union, Tuple, Sequence, Iterator


@dataclass
class Snap:
    idump: int
    dump_arr: DumpArrayOnGrid


class _SnapsView:
    """Iterator over snapshots."""

    def __init__(self, mdat: MusicData, items: Sequence[Union[int, slice]]):
        self._mdat = mdat
        self._items = items

    def _exists(self, idump: int) -> bool:
        try:
            self._mdat[idump]
        except IndexError:
            return False
        return True

    def __iter__(self) -> Iterator[Snap]:
        for item in self._items:
            if isinstance(item, slice):
                idx = item.indices(len(self._mdat))
                yield from (self._mdat[i] for i in range(*idx)
                            if self._exists(i))
            elif self._exists(item):
                yield self._mdat[item]


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
    def eos(self) -> eos.EoS:
        metallicity = self.params["physics"]["zz"]
        if self.params["scalars"].get("nscalars", 0) > 0:
            # soon to be deprecated logic in MUSIC, this always meant variable
            # He content in scalar_1, enough for now but should be revisited
            return eos.MesaCstMetalEos(metallicity)
        return eos.MesaCstCompoEos(metallicity, self.params["physics"]["yy"])

    @property
    def _out_pattern(self) -> str:
        return self.params["io"]["dataoutput"] + "*.music"

    def _outfile(self, idump: int) -> Path:
        return self.path / (
            self.params["io"]["dataoutput"] + f"{idump:08}.music"
        )

    @property
    def _dump_info(self) -> MusicDumpInfo:
        return MusicDumpInfo(
            num_space_dims=2,
            num_velocities=2,
            num_scalars=self.params["scalars"].get("nscalars", 0)
        )

    def _recenter_bc(self) -> list:
        return [PeriodicArrayBC(), PeriodicArrayBC()]

    @cached_property
    def sim_arr_on_grid(self) -> SimArrayOnGrid:
        sim = MusicSim.from_dump_file_names(
            file_names=sorted(self.path.glob(self._out_pattern)),
            dump_info=self._dump_info,
            recenter_bc_list=self._recenter_bc(),
        )
        return SimArrayOnGrid(sim)

    @cached_property
    def _len(self) -> int:
        last_dump = max(self.path.glob(self._out_pattern))
        ilast = int(last_dump.name[-14:-6])
        return ilast + 1

    def __len__(self) -> int:
        return self._len

    def _normalize_idump(self, idump: int) -> int:
        if idump < 0:
            idump += len(self)
        return idump

    @typing.overload
    def __getitem__(self, idump: int) -> Snap:
        ...

    @typing.overload
    def __getitem__(
        self, idump: Union[slice, Tuple[Union[int, slice]]]
    ) -> _SnapsView:
        ...

    def __getitem__(
        self, idump: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Union[Snap, _SnapsView]:
        if isinstance(idump, slice):
            return _SnapsView(self, (idump,))
        if isinstance(idump, tuple):
            return _SnapsView(self, idump)
        idump = self._normalize_idump(idump)
        if idump < 0 or idump >= len(self):
            raise IndexError(f"out of bounds index: {idump}")
        dump_file = self._outfile(idump)
        if not dump_file.exists():
            raise IndexError(f"{dump_file} doesn't exist")
        dump = MusicDump(
            MusicNewFormatDumpFile(dump_file, self._dump_info),
            self._recenter_bc(),
            KnownMusicVariables(),
        )
        return Snap(idump, DumpArrayOnGrid(dump))