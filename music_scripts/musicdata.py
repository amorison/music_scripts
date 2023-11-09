from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import f90nml
from music_pykg.format2 import MusicNewFormatDumpFile
from music_pykg.known_variables import KnownMusicVariables
from music_pykg.prof1d import Prof1d
from pymusic.big_array import BigArray, CachedArray
from pymusic.io import (
    MusicDump,
    MusicDumpArray,
    MusicSim,
    PeriodicArrayBC,
    ReflectiveArrayBC,
)

from . import eos
from .derived_fields import (
    BaseMusicData,
    TimeAveragedProfGetter,
    TimeSeriesGetter,
    _DataGetter,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Any, Iterator, Mapping, Sequence, Tuple, Union

    from pymusic.grid import Grid
    from pymusic.io import ArrayBC


@dataclass(frozen=True)
class Snap(BaseMusicData):
    mdat: MusicData
    idump: int
    dump: MusicDump

    @property
    def grid(self) -> Grid:
        return self.dump.grid

    @property
    def cartesian(self) -> bool:
        return self.mdat.cartesian

    @cached_property
    def big_array(self) -> BigArray:
        return CachedArray(MusicDumpArray(self.dump, False))

    @property
    def eos(self) -> eos.EoS:
        return self.mdat.eos


class _SnapsView(BaseMusicData):
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

    def _idumps(self) -> Iterator[int]:
        for item in self._items:
            if isinstance(item, slice):
                idx = item.indices(len(self._mdat))
                yield from range(*idx)
            else:
                yield self._mdat._normalize_idump(item)

    @cached_property
    def sim(self) -> MusicSim:
        return MusicSim.from_dump_file_names(
            file_names=sorted(
                file
                for idump in self._idumps()
                if (file := self._mdat._outfile(idump)).exists()
            ),
            recenter_bc_list=self._mdat._recenter_bc(),
        )

    def __iter__(self) -> Iterator[Snap]:
        return (self._mdat[idump] for idump in self._idumps() if self._exists(idump))

    @property
    def grid(self) -> Grid:
        return self.sim.grid

    @cached_property
    def big_array(self) -> BigArray:
        return self.sim.big_array()

    @cached_property
    def eos(self) -> eos.EoS:
        return self._mdat.eos

    @cached_property
    def cartesian(self) -> bool:
        return self._mdat.params["geometry"]["cartesian"]

    @cached_property
    def prof1d(self) -> Prof1d:
        return self._mdat.prof1d


class MusicData(BaseMusicData):
    """Data accessor of a MUSIC run."""

    def __init__(self, parfile: Union[str, PathLike]):
        self.parfile = Path(parfile).resolve()

    @property
    def path(self) -> Path:
        return self.parfile.parent

    @cached_property
    def params(self) -> Mapping[str, Any]:
        """Run parameters from Fortran namelist."""
        return f90nml.read(self.parfile)

    @cached_property
    def eos(self) -> eos.EoS:
        eos_name = self.params["microphysics"].get("eos", "mesa")
        if eos_name == "mesa":
            abd = self.params["abundances"]
            metallicity = abd.get("metals_mass_fraction", 0.02)
            he_scalar = abd.get("helium_scalar", 0)
            if he_scalar > 0:
                return eos.MesaCstMetalEos(metallicity, he_scalar)
            return eos.MesaCstCompoEos(
                metallicity, abd.get("helium_mass_fraction", 0.28)
            )
        elif eos_name == "ideal_gas_mix2":
            eos_nml = self.params["eos_ideal_mix2"]
            return eos.IdealGasMix2(
                gamma1=eos_nml["gamma1"],
                gamma2=eos_nml["gamma2"],
                mu1=eos_nml["mu1"],
                mu2=eos_nml["mu2"],
                c1_scalar=eos_nml["mass_frac_1_scalar"],
            )
        else:
            raise NotImplementedError(f"EoS: {eos_name}")

    @property
    def _out_pattern(self) -> str:
        return self.params["io"]["dataoutput"] + "*.music"

    def _outfile(self, idump: int) -> Path:
        return self.path / (self.params["io"]["dataoutput"] + f"{idump:08}.music")

    def _recenter_bc(self) -> Sequence[ArrayBC]:
        # very crude way to handle boundary conditions for now
        bcr = self.params["boundaryconditions"]["bc1"][0]
        bct = self.params["boundaryconditions"]["bc3"][0]
        return [
            PeriodicArrayBC() if bcr == "periodic" else ReflectiveArrayBC(),
            PeriodicArrayBC() if bct == "periodic" else ReflectiveArrayBC(),
        ]

    @cached_property
    def sim(self) -> MusicSim:
        return MusicSim.from_dump_file_names(
            file_names=sorted(self.path.glob(self._out_pattern)),
            recenter_bc_list=self._recenter_bc(),
        )

    @property
    def grid(self) -> Grid:
        return self.sim.grid

    @property
    def cartesian(self) -> bool:
        return self.params["geometry"]["cartesian"]

    @cached_property
    def big_array(self) -> BigArray:
        return self.sim.big_array()

    @cached_property
    def prof1d(self) -> Prof1d:
        return Prof1d.with_path_hint(self.path)

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
        self, idump: Union[slice, Tuple[Union[int, slice], ...]]
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
        dump = MusicDump.from_file(
            MusicNewFormatDumpFile(dump_file),
            self._recenter_bc(),
            KnownMusicVariables(),
        )
        return Snap(self, idump, dump)

    def __iter__(self) -> Iterator[Snap]:
        return iter(self[:])

    @property
    def rprof_avg(self) -> _DataGetter:
        return _DataGetter(self, TimeAveragedProfGetter)

    @property
    def tseries(self) -> _DataGetter:
        return _DataGetter(self, TimeSeriesGetter)
