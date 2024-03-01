from __future__ import annotations

import typing
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import constants as const
from .figure import SinglePlotFigure
from .fort_pp import Rprof, RprofPlot

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import BinaryIO, Dict, Tuple, Type, Union

    from .config import Config


@dataclass(frozen=True)
class Header:
    fsize: np.int32
    gms: np.float64
    model: np.int32
    dtn: np.float64
    time: np.float64
    n_mesh: np.int32
    n_1: np.int32
    n_species: np.int32

    @staticmethod
    def read_from(file: BinaryIO) -> Header:
        thints = typing.get_type_hints(Header)
        vals = {
            fld.name: np.fromfile(file, dtype=thints[fld.name], count=1)[0]
            for fld in fields(Header)
        }
        return Header(**vals)


@dataclass(frozen=True)
class Lyon1dData:
    header: Header
    yzi: NDArray[np.uint8]
    u: NDArray[np.float64]
    radius: NDArray[np.float64]
    rho: NDArray[np.float64]
    temperature: NDArray[np.float64]
    luminosity: NDArray[np.float64]
    v_u: NDArray[np.float64]
    v_r: NDArray[np.float64]
    v_rho: NDArray[np.float64]
    v_t: NDArray[np.float64]
    v_sl: NDArray[np.float64]
    pressure: NDArray[np.float64]
    mass: NDArray[np.float64]
    xmr: NDArray[np.float64]
    d_m: NDArray[np.float64]
    eint: NDArray[np.float64]
    v_enuc: NDArray[np.float64]
    v_eg: NDArray[np.float64]
    entropy: NDArray[np.float64]
    chem: NDArray[np.float64]
    nabla_adiab: NDArray[np.float64]
    nabla: NDArray[np.float64]
    c_sound: NDArray[np.float64]
    brunt_vaisala: NDArray[np.float64]

    @staticmethod
    def from_file(filepath: Union[str, PathLike]) -> Lyon1dData:
        with Path(filepath).open("rb") as fid:
            hdr = Header.read_from(fid)
            flds = list(fields(Lyon1dData))[1:]
            type_count: Dict[str, Tuple[Type, Union[int, np.integer]]] = {
                fld.name: (np.float64, 1) for fld in flds
            }
            type_count["yzi"] = (np.uint8, 1)
            type_count["chem"] = (np.float64, hdr.n_species)
            vals = {
                name: np.zeros((hdr.n_mesh, count), dtype=dtype).squeeze()
                for name, (dtype, count) in type_count.items()
            }
            for irow in range(hdr.n_mesh):
                for fld in flds:
                    dtype, count = type_count[fld.name]
                    vals[fld.name][irow] = np.fromfile(fid, dtype, count)
        return Lyon1dData(header=hdr, **vals)

    @property
    def he3(self) -> NDArray[np.float64]:
        return self.chem[:, 3]

    @property
    def he4(self) -> NDArray[np.float64]:
        return self.chem[:, 4]

    def get_rprof(self, name: str) -> Rprof:
        return Rprof(
            name=name,
            degree=1,
            values=getattr(self, name),
            radius=self.radius,
        )


@dataclass(frozen=True)
class Lyon1dStruc:
    file_path: Path
    msun: float

    @cached_property
    def _data(self) -> NDArray[np.floating]:
        return np.loadtxt(self.file_path)

    @property
    def mass_adim(self) -> NDArray[np.floating]:
        return self._data[:, 1]

    @property
    def radius(self) -> NDArray[np.floating]:
        return self._data[:, 2]

    @property
    def temperature(self) -> NDArray[np.floating]:
        return self._data[:, 3]

    @property
    def density(self) -> NDArray[np.floating]:
        return self._data[:, 4]

    @property
    def pressure(self) -> NDArray[np.floating]:
        return self._data[:, 5]

    @property
    def opacity(self) -> NDArray[np.floating]:
        return self._data[:, 7]

    @property
    def luminosity(self) -> NDArray[np.floating]:
        return self._data[:, 8]

    @property
    def height_press(self) -> NDArray[np.floating]:
        return self._data[:, 13]

    @property
    def nabla_adiab(self) -> NDArray[np.floating]:
        """d ln(temp) / d ln(press) at constant entropy"""
        return self._data[:, 19]

    @property
    def heat_capacity(self) -> NDArray[np.floating]:
        return self._data[:, 23]

    @property
    def nabla(self) -> NDArray[np.floating]:
        """d ln(temp) / d ln(press)"""
        return self._data[:, 24]

    @property
    def delta(self) -> NDArray[np.floating]:
        """-d ln(density) / d ln(temp) at constant pressure, mu"""
        return self._data[:, 25]

    @property
    def bv_ang_freq(self) -> NDArray[np.floating]:
        return np.sqrt(np.maximum(self._data[:, 32], 0.0))

    @cached_property
    def mass(self) -> NDArray[np.floating]:
        return self.mass_adim * self.msun * const.SUN_MASS

    @cached_property
    def gravity(self) -> NDArray[np.floating]:
        return const.GRAVITATIONAL_CONSTANT * self.mass / self.radius**2

    @cached_property
    def heat_conductivity(self) -> NDArray[np.floating]:
        return (
            16
            * const.STEFAN_BOLTZMANN
            * self.temperature**3
            / (3 * self.opacity * self.density)
        )

    @cached_property
    def heat_diffusivity(self) -> NDArray[np.floating]:
        return self.heat_conductivity / (self.density * self.heat_capacity)

    @cached_property
    def bv_ang_freq_thermal(self) -> NDArray[np.floating]:
        bvt2 = (
            self.gravity
            * self.delta
            * (self.nabla_adiab - self.nabla)
            / self.height_press
        )
        return np.sqrt(np.maximum(bvt2, 0.0))

    @property
    def r_star(self) -> float:
        return self.radius[-1]

    @property
    def luminosity_tot(self) -> float:
        return self.luminosity[-1]


def cmd(conf: Config) -> None:
    mesadata = Lyon1dData.from_file(conf.lyon1d.mfile)
    for var in conf.lyon1d.plot:
        SinglePlotFigure(
            plot=RprofPlot(
                rprof=mesadata.get_rprof(var),
                marks=conf.plotting.rmarks,
                scale="log" if conf.plotting.log else "linear",
            ),
        ).save_to(f"rprof_mesa_{var}.pdf")
