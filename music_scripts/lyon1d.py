from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import typing

from pymusic.plotting import SinglePlotFigure
import numpy as np
from numpy.typing import NDArray

from .fort_pp import Rprof, RprofPlot

if typing.TYPE_CHECKING:
    from typing import BinaryIO, Union
    from os import PathLike
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
class Mesa1dData:
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
    def from_file(filepath: Union[str, PathLike]) -> Mesa1dData:
        with Path(filepath).open("rb") as fid:
            hdr = Header.read_from(fid)
            flds = list(fields(Mesa1dData))[1:]
            type_count = {fld.name: (np.float64, 1) for fld in flds}
            type_count["yzi"] = (np.uint8, 1)
            type_count["chem"] = (np.float64, hdr.n_species)
            vals = {name: np.zeros((hdr.n_mesh, count), dtype=dtype).squeeze()
                    for name, (dtype, count) in type_count.items()}
            for irow in range(hdr.n_mesh):
                for fld in flds:
                    dtype, count = type_count[fld.name]
                    vals[fld.name][irow] = np.fromfile(fid, dtype, count)
        return Mesa1dData(header=hdr, **vals)

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


def cmd(conf: Config) -> None:
    mesadata = Mesa1dData.from_file(conf.mesa1d.mfile)
    for var in conf.mesa1d.plot:
        SinglePlotFigure(
            plot=RprofPlot(
                rprof=mesadata.get_rprof(var),
                marks=conf.plotting.rmarks,
                scale="log" if conf.plotting.log else "linear",
            ),
        ).save_to(f"rprof_mesa_{var}.pdf")
