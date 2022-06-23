from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import typing

import numpy as np
from pymusic.plotting import SinglePlotFigure

from .plots import RawSphericalScalarPlot

if typing.TYPE_CHECKING:
    from typing import Union, BinaryIO
    from os import PathLike
    from .config import Config


# factorize with lyon1d
@dataclass(frozen=True)
class Header:
    nrtot: np.int32
    nttot: np.int32
    nptot: np.int32
    time: np.float64

    @staticmethod
    def read_from(file: BinaryIO) -> Header:
        thints = typing.get_type_hints(Header)
        vals = {
            fld.name: np.fromfile(file, dtype=thints[fld.name], count=1)[0]
            for fld in fields(Header)
        }
        return Header(**vals)


@dataclass(frozen=True)
class BinData:
    header: Header
    rad_i: np.ndarray
    theta_i: np.ndarray
    u_r: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    temp_prof: np.ndarray
    temp_pert: np.ndarray

    @staticmethod
    def from_file(filepath: Union[str, PathLike]) -> BinData:
        with Path(filepath).open("rb") as fid:
            hdr = Header.read_from(fid)
            data = np.zeros((8, hdr.nrtot, hdr.nttot))
            for i in range(hdr.nrtot):
                for j in range(hdr.nttot):
                    data[:, i, j] = np.fromfile(
                        fid, dtype=np.float64, count=8)
        return BinData(
            header=hdr,
            rad_i=data[0, :, 0],
            theta_i=data[1, 0, :],
            u_r=data[2],
            rho=data[3],
            pressure=data[4],
            temperature=data[5],
            temp_prof=data[6],
            temp_pert=data[7],
        )


def cmd(conf: Config) -> None:
    data = BinData.from_file(conf.lscale.tfile)
    SinglePlotFigure(
        plot=RawSphericalScalarPlot(
            r_coord=data.rad_i,
            t_coord=data.theta_i,
            data=data.temp_pert[:-1, :-1] / data.temp_prof[:-1, :-1],
            cmap="RdBu_r",
        ),
    ).save_to("lscale_temp_pert.pdf")
