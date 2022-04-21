from __future__ import annotations
from dataclasses import dataclass
import typing

import h5py

if typing.TYPE_CHECKING:
    from typing import Union
    from os import PathLike
    import numpy as np


@dataclass(frozen=True)
class Contour:
    name: str
    values: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True)
class FortPpCheckpoint:
    master_h5: Union[str, PathLike]
    idump: int

    def contour_field(self, name: str) -> Contour:
        with h5py.File(self.master_h5) as h5f:
            chkp = h5f["checkpoints"][f"{self.idump:05d}"]
            contour = Contour(
                name=name,
                values=chkp["Contour_field"][name][()].squeeze(),
                theta=chkp["pp_parameters"]["eval_grid"]["theta"][()].squeeze()
            )
        return contour
