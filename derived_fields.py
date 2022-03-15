"""Helpers to treat output and derived fields in the same way.

Most functions/callables in this module accept a BigArray built from a MusicSim
or a MusicDump.  In particular, they expect this BigArray to contain a "var"
axis with "rho", "vel_N", and "e_int" labels.  These requirements cannot be
expressed through the type system and therefore are the responsibility of the
user.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass

import numpy as np

from pymusic.big_array import DerivedFieldArray

if typing.TYPE_CHECKING:
    from typing import Callable, Dict

    from pymusic.big_array import BigArray
    from pymusic.io import MusicDump


class DataFetcher(ABC):
    @abstractmethod
    def __call__(self, music_data: BigArray) -> BigArray:
        """Get some data from a dump."""


@dataclass(frozen=True)
class FieldGetter(DataFetcher):

    """Get a field from a MUSIC dump."""

    var_name: str
    _handlers: typing.ClassVar[Dict[str, Callable[[BigArray], BigArray]]] = {}

    @classmethod
    def register(
        cls, thunk: Callable[[BigArray], BigArray]
    ) -> Callable[[BigArray], BigArray]:
        cls._handlers[thunk.__name__] = thunk
        return thunk

    def __call__(self, music_data: BigArray) -> BigArray:
        try:
            return self._handlers[self.var_name](music_data)
        except KeyError:
            return music_data.xs(self.var_name, "var")


@FieldGetter.register
def vel_ampl(music_data: BigArray) -> BigArray:
    """Norm of velocity vector."""
    return DerivedFieldArray(
        music_data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2, vel_2**2)
    )
