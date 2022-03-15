import typing
from dataclasses import dataclass

from abc import ABC, abstractmethod

import numpy as np
from pymusic.io import MusicDump

if typing.TYPE_CHECKING:
    from typing import Callable, ClassVar, Dict


class DataFetcher(ABC):
    @abstractmethod
    def __call__(self, dump: MusicDump) -> np.ndarray:
        """Get some data from a dump."""


@dataclass(frozen=True)
class FieldGetter(DataFetcher):

    """Get a field from a MUSIC dump."""

    var_name: str
    _handlers: ClassVar[Dict[str, Callable[[MusicDump], np.ndarray]]] = {}

    @classmethod
    def register(
        cls, thunk: Callable[[MusicDump], np.ndarray]
    ) -> Callable[[MusicDump], np.ndarray]:
        cls._handlers[thunk.__name__] = thunk
        return thunk

    def __call__(self, dump: MusicDump) -> np.ndarray:
        try:
            return self._handlers[self.var_name](dump)
        except KeyError:
            return dump.big_array().xs(self.var_name, "var").array()


@FieldGetter.register
def vel_ampl(dump: MusicDump) -> np.ndarray:
    dump_array = dump.big_array()
    vel_1 = dump_array.xs("vel_1", "var").array()
    vel_2 = dump_array.xs("vel_2", "var").array()
    return np.sqrt(vel_1**2 + vel_2**2)
