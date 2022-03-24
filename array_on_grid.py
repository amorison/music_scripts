from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

from pymusic.big_array import BigArray
from pymusic.grid import Grid
from pymusic.io import MusicDump, MusicSim


class ArrayOnGrid(ABC):

    """Array with associated grid information.

    This typically represent MUSIC simulation data and the associated grid.
    """

    @property
    @abstractmethod
    def grid(self) -> Grid:
        """Grid object relevant for the wrapped data."""

    @property
    @abstractmethod
    def data(self) -> BigArray:
        """The data itself."""


class DumpArrayOnGrid(ArrayOnGrid):

    def __init__(self, dump: MusicDump):
        self.dump = dump

    @property
    def grid(self) -> Grid:
        return self.dump.grid

    @cached_property
    def data(self) -> BigArray:
        return self.dump.big_array()


class SimArrayOnGrid(ArrayOnGrid):

    def __init__(self, sim: MusicSim):
        self.sim = sim

    @property
    def grid(self) -> Grid:
        return self.grid

    @cached_property
    def data(self) -> BigArray:
        return self.sim.big_array()
