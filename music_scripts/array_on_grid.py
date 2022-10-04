from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from pymusic.big_array import BigArray
from pymusic.grid import Grid


class ArrayOnGrid(Protocol):

    """Array with associated grid information.

    This typically represent MUSIC simulation data and the associated grid.
    """

    @property
    @abstractmethod
    def grid(self) -> Grid:
        """Grid object relevant for the wrapped data."""

    @property
    @abstractmethod
    def big_array(self) -> BigArray:
        """The data itself."""
