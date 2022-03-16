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


class DataFetcher(ABC):

    def __init_subclass__(cls):
        cls._handlers: Dict[str, Callable[[BigArray], BigArray]] = {}

    @classmethod
    def register(
        cls, thunk: Callable[[BigArray], BigArray]
    ) -> Callable[[BigArray], BigArray]:
        cls._handlers[thunk.__name__] = thunk
        return thunk

    @abstractmethod
    def default_getter(self, music_data: BigArray) -> BigArray:
        """Fallback method to get the desired data."""

    def __call__(self, music_data: BigArray) -> BigArray:
        """Get some data from a dump."""
        try:
            return self._handlers[self.var_name](music_data)
        except KeyError:
            return self.default_getter(music_data)


@dataclass(frozen=True)
class FieldGetter(DataFetcher):

    """Get a field from a MUSIC dump."""

    var_name: str

    def default_getter(self, music_data: BigArray) -> BigArray:
        music_data.xs(self.var_name, "var")


@FieldGetter.register
def vel_ampl(music_data: BigArray) -> BigArray:
    """Norm of velocity vector."""
    return DerivedFieldArray(
        music_data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 + vel_2**2))


@FieldGetter.register
def vel_square(music_data: BigArray) -> BigArray:
    """Square of velocity amplitude."""
    return DerivedFieldArray(
        music_data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: vel_1**2 + vel_2**2)


@FieldGetter.register
def ekin(music_data: BigArray) -> BigArray:
    """Kinetic energy."""
    return DerivedFieldArray(
        music_data, "var", ["rho", "vel_1", "vel_2"],
        lambda rho, vel_1, vel_2: 0.5 * rho * (vel_1**2 + vel_2**2))


@FieldGetter.register
def vr_abs(music_data: BigArray) -> BigArray:
    """Absolute vr."""
    return DerivedFieldArray(music_data, "var", ["vel_1"], np.abs)


@FieldGetter.register
def vt_abs(music_data: BigArray) -> BigArray:
    """Absolute vt."""
    return DerivedFieldArray(music_data, "var", ["vel_2"], np.abs)


@FieldGetter.register
def vr_normalized(music_data: BigArray) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        music_data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 / (vel_1**2 + vel_2**2)))


@FieldGetter.register
def vt_normalized(music_data: BigArray) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        music_data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_2**2 / (vel_1**2 + vel_2**2)))
