"""Helpers to treat output and derived fields in the same way.

Most functions/callables in this module accept a ArrayOnGrid built from a MusicSim
or a MusicDump.  In particular, they expect this ArrayOnGrid to contain a "var"
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
from pymusic.big_array.dtyped_func import FixedDtypedFunc
from pymusic.math.spherical_quadrature import SphericalMidpointQuad1D

if typing.TYPE_CHECKING:
    from typing import Callable, Dict

    from pymusic.big_array import BigArray
    from array_on_grid import ArrayOnGrid


class DataFetcher(ABC):

    def __init_subclass__(cls):
        cls._handlers: Dict[str, Callable[[ArrayOnGrid], BigArray]] = {}

    @classmethod
    def register(
        cls, thunk: Callable[[ArrayOnGrid], BigArray]
    ) -> Callable[[ArrayOnGrid], BigArray]:
        cls._handlers[thunk.__name__] = thunk
        return thunk

    @abstractmethod
    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        """Fallback method to get the desired data."""

    def __call__(self, aog: ArrayOnGrid) -> BigArray:
        """Get some data from a dump."""
        try:
            return self._handlers[self.var_name](aog)
        except KeyError:
            return self.default_getter(aog)


@dataclass(frozen=True)
class FieldGetter(DataFetcher):

    """Get a field from a MUSIC dump."""

    var_name: str

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        return aog.data.xs(self.var_name, "var")


@dataclass(frozen=True)
class ProfGetter(DataFetcher):

    """Get a radial profile from a MUSIC dump."""

    var_name: str

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        field = FieldGetter(self.var_name)(aog)
        sph_quad = SphericalMidpointQuad1D(aog.grid.theta_grid)
        return field.collapse(
            FixedDtypedFunc(sph_quad.average, np.float64),
            axis="x2")


@FieldGetter.register
def vel_ampl(aog: ArrayOnGrid) -> BigArray:
    """Norm of velocity vector."""
    return DerivedFieldArray(
        aog.data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 + vel_2**2))


@FieldGetter.register
def vel_square(aog: ArrayOnGrid) -> BigArray:
    """Square of velocity amplitude."""
    return DerivedFieldArray(
        aog.data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: vel_1**2 + vel_2**2)


@FieldGetter.register
def ekin(aog: ArrayOnGrid) -> BigArray:
    """Kinetic energy."""
    return DerivedFieldArray(
        aog.data, "var", ["rho", "vel_1", "vel_2"],
        lambda rho, vel_1, vel_2: 0.5 * rho * (vel_1**2 + vel_2**2))


@FieldGetter.register
def vr_abs(aog: ArrayOnGrid) -> BigArray:
    """Absolute vr."""
    return DerivedFieldArray(aog.data, "var", ["vel_1"], np.abs)


@FieldGetter.register
def vt_abs(aog: ArrayOnGrid) -> BigArray:
    """Absolute vt."""
    return DerivedFieldArray(aog.data, "var", ["vel_2"], np.abs)


@FieldGetter.register
def vr_normalized(aog: ArrayOnGrid) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        aog.data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 / (vel_1**2 + vel_2**2)))


@FieldGetter.register
def vt_normalized(aog: ArrayOnGrid) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        aog.data, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_2**2 / (vel_1**2 + vel_2**2)))
