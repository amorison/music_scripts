"""Helpers to treat output and derived fields in the same way.

Most functions/callables in this module accept a ArrayOnGrid built from a MusicSim
or a MusicDump.  In particular, they expect this ArrayOnGrid to contain a "var"
axis with "rho", "vel_N", and "e_int" labels.  These requirements cannot be
expressed through the type system and therefore are the responsibility of the
user.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from pymusic.big_array import DerivedFieldArray
from pymusic.big_array.dtyped_func import FixedDtypedFunc
from pymusic.math.spherical_quadrature import SphericalMidpointQuad1D

if TYPE_CHECKING:
    from typing import Callable, Dict

    from pymusic.big_array import BigArray
    from array_on_grid import ArrayOnGrid


In = TypeVar("In")
Out = TypeVar("Out")


class DataFetcher(ABC, Generic[In, Out]):

    def __init_subclass__(cls):
        cls._handlers: Dict[str, Callable[[In], Out]] = {}

    @classmethod
    def register(cls, thunk: Callable[[In], Out]) -> Callable[[In], Out]:
        cls._handlers[thunk.__name__] = thunk
        return thunk

    @property
    @abstractmethod
    def var_name(self) -> str:
        """The variable name an instance is responsible of fetching."""

    @abstractmethod
    def default_getter(self, obj: In) -> Out:
        """Fallback method to get the desired data."""

    def __call__(self, obj: In) -> Out:
        """Get some data from a dump."""
        try:
            return self._handlers[self.var_name](obj)
        except KeyError:
            return self.default_getter(obj)


@dataclass(frozen=True)
class FieldGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a field from MUSIC data."""

    var_name: str

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        return aog.data.xs(self.var_name, "var")


@dataclass(frozen=True)
class ProfGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a radial profile from MUSIC data."""

    var_name: str

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        field = FieldGetter(self.var_name)(aog)
        sph_quad = SphericalMidpointQuad1D(aog.grid.theta_grid)
        return field.collapse(
            FixedDtypedFunc(sph_quad.average, np.float64),
            axis="x2")


@dataclass(frozen=True)
class TimeAveragedProfGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a radial profile from MUSIC data."""

    var_name: str

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        field = ProfGetter(self.var_name)(aog)
        if "time" in aog.data.axes:
            field = field.mean("time")
        return field


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


@ProfGetter.register
def vrms(aog: ArrayOnGrid) -> BigArray:
    """Vrms defined as vrms(r, t) = sqrt(mean_theta(v2))."""
    return ProfGetter("vel_square")(aog).sqrt()
