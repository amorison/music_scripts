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
from pymusic.math.spherical_quadrature import SphericalMidpointQuad1D

from pymusic.big_array import BigArray
from .array_on_grid import ArrayOnGrid

if TYPE_CHECKING:
    from typing import Callable, Dict


T_contra = TypeVar("T_contra", contravariant=True)
U_co = TypeVar("U_co", covariant=True)


# That dataclass is needed to circumvent a bug in mypy with
# generic abstract dataclasses
@dataclass(frozen=True)
class _DCWithVarName:
    var_name: str


class DataFetcher(ABC, Generic[T_contra, U_co], _DCWithVarName):

    def __init_subclass__(cls) -> None:
        # TYPE SAFETY: mypy doesn't seem to understand __init_subclass__
        cls._handlers: Dict[str, Callable[[T_contra], U_co]] = {}  # type: ignore

    @classmethod
    def register(
        cls, thunk: Callable[[T_contra], U_co]
    ) -> Callable[[T_contra], U_co]:
        cls._handlers[thunk.__name__] = thunk  # type: ignore
        return thunk

    @abstractmethod
    def default_getter(self, obj: T_contra) -> U_co:
        """Fallback method to get the desired data."""

    def __call__(self, obj: T_contra) -> U_co:
        """Get some data from a dump."""
        try:
            return self._handlers[self.var_name](obj)  # type: ignore
        except KeyError:
            return self.default_getter(obj)


@dataclass(frozen=True)
class FieldGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a field from MUSIC data."""

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        return aog.big_array.xs(self.var_name, "var")


@dataclass(frozen=True)
class ProfGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a radial profile from MUSIC data."""

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        field = FieldGetter(self.var_name)(aog)
        sph_quad = SphericalMidpointQuad1D(aog.grid.theta_grid)
        return field.collapse(sph_quad.average, axis="x2")


@dataclass(frozen=True)
class TimeAveragedProfGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a radial profile from MUSIC data."""

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        field = ProfGetter(self.var_name)(aog)
        if "time" in aog.big_array.axes:
            field = field.mean("time")
        return field


@dataclass(frozen=True)
class TimeSeriesGetter(DataFetcher[ArrayOnGrid, BigArray]):

    """Get a time series from MUSIC data."""

    def default_getter(self, aog: ArrayOnGrid) -> BigArray:
        r_grid = aog.grid.r_grid
        rad = r_grid.cell_centers()
        d_rad = r_grid.cell_widths()
        return ProfGetter(self.var_name)(aog).collapse(
            lambda w: np.average(w, weights=d_rad * rad**2), axis="x1")


@FieldGetter.register
def vel_ampl(aog: ArrayOnGrid) -> BigArray:
    """Norm of velocity vector."""
    return DerivedFieldArray(
        aog.big_array, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 + vel_2**2))


@FieldGetter.register
def vel_square(aog: ArrayOnGrid) -> BigArray:
    """Square of velocity amplitude."""
    return DerivedFieldArray(
        aog.big_array, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: vel_1**2 + vel_2**2)


@FieldGetter.register
def ekin(aog: ArrayOnGrid) -> BigArray:
    """Kinetic energy."""
    return DerivedFieldArray(
        aog.big_array, "var", ["rho", "vel_1", "vel_2"],
        lambda rho, vel_1, vel_2: 0.5 * rho * (vel_1**2 + vel_2**2))


@FieldGetter.register
def vr_abs(aog: ArrayOnGrid) -> BigArray:
    """Absolute vr."""
    return DerivedFieldArray(aog.big_array, "var", ["vel_1"], np.abs)


@FieldGetter.register
def vt_abs(aog: ArrayOnGrid) -> BigArray:
    """Absolute vt."""
    return DerivedFieldArray(aog.big_array, "var", ["vel_2"], np.abs)


@FieldGetter.register
def vr_normalized(aog: ArrayOnGrid) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        aog.big_array, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 / (vel_1**2 + vel_2**2)))


@FieldGetter.register
def vt_normalized(aog: ArrayOnGrid) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        aog.big_array, "var", ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_2**2 / (vel_1**2 + vel_2**2)))


@ProfGetter.register
def vrms(aog: ArrayOnGrid) -> BigArray:
    """Vrms defined as vrms(r, t) = sqrt(mean_theta(v2))."""
    return ProfGetter("vel_square")(aog).sqrt()
