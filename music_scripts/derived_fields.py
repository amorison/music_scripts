"""Helpers to treat output and derived fields in the same way.

Most functions/callables in this module accept a BaseMusicData, the main
implementations of which being Snap and MusicData.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from pymusic.big_array import BigArray, DerivedFieldArray
from pymusic.math.spherical_quadrature import SphericalMidpointQuad1D

if TYPE_CHECKING:
    from typing import Callable, Dict, Type

    from pymusic.grid import Grid

    from .eos import EoS


class BaseMusicData(ABC):

    """MUSIC simulation data wrapper, either a full sim or a single dump."""

    @property
    @abstractmethod
    def grid(self) -> Grid:
        """Grid object relevant for the wrapped data."""

    @property
    @abstractmethod
    def big_array(self) -> BigArray:
        """The data itself."""

    @property
    @abstractmethod
    def eos(self) -> EoS:
        """The relevant EoS."""

    @property
    @abstractmethod
    def cartesian(self) -> bool:
        """Whether the geometry is cartesian."""

    @property
    def field(self) -> _DataGetter:
        return _DataGetter(self, FieldGetter)

    @property
    def rprof(self) -> _DataGetter:
        return _DataGetter(self, ProfGetter)


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
    def register(cls, thunk: Callable[[T_contra], U_co]) -> Callable[[T_contra], U_co]:
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
class FieldGetter(DataFetcher[BaseMusicData, BigArray]):

    """Get a field from MUSIC data."""

    def default_getter(self, bmdat: BaseMusicData) -> BigArray:
        return bmdat.big_array.xs(self.var_name, "var")


@dataclass(frozen=True)
class ProfGetter(DataFetcher[BaseMusicData, BigArray]):

    """Get a radial profile from MUSIC data."""

    def default_getter(self, bmdat: BaseMusicData) -> BigArray:
        field = bmdat.field[self.var_name]
        if bmdat.cartesian:
            prof = field.mean("x2")
        else:
            sph_quad = SphericalMidpointQuad1D(bmdat.grid.grids[1])
            prof = field.collapse(sph_quad.average, axis="x2")
        if "time" in field.axes:
            return prof.slabbed("time", 10)
        return prof


@dataclass(frozen=True)
class TimeAveragedProfGetter(DataFetcher[BaseMusicData, BigArray]):

    """Get a radial profile from MUSIC data."""

    def default_getter(self, bmdat: BaseMusicData) -> BigArray:
        field = bmdat.rprof[self.var_name]
        if "time" in bmdat.big_array.axes:
            field = field.mean("time")
        return field


@dataclass(frozen=True)
class TimeSeriesGetter(DataFetcher[BaseMusicData, BigArray]):

    """Get a time series from MUSIC data."""

    def default_getter(self, bmdat: BaseMusicData) -> BigArray:
        prof = bmdat.rprof[self.var_name]
        if bmdat.cartesian:
            return prof.mean("x1")
        r_grid = bmdat.grid.grids[0]
        rad = r_grid.cell_centers()
        d_rad = r_grid.cell_widths()
        return prof.collapse(
            lambda w: np.average(w, weights=d_rad * rad**2), axis="x1"
        ).slabbed("time", 10)


@FieldGetter.register
def vel_ampl(bmdat: BaseMusicData) -> BigArray:
    """Norm of velocity vector."""
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 + vel_2**2),
    )


@FieldGetter.register
def vel_square(bmdat: BaseMusicData) -> BigArray:
    """Square of velocity amplitude."""
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["vel_1", "vel_2"],
        lambda vel_1, vel_2: vel_1**2 + vel_2**2,
    )


@FieldGetter.register
def ekin(bmdat: BaseMusicData) -> BigArray:
    """Kinetic energy."""
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["density", "vel_1", "vel_2"],
        lambda rho, vel_1, vel_2: 0.5 * rho * (vel_1**2 + vel_2**2),
    )


@FieldGetter.register
def vr_abs(bmdat: BaseMusicData) -> BigArray:
    """Absolute vr."""
    return DerivedFieldArray(bmdat.big_array, "var", ["vel_1"], np.abs)


@FieldGetter.register
def vt_abs(bmdat: BaseMusicData) -> BigArray:
    """Absolute vt."""
    return DerivedFieldArray(bmdat.big_array, "var", ["vel_2"], np.abs)


@FieldGetter.register
def vr_normalized(bmdat: BaseMusicData) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_1**2 / (vel_1**2 + vel_2**2)),
    )


@FieldGetter.register
def vt_normalized(bmdat: BaseMusicData) -> BigArray:
    """Radial velocity normalized by velocity amplitude."""
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["vel_1", "vel_2"],
        lambda vel_1, vel_2: np.sqrt(vel_2**2 / (vel_1**2 + vel_2**2)),
    )


@FieldGetter.register
def temp(bmdat: BaseMusicData) -> BigArray:
    """Temperature."""
    return bmdat.eos.temperature(bmdat.big_array)


@FieldGetter.register
def press(bmdat: BaseMusicData) -> BigArray:
    """Pressure."""
    return bmdat.eos.pressure(bmdat.big_array)


@FieldGetter.register
def pgas(bmdat: BaseMusicData) -> BigArray:
    """Gas pressure."""
    return bmdat.eos.pressure_gas(bmdat.big_array)


@FieldGetter.register
def adiab_grad(bmdat: BaseMusicData) -> BigArray:
    """Adiabatic gradient dlnT / dlnP as constant S."""
    return bmdat.eos.adiab_grad(bmdat.big_array)


@FieldGetter.register
def enthalpy(bmdat: BaseMusicData) -> BigArray:
    """Enthalpy."""
    return bmdat.eos.enthalpy(bmdat.big_array)


@FieldGetter.register
def gamma(bmdat: BaseMusicData) -> BigArray:
    """Heat capacity ratio."""
    return bmdat.eos.gamma(bmdat.big_array)


@FieldGetter.register
def gamma1(bmdat: BaseMusicData) -> BigArray:
    """First adiabatic index dlnP / dlnD at constant S."""
    return bmdat.eos.gamma_1(bmdat.big_array)


@FieldGetter.register
def enthalpy_flux(bmdat: BaseMusicData) -> BigArray:
    temp_field = bmdat.field["temp"].array()
    # FIXME: handle time axis
    temp_prof = bmdat.rprof["temp"].array()
    delta_temp = temp_field - temp_prof[:, np.newaxis]
    c_p = bmdat.eos.heat_cap_p(bmdat.big_array).array()
    return DerivedFieldArray(
        bmdat.big_array,
        "var",
        ["density", "vel_1"],
        lambda rho, v_r: rho * c_p * v_r * delta_temp,
    )


@ProfGetter.register
def vrms(bmdat: BaseMusicData) -> BigArray:
    """Vrms defined as vrms(r, t) = sqrt(mean_theta(v2))."""
    return bmdat.rprof["vel_square"].sqrt()


@dataclass(frozen=True)
class _DataGetter:
    _bmdat: BaseMusicData
    _fetcher: Type[DataFetcher]

    def __getitem__(self, var: str) -> BigArray:
        return self._fetcher(var)(self._bmdat)
