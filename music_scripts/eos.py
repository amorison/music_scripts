from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import music_mesa_tables as mmt
from pymusic.big_array import DerivedFieldArray

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


class EoS(ABC):
    """Equation of state."""

    @abstractmethod
    def temperature(self, array: BigArray) -> BigArray:
        ...

    @abstractmethod
    def pressure(self, array: BigArray) -> BigArray:
        ...

    @abstractmethod
    def pressure_gas(self, array: BigArray) -> BigArray:
        ...

    @abstractmethod
    def enthalpy(self, array: BigArray) -> BigArray:
        ...

    @abstractmethod
    def adiab_grad(self, array: BigArray) -> BigArray:
        """Adiabatic gradient dlnT / dlnP as constant S."""

    @abstractmethod
    def heat_cap_p(self, array: BigArray) -> BigArray:
        """Heat capacity at constant pressure."""


@dataclass(frozen=True)
class MesaCstMetalEos(EoS):
    """MESA EoS at constant metallicity."""

    metallicity: float
    he_scalar: int

    @cached_property
    def _eos(self) -> mmt.CstMetalEos:
        return mmt.CstMetalEos(self.metallicity)

    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray, he_frac: NDArray) -> NDArray:
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            return state.compute(var)

        return DerivedFieldArray(
            array,
            "var",
            ["density", "e_int_spec", f"scalar_{self.he_scalar}"],
            calculator,
        )

    def temperature(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogTemperature).apply(
            lambda v: 10**v
        )

    def pressure(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogPressure).apply(lambda v: 10**v)

    def pressure_gas(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogPgas).apply(lambda v: 10**v)

    def enthalpy(self, array: BigArray) -> BigArray:
        def calc(rho: NDArray, e_int: NDArray, he_frac: NDArray) -> NDArray:
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            press = 10 ** state.compute(mmt.StateVar.LogPressure)
            return e_int + press / rho

        return DerivedFieldArray(
            array,
            "var",
            ["density", "e_int_spec", f"scalar_{self.he_scalar}"],
            calc,
        )

    def adiab_grad(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.DTempDPresScst)

    def heat_cap_p(self, array: BigArray) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray, he_frac: NDArray) -> NDArray:
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            temp = 10 ** state.compute(mmt.StateVar.LogTemperature)
            c_v = e_int / (temp * state.compute(mmt.StateVar.DTempDEnerDcst))
            return c_v * state.compute(mmt.StateVar.Gamma)

        return DerivedFieldArray(
            array,
            "var",
            ["density", "e_int_spec", f"scalar_{self.he_scalar}"],
            calculator,
        )


@dataclass(frozen=True)
class MesaCstCompoEos(EoS):
    """MESA EoS at constant metallicity and helium fraction."""

    metallicity: float
    he_frac: float

    @cached_property
    def _eos(self) -> mmt.CstCompoEos:
        return mmt.CstCompoEos(self.metallicity, self.he_frac)

    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray) -> NDArray:
            state = mmt.CstCompoState(self._eos, rho, e_int)
            return state.compute(var)

        return DerivedFieldArray(array, "var", ["density", "e_int_spec"], calculator)

    def temperature(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogTemperature).apply(
            lambda v: 10**v
        )

    def pressure(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogPressure).apply(lambda v: 10**v)

    def pressure_gas(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.LogPgas).apply(lambda v: 10**v)

    def enthalpy(self, array: BigArray) -> BigArray:
        def calc(rho: NDArray, e_int: NDArray) -> NDArray:
            state = mmt.CstCompoState(self._eos, rho, e_int)
            press = 10 ** state.compute(mmt.StateVar.LogPressure)
            return e_int + press / rho

        return DerivedFieldArray(
            array,
            "var",
            ["density", "e_int_spec"],
            calc,
        )

    def adiab_grad(self, array: BigArray) -> BigArray:
        return self.derive_arr(array, mmt.StateVar.DTempDPresScst)

    def heat_cap_p(self, array: BigArray) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray) -> NDArray:
            state = mmt.CstCompoState(self._eos, rho, e_int)
            temp = 10 ** state.compute(mmt.StateVar.LogTemperature)
            c_v = e_int / (temp * state.compute(mmt.StateVar.DTempDEnerDcst))
            return c_v * state.compute(mmt.StateVar.Gamma)

        return DerivedFieldArray(array, "var", ["density", "e_int_spec"], calculator)


@dataclass(frozen=True)
class IdealGasMix2(EoS):
    """Mix of two ideal gases."""

    gamma1: float
    gamma2: float
    mu1: float
    mu2: float
    c1_scalar: int
    rgas: float = 8.314462618e7  # cgs

    @cached_property
    def _c1var(self) -> str:
        return f"scalar_{self.c1_scalar}"

    def _mu(self, c1: NDArray) -> NDArray:
        return 1 / (c1 / self.mu1 + (1 - c1) / self.mu2)

    def _gm1(self, c1: NDArray) -> NDArray:
        """gamma - 1 of mix."""
        n1 = c1 / self.mu1
        x1 = n1 / (n1 + (1 - c1) / self.mu2)
        inv_gm1 = x1 / (self.gamma1 - 1) + (1 - x1) / (self.gamma2 - 1)
        return 1 / inv_gm1

    def _adgrad(self, c1: NDArray) -> NDArray:
        gm1 = self._gm1(c1)
        return gm1 / (gm1 + 1)

    def _cp(self, c1: NDArray) -> NDArray:
        gm1 = self._gm1(c1)
        return (gm1 + 1) * self.rgas / (gm1 * self._mu(c1))

    def temperature(self, array: BigArray) -> BigArray:
        return DerivedFieldArray(
            array,
            "var",
            ["e_int_spec", self._c1var],
            lambda e_int, c1: self._gm1(c1) * e_int * self._mu(c1) / self.rgas,
        )

    def pressure(self, array: BigArray) -> BigArray:
        return DerivedFieldArray(
            array,
            "var",
            ["density", "e_int_spec", self._c1var],
            lambda rho, e_int, c1: self._gm1(c1) * rho * e_int,
        )

    def pressure_gas(self, array: BigArray) -> BigArray:
        return self.pressure(array)

    def enthalpy(self, array: BigArray) -> BigArray:
        return DerivedFieldArray(
            array,
            "var",
            ["e_int_spec", self._c1var],
            lambda e_int, c1: (self._gm1(c1) + 1.0) * e_int,
        )

    def adiab_grad(self, array: BigArray) -> BigArray:
        return DerivedFieldArray(array, "var", [self._c1var], self._adgrad)

    def heat_cap_p(self, array: BigArray) -> BigArray:
        return DerivedFieldArray(array, "var", [self._c1var], self._cp)
