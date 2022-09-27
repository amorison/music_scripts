from __future__ import annotations

from abc import ABC, abstractmethod
import typing

from pymusic.big_array import DerivedFieldArray
import music_mesa_tables as mmt

if typing.TYPE_CHECKING:
    from pymusic.big_array import BigArray


class EoS(ABC):
    """Equation of state."""

    @abstractmethod
    def temperature(self, array: BigArray) -> BigArray:
        """Compute temperature from MUSIC state."""

    @abstractmethod
    def pressure(self, array: BigArray) -> BigArray:
        """Compute pressure from MUSIC state."""


class MesaCstMetalEos(EoS):
    """MESA EoS at constant metallicity."""

    def __init__(self, metallicity: float):
        self._eos = mmt.CstMetalEos(metallicity)

    def _derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho, e_int, he_frac):
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            return 10**state.compute(var)
        return DerivedFieldArray(
            array, "var", ["rho", "e_int", "scalar_1"], calculator)

    def temperature(self, array: BigArray) -> BigArray:
        return self._derive_arr(array, mmt.StateVar.LogTemperature)

    def pressure(self, array: BigArray) -> BigArray:
        return self._derive_arr(array, mmt.StateVar.LogPressure)


class MesaCstCompoEos(EoS):
    """MESA EoS at constant metallicity and helium fraction."""

    def __init__(self, metallicity: float, he_frac: float):
        self._eos = mmt.CstCompoEos(metallicity, he_frac)

    def _derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho, e_int):
            state = mmt.CstCompoState(self._eos, rho, e_int)
            return 10**state.compute(var)
        return DerivedFieldArray(array, "var", ["rho", "e_int"], calculator)

    def temperature(self, array: BigArray) -> BigArray:
        return self._derive_arr(array, mmt.StateVar.LogTemperature)

    def pressure(self, array: BigArray) -> BigArray:
        return self._derive_arr(array, mmt.StateVar.LogPressure)
