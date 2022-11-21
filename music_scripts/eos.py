from __future__ import annotations

from abc import ABC, abstractmethod
import typing

from pymusic.big_array import DerivedFieldArray
import music_mesa_tables as mmt

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


class EoS(ABC):
    """Equation of state."""

    @abstractmethod
    def _derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        """Build an array with the desired state variable."""

    def temperature(self, array: BigArray) -> BigArray:
        """Compute temperature from MUSIC state."""
        return self._derive_arr(array, mmt.StateVar.LogTemperature)

    def pressure(self, array: BigArray) -> BigArray:
        """Compute pressure from MUSIC state."""
        return self._derive_arr(array, mmt.StateVar.LogPressure)

    def entropy(self, array: BigArray) -> BigArray:
        """Compute entropy from MUSIC state."""
        return self._derive_arr(array, mmt.StateVar.LogEntropy)


class MesaCstMetalEos(EoS):
    """MESA EoS at constant metallicity."""

    def __init__(self, metallicity: float):
        self._eos = mmt.CstMetalEos(metallicity)

    def _derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(
            rho: NDArray, e_int: NDArray, he_frac: NDArray
        ) -> NDArray:
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            return 10**state.compute(var)
        return DerivedFieldArray(
            array, "var", ["density", "e_spec_int", "scalar_1"], calculator)


class MesaCstCompoEos(EoS):
    """MESA EoS at constant metallicity and helium fraction."""

    def __init__(self, metallicity: float, he_frac: float):
        self._eos = mmt.CstCompoEos(metallicity, he_frac)

    def _derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray) -> NDArray:
            state = mmt.CstCompoState(self._eos, rho, e_int)
            return 10**state.compute(var)
        return DerivedFieldArray(array, "var", ["density", "e_spec_int"], calculator)
