from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import music_mesa_tables as mmt
from pymusic.big_array import DerivedFieldArray

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


class EoS(ABC):
    """Equation of state."""

    @abstractmethod
    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        """Build an array with the desired state variable."""


class MesaCstMetalEos(EoS):
    """MESA EoS at constant metallicity."""

    def __init__(self, metallicity: float):
        self._eos = mmt.CstMetalEos(metallicity)

    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray, he_frac: NDArray) -> NDArray:
            state = mmt.CstMetalState(self._eos, he_frac, rho, e_int)
            return state.compute(var)

        return DerivedFieldArray(
            array, "var", ["density", "e_int_spec", "scalar_1"], calculator
        )


class MesaCstCompoEos(EoS):
    """MESA EoS at constant metallicity and helium fraction."""

    def __init__(self, metallicity: float, he_frac: float):
        self._eos = mmt.CstCompoEos(metallicity, he_frac)

    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        def calculator(rho: NDArray, e_int: NDArray) -> NDArray:
            state = mmt.CstCompoState(self._eos, rho, e_int)
            return state.compute(var)

        return DerivedFieldArray(array, "var", ["density", "e_int_spec"], calculator)
