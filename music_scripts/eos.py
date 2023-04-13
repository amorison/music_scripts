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
    def derive_arr(self, array: BigArray, var: mmt.StateVar) -> BigArray:
        """Build an array with the desired state variable."""


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
