from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from tomso.fgong import FGONG, load_fgong

from .constants import STEFAN_BOLTZMANN

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class FgongModel:
    fgong_file: Path

    @cached_property
    def _fgong(self) -> FGONG:
        return load_fgong(str(self.fgong_file))

    @cached_property
    def r_star(self) -> float:
        return self._fgong.R

    @cached_property
    def luminosity_tot(self) -> float:
        return self._fgong.L

    @cached_property
    def mass_tot(self) -> float:
        return self._fgong.M

    @cached_property
    def bv_ang_freq(self) -> NDArray[np.floating]:
        n2 = np.flip(self._fgong.N2)
        return np.sqrt(np.maximum(n2, 0.0))

    @cached_property
    def radius(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.r)

    @cached_property
    def xmass(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.m)

    @cached_property
    def h_frac(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.X)

    @cached_property
    def metal_frac(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.Z)

    @cached_property
    def density(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.rho)

    @cached_property
    def bv_freq2(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.N2)

    @cached_property
    def temperature(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.T)

    @cached_property
    def pressure(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.P)

    @cached_property
    def luminosity(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.L_r)

    @cached_property
    def e_nuclear(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.epsilon)

    @cached_property
    def opacity(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.kappa)

    @cached_property
    def heat_capacity_press(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.cp)

    @cached_property
    def conductivity(self) -> NDArray[np.floating]:
        return (16 * STEFAN_BOLTZMANN * self.temperature**3) / (
            3 * self.opacity * self.density
        )

    @cached_property
    def diffusivity(self) -> NDArray[np.floating]:
        return self.conductivity / (self.density * self.heat_capacity_press)

    @cached_property
    def pressure_scale_height(self) -> NDArray[np.floating]:
        return np.flip(self._fgong.Hp)
