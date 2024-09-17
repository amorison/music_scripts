from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import h5py
import numpy as np
import pymusic.spec as pms
from pymusic.big_array import CachedArray, FFTPowerSpectrumArray, SphHarm1DArray
from pymusic.math import SphericalMidpointQuad1D
from scipy.integrate import cumulative_trapezoid

from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .config import Config
    from .lyon1d import Lyon1dStruc


@dataclass(frozen=True)
class _SpecData:
    spectrum: NDArray[np.floating]
    freqs: NDArray[np.floating]
    rads: NDArray[np.floating]
    ells: NDArray[np.integer]


@dataclass(frozen=True)
class SpectrumAnalysis:
    spectrum_h5: Path
    struc: Lyon1dStruc

    @cached_property
    def _data(self) -> _SpecData:
        with h5py.File(self.spectrum_h5, "r") as h5f:
            return _SpecData(
                spectrum=h5f.get("spectrum")[()],
                freqs=h5f.get("frequency")[()],
                rads=h5f.get("radius")[()],
                ells=h5f.get("ell")[()],
            )

    @property
    def spectrum(self) -> NDArray[np.floating]:
        """Spectrum indexed by (freq, rad, ell)."""
        return self._data.spectrum

    @property
    def freqs(self) -> NDArray[np.floating]:
        """Frequencies."""
        return self._data.freqs

    @cached_property
    def ang_freqs(self) -> NDArray[np.floating]:
        """Angular frequencies."""
        return 2 * np.pi * self.freqs

    @property
    def rads(self) -> NDArray[np.floating]:
        """Radial positions."""
        return self._data.rads

    @property
    def ells(self) -> NDArray[np.integer]:
        """Harmonic degrees."""
        return self._data.ells

    @cached_property
    def density(self) -> NDArray[np.floating]:
        """Density profile."""
        return np.interp(self.rads, self.struc.radius, self.struc.density)

    @cached_property
    def bv_ang_freq(self) -> NDArray[np.floating]:
        """Brunt-Vaisala angular frequency profile."""
        return np.interp(self.rads, self.struc.radius, self.struc.bv_ang_freq)

    @cached_property
    def k_h(self) -> NDArray[np.floating]:
        """Horizontal wavenumber, indexed by (rad, ell)."""
        ell = np.sqrt(self.ells * (self.ells + 1))
        return ell[np.newaxis, :] / self.rads[:, np.newaxis]

    @cached_property
    def luminosity(self) -> NDArray[np.floating]:
        """Wave luminosity, indexed by (freq, rad, ell)."""
        # indexed by (rad)
        rad_kh = 4 * np.pi * self.rads**2 * self.density
        # indexed by (rad, ell)
        rad_kh = rad_kh[:, np.newaxis] / self.k_h
        # indexed by (freq, rad)
        freq = np.sqrt(
            np.maximum(
                self.bv_ang_freq[np.newaxis, :] ** 2
                - self.ang_freqs[:, np.newaxis] ** 2,
                0.0,
            )
        )
        return rad_kh[np.newaxis, :, :] * freq[:, :, np.newaxis] * self.spectrum

    def theory_with_excitation_radius(self, r_e: float) -> LinearTheory:
        return LinearTheory(self, r_e * self.struc.r_star)


@dataclass(frozen=True)
class LinearTheory:
    spec: SpectrumAnalysis
    r_e: float

    @cached_property
    def _ire(self) -> int:
        return int(np.searchsorted(self.spec.struc.radius, self.r_e)) + 1

    @property
    def radius(self) -> NDArray[np.floating]:
        return self.spec.struc.radius[self._ire :]

    @property
    def density(self) -> NDArray[np.floating]:
        return self.spec.struc.density[self._ire :]

    @property
    def bv_ang_freq(self) -> NDArray[np.floating]:
        return self.spec.struc.bv_ang_freq[self._ire :]

    @property
    def bv_ang_freq_thermal(self) -> NDArray[np.floating]:
        return self.spec.struc.bv_ang_freq_thermal[self._ire :]

    @property
    def diffusivity(self) -> NDArray[np.floating]:
        return self.spec.struc.heat_diffusivity[self._ire :]

    @cached_property
    def k_h(self) -> NDArray[np.floating]:
        """Wavenumber, indexed by (rad, ell)."""
        ell = np.sqrt(self.spec.ells * (self.spec.ells + 1))
        return ell[np.newaxis, :] / self.radius[:, np.newaxis]

    @cached_property
    def damping(self) -> NDArray[np.floating]:
        """Damping coefficient, indexed by (freq, rad, ell)."""
        # indexed by (freq, rad)
        freq = np.sqrt(
            np.maximum(
                self.bv_ang_freq[np.newaxis, :] ** 2
                - self.spec.ang_freqs[:, np.newaxis] ** 2,
                1e-12,
            )
        )
        # indexed by (freq, rad, ell)
        integrand = (
            self.diffusivity[np.newaxis, :, np.newaxis]
            * self.k_h[np.newaxis, :, :] ** 3
            * self.bv_ang_freq[np.newaxis, :, np.newaxis] ** 2
            * self.bv_ang_freq_thermal[np.newaxis, :, np.newaxis] ** 2
            / self.spec.ang_freqs[:, np.newaxis, np.newaxis] ** 4
            / freq[:, :, np.newaxis]
        )
        return cumulative_trapezoid(
            y=integrand,
            x=self.radius,
            initial=0.0,
            axis=1,
        )

    @cached_property
    def _ire_spec(self) -> int:
        return int(np.searchsorted(self.spec.rads, self.radius[0]))

    @cached_property
    def wave_lum(self) -> NDArray[np.floating]:
        """Predicted wave luminosity."""
        lwave0 = self.spec.luminosity[:, (self._ire_spec,), :]
        return lwave0 * np.exp(-self.damping)

    @cached_property
    def v_r_no_damping(self) -> NDArray[np.floating]:
        """Predicted v_r without damping."""
        # indexed by (freq, rad, ell)
        v_r0 = np.sqrt(
            np.max(
                self.spec.spectrum[:, self._ire_spec : self._ire_spec + 5, :],
                axis=1,
                keepdims=True,
            )
        )
        # indexed by (freq, rad)
        freq = self.spec.ang_freqs[:, np.newaxis]
        freq_ratio = (self.bv_ang_freq[np.newaxis, :] ** 2 - freq**2) / (
            self.bv_ang_freq[0] ** 2 - freq**2
        )
        return (
            v_r0
            * (self.radius[0] / self.radius[np.newaxis, :, np.newaxis]) ** (3 / 2)
            * (self.density[0] / self.density[np.newaxis, :, np.newaxis]) ** (1 / 2)
            * freq_ratio[:, :, np.newaxis] ** (-1 / 4)
        )

    @cached_property
    def v_r(self) -> NDArray[np.floating]:
        """Predicted v_r with damping."""
        return self.v_r_no_damping * np.exp(-self.damping / 2)


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    assert len(conf.core.dumps) == 1
    dumps = conf.core.dumps[0]
    assert isinstance(dumps, slice)

    subsim = mdat[dumps]
    fld = subsim.field[conf.igw.field]

    times = np.array(fld.labels_along_axis("time"))
    d_time = np.mean(np.diff(times))

    fft = pms.NuFFT1D(
        window=pms.NormalizedWindow(
            window=pms.BlackmanWindow(),
            normalization=pms.PreservePower(),
        ),
        sampling_period=d_time,
        spacing_tol=0.1,
    )

    sh_xform = pms.SphericalHarmonicsTransform1D(
        quad=SphericalMidpointQuad1D(theta_grid=subsim.grid.grids[1]),
        ell_max=max(conf.igw.ells),
        tol=0.15,
    )

    power_spec = CachedArray(
        FFTPowerSpectrumArray(
            array=SphHarm1DArray(
                array=fld,
                sph_harm_xform=sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=conf.igw.ells,
            ).slabbed("time", 200),
            fft1d=fft,
            axis="time",
            freq_axis="freq",
        ).slabbed("x1", 256)
    )

    ells = power_spec.labels_along_axis("ell")
    freqs = power_spec.labels_along_axis("freq")
    rads = power_spec.labels_along_axis("x1")

    ells_str = "_".join(map(str, conf.igw.ells))
    dstart, dstop, dstep = dumps.indices(len(mdat))
    oname = (
        f"spectrum_{conf.igw.field}_ell_{ells_str}_dumps_{dstart}:{dstop}:{dstep}.h5"
    )

    with h5py.File(oname, "w") as hf:
        hf.create_dataset("spectrum", data=power_spec.array())
        hf.create_dataset("radius", data=rads)
        hf.create_dataset("ell", data=ells)
        hf.create_dataset("frequency", data=freqs)
