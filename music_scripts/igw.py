from __future__ import annotations

import typing

import h5py
import numpy as np
from pymusic.big_array import CachedArray, FFTPowerSpectrumArray, SphHarm1DArray
from pymusic.math import SphericalMidpointQuad1D
from pymusic.spec import BlackmanWindow, NuFFT1D, SphericalHarmonicsTransform1D

from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    assert len(conf.core.dumps) == 1
    dumps = conf.core.dumps[0]
    assert isinstance(dumps, slice)

    subsim = mdat[dumps]
    fld = subsim.field[conf.igw.field]

    times = np.array(fld.labels_along_axis("time"))
    d_time = np.mean(np.diff(times))

    fft = NuFFT1D(window=BlackmanWindow(), sampling_period=d_time, spacing_tol=0.1)

    sh_xform = SphericalHarmonicsTransform1D(
        quad=SphericalMidpointQuad1D(theta_grid=subsim.grid.grids[1]),
        ell_max=max(conf.igw.ells),
        tol=0.15,
    )

    power_spec = CachedArray(
        FFTPowerSpectrumArray(
            SphHarm1DArray(
                array=fld,
                sph_harm_xform=sh_xform,
                theta_axis="x2",
                ell_axis="ell",
                ells=conf.igw.ells,
            ).slabbed("time", 200),
            fft,
            "time",
            "freq",
        ).slabbed("x1", 256)
    )

    ells = power_spec.labels_along_axis("ell")
    freqs = power_spec.labels_along_axis("freq")
    rads = power_spec.labels_along_axis("x1")

    ells_str = "_".join(map(str, conf.igw.ells))
    dstart, dstop, dstep = dumps.indices(len(mdat))
    oname = f"igw_{conf.igw.field}_ell_{ells_str}_dumps_{dstart}:{dstop}:{dstep}.h5"

    with h5py.File(oname, "w") as hf:
        hf.create_dataset("spectrum", data=power_spec.array())
        hf.create_dataset("radius", data=rads)
        hf.create_dataset("ell", data=ells)
        hf.create_dataset("frequency", data=freqs)
