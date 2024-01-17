from __future__ import annotations

import typing

import numpy as np

from .globdiag import tau_conv
from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from typing import Union

    from .config import Config
    from .musicdata import _SnapsView


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    print("Run in:", mdat.path)
    view: Union[MusicData, _SnapsView] = (
        mdat if conf.core.dumps == () else mdat[conf.core.dumps]
    )
    rstar = mdat.prof1d.params["rad_surf"]
    print(f"rstar: {rstar:e}")

    rfaces = mdat.prof1d.profs["r_grid"].values
    rcore = mdat.prof1d.params["rcore/rtot"]
    renv = mdat.prof1d.params["renv/rtot"]

    rcenters = mdat.prof1d.profs["radc"].values[:-1]
    press = mdat.prof1d.profs["P"].values[:-1]
    press_scale_height = -np.diff(rcenters) / np.diff(np.log(press))

    print("rin/rstar:", rfaces[0] / rstar)
    if rcore > 0:
        print("rcore/rstar:", rcore)
        rc_dim = mdat.prof1d.params["rcore"]
        print(
            "Hp(rcore)/rstar:",
            np.interp(rc_dim, rfaces[1:-1], press_scale_height) / rstar,
        )
    if renv > 0:
        print("renv/rstar:", renv)
        renv_dim = mdat.prof1d.params["renv"]
        print(
            "Hp(renv)/rstar:",
            np.interp(renv_dim, rfaces[1:-1], press_scale_height) / rstar,
        )
    print("rout/rstar:", rfaces[-1] / rstar)

    # FIXME: duration needs to be computed over the view
    # print("dump 1 at t={:e}".format(t0 := mdat[1].dump.time))
    # print("dump {} at t={:e}".format(mdat[-1].idump, tf := mdat[-1].dump.time))
    # print("duration: {:e}".format(tf - t0))

    if conf.info.tconv:
        tconv = tau_conv(view)
        print(f"Convective timescale: {tconv:e}")
