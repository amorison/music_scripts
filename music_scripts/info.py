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
    rcenter = mdat.prof1d.profs["radc"].values[:-1]
    rcore = mdat.prof1d.params["rcore/rtot"]
    renv = mdat.prof1d.params["renv/rtot"]

    delta_r = np.mean(np.diff(rfaces))
    press = mdat.prof1d.profs["P"].values[:-1]
    density = mdat.prof1d.profs["rho"].values[:-1]
    gravity = mdat.prof1d.profs["g_used"].values[:-1]
    press_scale_height = press / (density * gravity)

    print("rin/rstar:", rfaces[0] / rstar)
    if rcore > 0:
        print("rcore/rstar:", rcore)
        rc_dim = mdat.prof1d.params["rcore"]
        hp_core = np.interp(rc_dim, rcenter, press_scale_height)
        print(f"Hp(rcore): {hp_core:e}")
        print("Hp(rcore)/rstar:", hp_core / rstar)
        print("Hp(rcore)/delta_r:", hp_core / delta_r)
    if renv > 0:
        print("renv/rstar:", renv)
        renv_dim = mdat.prof1d.params["renv"]
        hp_env = np.interp(renv_dim, rcenter, press_scale_height)
        print(f"Hp(renv): {hp_env:e}")
        print("Hp(renv)/rstar:", hp_env / rstar)
        print("Hp(renv)/delta_r:", hp_env / delta_r)
    print("rout/rstar:", rfaces[-1] / rstar)

    # FIXME: duration needs to be computed over the view
    # print("dump 1 at t={:e}".format(t0 := mdat[1].dump.time))
    # print("dump {} at t={:e}".format(mdat[-1].idump, tf := mdat[-1].dump.time))
    # print("duration: {:e}".format(tf - t0))

    if conf.info.tconv:
        tconv = tau_conv(view)
        print(f"Convective timescale: {tconv:e}")
